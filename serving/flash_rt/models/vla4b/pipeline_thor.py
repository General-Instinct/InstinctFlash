"""FlashRT — LingBot-VLA-4B Thor SM110 pipeline (B=1, static FP8).

Three forward stages, all pure fvk pointer ops (no torch inside the
forward functions — the frontend owns buffers and graph capture):

    vit_forward         — Qwen2.5-VL ViT, 32 blocks (window/fullatt) + merger
    lm_prefill_forward  — Qwen2.5-VL LM, 36 layers, KV-cache fill only
    expert_forward      — action expert, 10 Euler steps × 36 layers,
                          joint attention against the shared KV slab

Numerics contract (ground truth: lingbotvla modeling_lingbot_vla.py):
    * RMSNorm eps 1e-6 everywhere; LM/ViT norm weights folded into the
      consuming GEMM weights at load time (fp32 fold) → noweight RMS
      kernels here.
    * Expert AdaRMS: style tables carry scale = weight*(1+gamma(t))-1,
      shift = beta(t), gate = 1 → the pi05 adarms kernels compute
      norm*(1+scale)+shift exactly.
    * FFN is true SiLU (silu_mul_split), NOT the Gemma GELU-tanh; gate
      and up run as separate GEMMs.
    * 1D rope theta 1e4, split-half (rope_rotate_half kernel) for
      LM + expert; ViT 2D rope tables from models/vla4b/rope_table.py.
    * GQA 16q/2kv is handled natively by FA2 (q head h → kv head h//8,
      identical to the reference einops repeat); KV cache stays 2-head.
    * Suffix state row must not attend action tokens → two FA2 calls
      (state row kv=Se+1, action rows kv=Se+51).
    * Prefill is fully bidirectional (vlm_causal=false ckpt config);
      last LM layer is KV-write-only (LM output unused).

Calibration: ``calibrate=True`` replaces every static quantize with
``quantize_fp8_device_fp16`` writing amax/448 into the same act-scale
slot the static path reads — so scales land in pointer-stable device
memory and the captured graph never needs recapture after
recalibration.
"""

import math


# ════════════════════════════════════════════════════════════════
# FA2 attention adapter
# ════════════════════════════════════════════════════════════════

class Vla4bFa2Attn:
    """Thin dispatch over the vendored FA2 forward for the three sites.

    Pipeline-owned memory model: all pointers are ints supplied by the
    frontend at construction. ``num_sms=0`` → no splitkv → single
    kernel launch per call (CUDA-graph safe).
    """

    def __init__(self, fa2_mod, *, vit, lm, suffix):
        """
        Args:
            fa2_mod: the ``flash_rt.flash_rt_fa2`` module.
            vit: dict(q, k, v, o, lse — int ptrs; nv — num images)
                q/k/v are the separated [768, 1280] fp16 buffers.
            lm: dict(q, o, lse, Kc, Vc, layer_stride_elems)
                Kc/Vc are the 2-head cache bases; per-layer offset =
                l * layer_stride_elems (fp16 elements).
            suffix: dict(q, o, lse1, lse2, Kc, Vc, layer_stride_elems)
        """
        self._fa2 = fa2_mod
        self._vit = dict(vit)
        self._lm = dict(lm)
        self._sfx = dict(suffix)

    # ── ViT: window-batched or per-image fullatt ──
    def run_vit(self, *, full: bool, stream: int = 0):
        s = self._vit
        nv = int(s["nv"])
        if full:
            batch, seq = nv, 256
        else:
            batch, seq = nv * 4, 64
        D = 1280
        scale = 1.0 / math.sqrt(80.0)
        st = (seq * D, D, 80)
        self._fa2.fwd_fp16(
            s["q"], s["k"], s["v"], s["o"], s["lse"], 0, 0,
            batch, seq, seq, 16, 16, 80,
            st, st, st, st, scale, 0, stream)

    # ── LM prefill: bidirectional GQA self-attention ──
    def run_prefix(self, layer: int, *, se: int, stream: int = 0):
        s = self._lm
        kv_off = layer * int(s["layer_stride_elems"]) * 2  # bytes
        DQ, DKV, HD = 2048, 256, 128
        scale = 1.0 / math.sqrt(float(HD))
        self._fa2.fwd_fp16(
            s["q"], s["Kc"] + kv_off, s["Vc"] + kv_off, s["o"], s["lse"], 0, 0,
            1, se, se, 16, 2, HD,
            (se * DQ, DQ, HD), (se * DKV, DKV, HD), (se * DKV, DKV, HD),
            (se * DQ, DQ, HD), scale, 0, stream)

    # ── Suffix joint attention: state row + action rows ──
    def run_suffix(self, layer: int, *, se: int, stream: int = 0):
        s = self._sfx
        kv_off = layer * int(s["layer_stride_elems"]) * 2  # bytes
        DQ, DKV, HD = 2048, 256, 128
        scale = 1.0 / math.sqrt(float(HD))
        K = s["Kc"] + kv_off
        V = s["Vc"] + kv_off
        # state row: attends prefix + itself (kv rows 0..Se)
        sk1 = se + 1
        self._fa2.fwd_fp16(
            s["q"], K, V, s["o"], s["lse1"], 0, 0,
            1, 1, sk1, 16, 2, HD,
            (DQ, DQ, HD), (sk1 * DKV, DKV, HD), (sk1 * DKV, DKV, HD),
            (DQ, DQ, HD), scale, 0, stream)
        # action rows: attend everything (kv rows 0..Se+50)
        sk2 = se + 51
        qb = s["q"] + DQ * 2   # byte offset: skip state row
        ob = s["o"] + DQ * 2
        self._fa2.fwd_fp16(
            qb, K, V, ob, s["lse2"], 0, 0,
            1, 50, sk2, 16, 2, HD,
            (50 * DQ, DQ, HD), (sk2 * DKV, DKV, HD), (sk2 * DKV, DKV, HD),
            (50 * DQ, DQ, HD), scale, 0, stream)


# ════════════════════════════════════════════════════════════════
# Shared quantize-point helper
# ════════════════════════════════════════════════════════════════

def _quant_point(fvk, src_fp16, dst_fp8, slot_ptr, n, stream, calibrate):
    """FP16 buffer → FP8 with the act scale at ``slot_ptr``.

    calibrate=True: measure amax/448 into the slot AND quantize with it.
    """
    if calibrate:
        fvk.quantize_fp8_device_fp16(src_fp16, dst_fp8, slot_ptr, n, stream)
    else:
        fvk.quantize_fp8_static_fp16(src_fp16, dst_fp8, slot_ptr, n, stream)


# ════════════════════════════════════════════════════════════════
# ViT (32 blocks + merger)
# ════════════════════════════════════════════════════════════════

def vit_forward(fvk, ctx, bufs, weights, dims, stream=0, *, attn):
    """Qwen2.5-VL vision tower on the window-permuted token sequence.

    FULL FP16 tower (T2 recipe decision, gate_p4_ablate_*: ViT
    activation FP8 alone costs action_meandiff 0.041 vs the 0.006
    fp16 floor — Qwen2.5-VL vision outliers break per-tensor act
    quant; ViT fp16 adds only ~0.7GB weight traffic ≈ 3ms on Thor).
    LM + expert stay FP8 (measured ~free / moderate respectively).

    bufs (int ptrs):
        x        [S, D] fp16 — patch embeddings in (residual stream)
        xn       [S, D] fp16 — norm output
        qkv      [S, 3D] fp16
        q, k, v  [S, D] fp16 (separated for rope + FA2)
        attn_out [S, D] fp16
        fg       [S, D] fp16 (o/down GEMM output pre-residual)
        gate     [S, H] fp16
        up       [S, H] fp16
        hid      [S, H] fp16 (silu(gate)*up; zeroed then gate_res)
        m0       [S/4, D_m] fp16 (merger hidden)
        vis_emb  [S/4, D_enc] fp16 (merger output, window order)
        ones     [>=D] fp16 all-1.0 (noweight RMS — norm weights are
                 folded into the fp16 GEMM weights at load)
    weights (int ptrs / lists): qkv_w[L], qkv_b[L], o_w[L], o_b[L],
        gate_w[L], gate_b[L], up_w[L], up_b[L], down_w[L], down_b[L]
        (fp16 [K,N]), m0_w, m0_b, m2_w, m2_b, cos, sin ([S,80] fp16)
    dims: S, D, H (raw 3420 — fp16 GEMMs need no padding), L,
          fullatt (set), S_m, D_m, D_enc
    """
    S = dims["S"]; D = dims["D"]; H = dims["H"]; L = dims["L"]
    fullatt = dims["fullatt"]
    S_m = dims["S_m"]; D_m = dims["D_m"]; D_enc = dims["D_enc"]

    x = bufs["x"]; xn = bufs["xn"]; qkv = bufs["qkv"]
    q = bufs["q"]; k = bufs["k"]; v = bufs["v"]
    attn_out = bufs["attn_out"]; fg = bufs["fg"]
    gate = bufs["gate"]; up = bufs["up"]; hid = bufs["hid"]
    m0 = bufs["m0"]; vis_emb = bufs["vis_emb"]
    ones = bufs["ones"]
    cos = weights["cos"]; sin = weights["sin"]

    for l in range(L):
        # ── norm1 (folded) → QKV GEMM + bias ──
        fvk.rms_norm_fp16(x, ones, xn, S, D, 1e-6, stream)
        fvk.gmm_fp16(ctx, xn, weights["qkv_w"][l], qkv, S, 3 * D, D,
                     0.0, stream)
        fvk.add_bias_fp16(qkv, weights["qkv_b"][l], S, 3 * D, stream)

        # ── split + 2D rope + FA2 ──
        fvk.qkv_split_fp16(qkv, q, k, v, S, D, D, D, stream)
        fvk.rope_rotate_half_fp16(q, cos, sin, S, 16, 80, stream)
        fvk.rope_rotate_half_fp16(k, cos, sin, S, 16, 80, stream)
        attn.run_vit(full=(l in fullatt), stream=stream)

        # ── proj + bias + residual ──
        fvk.gmm_fp16(ctx, attn_out, weights["o_w"][l], fg, S, D, D,
                     0.0, stream)
        fvk.bias_residual_fp16(x, fg, weights["o_b"][l], S, D, stream)

        # ── norm2 (folded) → gate/up GEMMs + biases ──
        fvk.rms_norm_fp16(x, ones, xn, S, D, 1e-6, stream)
        fvk.gmm_fp16(ctx, xn, weights["gate_w"][l], gate, S, H, D,
                     0.0, stream)
        fvk.add_bias_fp16(gate, weights["gate_b"][l], S, H, stream)
        fvk.gmm_fp16(ctx, xn, weights["up_w"][l], up, S, H, D,
                     0.0, stream)
        fvk.add_bias_fp16(up, weights["up_b"][l], S, H, stream)

        # ── SiLU(gate)·up (fp16) → down GEMM + bias + residual ──
        fvk.silu_inplace_fp16(gate, S * H, stream)
        _zero_fp16(fvk, hid, S * H, stream)
        fvk.gate_res_fp16(up, gate, hid, S * H, stream)
        fvk.gmm_fp16(ctx, hid, weights["down_w"][l], fg, S, D, H,
                     0.0, stream)
        fvk.bias_residual_fp16(x, fg, weights["down_b"][l], S, D, stream)

    # ── merger: RMS (ln_q folded) → mlp0 → GELU → mlp2 (all fp16) ──
    fvk.rms_norm_fp16(x, ones, xn, S, D, 1e-6, stream)
    fvk.gmm_fp16(ctx, xn, weights["m0_w"], m0, S_m, D_m, D_m, 0.0, stream)
    fvk.add_bias_fp16(m0, weights["m0_b"], S_m, D_m, stream)
    fvk.gelu_inplace_fp16(m0, S_m * D_m, stream)
    fvk.gmm_fp16(ctx, m0, weights["m2_w"], vis_emb, S_m, D_enc, D_m,
                 0.0, stream)
    fvk.add_bias_fp16(vis_emb, weights["m2_b"], S_m, D_enc, stream)
    # NOTE: vis_emb rows are in window order; the frontend un-permutes
    # (index_select with the precomputed reverse_index) into enc_x.


import ctypes as _ctypes
_crt = _ctypes.CDLL("libcudart.so")


def _zero_fp16(fvk, ptr, n, stream):
    """Zero n fp16 elements (cudaMemsetAsync; CUDA-graph capturable)."""
    _crt.cudaMemsetAsync(_ctypes.c_void_p(ptr), 0,
                         _ctypes.c_size_t(n * 2), _ctypes.c_void_p(stream))


# ════════════════════════════════════════════════════════════════
# LM prefill (36 layers, KV fill; bidirectional attention)
# ════════════════════════════════════════════════════════════════

def lm_prefill_forward(fvk, bufs, weights, dims, stream=0, *, attn,
                       calibrate=False):
    """Qwen2.5-VL LM prefill over [vision(192) | lang] tokens.

    Fills the 2-head KV cache (rows 0..Se-1 of every layer). The LM
    hidden output is not consumed — the last layer computes only
    QKV + rope + KV write (same last-layer cut as the pi05 encoder).

    bufs: x (enc_x [Se_max, D]), x_fp8 (u8 scratch >= Se*H), qkv
          [Se, 2560], q [Se, 2048], attn_out [Se, 2048], fg [Se, D],
          gate [Se, H], up [Se, H], calib_hid [Se, H], ones [D]
    weights: qkv_w[L], qkv_b[L], o_w[L], gate_w[L], up_w[L], down_w[L],
             Kc, Vc (base ptrs), cos, sin ([Se,128] prefix tables),
             act_scales, w_scales
    dims: Se, D (2048), H (11008), L (36), total_keys, kv_row (256)
    """
    Se = dims["Se"]; D = dims["D"]; H = dims["H"]; L = dims["L"]
    total_keys = dims["total_keys"]; kv_row = dims["kv_row"]

    x = bufs["x"]; x_fp8 = bufs["x_fp8"]; qkv = bufs["qkv"]
    q = bufs["q"]; attn_out = bufs["attn_out"]; fg = bufs["fg"]
    gate = bufs["gate"]; up = bufs["up"]

    acts = weights["act_scales"]; ws = weights["w_scales"]
    cos = weights["cos"]; sin = weights["sin"]
    Kc = weights["Kc"]; Vc = weights["Vc"]

    def _norm_fp8_from(x_ptr, slot):
        if calibrate:
            fvk.rms_norm_fp16(x_ptr, bufs["ones"], bufs["calib_hid"],
                              Se, D, 1e-6, stream)
            fvk.quantize_fp8_device_fp16(bufs["calib_hid"], x_fp8, slot,
                                         Se * D, stream)
        else:
            fvk.rms_norm_fp8_noweight_fp16(x_ptr, x_fp8, Se, D, slot, stream)

    def _res_norm_fp8(slot):
        if calibrate:
            fvk.residual_add_fp16(x, fg, Se * D, stream)
            fvk.rms_norm_fp16(x, bufs["ones"], bufs["calib_hid"],
                              Se, D, 1e-6, stream)
            fvk.quantize_fp8_device_fp16(bufs["calib_hid"], x_fp8, slot,
                                         Se * D, stream)
        else:
            fvk.residual_add_rms_norm_fp8_noweight_fp16(
                x, fg, x_fp8, Se, D, slot, stream)

    # layer 0 norm
    _norm_fp8_from(x, acts + 0 * 4)

    for l in range(L):
        last = (l == L - 1)
        a_qkv = acts + (l * 4 + 0) * 4
        a_o   = acts + (l * 4 + 1) * 4
        a_gu  = acts + (l * 4 + 2) * 4
        a_dn  = acts + (l * 4 + 3) * 4
        w_qkv = ws + (l * 5 + 0) * 4
        w_o   = ws + (l * 5 + 1) * 4
        w_g   = ws + (l * 5 + 2) * 4
        w_u   = ws + (l * 5 + 3) * 4
        w_d   = ws + (l * 5 + 4) * 4

        # x_fp8 currently holds norm(x) for this layer (set by the
        # previous tail / the pre-loop call).
        fvk.fp8_gemm_descale_fp16(x_fp8, weights["qkv_w"][l], qkv,
                                  Se, 2560, D, a_qkv, w_qkv, stream)
        fvk.add_bias_fp16(qkv, weights["qkv_b"][l], Se, 2560, stream)

        # split → q buf + KV cache rows [0, Se) of layer l; rope both
        k_dst = Kc + (l * total_keys) * kv_row * 2
        v_dst = Vc + (l * total_keys) * kv_row * 2
        fvk.qkv_split_fp16(qkv, q, k_dst, v_dst, Se, 2048, 256, 256, stream)
        fvk.rope_rotate_half_fp16(q, cos, sin, Se, 16, 128, stream)
        fvk.rope_rotate_half_fp16(k_dst, cos, sin, Se, 2, 128, stream)

        if last:
            break  # KV written; LM output unused

        attn.run_prefix(l, se=Se, stream=stream)

        _quant_point(fvk, attn_out, x_fp8, a_o, Se * 2048, stream, calibrate)
        fvk.fp8_gemm_descale_fp16(x_fp8, weights["o_w"][l], fg,
                                  Se, D, 2048, a_o, w_o, stream)
        _res_norm_fp8(a_gu)

        fvk.fp8_gemm_descale_fp16(x_fp8, weights["gate_w"][l], gate,
                                  Se, H, D, a_gu, w_g, stream)
        fvk.fp8_gemm_descale_fp16(x_fp8, weights["up_w"][l], up,
                                  Se, H, D, a_gu, w_u, stream)
        if calibrate:
            fvk.silu_inplace_fp16(gate, Se * H, stream)
            _zero_fp16(fvk, bufs["calib_hid"], Se * H, stream)
            fvk.gate_res_fp16(up, gate, bufs["calib_hid"], Se * H, stream)
            fvk.quantize_fp8_device_fp16(bufs["calib_hid"], x_fp8, a_dn,
                                         Se * H, stream)
        else:
            fvk.silu_mul_split_fp8_fp16(gate, up, x_fp8, Se * H, a_dn, stream)
        fvk.fp8_gemm_descale_fp16(x_fp8, weights["down_w"][l], fg,
                                  Se, D, H, a_dn, w_d, stream)

        # tail: residual + next layer's norm → fp8
        _res_norm_fp8(acts + ((l + 1) * 4 + 0) * 4)


# ════════════════════════════════════════════════════════════════
# Action expert (10 steps × 36 layers, joint attention)
# ════════════════════════════════════════════════════════════════

def expert_forward(fvk, ctx, bufs, weights, dims, stream=0, *, attn,
                   calibrate=False):
    """10-step flow-matching loop of the 36-layer action expert.

    Style tables (frontend-precomputed, [steps, L, 51, 3*768] fp16):
        scale = adaW * (1 + gamma(t_s)) - 1, shift = beta(t_s), gate = 1

    bufs: x_t [50, 75] fp16 (noise in, actions out — Euler-accumulated),
          state_emb [1, 768] (precomputed once per infer by frontend),
          a_emb [50, 768], tmp [50, 768],
          x [51, 768] (suffix residual stream), x_fp8 (u8 >= 51*2752),
          gatebuf [51, 768], qkv [51, 2560], q [51, 2048],
          attn_out [51, 2048], fg [51, 768], gate [51, H], up [51, H],
          xn [51, 768] (final norm out), calib_hid [51, H], ones [768]
    weights: qkv_w[L], qkv_b[L], o_w[L], gate_w[L], up_w[L], down_w[L],
             style_attn, style_ffn (base ptrs), Kc, Vc, cos, sin
             ([51,128] suffix tables), final_norm_w,
             ain_w [75,768], ain_b, atm_a_w [768,768] (action half of
             action_time_mlp_in), t_contrib [10,768] (time half + bias),
             atm_out_w [768,768], atm_out_b,
             aout_w_dt [768,75] (action_out * dt), aout_b_dt,
             act_scales, w_scales
    dims: D (768), H (2752), L (36), steps (10), Se, total_keys,
          kv_row (256)
    """
    D = dims["D"]; H = dims["H"]; L = dims["L"]; steps = dims["steps"]
    Se = dims["Se"]; total_keys = dims["total_keys"]; kv_row = dims["kv_row"]
    S = 51
    D3 = 3 * D

    x_t = bufs["x_t"]; a_emb = bufs["a_emb"]; tmp = bufs["tmp"]
    x = bufs["x"]; x_fp8 = bufs["x_fp8"]; gatebuf = bufs["gatebuf"]
    qkv = bufs["qkv"]; q = bufs["q"]; attn_out = bufs["attn_out"]
    fg = bufs["fg"]; gate = bufs["gate"]; up = bufs["up"]; xn = bufs["xn"]

    acts = weights["act_scales"]; ws = weights["w_scales"]
    cos = weights["cos"]; sin = weights["sin"]
    Kc = weights["Kc"]; Vc = weights["Vc"]
    style_a = weights["style_attn"]; style_f = weights["style_ffn"]

    style_stride = S * D3 * 2  # bytes per (step, layer) slice

    def _adarms_fp8(x_ptr, style_ptr, slot):
        if calibrate:
            fvk.adarms_fp16(x_ptr, style_ptr, xn, gatebuf, S, D, stream)
            fvk.quantize_fp8_device_fp16(xn, x_fp8, slot, S * D, stream)
        else:
            fvk.fused_adarms_fp8_static_fp16(x_ptr, style_ptr, x_fp8,
                                             gatebuf, S, D, slot, stream)

    def _gate_res_adarms_fp8(style_ptr, slot):
        if calibrate:
            fvk.gate_res_fp16(fg, gatebuf, x, S * D, stream)
            fvk.adarms_fp16(x, style_ptr, xn, gatebuf, S, D, stream)
            fvk.quantize_fp8_device_fp16(xn, x_fp8, slot, S * D, stream)
        else:
            fvk.gate_res_adarms_fp8_static_fp16(fg, gatebuf, x, style_ptr,
                                                x_fp8, gatebuf, S, D, slot,
                                                stream)

    for s in range(steps):
        # ── suffix embedding rebuild ──
        # row 0: state token (constant across steps)
        fvk.gpu_copy(x, bufs["state_emb"], D * 2, stream)
        # rows 1..50: action_time_mlp(action_in(x_t), time_emb_s)
        fvk.gmm_fp16(ctx, x_t, weights["ain_w"], a_emb, 50, D, 75, 0.0, stream)
        fvk.add_bias_fp16(a_emb, weights["ain_b"], 50, D, stream)
        fvk.gmm_fp16(ctx, a_emb, weights["atm_a_w"], tmp, 50, D, D, 0.0, stream)
        fvk.add_bias_fp16(tmp, weights["t_contrib"] + s * D * 2, 50, D, stream)
        fvk.silu_inplace_fp16(tmp, 50 * D, stream)
        fvk.gmm_fp16(ctx, tmp, weights["atm_out_w"], x + D * 2, 50, D, D,
                     0.0, stream)
        fvk.add_bias_fp16(x + D * 2, weights["atm_out_b"], 50, D, stream)

        for l in range(L):
            a_qkv = acts + (l * 4 + 0) * 4
            a_o   = acts + (l * 4 + 1) * 4
            a_gu  = acts + (l * 4 + 2) * 4
            a_dn  = acts + (l * 4 + 3) * 4
            w_qkv = ws + (l * 5 + 0) * 4
            w_o   = ws + (l * 5 + 1) * 4
            w_g   = ws + (l * 5 + 2) * 4
            w_u   = ws + (l * 5 + 3) * 4
            w_d   = ws + (l * 5 + 4) * 4
            sa_ptr = style_a + (s * L + l) * style_stride
            sf_ptr = style_f + (s * L + l) * style_stride

            if l == 0:
                _adarms_fp8(x, sa_ptr, a_qkv)
            # else: tail of previous layer already produced x_fp8

            fvk.fp8_gemm_descale_fp16(x_fp8, weights["qkv_w"][l], qkv,
                                      S, 2560, D, a_qkv, w_qkv, stream)
            fvk.add_bias_fp16(qkv, weights["qkv_b"][l], S, 2560, stream)

            # split → q + suffix KV rows [Se, Se+51) of layer l; rope
            k_dst = Kc + (l * total_keys + Se) * kv_row * 2
            v_dst = Vc + (l * total_keys + Se) * kv_row * 2
            fvk.qkv_split_fp16(qkv, q, k_dst, v_dst, S, 2048, 256, 256,
                               stream)
            fvk.rope_rotate_half_fp16(q, cos, sin, S, 16, 128, stream)
            fvk.rope_rotate_half_fp16(k_dst, cos, sin, S, 2, 128, stream)

            attn.run_suffix(l, se=Se, stream=stream)

            _quant_point(fvk, attn_out, x_fp8, a_o, S * 2048, stream,
                         calibrate)
            fvk.fp8_gemm_descale_fp16(x_fp8, weights["o_w"][l], fg,
                                      S, D, 2048, a_o, w_o, stream)

            _gate_res_adarms_fp8(sf_ptr, a_gu)

            fvk.fp8_gemm_descale_fp16(x_fp8, weights["gate_w"][l], gate,
                                      S, H, D, a_gu, w_g, stream)
            fvk.fp8_gemm_descale_fp16(x_fp8, weights["up_w"][l], up,
                                      S, H, D, a_gu, w_u, stream)
            if calibrate:
                fvk.silu_inplace_fp16(gate, S * H, stream)
                _zero_fp16(fvk, bufs["calib_hid"], S * H, stream)
                fvk.gate_res_fp16(up, gate, bufs["calib_hid"], S * H, stream)
                fvk.quantize_fp8_device_fp16(bufs["calib_hid"], x_fp8, a_dn,
                                             S * H, stream)
            else:
                fvk.silu_mul_split_fp8_fp16(gate, up, x_fp8, S * H, a_dn,
                                            stream)
            fvk.fp8_gemm_descale_fp16(x_fp8, weights["down_w"][l], fg,
                                      S, D, H, a_dn, w_d, stream)

            if l < L - 1:
                sa_next = style_a + (s * L + l + 1) * style_stride
                _gate_res_adarms_fp8(sa_next, acts + ((l + 1) * 4 + 0) * 4)
            else:
                fvk.gate_res_fp16(fg, gatebuf, x, S * D, stream)

        # ── final plain RMS (FixQwen2RMSNorm) + Euler accumulate ──
        fvk.rms_norm_fp16(x, weights["final_norm_w"], xn, S, D, 1e-6, stream)
        # x_t += dt * (xn[1:] @ aout_w + aout_b)   (dt baked into weights)
        fvk.gmm_fp16(ctx, xn + D * 2, weights["aout_w_dt"], x_t,
                     50, 75, D, 1.0, stream)
        fvk.add_bias_fp16(x_t, weights["aout_b_dt"], 50, 75, stream)

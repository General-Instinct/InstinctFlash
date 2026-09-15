"""FlashRT — LingBot-VLA-V2-6B Thor SM110 pipeline (B=1, static; STAGE-1 SCAFFOLD).

Structure cloned from ``models/vla4b/pipeline_thor.py`` (the T2-V1
template) with V2 shapes filled in from the real checkpoint + config.
Three forward stages, all pure fvk pointer ops:

    vit_forward         — Qwen3-VL ViT, 24 blocks (full attention per
                          image) + deepstack taps [5,11,17] + merger
    lm_prefill_forward  — Qwen3-VL LM, 36 layers, CAUSAL prefill,
                          KV-cache fill + deepstack injection (L0-2)
    expert_forward      — MoE action expert, 10 Euler steps × 36
                          layers, joint attention against the shared
                          KV slab; routed MoE delegated to an injected
                          callable (STAGE-2: engine kernels)

Numerics contract (ground truth: lingbotvla modeling_lingbot_vla_v2.py
+ qwen2_action_expert.py + qwen3vl_in_vla.py; see
/home/ubuntu/iwm_distill/thor_t2v2/mapping_memo.md for file:line):
    * LM RMSNorm eps 1e-6, weights foldable into consuming GEMMs.
      ViT uses LayerNorm(eps 1e-6, affine+bias) — scaffold keeps
      explicit affine LN (``layer_norm_fp16``); γ/β-fold is a Stage-2
      micro-opt.
    * LM attention has per-head q_norm/k_norm (RMS over head_dim=128)
      AFTER the projections — applied with ``rms_norm_fp16`` treating
      (S·NH, 128) rows. The expert has NO q/k norms.
    * RoPE: Qwen3-VL interleaved 3D M-RoPE, theta 5e6 (tables from
      models/vla2/rope_table.py) applied to LM prefix q/k AND expert
      suffix q/k. Split-half rotation → ``rope_rotate_half_fp16``.
    * Prefill is CAUSAL (vlm_causal=true in the shipped yaml) —
      opposite of vla4b. Suffix: state row must not attend action
      tokens → two FA2 calls (state kv=Se+1, actions kv=Se+51).
    * GQA 32q/8kv native in FA2 (q head h → kv head h//4);
      KV cache stays 8-head (kv_row 1024 fp16).
    * Expert AdaRMS style tables: scale = weight*(1+gamma(t))-1,
      shift = beta(t), gate = 1 (identical fold to vla4b).
    * Expert qkv projections HAVE bias (Qwen2 skeleton); LM has none.
    * MoE per expert layer: fp32 sigmoid router + e_score_correction
      bias top-4 (STAGE-2 kernel), dense expert slab via batched fp8
      GEMM (STAGE-2 wrapper entry), gather-weighted combine (STAGE-2
      kernel), plus an UNGATED shared expert (existing kernels).
      Torch parity baseline: models/vla2/moe_ref.py.
    * LM last layer is KV-write-only; LM final norm skipped (its
      output only feeds training-time align heads).
    * ViT MLP + mergers use gelu_pytorch_tanh (``gelu_inplace_fp16``
      implements the tanh approximation) and have NO gate projection.

Calibration follows the vla4b pattern: ``calibrate=True`` replaces
static quantizes with ``quantize_fp8_device_fp16`` writing amax/448
into the same device act-scale slots — recalibration never recaptures.
"""

import math

# ── architecture constants (locked by ckpt shapes + Qwen3-VL-4B config) ──
VIS_L, VIS_D, VIS_H = 24, 1024, 4096
VIS_NH, VIS_HD = 16, 64
VIS_DEEPSTACK = (5, 11, 17)          # tap AFTER these blocks
LM_L, LM_D, LM_H = 36, 2560, 9728
LM_NH, LM_NKV, LM_HD = 32, 8, 128
LM_DQ, LM_DKV = LM_NH * LM_HD, LM_NKV * LM_HD          # 4096 / 1024
LM_QKV_OUT = LM_DQ + 2 * LM_DKV                        # 6144
EXP_L, EXP_D = 36, 768
EXP_QKV_OUT = LM_DQ + 2 * LM_DKV                       # joint geometry
MOE_E, MOE_TOPK, MOE_H, SHARED_H = 32, 4, 512, 704
CHUNK, ADIM, SDIM = 50, 55, 55
STEPS = 10
SUF = CHUNK + 1
KV_ROW = LM_DKV                                        # 1024


# ════════════════════════════════════════════════════════════════
# FA2 attention adapter
# ════════════════════════════════════════════════════════════════

class Vla2Fa2Attn:
    """Thin dispatch over the vendored FA2 forward for the three sites.

    Same pointer-owning idiom as ``Vla4bFa2Attn``; ``num_sms=0`` → no
    splitkv → single launch per call (CUDA-graph safe). hd64 (ViT) and
    hd128 (LM/expert) are both first-class FA2 head dims — no padding.

    CAUSAL NOTE (found before first Thor run): the vendored FA2 binding
    has NO is_causal flag — the trailing int is ``num_sms``. Causal is
    the SEPARATE ``fwd_bf16_causal`` entry, instantiated ONLY for
    (bf16, hd128). The LM prefill therefore casts q/K/V through bf16
    scratch (cast_fp16_to_bf16 / cast_bf16_to_fp16) around the causal
    call — ~10 MB/layer extra traffic, a Stage-3 micro-opt would build
    the fp16 causal instantiation instead.
    """

    def __init__(self, fa2_mod, *, vit, lm, suffix, fvk_mod=None):
        """
        Args:
            fa2_mod: the ``flash_rt.flash_rt_fa2`` module.
            vit: dict(q, k, v, o, lse — int ptrs; nv — num images).
                q/k/v are separated [768, 1024] fp16 buffers; per-image
                full attention == batched FMHA with batch=nv, seq=256.
            lm: dict(q, o, lse, Kc, Vc, layer_stride_elems, qb, kb, vb,
                ob — bf16 scratch for the causal call). CAUSAL prefill.
            suffix: dict(q, o, lse1, lse2, Kc, Vc, layer_stride_elems).
            fvk_mod: flash_rt_kernels (for the fp16↔bf16 casts).
        """
        self._fa2 = fa2_mod
        self._fvk = fvk_mod
        self._vit = dict(vit)
        self._lm = dict(lm)
        self._sfx = dict(suffix)

    # ── ViT: per-image full attention (batch = num views) ──
    def run_vit(self, *, stream: int = 0):
        s = self._vit
        nv = int(s["nv"])
        batch, seq = nv, 256
        D = VIS_D
        if s.get("use_fmha"):
            # groot_n17 vit idiom: cutlass strided FMHA
            # (libfmha_fp16_strided.so; internal scale = hd^-0.5). The
            # vendored FA2 build has NO hd64 instantiation (only 96/128)
            # — found before the first Thor ViT run.
            self._fvk.fmha_strided_full(
                s["q"], s["k"], s["v"], s["o"],
                batch, seq, seq, VIS_NH, VIS_NH, VIS_HD,
                D, D, stream)
            return
        scale = 1.0 / math.sqrt(float(VIS_HD))
        st = (seq * D, D, VIS_HD)
        self._fa2.fwd_fp16(
            s["q"], s["k"], s["v"], s["o"], s["lse"], 0, 0,
            batch, seq, seq, VIS_NH, VIS_NH, VIS_HD,
            st, st, st, st, scale, 0, stream)

    # ── LM prefill: CAUSAL GQA self-attention (bf16 causal entry) ──
    def run_prefix(self, layer: int, *, se: int, stream: int = 0):
        s = self._lm
        kv_off = layer * int(s["layer_stride_elems"]) * 2  # bytes
        scale = 1.0 / math.sqrt(float(LM_HD))
        fvk = self._fvk
        fvk.cast_fp16_to_bf16(s["q"], s["qb"], se * LM_DQ, stream)
        fvk.cast_fp16_to_bf16(s["Kc"] + kv_off, s["kb"], se * LM_DKV,
                              stream)
        fvk.cast_fp16_to_bf16(s["Vc"] + kv_off, s["vb"], se * LM_DKV,
                              stream)
        self._fa2.fwd_bf16_causal(
            s["qb"], s["kb"], s["vb"], s["ob"], s["lse"],
            0, 0,
            1, se, se, LM_NH, LM_NKV, LM_HD,
            (se * LM_DQ, LM_DQ, LM_HD), (se * LM_DKV, LM_DKV, LM_HD),
            (se * LM_DKV, LM_DKV, LM_HD), (se * LM_DQ, LM_DQ, LM_HD),
            scale, 0, stream)          # num_sms=0 → no splitkv
        fvk.cast_bf16_to_fp16(s["ob"], s["o"], se * LM_DQ, stream)

    # ── Suffix joint attention: state row + action rows ──
    def run_suffix(self, layer: int, *, se: int, stream: int = 0):
        s = self._sfx
        kv_off = layer * int(s["layer_stride_elems"]) * 2  # bytes
        scale = 1.0 / math.sqrt(float(LM_HD))
        K = s["Kc"] + kv_off
        V = s["Vc"] + kv_off
        # state row: attends prefix + itself (kv rows 0..Se)
        sk1 = se + 1
        self._fa2.fwd_fp16(
            s["q"], K, V, s["o"], s["lse1"], 0, 0,
            1, 1, sk1, LM_NH, LM_NKV, LM_HD,
            (LM_DQ, LM_DQ, LM_HD), (sk1 * LM_DKV, LM_DKV, LM_HD),
            (sk1 * LM_DKV, LM_DKV, LM_HD), (LM_DQ, LM_DQ, LM_HD),
            scale, 0, stream)
        # action rows: attend everything (kv rows 0..Se+50)
        sk2 = se + SUF
        qb = s["q"] + LM_DQ * 2       # byte offset: skip state row
        ob = s["o"] + LM_DQ * 2
        self._fa2.fwd_fp16(
            qb, K, V, ob, s["lse2"], 0, 0,
            1, CHUNK, sk2, LM_NH, LM_NKV, LM_HD,
            (CHUNK * LM_DQ, LM_DQ, LM_HD), (sk2 * LM_DKV, LM_DKV, LM_HD),
            (sk2 * LM_DKV, LM_DKV, LM_HD), (CHUNK * LM_DQ, LM_DQ, LM_HD),
            scale, 0, stream)


# ════════════════════════════════════════════════════════════════
# Shared helpers (vla4b idiom)
# ════════════════════════════════════════════════════════════════

def _quant_point(fvk, src_fp16, dst_fp8, slot_ptr, n, stream, calibrate):
    """FP16 buffer → FP8 with the act scale at ``slot_ptr``."""
    if calibrate:
        fvk.quantize_fp8_device_fp16(src_fp16, dst_fp8, slot_ptr, n, stream)
    else:
        fvk.quantize_fp8_static_fp16(src_fp16, dst_fp8, slot_ptr, n, stream)


import ctypes as _ctypes

_crt = None
for _n in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
    try:                              # import-clean on CUDA-less hosts;
        _crt = _ctypes.CDLL(_n)       # versioned names bind to the copy
        break                         # torch already loaded (Thor has no
    except OSError:                   # unversioned dev symlink on the
        continue                      # default loader path)


def _zero_fp16(fvk, ptr, n, stream):
    """Zero n fp16 elements (cudaMemsetAsync; CUDA-graph capturable)."""
    assert _crt is not None, "libcudart.so unavailable"
    _crt.cudaMemsetAsync(_ctypes.c_void_p(ptr), 0,
                         _ctypes.c_size_t(n * 2), _ctypes.c_void_p(stream))


# ════════════════════════════════════════════════════════════════
# ViT (24 blocks, full attention per image, deepstack + merger)
# ════════════════════════════════════════════════════════════════

def vit_forward(fvk, ctx, bufs, weights, dims, stream=0, *, attn):
    """Qwen3-VL vision tower in raw HF token order (no window permute).

    FULL FP16 tower (T2-V1 recipe carried over: vision-tower act FP8
    was the single measured recipe-breaker on V1; V2 keeps ViT fp16
    until an ablation says otherwise — ~0.6 GB weight traffic).

    Input ``x`` must already hold patch_embed(pixels) + interpolated
    pos_embed (both frontend-precomputed; the 256px grid is fixed so
    ``fast_pos_embed_interpolate`` collapses to a constant table).

    bufs (int ptrs, fp16 unless noted):
        x         [S, D]  residual stream
        xn        [S, D]  norm output
        qkv       [S, 3D]
        q, k, v   [S, D]  (separated for rope + FA2)
        attn_out  [S, D]
        fg        [S, D]  (o/fc2 GEMM output pre-residual)
        hid       [S, H]  (fc1 output / GELU buffer)
        mrg_in    [S/4, 4D]  merge-unit view scratch (LN output)
        m0        [S/4, 4D]  merger fc1 output
        vis_emb   [S/4, D_enc] final merger output (row-major order)
        ds_out[i] [S/4, D_enc] ×3 — deepstack merger outputs
    weights (int ptrs / lists):
        qkv_w[L] qkv_b[L] o_w[L] o_b[L] fc1_w[L] fc1_b[L] fc2_w[L]
        fc2_b[L] n1_w[L] n1_b[L] n2_w[L] n2_b[L]  (fp16, [K,N] GEMMs)
        ds_n_w[i] ds_n_b[i] ds_fc1_w[i] ds_fc1_b[i] ds_fc2_w[i]
        ds_fc2_b[i]  (deepstack mergers, i in 0..2, LN over 4D)
        m_n_w m_n_b m0_w m0_b m2_w m2_b  (final merger; LN over D)
        cos, sin  ([S, 64] fp16 2D-rope tables)
    dims: S (768), D (1024), H (4096), L (24), S_m (192), D_enc (2560)
    """
    S = dims["S"]; D = dims["D"]; H = dims["H"]; L = dims["L"]
    S_m = dims["S_m"]; D_enc = dims["D_enc"]
    D4 = 4 * D

    x = bufs["x"]; xn = bufs["xn"]; qkv = bufs["qkv"]
    q = bufs["q"]; k = bufs["k"]; v = bufs["v"]
    attn_out = bufs["attn_out"]; fg = bufs["fg"]; hid = bufs["hid"]
    mrg_in = bufs["mrg_in"]; m0 = bufs["m0"]; vis_emb = bufs["vis_emb"]
    cos = weights["cos"]; sin = weights["sin"]

    def _deepstack(i):
        """merge-unit LN(4D) → fc1(4D→4D) → GELU → fc2(4D→D_enc)."""
        fvk.layer_norm_fp16(x, weights["ds_n_w"][i], weights["ds_n_b"][i],
                            mrg_in, S_m, D4, 1e-6, stream)
        fvk.gmm_fp16(ctx, mrg_in, weights["ds_fc1_w"][i], m0,
                     S_m, D4, D4, 0.0, stream)
        fvk.add_bias_fp16(m0, weights["ds_fc1_b"][i], S_m, D4, stream)
        fvk.gelu_inplace_fp16(m0, S_m * D4, stream)
        fvk.gmm_fp16(ctx, m0, weights["ds_fc2_w"][i], bufs["ds_out"][i],
                     S_m, D_enc, D4, 0.0, stream)
        fvk.add_bias_fp16(bufs["ds_out"][i], weights["ds_fc2_b"][i],
                          S_m, D_enc, stream)

    ds_idx = 0
    for l in range(L):
        # ── norm1 (affine LN) → merged QKV GEMM + bias ──
        fvk.layer_norm_fp16(x, weights["n1_w"][l], weights["n1_b"][l],
                            xn, S, D, 1e-6, stream)
        fvk.gmm_fp16(ctx, xn, weights["qkv_w"][l], qkv, S, 3 * D, D,
                     0.0, stream)
        fvk.add_bias_fp16(qkv, weights["qkv_b"][l], S, 3 * D, stream)

        # ── split + 2D rope + per-image FA2 ──
        fvk.qkv_split_fp16(qkv, q, k, v, S, D, D, D, stream)
        fvk.rope_rotate_half_fp16(q, cos, sin, S, VIS_NH, VIS_HD, stream)
        fvk.rope_rotate_half_fp16(k, cos, sin, S, VIS_NH, VIS_HD, stream)
        attn.run_vit(stream=stream)

        # ── proj + bias + residual ──
        fvk.gmm_fp16(ctx, attn_out, weights["o_w"][l], fg, S, D, D,
                     0.0, stream)
        fvk.bias_residual_fp16(x, fg, weights["o_b"][l], S, D, stream)

        # ── norm2 → fc1 → GELU(tanh) → fc2 + residual (NO gate) ──
        fvk.layer_norm_fp16(x, weights["n2_w"][l], weights["n2_b"][l],
                            xn, S, D, 1e-6, stream)
        fvk.gmm_fp16(ctx, xn, weights["fc1_w"][l], hid, S, H, D,
                     0.0, stream)
        fvk.add_bias_fp16(hid, weights["fc1_b"][l], S, H, stream)
        fvk.gelu_inplace_fp16(hid, S * H, stream)
        fvk.gmm_fp16(ctx, hid, weights["fc2_w"][l], fg, S, D, H,
                     0.0, stream)
        fvk.bias_residual_fp16(x, fg, weights["fc2_b"][l], S, D, stream)

        # ── deepstack tap AFTER blocks 5/11/17 ──
        if l in VIS_DEEPSTACK:
            _deepstack(ds_idx)
            ds_idx += 1

    # ── final merger: LN(D) → reshape (S/4, 4D) → fc1 → GELU → fc2 ──
    # (the LN is per patch token pre-merge; the reshape is a no-op on
    #  a contiguous [S, D] buffer viewed as [S/4, 4D])
    fvk.layer_norm_fp16(x, weights["m_n_w"], weights["m_n_b"], xn,
                        S, D, 1e-6, stream)
    fvk.gmm_fp16(ctx, xn, weights["m0_w"], m0, S_m, D4, D4, 0.0, stream)
    fvk.add_bias_fp16(m0, weights["m0_b"], S_m, D4, stream)
    fvk.gelu_inplace_fp16(m0, S_m * D4, stream)
    fvk.gmm_fp16(ctx, m0, weights["m2_w"], vis_emb, S_m, D_enc, D4,
                 0.0, stream)
    fvk.add_bias_fp16(vis_emb, weights["m2_b"], S_m, D_enc, stream)
    # NOTE: rows are already in raw row-major merged order (no window
    # un-permute needed — Qwen3-VL difference vs vla4b).


# ════════════════════════════════════════════════════════════════
# LM prefill (36 layers, CAUSAL, KV fill + deepstack injection)
# ════════════════════════════════════════════════════════════════

def lm_prefill_forward(fvk, bufs, weights, dims, stream=0, *, attn,
                       calibrate=False, ctx=None, precision="fp8"):
    """Qwen3-VL LM prefill over the dense V2 prefix.

    Prefix rows (frontend-staged into ``x``):
        3 × [vision_start | 64 merged patches | vision_end]  (198)
      | language tokens (dense, pads dropped)
      | 16 align-query constants (current-task 8 + future-task 8)

    Fills the 8-head KV cache rows [0, Se) of every layer. Keys are
    written POST q/k-norm POST-rope (that is what the reference caches
    — handle_kv_cache runs after apply_mrope). Deepstack merger
    outputs are residual-added at the patch rows after layers 0/1/2.
    The last layer computes only QKV + norms + rope + KV write.

    bufs: x [Se,D], x_fp8 (u8 ≥ Se*H), qkv [Se,6144], q [Se,4096],
          q2 [Se,4096] (post-q_norm), kt [Se,1024] (pre-k_norm),
          attn_out [Se,4096], fg [Se,D], gate [Se,H], up [Se,H],
          calib_hid, ones [D]
    weights: qkv_w[L] (NO bias — attention_bias=false), o_w[L],
             gate_w[L], up_w[L], down_w[L], qnorm_w[L], knorm_w[L]
             ([128] fp16 each), Kc, Vc, cos, sin ([Se,128] prefix
             M-RoPE), ds_rows (list of (row0, nrows) per image),
             ds_out[3] (ViT deepstack outputs, [192, D] fp16),
             act_scales, w_scales
    dims: Se, D (2560), H (9728), L (36), total_keys, kv_row (1024)

    precision:
        "fp8"  — fp8 W+A GEMMs, RMS folds baked into the weights
                 (vla4b recipe; act scales from the tier-2 calib).
                 cuBLASLt NN fp8 → Thor/cu13 only (F1: sm_90/cu12.8
                 rejects the layout).
        "fp16" — the P1 parity arm (runs on BOTH boxes): plain
                 gmm_fp16 on UNFOLDED fp16 weights with explicit
                 rms_norm_fp16 (needs ctx + bufs["xn"] [Se, D] and
                 weights n_in_w[L]/n_post_w[L]/qkv_w16[L]/o_w16[L]/
                 gate_w16[L]/up_w16[L]/down_w16[L]).
    """
    if precision == "fp16":
        _lm_prefill_forward_fp16(fvk, ctx, bufs, weights, dims, stream,
                                 attn=attn)
        return
    assert precision == "fp8", precision
    Se = dims["Se"]; D = dims["D"]; H = dims["H"]; L = dims["L"]
    total_keys = dims["total_keys"]; kv_row = dims["kv_row"]

    x = bufs["x"]; x_fp8 = bufs["x_fp8"]; qkv = bufs["qkv"]
    q = bufs["q"]; q2 = bufs["q2"]; kt = bufs["kt"]
    attn_out = bufs["attn_out"]; fg = bufs["fg"]
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

        # merged QKV GEMM (2560 → 6144, no bias)
        fvk.fp8_gemm_descale_fp16(x_fp8, weights["qkv_w"][l], qkv,
                                  Se, LM_QKV_OUT, D, a_qkv, w_qkv, stream)

        # split: q buf, k scratch, v straight into cache rows [0, Se)
        k_dst = Kc + (l * total_keys) * kv_row * 2
        v_dst = Vc + (l * total_keys) * kv_row * 2
        fvk.qkv_split_fp16(qkv, q, kt, v_dst, Se, LM_DQ, LM_DKV, LM_DKV,
                           stream)
        # per-head q/k RMS norms (Qwen3): rows = tokens × heads
        fvk.rms_norm_fp16(q, weights["qnorm_w"][l], q2,
                          Se * LM_NH, LM_HD, 1e-6, stream)
        fvk.rms_norm_fp16(kt, weights["knorm_w"][l], k_dst,
                          Se * LM_NKV, LM_HD, 1e-6, stream)
        # interleaved M-RoPE on q + cached k
        fvk.rope_rotate_half_fp16(q2, cos, sin, Se, LM_NH, LM_HD, stream)
        fvk.rope_rotate_half_fp16(k_dst, cos, sin, Se, LM_NKV, LM_HD,
                                  stream)

        if last:
            break  # KV written; LM hidden output unused

        attn.run_prefix(l, se=Se, stream=stream)

        _quant_point(fvk, attn_out, x_fp8, a_o, Se * LM_DQ, stream,
                     calibrate)
        fvk.fp8_gemm_descale_fp16(x_fp8, weights["o_w"][l], fg,
                                  Se, D, LM_DQ, a_o, w_o, stream)
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
            fvk.silu_mul_split_fp8_fp16(gate, up, x_fp8, Se * H, a_dn,
                                        stream)
        fvk.fp8_gemm_descale_fp16(x_fp8, weights["down_w"][l], fg,
                                  Se, D, H, a_dn, w_d, stream)

        # tail: residual (+ deepstack injection after layers 0..2)
        if l < len(weights["ds_out"]):
            fvk.residual_add_fp16(x, fg, Se * D, stream)
            # add deepstack[l] at the 3 × 64 patch-row spans
            ds = weights["ds_out"][l]
            for i, (row0, nrows) in enumerate(weights["ds_rows"]):
                fvk.residual_add_fp16(
                    x + row0 * D * 2,
                    ds + (i * nrows) * D * 2,
                    nrows * D, stream)
            _norm_fp8_from(x, acts + ((l + 1) * 4 + 0) * 4)
        else:
            _res_norm_fp8(acts + ((l + 1) * 4 + 0) * 4)


def _lm_prefill_forward_fp16(fvk, ctx, bufs, weights, dims, stream=0, *,
                             attn):
    """fp16-substrate LM prefill (the P1 parity arm; see
    ``lm_prefill_forward``). Same dataflow as the fp8 path with every
    quantize point removed: explicit input/post RMS norms (unfolded
    weights), gmm_fp16 GEMMs, silu·mul via the zero+gate_res pair
    (the calibrate-path idiom). KV write semantics identical."""
    Se = dims["Se"]; D = dims["D"]; H = dims["H"]; L = dims["L"]
    total_keys = dims["total_keys"]; kv_row = dims["kv_row"]

    x = bufs["x"]; xn = bufs["xn"]; qkv = bufs["qkv"]
    q = bufs["q"]; q2 = bufs["q2"]; kt = bufs["kt"]
    attn_out = bufs["attn_out"]; fg = bufs["fg"]
    gate = bufs["gate"]; up = bufs["up"]; hid = bufs["calib_hid"]
    cos = weights["cos"]; sin = weights["sin"]
    Kc = weights["Kc"]; Vc = weights["Vc"]

    for l in range(L):
        last = (l == L - 1)

        # input RMS norm (explicit weight) → merged QKV GEMM (no bias)
        fvk.rms_norm_fp16(x, weights["n_in_w"][l], xn, Se, D, 1e-6, stream)
        fvk.gmm_fp16(ctx, xn, weights["qkv_w16"][l], qkv,
                     Se, LM_QKV_OUT, D, 0.0, stream)

        # split: q buf, k scratch, v straight into cache rows [0, Se)
        k_dst = Kc + (l * total_keys) * kv_row * 2
        v_dst = Vc + (l * total_keys) * kv_row * 2
        fvk.qkv_split_fp16(qkv, q, kt, v_dst, Se, LM_DQ, LM_DKV, LM_DKV,
                           stream)
        # per-head q/k RMS norms (Qwen3): rows = tokens × heads
        fvk.rms_norm_fp16(q, weights["qnorm_w"][l], q2,
                          Se * LM_NH, LM_HD, 1e-6, stream)
        fvk.rms_norm_fp16(kt, weights["knorm_w"][l], k_dst,
                          Se * LM_NKV, LM_HD, 1e-6, stream)
        # interleaved M-RoPE on q + cached k
        fvk.rope_rotate_half_fp16(q2, cos, sin, Se, LM_NH, LM_HD, stream)
        fvk.rope_rotate_half_fp16(k_dst, cos, sin, Se, LM_NKV, LM_HD,
                                  stream)

        if last:
            break  # KV written; LM hidden output unused

        attn.run_prefix(l, se=Se, stream=stream)

        fvk.gmm_fp16(ctx, attn_out, weights["o_w16"][l], fg,
                     Se, D, LM_DQ, 0.0, stream)
        fvk.residual_add_fp16(x, fg, Se * D, stream)

        # post-attn RMS norm → gate/up → SiLU·mul → down
        fvk.rms_norm_fp16(x, weights["n_post_w"][l], xn, Se, D, 1e-6,
                          stream)
        fvk.gmm_fp16(ctx, xn, weights["gate_w16"][l], gate, Se, H, D,
                     0.0, stream)
        fvk.gmm_fp16(ctx, xn, weights["up_w16"][l], up, Se, H, D,
                     0.0, stream)
        fvk.silu_inplace_fp16(gate, Se * H, stream)
        _zero_fp16(fvk, hid, Se * H, stream)
        fvk.gate_res_fp16(up, gate, hid, Se * H, stream)   # hid += up·silu
        fvk.gmm_fp16(ctx, hid, weights["down_w16"][l], fg, Se, D, H,
                     0.0, stream)

        # tail: residual (+ deepstack injection after layers 0..2)
        fvk.residual_add_fp16(x, fg, Se * D, stream)
        if l < len(weights["ds_out"]):
            ds = weights["ds_out"][l]
            for i, (row0, nrows) in enumerate(weights["ds_rows"]):
                fvk.residual_add_fp16(
                    x + row0 * D * 2,
                    ds + (i * nrows) * D * 2,
                    nrows * D, stream)


# ════════════════════════════════════════════════════════════════
# Action expert (10 steps × 36 layers, joint attention + MoE)
# ════════════════════════════════════════════════════════════════

def routed_moe_stub(layer, step, x_fp8, moe_out, stream):
    """STAGE-2 placeholder for the routed-expert path.

    Contract (see mapping_memo.md §MoE):
        inputs : x_fp8   — post-AdaRMS suffix activations, quantized
                           (the same buffer the shared expert reads)
        output : moe_out — [51, 768] fp16 routed-expert contribution
                           (weighted top-4 combine, pre-shared-expert)
    Engine plan v0: per-expert loop of existing fp8 GEMMs + STAGE-2
    router/combine kernels; v0.5: strided-batched cuBLASLt entry;
    v1: grouped GEMM. Torch parity baseline: models/vla2/moe_ref.py.
    """
    raise NotImplementedError(
        "routed MoE is Stage-2 work: needs moe_router_topk (fp32 sigmoid "
        "+ e_score bias + top-4 + norm + ×4.0) and moe_combine kernels; "
        "see /home/ubuntu/iwm_distill/thor_t2v2/mapping_memo.md")


def expert_forward(fvk, ctx, bufs, weights, dims, stream=0, *, attn,
                   routed_moe_fn=routed_moe_stub, calibrate=False,
                   ffn_xn_fp16=False, calibration_observer=None):
    """10-step flow-matching loop of the 36-layer MoE action expert.

    Differences vs vla4b's expert_forward:
        * qkv GEMM 768 → 6144 (32q/8kv × 128 joint geometry) + bias
        * suffix rope = interleaved M-RoPE theta 5e6 tables
        * the FFN is a token-MoE: routed part via ``routed_moe_fn``
          (STAGE-2), shared expert (704, ungated) with existing
          kernels, summed then residual-added.

    bufs: x_t [50, 55], state_emb [1, 768], a_emb [50, 768],
          tmp [50, 768], x [51, 768], x_fp8 (u8 ≥ 51*768),
          gatebuf [51, 768], qkv [51, 6144], q [51, 4096],
          attn_out [51, 4096], fg [51, 768], sh_gate [51, 704],
          sh_up [51, 704], moe_out [51, 768], xn [51, 768],
          calib_hid, ones [768]
    weights: qkv_w[L], qkv_b[L], o_w[L], sh_gate_w[L], sh_up_w[L],
             sh_down_w[L], style_attn, style_ffn, Kc, Vc, cos, sin
             ([51,128] suffix M-RoPE), final_norm_w, ain_w [55,768],
             ain_b, atm_a_w [768,768], t_contrib [10,768],
             atm_out_w, atm_out_b, aout_w_dt [768,55], aout_b_dt,
             act_scales, w_scales
    dims: D (768), L (36), steps (10), Se, total_keys, kv_row (1024)
    """
    D = dims["D"]; L = dims["L"]; steps = dims["steps"]
    if calibration_observer is not None and not calibrate:
        raise ValueError("calibration observer requires dynamic calibration")
    Se = dims["Se"]; total_keys = dims["total_keys"]; kv_row = dims["kv_row"]
    S = SUF
    D3 = 3 * D

    x_t = bufs["x_t"]; a_emb = bufs["a_emb"]; tmp = bufs["tmp"]
    x = bufs["x"]; x_fp8 = bufs["x_fp8"]; gatebuf = bufs["gatebuf"]
    qkv = bufs["qkv"]; q = bufs["q"]; attn_out = bufs["attn_out"]
    fg = bufs["fg"]; xn = bufs["xn"]
    sh_gate = bufs["sh_gate"]; sh_up = bufs["sh_up"]
    moe_out = bufs["moe_out"]

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

    def _gate_res_adarms_fp8(style_ptr, slot, want_xn=False):
        # want_xn (M2d F2 fp16-router wiring): materialize the post-AdaRMS
        # fp16 ``xn`` alongside the fp8 quantize so a fp16 router source
        # exists at the ffn site (moe_router_gemm_topk_fp16x reads it via
        # Vla2MoeEngine.bind_router_fp16). Same math as the fused static
        # entry, split into the existing calibrate-path launch pair.
        if calibrate:
            fvk.gate_res_fp16(fg, gatebuf, x, S * D, stream)
            fvk.adarms_fp16(x, style_ptr, xn, gatebuf, S, D, stream)
            fvk.quantize_fp8_device_fp16(xn, x_fp8, slot, S * D, stream)
        elif want_xn:
            fvk.gate_res_fp16(fg, gatebuf, x, S * D, stream)
            fvk.adarms_fp16(x, style_ptr, xn, gatebuf, S, D, stream)
            fvk.quantize_fp8_static_fp16(xn, x_fp8, slot, S * D, stream)
        else:
            fvk.gate_res_adarms_fp8_static_fp16(fg, gatebuf, x, style_ptr,
                                                x_fp8, gatebuf, S, D, slot,
                                                stream)

    for s in range(steps):
        # ── suffix embedding rebuild (dims 55, otherwise vla4b) ──
        fvk.gpu_copy(x, bufs["state_emb"], D * 2, stream)
        fvk.gmm_fp16(ctx, x_t, weights["ain_w"], a_emb, CHUNK, D, ADIM,
                     0.0, stream)
        fvk.add_bias_fp16(a_emb, weights["ain_b"], CHUNK, D, stream)
        fvk.gmm_fp16(ctx, a_emb, weights["atm_a_w"], tmp, CHUNK, D, D,
                     0.0, stream)
        fvk.add_bias_fp16(tmp, weights["t_contrib"] + s * D * 2, CHUNK, D,
                          stream)
        fvk.silu_inplace_fp16(tmp, CHUNK * D, stream)
        fvk.gmm_fp16(ctx, tmp, weights["atm_out_w"], x + D * 2, CHUNK, D, D,
                     0.0, stream)
        fvk.add_bias_fp16(x + D * 2, weights["atm_out_b"], CHUNK, D, stream)

        for l in range(L):
            # 4 act slots per layer: qkv-in, o-in, ffn-in (post-AdaRMS,
            # shared by router/experts/shared gate+up), shared-down-in
            a_qkv = acts + (l * 4 + 0) * 4
            a_o   = acts + (l * 4 + 1) * 4
            a_ffn = acts + (l * 4 + 2) * 4
            a_shd = acts + (l * 4 + 3) * 4
            w_qkv = ws + (l * 5 + 0) * 4
            w_o   = ws + (l * 5 + 1) * 4
            w_shg = ws + (l * 5 + 2) * 4
            w_shu = ws + (l * 5 + 3) * 4
            w_shd = ws + (l * 5 + 4) * 4
            sa_ptr = style_a + (s * L + l) * style_stride
            sf_ptr = style_f + (s * L + l) * style_stride

            if l == 0:
                _adarms_fp8(x, sa_ptr, a_qkv)

            # merged QKV (768 → 6144) + bias (Qwen2 skeleton has bias)
            fvk.fp8_gemm_descale_fp16(x_fp8, weights["qkv_w"][l], qkv,
                                      S, EXP_QKV_OUT, D, a_qkv, w_qkv,
                                      stream)
            fvk.add_bias_fp16(qkv, weights["qkv_b"][l], S, EXP_QKV_OUT,
                              stream)

            # split → q + suffix KV rows [Se, Se+51); NO q/k norm here
            k_dst = Kc + (l * total_keys + Se) * kv_row * 2
            v_dst = Vc + (l * total_keys + Se) * kv_row * 2
            fvk.qkv_split_fp16(qkv, q, k_dst, v_dst, S, LM_DQ, LM_DKV,
                               LM_DKV, stream)
            fvk.rope_rotate_half_fp16(q, cos, sin, S, LM_NH, LM_HD, stream)
            fvk.rope_rotate_half_fp16(k_dst, cos, sin, S, LM_NKV, LM_HD,
                                      stream)

            attn.run_suffix(l, se=Se, stream=stream)

            _quant_point(fvk, attn_out, x_fp8, a_o, S * LM_DQ, stream,
                         calibrate)
            fvk.fp8_gemm_descale_fp16(x_fp8, weights["o_w"][l], fg,
                                      S, D, LM_DQ, a_o, w_o, stream)

            # post-attn AdaRMS → x_fp8 feeds BOTH MoE paths
            # (ffn_xn_fp16 also leaves the fp16 xn for the fp16 router)
            _gate_res_adarms_fp8(sf_ptr, a_ffn, want_xn=ffn_xn_fp16)

            # ── routed experts (STAGE-2 kernels) ──
            routed_moe_fn(l, s, x_fp8, moe_out, stream)

            # ── shared expert (704, ungated) — existing kernels ──
            fvk.fp8_gemm_descale_fp16(x_fp8, weights["sh_gate_w"][l],
                                      sh_gate, S, SHARED_H, D, a_ffn,
                                      w_shg, stream)
            fvk.fp8_gemm_descale_fp16(x_fp8, weights["sh_up_w"][l],
                                      sh_up, S, SHARED_H, D, a_ffn,
                                      w_shu, stream)
            fvk.silu_inplace_fp16(sh_gate, S * SHARED_H, stream)
            _zero_fp16(fvk, bufs["calib_hid"], S * SHARED_H, stream)
            fvk.gate_res_fp16(sh_up, sh_gate, bufs["calib_hid"],
                              S * SHARED_H, stream)
            # shared-expert intermediate gets its OWN act slot (a_shd);
            # this quantize overwrites x_fp8, so routed_moe_fn above has
            # already consumed the post-AdaRMS fp8 (stream-ordered).
            # Stage-2 micro-opt: fuse via silu_mul_split_fp8_fp16.
            _quant_point(fvk, bufs["calib_hid"], x_fp8, a_shd,
                         S * SHARED_H, stream, calibrate)
            fvk.fp8_gemm_descale_fp16(x_fp8, weights["sh_down_w"][l], fg,
                                      S, D, SHARED_H, a_shd, w_shd, stream)
            # fg += moe_out  (routed + shared, before the residual)
            fvk.residual_add_fp16(fg, moe_out, S * D, stream)

            # Observe this layer before another step overwrites its scales.
            # The caller must enqueue observations on the same CUDA stream.
            if calibration_observer is not None:
                calibration_observer(l, s)

            if l < L - 1:
                sa_next = style_a + (s * L + l + 1) * style_stride
                _gate_res_adarms_fp8(sa_next, acts + ((l + 1) * 4 + 0) * 4)
            else:
                fvk.gate_res_fp16(fg, gatebuf, x, S * D, stream)

        # ── final plain RMS (FixQwen2RMSNorm) + Euler accumulate ──
        fvk.rms_norm_fp16(x, weights["final_norm_w"], xn, S, D, 1e-6, stream)
        fvk.gmm_fp16(ctx, xn + D * 2, weights["aout_w_dt"], x_t,
                     CHUNK, ADIM, D, 1.0, stream)
        fvk.add_bias_fp16(x_t, weights["aout_b_dt"], CHUNK, ADIM, stream)

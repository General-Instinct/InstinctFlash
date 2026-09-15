"""FlashRT — LingBot-VA (wan_va) Thor SM110 DiT program (Stage 2, 2026-09-02).

One forward stage, run ``point.forwards_per_cycle`` times per 32-control-step
cycle (10 at 2V/4A@w5, 8 at 2V/2A@w1):

    dit_forward — 30 blocks over the current chunk (video 240/360 tok or
                  action 32 tok, ``B`` = cfg_batch streams stream-major in the
                  row dimension), self-attention against the linear episode
                  KV slab window [head, tail), cross-attention against the
                  hoisted episode cross-KV, AdaLN step tables (span-wise for
                  the cycle-0 mixed-t forwards), gelu-tanh ffn, gated
                  residuals; then the norm_out + proj head. The same program
                  serves denoise (transient), pred-commit and kv-commit
                  forwards — the slab bookkeeping (what happens to the rows
                  written at [tail-S, tail)) is HOST state in the frontend,
                  never inside this program and never inside a captured graph.

Numerics contract (ground truth wan_va/modules/model.py; mapping_memo §0/§A):
    * norm1/norm3: FP32LayerNorm no-affine * (1+scale) + shift, scale/shift =
      step-table rows [D] -> ``ada_layer_norm_fp16`` (fp32 stats, eps 1e-6).
    * attn1 qkv merged [3D] + bias; q/k RMSNorm over the FULL 3072 before the
      head split (eps 1e-6, affine, weights P-permuted) -> ``rms_norm_fp16``;
      rope split-half on P-permuted q/k -> ``rope_rotate_half_fp16``; K written
      post-norm post-rope, V raw, into slab rows [tail-S, tail) per stream.
    * self attention: FA2 ``fwd_fp16`` batch B over the contiguous slab window,
      scale hd^-0.5, no mask (inference attention is unmasked, memo §0.4).
    * gated residuals x += y * gate[c] in fp32 -> fp16: the ONE new kernel
      ``gate_row_mul_residual_fp16`` (csrc/kernels/wan_va_fused.cu); when the
      built .so lacks it, the fallback is a row broadcast
      (``gpu_strided_copy_fp16`` src_stride=0) + ``gate_res_fp16`` — same math,
      one extra S*D write+read per gate.
    * norm2: FP32LayerNorm WITH affine -> ``layer_norm_fp16``; cross residual is
      UNGATED (``residual_add_fp16``).
    * ffn: gelu-tanh (``gelu_inplace_fp16`` is the tanh form) between two GEMMs.
    * head: norm_out no-affine + root-table row (shift, scale from temb) ->
      ``ada_layer_norm_fp16`` + proj GEMM + bias -> [B*S, out_dim] fp16. CFG
      combine, Euler step (bf16 re-round), patch permutes and the cycle-0 pins
      are torch-side in the frontend on the tiny head output (exact stock
      dtype chain, latency-irrelevant).

Precision arms: "fp16" (parity: fp16 [K,N] weights, gmm_fp16) and "fp8"
(ship: per-tensor amax/448 E4M3 W+A on the 6 loop GEMM families; act-scale
slots 6/layer: qkv-in, o-in, cq-in, co-in, ff1-in, ff2-in(post-gelu);
``calibrate=True`` writes amax/448 into the SAME device slots via
``quantize_fp8_device_fp16`` — recalibration never recaptures).
"""

import math

# ── architecture constants (locked by ckpt shapes + configs) ──
L, D, NH, HD = 30, 3072, 24, 128
FFN = 14336
QKV_OUT = 3 * D
TEXT_LEN = 512
VIDEO_TOKENS, ACTION_TOKENS = 240, 32
SCALE = 1.0 / math.sqrt(HD)
EPS = 1e-6
ACT_SLOTS = 6          # per layer: qkv-in, o-in, cq-in, co-in, ff1-in, ff2-in
W_SLOTS = 6            # per layer: qkv, o, cq, co, ff1, ff2 (loader order)
MOD_ROWS = 6           # shift, scale, gate, c_shift, c_scale, c_gate (model:524-532)


# ════════════════════════════════════════════════════════════════
# FA2 attention adapter
# ════════════════════════════════════════════════════════════════

class WanFa2Attn:
    """Dispatch over the vendored FA2 forward for the two sites.

    self site: q = current chunk (S rows per stream), K/V = slab window
    [head, tail) of THIS layer — one contiguous interval per stream (linear
    slab, memo §A.3) — batch B with stride slab_rows*D between streams.
    ``num_sms`` (the trailing int; the binding has NO is_causal flag) is a
    split-kv knob: 0 = single launch, capture-safe (measured decision, M2c).

    cross site: K/V = the hoisted episode cross slab (512 rows), batch B.
    """

    def __init__(self, fa2_mod, *, q, o, lse, Kc, Vc, cK, cV, B, slab_rows,
                 num_sms=0):
        self._fa2 = fa2_mod
        self.q, self.o, self.lse = q, o, lse
        self.Kc, self.Vc, self.cK, self.cV = Kc, Vc, cK, cV
        self.B = int(B)
        self.slab_rows = int(slab_rows)
        self.num_sms = int(num_sms)
        self.text_len = None          # debug override (M2 broken control); None = TEXT_LEN

    def run_self(self, layer, *, s_cur, head, tail, stream=0):
        B = self.B
        kv_len = tail - head
        assert kv_len >= s_cur > 0, (head, tail, s_cur)
        layer_off = (layer * B) * self.slab_rows * D          # elements
        k_base = self.Kc + (layer_off + head * D) * 2         # bytes
        v_base = self.Vc + (layer_off + head * D) * 2
        st_q = (s_cur * D, D, HD)
        st_kv = (self.slab_rows * D, D, HD)
        self._fa2.fwd_fp16(
            self.q, k_base, v_base, self.o, self.lse, 0, 0,
            B, s_cur, kv_len, NH, NH, HD,
            st_q, st_kv, st_kv, st_q,
            SCALE, self.num_sms, stream)

    def run_cross(self, layer, *, s_cur, stream=0):
        B = self.B
        off = (layer * B) * TEXT_LEN * D * 2
        st_q = (s_cur * D, D, HD)
        st_kv = (TEXT_LEN * D, D, HD)
        kv_len = TEXT_LEN if self.text_len is None else int(self.text_len)
        self._fa2.fwd_fp16(
            self.q, self.cK + off, self.cV + off, self.o, self.lse, 0, 0,
            B, s_cur, kv_len, NH, NH, HD,
            st_q, st_kv, st_kv, st_q,
            SCALE, 0, stream)


# ════════════════════════════════════════════════════════════════
# Gated residual: the one new kernel, with a no-rebuild fallback
# ════════════════════════════════════════════════════════════════

def make_gate_fn(fvk, gate_bcast_ptr):
    """res[s, c] += x[s, c] * gate[c] over S rows (fp32 math, fp16 out)."""
    if hasattr(fvk, "gate_row_mul_residual_fp16"):
        def gate_fn(res, x, gate_row, s_rows, d, stream):
            fvk.gate_row_mul_residual_fp16(res, x, gate_row, s_rows, d, stream)
        gate_fn.kind = "kernel"
        return gate_fn

    def gate_fn(res, x, gate_row, s_rows, d, stream):
        # broadcast the [D] row into an [S, D] buffer, then elementwise gate_res
        fvk.gpu_strided_copy_fp16(gate_row, gate_bcast_ptr, s_rows, d, 0, 0, stream)
        fvk.gate_res_fp16(x, gate_bcast_ptr, res, s_rows * d, stream)
    gate_fn.kind = "fallback(strided_copy+gate_res)"
    return gate_fn


def _quant_point(fvk, src_fp16, dst_fp8, slot_ptr, n, stream, calibrate):
    if calibrate:
        fvk.quantize_fp8_device_fp16(src_fp16, dst_fp8, slot_ptr, n, stream)
    else:
        fvk.quantize_fp8_static_fp16(src_fp16, dst_fp8, slot_ptr, n, stream)


# ════════════════════════════════════════════════════════════════
# The DiT forward program
# ════════════════════════════════════════════════════════════════

def expand_spans(spans, S, B):
    """spans: ((row0, nrows, t_idx), ...) within ONE stream (frame-major token
    order; one span except the cycle-0 mixed-t forwards). Returns the row
    ranges over the stream-major [B*S] buffer. A single span covers all B
    streams in ONE call (the modulation tables do not depend on the text
    stream) — that is the common case."""
    if len(spans) == 1:
        r0, n, t = spans[0]
        assert r0 == 0 and n == S, spans
        return ((0, B * S, t),)
    out = []
    for b in range(B):
        for (r0, n, t) in spans:
            out.append((b * S + r0, n, t))
    return tuple(out)


def dit_forward(fvk, ctx, bufs, weights, dims, stream=0, *, attn, gate_fn,
                calibrate=False, precision="fp8"):
    """One DiT forward over the current chunk (all B streams) + the head.

    bufs (int ptrs, fp16 unless noted): x [R, D] residual (pre-filled with the
    embedded tokens), xn [R, D], qkv [R, 3D], qraw [R, D], q [R, D], kt [R, D],
    attn_out [R, D], fg [R, D], hid [R, FFN], x_fp8 (u8, >= R*FFN), head_out
    [R, out_dim]. R = B*S.

    weights: per-layer lists qkv_w, qkv_b, nq_w, nk_w, o_w, o_b, cq_w, cq_b,
    cnq_w, co_w, co_b, n2_w, n2_b, ff1_w, ff1_b, ff2_w, ff2_b (fp8 [K,N] u8
    or fp16 [K,N]); optional fp16_families = names of GEMM families kept fp16
    inside the fp8 arm (the pre-committed per-family fallback, memo §C); act_scales / w_scales (fp32 device bases, 6 per layer);
    Kc, Vc (slab bases [L, B, slab_rows, D]); mod (this stream's step table
    base, fp16 (T, L, 6, D)); out_tab ((T, 2, D)); cos, sin ((S, HD) tables
    for THIS forward); proj_w [D, out_dim], proj_b.

    dims: S, B, spans (see expand_spans), head, tail (slab window AFTER the
    host append: the rows written this forward are [tail-S, tail)), slab_rows,
    out_dim.
    """
    S = int(dims["S"]); B = int(dims["B"]); R = B * S
    head = int(dims["head"]); tail = int(dims["tail"])
    slab_rows = int(dims["slab_rows"]); out_dim = int(dims["out_dim"])
    rows = expand_spans(dims["spans"], S, B)
    fp8 = precision == "fp8"
    assert precision in ("fp8", "fp16"), precision

    x = bufs["x"]; xn = bufs["xn"]; qkv = bufs["qkv"]; qraw = bufs["qraw"]
    q = bufs["q"]; kt = bufs["kt"]; attn_out = bufs["attn_out"]
    fg = bufs["fg"]; hid = bufs["hid"]; x_fp8 = bufs["x_fp8"]
    head_out = bufs["head_out"]
    acts = weights["act_scales"]; ws = weights["w_scales"]
    Kc = weights["Kc"]; Vc = weights["Vc"]
    cos = weights["cos"]; sin = weights["sin"]
    mod = weights["mod"]; out_tab = weights["out_tab"]

    def mod_ptr(t_idx, layer, k):
        return mod + (((t_idx * L + layer) * MOD_ROWS + k) * D) * 2

    def out_ptr(t_idx, k):
        return out_tab + ((t_idx * 2 + k) * D) * 2

    def slab_row_ptr(base, layer, b, row):
        return base + (((layer * B + b) * slab_rows + row) * D) * 2

    fp16_fams = set(weights.get("fp16_families", ()))   # per-family fp16 fallback (memo §C)

    def gemm(a_fp16, a_fp8_n, w_list, layer, out, M, N, K, a_slot, w_slot, fam=None):
        if fp8 and fam not in fp16_fams:
            _quant_point(fvk, a_fp16, x_fp8, acts + (layer * ACT_SLOTS + a_slot) * 4,
                         a_fp8_n, stream, calibrate)
            fvk.fp8_gemm_descale_fp16(x_fp8, w_list[layer], out, M, N, K,
                                      acts + (layer * ACT_SLOTS + a_slot) * 4,
                                      ws + (layer * W_SLOTS + w_slot) * 4, stream)
        else:
            fvk.gmm_fp16(ctx, a_fp16, w_list[layer], out, M, N, K, 0.0, stream)

    wr = tail - S   # first slab row written this forward

    for l in range(L):
        # ── 1. norm1 + AdaLN modulation (span-wise) ──
        for (r0, n, t) in rows:
            fvk.ada_layer_norm_fp16(x + r0 * D * 2, mod_ptr(t, l, 1), mod_ptr(t, l, 0),
                                    xn + r0 * D * 2, n, D, EPS, stream)
        # ── 2. merged qkv GEMM + bias ──
        gemm(xn, R * D, weights["qkv_w"], l, qkv, R, QKV_OUT, D, 0, 0, "qkv_w")
        fvk.add_bias_fp16(qkv, weights["qkv_b"][l], R, QKV_OUT, stream)
        # ── 3. split: q, k scratch, v straight into the slab ──
        fvk.gpu_strided_copy_fp16(qkv, qraw, R, D, QKV_OUT, 0, stream)
        fvk.gpu_strided_copy_fp16(qkv, kt, R, D, QKV_OUT, D, stream)
        for b in range(B):
            fvk.gpu_strided_copy_fp16(qkv + (b * S) * QKV_OUT * 2,
                                      slab_row_ptr(Vc, l, b, wr),
                                      S, D, QKV_OUT, 2 * D, stream)
        # ── 4. full-3072 q/k RMS (P-permuted weights) + split-half rope ──
        fvk.rms_norm_fp16(qraw, weights["nq_w"][l], q, R, D, EPS, stream)
        for b in range(B):
            k_dst = slab_row_ptr(Kc, l, b, wr)
            fvk.rms_norm_fp16(kt + (b * S) * D * 2, weights["nk_w"][l], k_dst,
                              S, D, EPS, stream)
            fvk.rope_rotate_half_fp16(q + (b * S) * D * 2, cos, sin, S, NH, HD, stream)
            fvk.rope_rotate_half_fp16(k_dst, cos, sin, S, NH, HD, stream)
        # ── 5. self attention vs the slab window ──
        attn.run_self(l, s_cur=S, head=head, tail=tail, stream=stream)
        # ── 6. o proj + bias + GATED residual ──
        gemm(attn_out, R * D, weights["o_w"], l, fg, R, D, D, 1, 1, "o_w")
        fvk.add_bias_fp16(fg, weights["o_b"][l], R, D, stream)
        for (r0, n, t) in rows:
            gate_fn(x + r0 * D * 2, fg + r0 * D * 2, mod_ptr(t, l, 2), n, D, stream)
        # ── 7. norm2 (affine) → cross q → cross attention → UNGATED residual ──
        fvk.layer_norm_fp16(x, weights["n2_w"][l], weights["n2_b"][l], xn, R, D, EPS,
                            stream)
        gemm(xn, R * D, weights["cq_w"], l, qraw, R, D, D, 2, 2, "cq_w")
        fvk.add_bias_fp16(qraw, weights["cq_b"][l], R, D, stream)
        fvk.rms_norm_fp16(qraw, weights["cnq_w"][l], q, R, D, EPS, stream)
        attn.run_cross(l, s_cur=S, stream=stream)
        gemm(attn_out, R * D, weights["co_w"], l, fg, R, D, D, 3, 3, "co_w")
        fvk.add_bias_fp16(fg, weights["co_b"][l], R, D, stream)
        fvk.residual_add_fp16(x, fg, R * D, stream)
        # ── 8. norm3 + modulation → ffn (gelu-tanh) → GATED residual ──
        for (r0, n, t) in rows:
            fvk.ada_layer_norm_fp16(x + r0 * D * 2, mod_ptr(t, l, 4), mod_ptr(t, l, 3),
                                    xn + r0 * D * 2, n, D, EPS, stream)
        gemm(xn, R * D, weights["ff1_w"], l, hid, R, FFN, D, 4, 4, "ff1_w")
        fvk.add_bias_fp16(hid, weights["ff1_b"][l], R, FFN, stream)
        fvk.gelu_inplace_fp16(hid, R * FFN, stream)
        gemm(hid, R * FFN, weights["ff2_w"], l, fg, R, D, FFN, 5, 5, "ff2_w")
        fvk.add_bias_fp16(fg, weights["ff2_b"][l], R, D, stream)
        for (r0, n, t) in rows:
            gate_fn(x + r0 * D * 2, fg + r0 * D * 2, mod_ptr(t, l, 5), n, D, stream)

    # ── head: norm_out + (shift, scale) from the root table → proj ──
    for (r0, n, t) in rows:
        fvk.ada_layer_norm_fp16(x + r0 * D * 2, out_ptr(t, 1), out_ptr(t, 0),
                                xn + r0 * D * 2, n, D, EPS, stream)
    fvk.gmm_fp16(ctx, xn, weights["proj_w"], head_out, R, out_dim, D, 0.0, stream)
    fvk.add_bias_fp16(head_out, weights["proj_b"], R, out_dim, stream)


def launches_per_forward(B: int, spans: int, fp8: bool, gate_kernel: bool) -> int:
    """Host-side launch count of one forward (the M2c eager-overhead input)."""
    span_calls = 1 if spans == 1 else spans * B
    per_layer = (span_calls          # ada1
                 + (2 if fp8 else 1) + 1          # qkv gemm (+quant) + bias
                 + 2 + B                          # splits
                 + 1 + 3 * B                      # rms q, rms k / rope q / rope k
                 + 1                              # FA2 self
                 + (2 if fp8 else 1) + 1          # o gemm + bias
                 + span_calls * (1 if gate_kernel else 2)
                 + 1 + (2 if fp8 else 1) + 1 + 1  # ln2, cq gemm, bias, rms
                 + 1                              # FA2 cross
                 + (2 if fp8 else 1) + 1 + 1      # co gemm, bias, residual
                 + span_calls                     # ada3
                 + (2 if fp8 else 1) + 1 + 1      # ff1 gemm, bias, gelu
                 + (2 if fp8 else 1) + 1          # ff2 gemm, bias
                 + span_calls * (1 if gate_kernel else 2))
    return L * per_layer + span_calls + 2


__all__ = ["WanFa2Attn", "make_gate_fn", "dit_forward", "expand_spans",
           "launches_per_forward", "L", "D", "NH", "HD", "FFN", "VIDEO_TOKENS",
           "ACTION_TOKENS", "TEXT_LEN", "SCALE", "EPS", "ACT_SLOTS", "W_SLOTS",
           "MOD_ROWS"]

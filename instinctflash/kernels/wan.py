"""Wan-DiT projection epilogue: full-width Q/K RMSNorm, interleaved 3D RoPE, pooled K/V.

For one token row of a raw fused projection accumulator ``[Q | K | V]`` (each
``H * HD`` wide) this kernel applies the row/column dequantization and bias,
RMS-normalizes Q and K over their *full* width (Wan's ``norm_q`` / ``norm_k`` act
before the head split), rotates adjacent channel pairs ``(2j, 2j+1)`` of every
head by ``pos[t, section[j]] * inv_freq[j]``, and stores Q densely and K/V into
per-branch KV-pool rows ``slot[t]``. Missing parts (no K/V, no rotary) are
selected by flags, which also covers cross-attention Q and text K/V.

Every tile is loaded contiguously and split into (even, odd) lanes in registers.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _part(RAW, COL, BIAS, NW, POS, SEC, INV, raw_row, col0, t, r, stride_pos, eps,
          H: tl.constexpr, HD: tl.constexpr, BH: tl.constexpr, NORM: tl.constexpr, ROPE: tl.constexpr,
          HAS_COL: tl.constexpr, HAS_BIAS: tl.constexpr):
    h = tl.arange(0, BH)
    d = tl.arange(0, HD)
    m = (h < H)[:, None]
    offs = h[:, None] * HD + d[None, :]
    x = tl.load(RAW + raw_row + col0 + offs, mask=m, other=0.0).to(tl.float32) * r
    if HAS_COL:
        x = x * tl.load(COL + col0 + offs, mask=m, other=0.0)
    if HAS_BIAS:
        x = x + tl.load(BIAS + col0 + offs, mask=m, other=0.0)
    if NORM:
        rs = tl.math.rsqrt(tl.sum(tl.sum(x * x, 1), 0) / (H * HD) + eps)
        x = x * rs * tl.load(NW + offs, mask=m, other=0.0)
    if ROPE:
        HALF: tl.constexpr = HD // 2
        e, o = tl.split(tl.reshape(x, (BH, HALF, 2)))
        j = tl.arange(0, HALF)
        ang = tl.load(POS + t * stride_pos + tl.load(SEC + j)) * tl.load(INV + j)
        c = tl.cos(ang)[None, :]
        s = tl.sin(ang)[None, :]
        x = tl.reshape(tl.join(e * c - o * s, e * s + o * c), (BH, HD))
    return x, offs, m


@triton.jit
def _wan_qkv_kernel(RAW, ROW, COL, BIAS, QNW, KNW, QO, KP, VP, SLOT, POS, SEC, INV,
                    stride_raw, sq_row, sk_b, sk_s, sv_b, sv_s, stride_pos, T, eps,
                    H: tl.constexpr, HD: tl.constexpr, BH: tl.constexpr,
                    HAS_K: tl.constexpr, HAS_V: tl.constexpr, NORM_Q: tl.constexpr, NORM_K: tl.constexpr,
                    ROPE: tl.constexpr, HAS_ROW: tl.constexpr, HAS_COL: tl.constexpr, HAS_BIAS: tl.constexpr,
                    HAS_SLOT: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)          # 0 = Q, 1 = K, 2 = V
    b = row // T
    t = row % T
    D: tl.constexpr = H * HD
    raw_row = row * stride_raw
    if HAS_ROW:
        r = tl.load(ROW + row)
    else:
        r = 1.0
    if HAS_SLOT:
        slot = tl.load(SLOT + t)
    else:
        slot = t
    if part == 0:
        x, offs, m = _part(RAW, COL, BIAS, QNW, POS, SEC, INV, raw_row, 0, t, r, stride_pos, eps,
                           H, HD, BH, NORM_Q, ROPE, HAS_COL, HAS_BIAS)
        tl.store(QO + row * sq_row + offs, x.to(QO.dtype.element_ty), mask=m)
    elif part == 1:
        if HAS_K:
            x, offs, m = _part(RAW, COL, BIAS, KNW, POS, SEC, INV, raw_row, D, t, r, stride_pos, eps,
                               H, HD, BH, NORM_K, ROPE, HAS_COL, HAS_BIAS)
            tl.store(KP + b * sk_b + slot * sk_s + offs, x.to(KP.dtype.element_ty), mask=m)
    else:
        if HAS_V:
            x, offs, m = _part(RAW, COL, BIAS, KNW, POS, SEC, INV, raw_row, 2 * D, t, r, stride_pos, eps,
                               H, HD, BH, False, False, HAS_COL, HAS_BIAS)
            tl.store(VP + b * sv_b + slot * sv_s + offs, x.to(VP.dtype.element_ty), mask=m)


def wan_qkv(raw, q_out, *, heads, head_dim, tokens, k_pool=None, v_pool=None, slots=None, row=None, col=None,
            bias=None, q_norm=None, k_norm=None, pos=None, sections=None, inv_freq=None, eps=1e-6, num_warps=8):
    """raw [B*T, parts*H*HD] (parts: Q, then K, then V); q_out [B*T, H*HD];
    k_pool / v_pool [B, S, H*HD] written at rows ``slots[t]`` (or ``t``)."""
    rows = raw.shape[0]
    parts = 1 + (k_pool is not None) + (v_pool is not None)
    dummy = raw
    _wan_qkv_kernel[(rows, parts)](
        raw, row if row is not None else dummy, col if col is not None else dummy,
        bias if bias is not None else dummy, q_norm if q_norm is not None else dummy,
        k_norm if k_norm is not None else dummy, q_out, k_pool if k_pool is not None else q_out,
        v_pool if v_pool is not None else q_out, slots if slots is not None else dummy,
        pos if pos is not None else dummy, sections if sections is not None else dummy,
        inv_freq if inv_freq is not None else dummy,
        raw.stride(0), q_out.stride(0),
        k_pool.stride(0) if k_pool is not None else 0, k_pool.stride(1) if k_pool is not None else 0,
        v_pool.stride(0) if v_pool is not None else 0, v_pool.stride(1) if v_pool is not None else 0,
        pos.stride(0) if pos is not None else 0, int(tokens), float(eps),
        H=heads, HD=head_dim, BH=triton.next_power_of_2(heads), HAS_K=k_pool is not None,
        HAS_V=v_pool is not None, NORM_Q=q_norm is not None, NORM_K=k_norm is not None,
        ROPE=inv_freq is not None, HAS_ROW=row is not None, HAS_COL=col is not None,
        HAS_BIAS=bias is not None, HAS_SLOT=slots is not None, num_warps=num_warps)

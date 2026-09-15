"""Shared BF16 pointwise fusions with explicit eager rounding boundaries.

Only contiguous SM110 inference inputs use the kernels; other inputs use the
specified PyTorch expressions. Qualification is per model/checkpoint, not implied
by dtype. Adapters own installation and first-use byte checks. No global patches.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _square_float(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    tl.store(Y + i, x * x, i < N)


@triton.jit
def _scale_norm(X, S, W, Y, N: tl.constexpr, D: tl.constexpr,
                ROUND_BEFORE_WEIGHT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    scale = tl.load(S + i // D, i < N, 0)
    w = tl.load(W + i % D).to(tl.float32)
    normalized = x * scale
    if ROUND_BEFORE_WEIGHT:
        normalized = normalized.to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, w * normalized, i < N)


@triton.jit
def _relu2(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    positive = tl.maximum(x, 0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(Y + i, positive * positive, i < N)


@triton.jit
def _swiglu(G, U, LUT, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Every BF16 bit pattern indexes its exact eager SiLU result. This avoids
    # changing exp/div implementations, including their rounding boundaries.
    bits = tl.load(G.to(tl.pointer_type(tl.uint16)) + i, i < N, 0).to(tl.int32)
    activated = tl.load(LUT + bits).to(tl.float32)
    up = tl.load(U + i, i < N, 0).to(tl.float32)
    tl.store(Y + i, activated * up, i < N)


@triton.jit
def _rope(X, C, S, Y, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
          R: tl.constexpr, CS0: tl.constexpr, CS1: tl.constexpr,
          SS0: tl.constexpr, SS1: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % D
    row = i // (H * D)
    peer = i - d + (d + R // 2) % R
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    rotated = tl.load(X + peer, (i < N) & (d < R), 0).to(tl.float32)
    rotated = tl.where(d < R // 2, -rotated, rotated)
    c = tl.load(C + row * CS0 + d * CS1, (i < N) & (d < R), 0).to(tl.float32)
    s = tl.load(S + row * SS0 + d * SS1, (i < N) & (d < R), 0).to(tl.float32)
    # Eager BF16 multiply, BF16 multiply, then BF16 add; never fuse into FMA.
    left = (x * c).to(Y.dtype.element_ty).to(tl.float32)
    right = (rotated * s).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, tl.where(d < R, left + right, x), i < N)


def _norm_cuda(x, weight, eps, round_before_weight):
    squared = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _square_float[(triton.cdiv(x.numel(), 1024),)](x, squared, x.numel(), 1024, enable_fp_fusion=False)
    variance = squared.mean(-1, keepdim=True)
    scale = torch.rsqrt(variance + eps)
    out = torch.empty_like(x)
    _scale_norm[(triton.cdiv(x.numel(), 1024),)](
        x, scale, weight, out, x.numel(), x.shape[-1], round_before_weight, 1024,
        enable_fp_fusion=False)
    return out


def _relu2_cuda(x):
    out = torch.empty_like(x)
    _relu2[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), 1024, enable_fp_fusion=False)
    return out


def silu_table(device):
    bits = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16)
    return torch.nn.functional.silu(bits.view(torch.bfloat16))


def _swiglu_cuda(gate, up, table):
    out = torch.empty_like(gate)
    _swiglu[(triton.cdiv(gate.numel(), 1024),)](gate, up, table, out, gate.numel(), 1024, enable_fp_fusion=False)
    return out


def _rope_cuda(q, k, cos, sin):
    result = []
    for x in (q, k):
        out = torch.empty_like(x)
        if x.numel():
            _rope[(triton.cdiv(x.numel(), 1024),)](
                x, cos, sin, out, x.numel(), x.shape[1], x.shape[2], cos.shape[-1],
                *cos.stride(), *sin.stride(), 1024, enable_fp_fusion=False)
        result.append(out)
    return tuple(result)


def _supported(*xs):
    return (not torch.is_grad_enabled() and bool(xs)
            and all(x.is_cuda and x.dtype == torch.bfloat16
                    and x.device == xs[0].device and x.is_contiguous()
                    and 0 < x.numel() < 2**31 for x in xs)
            and torch.cuda.get_device_capability(xs[0].device) == (11, 0))


def norm(x, weight, eps, round_before_weight):
    """FP32 square/mean/rsqrt; optional BF16 rounding before weight multiply.

    Reduction remains PyTorch: substituting a different reduction is a separate
    numerical optimization. Weight must be a vector matching the last dimension.
    """
    if x.ndim < 1 or weight.shape != x.shape[-1:]:
        raise ValueError('RMSNorm requires a weight vector matching the last dimension')
    if _supported(x, weight):
        return _norm_cuda(x, weight, eps, round_before_weight)
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    if round_before_weight:
        return weight * normalized.to(x.dtype)
    return (weight.float() * normalized).to(x.dtype)


def relu2(x):
    """ReLU then square, with the original dtype's output rounding."""
    return _relu2_cuda(x) if _supported(x) else torch.relu(x).square()


def swiglu(gate, up, table=None):
    """BF16 SiLU rounding before multiplication, using an eager-generated LUT.

    Construct silu_table(device) before graph capture; the caller owns its
    lifetime and must not modify it. Missing tables use the native expression.
    """
    if gate.shape != up.shape:
        raise ValueError('SwiGLU requires matching gate and up shapes')
    if (table is not None and table.shape == (65536,)
            and _supported(gate, up, table)):
        return _swiglu_cuda(gate, up, table)
    return torch.nn.functional.silu(gate) * up


def rope(q, k, cos, sin):
    """Split-half partial RoPE on [tokens, heads, dim], with BF16 mul/add.

    cos/sin are [tokens, rotary_dim]; trailing non-rotary features are retained.
    This is not interleaved RoPE and performs no position lookup.
    """
    if (q.ndim != 3 or k.ndim != 3 or cos.ndim != 2 or sin.shape != cos.shape
            or q.shape[0] != k.shape[0] or q.shape[0] != cos.shape[0]
            or q.shape[-1] != k.shape[-1]
            or not 0 < cos.shape[-1] <= q.shape[-1] or cos.shape[-1] % 2):
        raise ValueError('Unsupported split-half RoPE geometry')
    if _supported(q, k, cos, sin):
        return _rope_cuda(q, k, cos, sin)
    r = cos.shape[-1]
    def eager(x):
        a = x[..., :r]
        rotated = torch.cat((-a[..., r // 2:], a[..., :r // 2]), -1)
        return torch.cat((a * cos[:, None, :] + rotated * sin[:, None, :], x[..., r:]), -1)
    return eager(q), eager(k)


@triton.jit
def _residual_square(X, R, SUM, SQUARED, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    r = tl.load(R + i, i < N, 0).to(tl.float32)
    # The eager residual addition rounds before entering the FP32 norm.
    added = (x + r).to(SUM.dtype.element_ty).to(tl.float32)
    tl.store(SUM + i, added, i < N)
    tl.store(SQUARED + i, added * added, i < N)


def residual_norm(x, residual, weight, eps, round_before_weight):
    """Return (normalized, residual_sum), with owned outputs and no mutation.

    Semantics: added = x + residual (BF16 rounded), then norm(added, ...).
    The FP32 reduction stays in PyTorch. This does not implement FP32 residual
    accumulation, dropout, scaling, or a norm-before-add architecture.
    """
    if x.shape != residual.shape or x.dtype != residual.dtype or x.device != residual.device:
        raise ValueError('Residual requires matching shape, dtype and device')
    if x.ndim < 1 or weight.shape != x.shape[-1:]:
        raise ValueError('RMSNorm requires a weight vector matching the last dimension')
    if not _supported(x, residual, weight):
        added = x + residual
        return norm(added, weight, eps, round_before_weight), added
    added = torch.empty_like(x)
    squared = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _residual_square[(triton.cdiv(x.numel(), 1024),)](
        x, residual, added, squared, x.numel(), 1024, enable_fp_fusion=False)
    scale = torch.rsqrt(squared.mean(-1, keepdim=True) + eps)
    out = torch.empty_like(x)
    _scale_norm[(triton.cdiv(x.numel(), 1024),)](
        added, scale, weight, out, x.numel(), x.shape[-1], round_before_weight,
        1024, enable_fp_fusion=False)
    return out, added

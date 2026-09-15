"""BF16-to-E4M3 packing with fixed or dynamic per-tensor scales.

The dynamic reduction preserves the original scale arithmetic. This preserves
an FP8 recipe, not native BF16 model equivalence.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_kernel(X, SCALE, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    scale = tl.load(SCALE)
    # Round-to-nearest division matches the explicit PyTorch float32 reference;
    # do not replace it with reciprocal multiplication or approximate division.
    divided = tl.div_rn(x, scale)
    # PyTorch clamp propagates NaN. Triton's min/max default suppresses it,
    # which could hide a nonfinite upstream activation as a finite FP8 value.
    clipped = tl.minimum(
        tl.maximum(divided, -448.0, propagate_nan=tl.PropagateNan.ALL),
        448.0, propagate_nan=tl.PropagateNan.ALL,
    )
    tl.store(Y + offsets, clipped.to(tl.float8e4nv), offsets < N)


def pack_bf16_e4m3(x, scale):
    if x.device.type != 'cuda' or torch.cuda.get_device_capability(x.device) not in (
        (9, 0), (11, 0), (12, 0),
    ):
        raise ValueError('FP8 packing requires SM90, SM110, or SM120')
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError('FP8 packing requires contiguous BF16 inputs')
    if scale.dtype != torch.float32 or scale.numel() != 1 or scale.device != x.device:
        raise ValueError('FP8 packing requires one FP32 scale on the input device')
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    if x.numel():
        _pack_kernel[(triton.cdiv(x.numel(), 1024),)](x, scale, out, x.numel(), 1024)
    return out


@triton.jit
def _amax_partial(X, PARTIAL, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    x = tl.load(X+i, i<N, other=0).to(tl.float32)
    nan = tl.sum((x != x).to(tl.int32),0) != 0
    value = tl.max(tl.where(x != x, 0., tl.abs(x)),0)
    tl.store(PARTIAL+tl.program_id(0),tl.where(nan,float('nan'),value))


@triton.jit
def _finish_scale(PARTIAL, SCALE, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    i=tl.arange(0,BLOCK)
    x=tl.load(PARTIAL+i,i<COUNT,other=0.)
    nan=tl.sum((x != x).to(tl.int32),0) != 0
    value=tl.maximum(tl.max(tl.where(x != x,0.,x),0),1.e-12)
    # PyTorch's division by a host scalar uses its FP32 reciprocal.
    scale=value*(1.0/448.0)
    tl.store(SCALE,tl.where(nan,float('nan'),scale))


def dynamic_pack_bf16_e4m3(x):
    """Same per-tensor recipe, without full activation abs/FP32 temporaries."""
    if x.device.type != 'cuda' or x.dtype != torch.bfloat16 or not x.is_contiguous() or not x.numel():
        raise ValueError('Dynamic FP8 packing requires nonempty contiguous CUDA BF16 input')
    count=triton.cdiv(x.numel(),4096)
    partial=torch.empty((count,),device=x.device,dtype=torch.float32)
    scale=torch.empty((1,),device=x.device,dtype=torch.float32)
    _amax_partial[(count,)](x,partial,x.numel(),4096)
    _finish_scale[(1,)](partial,scale,count,triton.next_power_of_2(count))
    return pack_bf16_e4m3(x,scale),scale

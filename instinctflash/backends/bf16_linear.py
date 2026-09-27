"""Shared, opt-in NUMERIC BF16 linear/activation dispatch.

Only the measured ReLU² shape has a fused implementation: the persistent
warp-specialized Triton GEMM in :mod:`instinctflash.kernels.gemm_bf16`, whose
epilogue rounds to BF16 before squaring, so it reproduces the eager
``relu(linear(x)).square()`` bit for bit. All other cases use an explicitly
supplied original activation. No module-name guessing, global patching or
fallback after a kernel launch failure.
"""
from __future__ import annotations

import torch

QUALIFIED_SHAPES = frozenset({(3093, 9216, 2048)})
# BM, BN, BK, GROUP_M, stages, warps, epilogue column slices, warp specialization.
# GROUP_M=32 covers all 25 row tiles, so every weight is streamed from DRAM once. Ranked on
# Thor with real Edge operands under a serving duty cycle (idle gaps between calls; back-to-back
# loops are power-throttled and hide the L2 effect): 925 us/call vs 942 us for the previous
# CUTLASS library and ~1590 us for eager linear + relu + square.
THOR_CONFIG = (128, 256, 64, 32, 3, 8, 2, True)
KERNEL = 'triton-persistent-ws'


def supports(*, activation, gated, bias, capability, dtype, shape):
    return (activation == 'relu2' and not gated and not bias and capability == (11, 0)
            and dtype == torch.bfloat16 and tuple(shape) in QUALIFIED_SHAPES)


_CAPABILITY = {}


def _capability(device):
    cap = _CAPABILITY.get(device)
    if cap is None:
        cap = _CAPABILITY[device] = torch.cuda.get_device_capability(device)
    return cap


def _validate(x, weight):
    if (x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
            or not x.is_contiguous() or not weight.is_contiguous()):
        raise ValueError('BF16 fused linear requires contiguous rank-2 compatible tensors')
    if not x.is_cuda or x.device != weight.device or x.dtype != weight.dtype:
        raise ValueError('BF16 fused linear requires matching CUDA device and dtype')
    if not supports(activation='relu2', gated=False, bias=False,
                    capability=_capability(x.device), dtype=x.dtype,
                    shape=(x.shape[0], weight.shape[0], x.shape[1])):
        raise ValueError('Unqualified BF16 fused linear hardware, dtype or shape')
    if x.data_ptr() % 16 or weight.data_ptr() % 16:
        raise ValueError('BF16 fused linear requires 16-byte aligned operands')


@torch.library.custom_op('instinctflash::bf16_linear_relu2', mutates_args=())
def _linear_relu2(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _validate(x, weight)
    from instinctflash.kernels.gemm_bf16 import RELU2, bf16_gemm_act_persistent
    if torch.cuda.current_device() == x.device.index:
        return bf16_gemm_act_persistent(x, weight, act=RELU2, config=THOR_CONFIG)
    with torch.cuda.device(x.device):
        return bf16_gemm_act_persistent(x, weight, act=RELU2, config=THOR_CONFIG)


@_linear_relu2.register_fake
def _fake(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0]))


class BF16LinearActivation:
    """Adapter-owned callable; explicit semantics and NUMERIC permission required.

    The adapter must preserve native eager admission separately. Only immutable
    inference weights are supported. Non-qualified cases execute the original
    linear + activation before any custom kernel is launched.
    """
    def __init__(self, linear, activation, *, activation_name, plan):
        from instinctflash.planners.planner import Tier
        from instinctflash.runtime.precision import require_transform_permission
        require_transform_permission(plan, Tier.NUMERIC, 'BF16 linear activation fusion')
        self.linear, self.activation = linear, activation
        self.activation_name = activation_name
        self.capability = (torch.cuda.get_device_capability(linear.weight.device)
                           if linear.weight.is_cuda else None)
        self.eligible = (activation_name == 'relu2' and linear.bias is None
                         and self.capability == (11, 0) and linear.weight.dtype == torch.bfloat16
                         and tuple(linear.weight.shape) == (9216, 2048))

    def __call__(self, x):
        weight = self.linear.weight
        if (self.eligible and not torch.is_grad_enabled()
                and x.ndim == 2 and tuple(x.shape) == (3093, 2048)
                and x.dtype == weight.dtype and x.device == weight.device
                and x.is_contiguous() and weight.is_contiguous()):
            return _linear_relu2(x, weight)
        return self.activation(self.linear(x))

    def report(self):
        return dict(activation=self.activation_name, eligible=self.eligible,
                    kernel=KERNEL if self.eligible else None, config=list(THOR_CONFIG),
                    transformation='NUMERIC', precision='bf16',
                    qualified_shapes=sorted(QUALIFIED_SHAPES), task_quality_certified=False)

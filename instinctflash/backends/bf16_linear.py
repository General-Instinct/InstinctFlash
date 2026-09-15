"""Shared, opt-in NUMERIC BF16 linear/activation dispatch.

Only the measured ReLU² shape has a fused implementation. All other cases use
an explicitly supplied original activation. No module-name guessing, global
patching, automatic builds, or fallback after a native launch failure.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import torch

ABI_VERSION = 1
LIBRARY_ENV = 'IFL_BF16_KERNEL_LIBRARY'
QUALIFIED_SHAPES = frozenset({(3093, 9216, 2048)})


def supports(*, activation, gated, bias, capability, dtype, shape):
    return (activation == 'relu2' and not gated and not bias and capability == (11, 0)
            and dtype == torch.bfloat16 and tuple(shape) in QUALIFIED_SHAPES)


@lru_cache(maxsize=8)
def _library(path):
    library = ctypes.CDLL(path)
    version = library.instinctflash_bf16_abi_version
    version.argtypes, version.restype = [], ctypes.c_int
    if version() != ABI_VERSION:
        raise RuntimeError('Unsupported InstinctFlash BF16 kernel ABI')
    fn = library.instinctflash_bf16_linear_relu2
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return library


def resolve_library(path=None):
    selected = Path(path or os.environ.get(LIBRARY_ENV) or
                    Path(__file__).resolve().parents[1] / 'native/libinstinctflash_bf16.so').resolve()
    if not selected.is_file():
        raise RuntimeError(f'BF16 kernel library missing: {selected}; build instinctflash/native/bf16')
    _library(str(selected))
    return str(selected)


def _validate(x, weight):
    if (x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
            or not x.is_contiguous() or not weight.is_contiguous()):
        raise ValueError('BF16 fused linear requires contiguous rank-2 compatible tensors')
    if not x.is_cuda or x.device != weight.device or x.dtype != weight.dtype:
        raise ValueError('BF16 fused linear requires matching CUDA device and dtype')
    if not supports(activation='relu2', gated=False, bias=False,
                    capability=torch.cuda.get_device_capability(x.device), dtype=x.dtype,
                    shape=(x.shape[0], weight.shape[0], x.shape[1])):
        raise ValueError('Unqualified BF16 fused linear hardware, dtype or shape')
    if x.data_ptr() % 16 or weight.data_ptr() % 16:
        raise ValueError('BF16 fused linear requires 16-byte aligned operands')


@torch.library.custom_op('instinctflash::bf16_linear_relu2', mutates_args=())
def _linear_relu2(x: torch.Tensor, weight: torch.Tensor, library_path: str) -> torch.Tensor:
    _validate(x, weight)
    with torch.cuda.device(x.device):
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
        # Per-call allocation on the current stream: no process-global scratch,
        # cross-stream races, retained device workspace or explicit close needed.
        workspace = torch.empty(32 * 1024 * 1024, device=x.device, dtype=torch.uint8)
        fn = _library(library_path).instinctflash_bf16_linear_relu2
        status = fn(x.data_ptr(), weight.data_ptr(), out.data_ptr(), x.shape[0],
                    weight.shape[0], x.shape[1], workspace.data_ptr(), workspace.numel(),
                    torch.cuda.current_stream(x.device).cuda_stream)
        if status:
            raise RuntimeError(f'BF16 fused linear failed with native status {status}')
    return out


@_linear_relu2.register_fake
def _fake(x, weight, library_path):
    return x.new_empty((x.shape[0], weight.shape[0]))


class BF16LinearActivation:
    """Adapter-owned callable; explicit semantics and NUMERIC permission required.

    The adapter must preserve native eager admission separately. Only immutable
    inference weights are supported. Non-qualified cases execute the original
    linear + activation before any custom kernel is launched.
    """
    def __init__(self, linear, activation, *, activation_name, plan, library=None):
        from instinctflash.planners.planner import Tier
        from instinctflash.runtime.precision import require_transform_permission
        require_transform_permission(plan, Tier.NUMERIC, 'BF16 linear activation fusion')
        self.linear, self.activation = linear, activation
        self.activation_name = activation_name
        self.capability = (torch.cuda.get_device_capability(linear.weight.device)
                           if linear.weight.is_cuda else None)
        eligible = (activation_name == 'relu2' and linear.bias is None
                    and self.capability == (11, 0) and linear.weight.dtype == torch.bfloat16
                    and tuple(linear.weight.shape) == (9216, 2048))
        self.library = resolve_library(library) if eligible else None

    def __call__(self, x):
        weight = self.linear.weight
        if (self.library is not None and not torch.is_grad_enabled()
                and x.ndim == 2 and tuple(x.shape) == (3093, 2048)
                and x.dtype == weight.dtype and x.device == weight.device
                and x.is_contiguous() and weight.is_contiguous()):
            return _linear_relu2(x, weight, self.library)
        return self.activation(self.linear(x))

    def report(self):
        return dict(activation=self.activation_name, eligible=self.library is not None,
                    library=self.library, transformation='NUMERIC', precision='bf16',
                    qualified_shapes=sorted(QUALIFIED_SHAPES), task_quality_certified=False)

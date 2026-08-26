"""Fused inference RMSNorm for LingBot's 768-wide action expert."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    x = tl.load(x_ptr + row * D + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / D
    normalized = (x * tl.rsqrt(variance + EPS)).to(x_ptr.dtype.element_ty)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + row * D + offsets, normalized * weight, mask=mask)


@triton.jit
def _ada_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    D: tl.constexpr,
    ROWS_PER_BATCH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // ROWS_PER_BATCH
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    x = tl.load(x_ptr + row * D + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / D
    normalized = x * tl.rsqrt(variance + EPS)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + batch * D + offsets, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + batch * D + offsets, mask=mask, other=0.0).to(tl.float32)
    value = (1.0 + gamma) * (weight * normalized) + beta
    tl.store(out_ptr + row * D + offsets, value, mask=mask)


def lingbot_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if not hidden_states.is_cuda or hidden_states.ndim < 2:
        raise ValueError("LingBot fused RMSNorm expects a rank>=2 CUDA tensor")
    hidden = hidden_states.shape[-1]
    if weight.numel() != hidden:
        raise ValueError("RMSNorm weight width does not match the hidden state")
    if not hidden_states.is_contiguous() or not weight.is_contiguous():
        raise ValueError("LingBot fused RMSNorm requires contiguous tensors")
    rows = hidden_states.numel() // hidden
    out = torch.empty_like(hidden_states)
    _rmsnorm_kernel[(rows,)](
        hidden_states,
        weight,
        out,
        D=hidden,
        EPS=float(eps),
        BLOCK_D=triton.next_power_of_2(hidden),
        num_warps=8,
    )
    return out


def lingbot_ada_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    hidden = hidden_states.shape[-1]
    batch = gamma.shape[0]
    if (
        not hidden_states.is_cuda
        or hidden_states.ndim < 2
        or weight.numel() != hidden
        or gamma.shape != beta.shape
        or gamma.numel() != batch * hidden
    ):
        raise ValueError("invalid LingBot AdaRMSNorm tensor signature")
    if not all(tensor.is_contiguous() for tensor in (hidden_states, weight, gamma, beta)):
        raise ValueError("LingBot fused AdaRMSNorm requires contiguous tensors")
    rows = hidden_states.numel() // hidden
    if rows % batch:
        raise ValueError("AdaRMSNorm rows must divide evenly across the batch")
    out = torch.empty_like(hidden_states)
    _ada_rmsnorm_kernel[(rows,)](
        hidden_states,
        weight,
        gamma,
        beta,
        out,
        D=hidden,
        ROWS_PER_BATCH=rows // batch,
        EPS=float(eps),
        BLOCK_D=triton.next_power_of_2(hidden),
        num_warps=8,
    )
    return out


@dataclass(frozen=True)
class RMSNormKernelInstall:
    modules: int
    hidden_size: int


def install_lingbot_rmsnorm_kernel(
    model: nn.Module,
    *,
    hidden_size: int = 768,
) -> RMSNormKernelInstall:
    """Patch only the action expert's 768-wide RMSNorm instances after weight loading."""
    _refuse_on_sm110()
    candidates = []
    for module in model.modules():
        weight = getattr(module, "weight", None)
        class_name = type(module).__name__
        if (
            class_name in {"Qwen2RMSNorm", "FixQwen2RMSNorm", "AdaRMSNorm", "FixAdaRMSNorm"}
            and isinstance(weight, torch.Tensor)
            and weight.numel() == hidden_size
        ):
            candidates.append(module)
    if not candidates:
        raise RuntimeError(f"no {hidden_size}-wide LingBot RMSNorm modules found")

    for module in candidates:
        if hasattr(module, "_instinctflash_rmsnorm_forward"):
            continue
        original = module.forward
        eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))
        is_ada = type(module).__name__ in {"AdaRMSNorm", "FixAdaRMSNorm"}
        fix_ada = type(module).__name__ == "FixAdaRMSNorm"

        def fast_forward(
            self,
            hidden_states,
            *args,
            _original=original,
            _eps=eps,
            _is_ada=is_ada,
            _fix_ada=fix_ada,
            **kwargs,
        ):
            if (
                self.training
                or torch.is_grad_enabled()
                or not hidden_states.is_cuda
                or not hidden_states.is_contiguous()
            ):
                return _original(hidden_states, *args, **kwargs)
            if _is_ada:
                cond = args[0] if args else kwargs.get("cond")
                if cond is None or len(args) > 1 or set(kwargs) - {"cond"}:
                    return _original(hidden_states, *args, **kwargs)
                cond_input = cond.to(torch.float32) if _fix_ada else cond
                gamma = self.gamma(cond_input)
                beta = self.beta(cond_input)
                return lingbot_ada_rmsnorm(
                    hidden_states,
                    self.weight,
                    gamma,
                    beta,
                    _eps,
                )
            if args or kwargs:
                return _original(hidden_states, *args, **kwargs)
            return lingbot_rmsnorm(hidden_states, self.weight, _eps)

        object.__setattr__(module, "_instinctflash_rmsnorm_forward", original)
        object.__setattr__(module, "forward", MethodType(fast_forward, module))
    return RMSNormKernelInstall(len(candidates), hidden_size)


def _refuse_on_sm110() -> None:
    """Triton is measured-dead on Thor sm_110a (PTXAS internal codegen error), and the vendor
    fallback path crashes (undefined ``logger`` in the MoE except-handler; the RMSNorm patch has
    no try/except). Installing here would fail later and worse; refuse up front instead."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (11, 0):
        raise RuntimeError(
            "LingBot-VLA-V2 Triton kernels are not supported on SM110 (Thor); use the Thor "
            "engine arm instead.")


__all__ = [
    "RMSNormKernelInstall",
    "install_lingbot_rmsnorm_kernel",
    "lingbot_ada_rmsnorm",
    "lingbot_rmsnorm",
]

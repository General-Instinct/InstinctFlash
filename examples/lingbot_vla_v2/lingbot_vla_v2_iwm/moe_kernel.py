"""Inference-only sparse-MoE kernels specialized for LingBot-VLA-V2.

The published checkpoint currently instantiates contiguous fused expert weights.  This module also
supports older/eager layouts by converting the already-loaded weights one layer at a time, then
installs an exact top-k Triton backend without changing checkpoint files.

For LingBot's routing contract, ``torch.topk`` returns distinct experts for a token.  Consequently
one expert can receive at most ``T`` routes, not ``T * TOPK``.  Using that tight bound removes most
of the empty grouped-GEMM programs.  The down projection writes one result per route and a final
fixed-order top-k kernel reduces them, avoiding output atomics and their nondeterminism.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading

import torch
from torch import nn
import triton
import triton.language as tl


_PATCH_LOCK = threading.Lock()
_PATCH_USERS = 0
_PATCH_ORIGINAL = None


@triton.jit
def _zero_i32_kernel(out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.int32), mask=offsets < N)


@triton.jit
def _pack_topk_kernel(
    selected_ptr,
    counts_ptr,
    rows_ptr,
    slots_ptr,
    TOPK: tl.constexpr,
    MAX_ROUTES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, BLOCK_K)
    mask = slots < TOPK
    experts = tl.load(selected_ptr + row * TOPK + slots, mask=mask, other=0).to(tl.int32)
    positions = tl.atomic_add(counts_ptr + experts, 1, sem="relaxed", mask=mask)
    # MAX_ROUTES == T is exact because torch.topk cannot select one expert twice per token.
    store_mask = mask & (positions < MAX_ROUTES)
    tl.store(rows_ptr + experts * MAX_ROUTES + positions, row, mask=store_mask)
    tl.store(slots_ptr + experts * MAX_ROUTES + positions, slots, mask=store_mask)


@triton.jit
def _gate_up_grouped_kernel(
    x_ptr,
    gate_ptr,
    up_ptr,
    counts_ptr,
    rows_ptr,
    slots_ptr,
    route_ptr,
    inter_ptr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    I: tl.constexpr,
    MAX_ROUTES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    expert = tl.program_id(0)
    block_m = tl.program_id(1)
    block_i = tl.program_id(2)
    count = tl.load(counts_ptr + expert).to(tl.int32)
    start_m = block_m * BLOCK_M
    if start_m >= count:
        return

    route_indices = start_m + tl.arange(0, BLOCK_M)
    offsets_i = block_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offsets_d = tl.arange(0, BLOCK_D)
    valid_m = route_indices < count
    rows = tl.load(
        rows_ptr + expert * MAX_ROUTES + route_indices, mask=valid_m, other=0
    ).to(tl.int32)
    slots = tl.load(
        slots_ptr + expert * MAX_ROUTES + route_indices, mask=valid_m, other=0
    ).to(tl.int32)

    gate_acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    for start_d in range(0, D, BLOCK_D):
        ds = start_d + offsets_d
        x = tl.load(
            x_ptr + rows[:, None] * D + ds[None, :],
            mask=valid_m[:, None] & (ds[None, :] < D),
            other=0.0,
        )
        gate = tl.load(
            gate_ptr + (expert * I + offsets_i[None, :]) * D + ds[:, None],
            mask=(offsets_i[None, :] < I) & (ds[:, None] < D),
            other=0.0,
        )
        up = tl.load(
            up_ptr + (expert * I + offsets_i[None, :]) * D + ds[:, None],
            mask=(offsets_i[None, :] < I) & (ds[:, None] < D),
            other=0.0,
        )
        gate_acc += tl.dot(x, gate)
        up_acc += tl.dot(x, up)

    route = tl.load(
        route_ptr + rows * TOPK + slots, mask=valid_m, other=0.0
    ).to(tl.float32)
    silu = gate_acc * (1.0 / (1.0 + tl.exp(-gate_acc)))
    value = silu * up_acc * route[:, None]
    tl.store(
        inter_ptr
        + ((rows[:, None] * TOPK + slots[:, None]) * I + offsets_i[None, :]),
        value.to(inter_ptr.dtype.element_ty),
        mask=valid_m[:, None] & (offsets_i[None, :] < I),
    )


@triton.jit
def _down_route_kernel(
    inter_ptr,
    down_ptr,
    counts_ptr,
    rows_ptr,
    slots_ptr,
    route_out_ptr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    I: tl.constexpr,
    MAX_ROUTES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    expert = tl.program_id(0)
    block_m = tl.program_id(1)
    block_d = tl.program_id(2)
    count = tl.load(counts_ptr + expert).to(tl.int32)
    start_m = block_m * BLOCK_M
    if start_m >= count:
        return

    route_indices = start_m + tl.arange(0, BLOCK_M)
    offsets_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offsets_i = tl.arange(0, BLOCK_I)
    valid_m = route_indices < count
    rows = tl.load(
        rows_ptr + expert * MAX_ROUTES + route_indices, mask=valid_m, other=0
    ).to(tl.int32)
    slots = tl.load(
        slots_ptr + expert * MAX_ROUTES + route_indices, mask=valid_m, other=0
    ).to(tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for start_i in range(0, I, BLOCK_I):
        indices_i = start_i + offsets_i
        x = tl.load(
            inter_ptr
            + ((rows[:, None] * TOPK + slots[:, None]) * I + indices_i[None, :]),
            mask=valid_m[:, None] & (indices_i[None, :] < I),
            other=0.0,
        )
        weight = tl.load(
            down_ptr + (expert * D + offsets_d[None, :]) * I + indices_i[:, None],
            mask=(offsets_d[None, :] < D) & (indices_i[:, None] < I),
            other=0.0,
        )
        acc += tl.dot(x, weight)

    tl.store(
        route_out_ptr
        + ((rows[:, None] * TOPK + slots[:, None]) * D + offsets_d[None, :]),
        acc,
        mask=valid_m[:, None] & (offsets_d[None, :] < D),
    )


@triton.jit
def _reduce_topk_kernel(
    route_out_ptr,
    selected_ptr,
    out_ptr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offsets_d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets_d < D
    # LingBot uses TOPK=4.  Expert-sorted reduction approximates the upstream expert-major block
    # arrival order while making the result independent of SM scheduling.
    tl.static_assert(TOPK == 4)
    e0 = tl.load(selected_ptr + row * TOPK + 0).to(tl.int32)
    e1 = tl.load(selected_ptr + row * TOPK + 1).to(tl.int32)
    e2 = tl.load(selected_ptr + row * TOPK + 2).to(tl.int32)
    e3 = tl.load(selected_ptr + row * TOPK + 3).to(tl.int32)
    v0 = tl.load(route_out_ptr + (row * TOPK + 0) * D + offsets_d, mask=mask, other=0.0)
    v1 = tl.load(route_out_ptr + (row * TOPK + 1) * D + offsets_d, mask=mask, other=0.0)
    v2 = tl.load(route_out_ptr + (row * TOPK + 2) * D + offsets_d, mask=mask, other=0.0)
    v3 = tl.load(route_out_ptr + (row * TOPK + 3) * D + offsets_d, mask=mask, other=0.0)

    swap = e0 > e1
    e0, e1 = tl.where(swap, e1, e0), tl.where(swap, e0, e1)
    v0, v1 = tl.where(swap, v1, v0), tl.where(swap, v0, v1)
    swap = e2 > e3
    e2, e3 = tl.where(swap, e3, e2), tl.where(swap, e2, e3)
    v2, v3 = tl.where(swap, v3, v2), tl.where(swap, v2, v3)
    swap = e0 > e2
    e0, e2 = tl.where(swap, e2, e0), tl.where(swap, e0, e2)
    v0, v2 = tl.where(swap, v2, v0), tl.where(swap, v0, v2)
    swap = e1 > e3
    e1, e3 = tl.where(swap, e3, e1), tl.where(swap, e1, e3)
    v1, v3 = tl.where(swap, v3, v1), tl.where(swap, v1, v3)
    swap = e1 > e2
    v1, v2 = tl.where(swap, v2, v1), tl.where(swap, v1, v2)

    acc = v0 + v1
    acc += v2
    acc += v3
    tl.store(out_ptr + row * D + offsets_d, acc, mask=mask)


def lingbot_sparse_moe(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    workspace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute LingBot's routed experts without evaluating unselected experts."""
    if hidden_states.ndim != 2 or not hidden_states.is_cuda:
        raise ValueError("LingBot sparse MoE expects a 2D CUDA hidden-state tensor")
    if selected_experts.ndim != 2 or routing_weights.shape != selected_experts.shape:
        raise ValueError("routing weights and selected experts must have the same 2D shape")

    tokens, hidden = hidden_states.shape
    experts, intermediate, weight_hidden = gate_weight.shape
    top_k = selected_experts.shape[1]
    if weight_hidden != hidden or up_weight.shape != gate_weight.shape:
        raise ValueError("gate/up expert weights do not match hidden-state dimensions")
    if down_weight.shape != (experts, hidden, intermediate):
        raise ValueError("down expert weights do not match hidden-state dimensions")
    if top_k > experts:
        raise ValueError("top_k cannot exceed the number of experts")

    if workspace is None:
        workspace = _allocate_workspace(
            tokens, hidden, experts, top_k, intermediate,
            hidden_states.device, hidden_states.dtype,
        )
    counts = workspace["counts"]
    rows = workspace["rows"]
    slots = workspace["slots"]
    inter = workspace["inter"]
    route_out = workspace.get("route_out")
    if route_out is None or route_out.shape != (tokens, top_k, hidden):
        route_out = torch.empty(
            (tokens, top_k, hidden), device=hidden_states.device, dtype=torch.float32
        )
        workspace["route_out"] = route_out
    out = workspace["out"]

    selected = selected_experts if selected_experts.is_contiguous() else selected_experts.contiguous()
    route = routing_weights if routing_weights.is_contiguous() else routing_weights.contiguous()
    max_routes = tokens

    _zero_i32_kernel[(1,)](
        counts, experts, BLOCK=triton.next_power_of_2(experts), num_warps=1
    )
    _pack_topk_kernel[(tokens,)](
        selected,
        counts,
        rows,
        slots,
        top_k,
        max_routes,
        BLOCK_K=triton.next_power_of_2(top_k),
        num_warps=1,
    )
    _gate_up_grouped_kernel[
        (experts, triton.cdiv(max_routes, 16), triton.cdiv(intermediate, 64))
    ](
        hidden_states,
        gate_weight,
        up_weight,
        counts,
        rows,
        slots,
        route,
        inter,
        hidden,
        top_k,
        intermediate,
        max_routes,
        BLOCK_M=16,
        BLOCK_I=64,
        BLOCK_D=64,
        num_warps=4,
        num_stages=4,
    )
    _down_route_kernel[
        (experts, triton.cdiv(max_routes, 16), triton.cdiv(hidden, 64))
    ](
        inter,
        down_weight,
        counts,
        rows,
        slots,
        route_out,
        hidden,
        top_k,
        intermediate,
        max_routes,
        BLOCK_M=16,
        BLOCK_D=64,
        BLOCK_I=64,
        num_warps=4,
        num_stages=2,
    )
    _reduce_topk_kernel[(tokens, triton.cdiv(hidden, 256))](
        route_out, selected, out, hidden, top_k, BLOCK_D=256, num_warps=4
    )
    return out.reshape_as(hidden_states)


def _allocate_workspace(tokens, hidden, experts, top_k, intermediate, device, dtype):
    return {
        "counts": torch.empty((experts,), device=device, dtype=torch.int32),
        "rows": torch.empty((experts, tokens), device=device, dtype=torch.int32),
        "slots": torch.empty((experts, tokens), device=device, dtype=torch.int32),
        "inter": torch.empty((tokens, top_k, intermediate), device=device, dtype=dtype),
        "route_out": torch.empty((tokens, top_k, hidden), device=device, dtype=torch.float32),
        "out": torch.empty((tokens, hidden), device=device, dtype=torch.float32),
    }


class _FusedInferenceExperts(nn.Module):
    """Contiguous expert weights plus stable workspaces for CUDA Graph capture."""

    def __init__(self, eager_experts: nn.ModuleList):
        super().__init__()
        if not eager_experts:
            raise ValueError("cannot convert an empty expert list")
        self.num_experts = len(eager_experts)
        self.intermediate_size = eager_experts[0].gate_proj.weight.shape[0]
        with torch.no_grad():
            self.gate_proj = nn.Parameter(
                torch.stack([expert.gate_proj.weight for expert in eager_experts]),
                requires_grad=False,
            )
            self.up_proj = nn.Parameter(
                torch.stack([expert.up_proj.weight for expert in eager_experts]),
                requires_grad=False,
            )
            self.down_proj = nn.Parameter(
                torch.stack([expert.down_proj.weight for expert in eager_experts]),
                requires_grad=False,
            )
        self._workspace = None
        self._workspace_key = None

    def _get_robby_moe_workspace(self, hidden_states, top_k):
        tokens, hidden = hidden_states.shape
        key = (tokens, int(top_k), hidden, hidden_states.dtype, hidden_states.device)
        if self._workspace is None or self._workspace_key != key:
            self._workspace = _allocate_workspace(
                tokens,
                hidden,
                self.num_experts,
                int(top_k),
                self.intermediate_size,
                hidden_states.device,
                hidden_states.dtype,
            )
            self._workspace_key = key
        return self._workspace

    def forward(self, module, num_experts, routing_weights, selected_experts, hidden_states):
        # Retain upstream's fallback seam if a future shape falls outside this kernel's contract.
        from lingbotvla.ops.fused_moe import fused_moe_forward

        return fused_moe_forward(
            module=module,
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            fc1_1_weight=self.gate_proj,
            fc1_2_weight=self.up_proj,
            fc2_weight=self.down_proj,
        )


@dataclass
class MoeKernelInstall:
    layers: int
    converted_layers: int
    weight_bytes: int
    _module: object
    _closed: bool = False

    def close(self) -> None:
        global _PATCH_USERS, _PATCH_ORIGINAL

        with _PATCH_LOCK:
            if self._closed:
                return
            self._closed = True
            _PATCH_USERS -= 1
            if _PATCH_USERS == 0:
                if self._module.robby_moe_forward is lingbot_sparse_moe:
                    self._module.robby_moe_forward = _PATCH_ORIGINAL
                _PATCH_ORIGINAL = None


def install_lingbot_moe_kernel(model: nn.Module) -> MoeKernelInstall:
    """Convert eager routed experts and patch LingBot's module-global inference callable."""
    _refuse_on_sm110()
    global _PATCH_USERS, _PATCH_ORIGINAL

    from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert as qwen_moe

    blocks = [module for module in model.modules() if isinstance(module, qwen_moe.Qwen2TokenMoeBlock)]
    if not blocks:
        raise RuntimeError("no LingBot Qwen2TokenMoeBlock modules found")
    if any(int(block.top_k) != 4 for block in blocks):
        raise RuntimeError("the LingBot sparse-MoE CUDA kernel currently requires top_k=4")

    converted = 0
    weight_bytes = 0
    for block in blocks:
        if isinstance(block.experts, nn.ModuleList):
            eager_experts = block.experts
            fused = _FusedInferenceExperts(eager_experts)
            weight_bytes += sum(
                parameter.numel() * parameter.element_size() for parameter in fused.parameters()
            )
            block.experts = fused
            block._moe_implementation = "fused"
            converted += 1
        elif not all(hasattr(block.experts, name) for name in ("gate_proj", "up_proj", "down_proj")):
            raise RuntimeError(f"unsupported LingBot expert container: {type(block.experts).__name__}")

    with _PATCH_LOCK:
        if _PATCH_USERS == 0:
            _PATCH_ORIGINAL = qwen_moe.robby_moe_forward
            qwen_moe.robby_moe_forward = lingbot_sparse_moe
        elif qwen_moe.robby_moe_forward is not lingbot_sparse_moe:
            raise RuntimeError("LingBot's process-global MoE callable changed while in use")
        _PATCH_USERS += 1
    return MoeKernelInstall(len(blocks), converted, weight_bytes, qwen_moe)


def _refuse_on_sm110() -> None:
    """Triton is measured-dead on Thor sm_110a (PTXAS internal codegen error), and the vendor
    fallback path crashes (undefined ``logger`` in the MoE except-handler; the RMSNorm patch has
    no try/except). Installing here would fail later and worse; refuse up front instead."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (11, 0):
        raise RuntimeError(
            "LingBot-VLA-V2 Triton kernels are not supported on SM110 (Thor); use the Thor "
            "engine arm instead.")


__all__ = ["MoeKernelInstall", "install_lingbot_moe_kernel", "lingbot_sparse_moe"]

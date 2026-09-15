"""Correct-by-construction torch reference for the V2 token-MoE block.

Ground truth: ``Qwen2TokenMoeBlock.forward`` (qwen2_action_expert.py
:274-362) under the shipped checkpoint config (lingbotvla_cli.yaml):

    num_experts            = 32
    top_k                  = 4
    moe_intermediate_size  = 512
    shared_intermediate    = 704
    router_activation      = sigmoid
    routed_scaling_factor  = 4.0
    norm_topk_prob         = True   (hard-coded in _install_moe_blocks)
    use_shared_expert_gate = False  (no shared_expert_gate weight in ckpt)
    e_score_correction_bias: persistent buffer, LOADED FROM CKPT —
        added to the sigmoid scores for top-k SELECTION only; the
        routing weights are the raw sigmoid scores gathered at the
        selected indices (:292-294).

Reference semantics replicated exactly:
    1. router logits in TRUE fp32 (autocast disabled, :284-285) —
       bf16 logits can flip top-k selection on near-ties;
    2. scores = sigmoid(logits);
    3. selection on scores + bias; weights = scores.gather(sel);
    4. weights /= (weights.sum(-1, keepdim=True) + 1e-20);
    5. weights *= 4.0; cast to hidden dtype;
    6. routed experts (fused 3D weights: gate/up [E, 512, 768],
       down [E, 768, 512]); SwiGLU per expert;
    7. shared expert (704) added UNGATED.

Two independent combine implementations are provided (dense one-hot
einsum — the reference's own eager path — and a gather loop that
mirrors the planned engine dataflow). ``self_check()`` asserts they
agree on CPU, which is the import-time-verifiable "numerically
checkable baseline" Stage 2 diffs the engine against.

This module needs no GPU and no flash_rt_kernels.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

NUM_EXPERTS = 32
TOP_K = 4
MOE_H = 512
SHARED_H = 704
HIDDEN = 768
ROUTED_SCALING = 4.0
NORM_EPS = 1e-20


class MoeIntermediates(NamedTuple):
    """Per-call intermediates for engine-side layerwise diffing."""

    router_logits: torch.Tensor      # (T, E) fp32
    routing_scores: torch.Tensor     # (T, E) fp32 (sigmoid)
    selected_experts: torch.Tensor   # (T, top_k) int64
    routing_weights: torch.Tensor    # (T, top_k) hidden dtype (post norm+scale)
    expert_slab: torch.Tensor        # (E, T, HIDDEN) all-expert outputs
    routed_out: torch.Tensor         # (T, HIDDEN) pre-shared
    shared_out: torch.Tensor         # (T, HIDDEN)


def route(
    hidden_flat: torch.Tensor,
    gate_w: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    *,
    top_k: int = TOP_K,
    routed_scaling_factor: float = ROUTED_SCALING,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 sigmoid router with bias-corrected top-k selection.

    Returns (router_logits fp32, routing_scores fp32, selected int64,
    routing_weights in hidden dtype).
    """
    router_logits = F.linear(hidden_flat.float(), gate_w.float())
    routing_scores = torch.sigmoid(router_logits)
    scores_for_choice = routing_scores + e_score_correction_bias.float().unsqueeze(0)
    _, selected = torch.topk(scores_for_choice, top_k, dim=-1)
    weights = routing_scores.gather(1, selected)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + NORM_EPS)
    if routed_scaling_factor != 1.0:
        weights = weights * routed_scaling_factor
    return router_logits, routing_scores, selected, weights.to(hidden_flat.dtype)


def experts_dense_slab(
    hidden_flat: torch.Tensor,
    gate_proj: torch.Tensor,   # (E, MOE_H, HIDDEN)
    up_proj: torch.Tensor,     # (E, MOE_H, HIDDEN)
    down_proj: torch.Tensor,   # (E, HIDDEN, MOE_H)
) -> torch.Tensor:
    """All-expert dense outputs, (E, T, HIDDEN).

    This is the exact dataflow of the planned engine dense loop:
    batched GEMM (E, T, HIDDEN) @ (E, HIDDEN, MOE_H)ᵀ → SwiGLU →
    batched GEMM → slab. torch.bmm here; strided-batched cuBLASLt fp8
    on the engine side.
    """
    x = hidden_flat.unsqueeze(0).expand(gate_proj.shape[0], -1, -1)
    g = torch.bmm(x, gate_proj.transpose(1, 2))
    u = torch.bmm(x, up_proj.transpose(1, 2))
    h = F.silu(g) * u
    return torch.bmm(h, down_proj.transpose(1, 2))


def combine_onehot(
    expert_slab: torch.Tensor,       # (E, T, HIDDEN)
    selected: torch.Tensor,          # (T, top_k)
    weights: torch.Tensor,           # (T, top_k)
    num_experts: int = NUM_EXPERTS,
) -> torch.Tensor:
    """The reference's own eager combine (:344-351): one-hot einsum."""
    expert_mask = F.one_hot(selected, num_classes=num_experts).float()
    w = (expert_mask * weights.unsqueeze(-1).float()).sum(dim=1)
    return torch.einsum("ebd,be->bd", expert_slab.float(), w).to(expert_slab.dtype)


def combine_gather(
    expert_slab: torch.Tensor,
    selected: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Gather-weighted combine — mirrors the planned engine kernel:
    out[t] = sum_k w[t,k] * slab[sel[t,k], t]."""
    t_idx = torch.arange(selected.shape[0], device=selected.device)
    out = torch.zeros(
        selected.shape[0], expert_slab.shape[-1],
        dtype=torch.float32, device=expert_slab.device)
    for k in range(selected.shape[1]):
        out += weights[:, k].float().unsqueeze(-1) \
            * expert_slab[selected[:, k], t_idx].float()
    return out.to(expert_slab.dtype)


def shared_expert_forward(
    hidden_flat: torch.Tensor,
    shared_gate_w: torch.Tensor,   # (SHARED_H, HIDDEN)
    shared_up_w: torch.Tensor,     # (SHARED_H, HIDDEN)
    shared_down_w: torch.Tensor,   # (HIDDEN, SHARED_H)
) -> torch.Tensor:
    """Ungated shared expert (use_shared_expert_gate=false in ckpt)."""
    g = F.linear(hidden_flat, shared_gate_w)
    u = F.linear(hidden_flat, shared_up_w)
    return F.linear(F.silu(g) * u, shared_down_w)


class Vla2TokenMoeRef(nn.Module):
    """One V2 MoE layer, loadable straight from the hf_ckpt tensors.

    Expected state-dict keys (relative), dtypes as shipped (fp32):
        gate.weight                 (32, 768)
        e_score_correction_bias     (32,)
        experts.gate_proj           (32, 512, 768)
        experts.up_proj             (32, 512, 768)
        experts.down_proj           (32, 768, 512)
        shared_expert.gate_proj.weight  (704, 768)
        shared_expert.up_proj.weight    (704, 768)
        shared_expert.down_proj.weight  (768, 704)
    """

    def __init__(self, *, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.gate_w = nn.Parameter(
            torch.zeros(NUM_EXPERTS, HIDDEN, dtype=dtype), requires_grad=False)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(NUM_EXPERTS, dtype=torch.float32), requires_grad=False)
        self.experts_gate = nn.Parameter(
            torch.zeros(NUM_EXPERTS, MOE_H, HIDDEN, dtype=dtype),
            requires_grad=False)
        self.experts_up = nn.Parameter(
            torch.zeros(NUM_EXPERTS, MOE_H, HIDDEN, dtype=dtype),
            requires_grad=False)
        self.experts_down = nn.Parameter(
            torch.zeros(NUM_EXPERTS, HIDDEN, MOE_H, dtype=dtype),
            requires_grad=False)
        self.shared_gate_w = nn.Parameter(
            torch.zeros(SHARED_H, HIDDEN, dtype=dtype), requires_grad=False)
        self.shared_up_w = nn.Parameter(
            torch.zeros(SHARED_H, HIDDEN, dtype=dtype), requires_grad=False)
        self.shared_down_w = nn.Parameter(
            torch.zeros(HIDDEN, SHARED_H, dtype=dtype), requires_grad=False)

    @classmethod
    def from_tensors(cls, t: dict, *, dtype: torch.dtype = torch.float32):
        m = cls(dtype=dtype)
        with torch.no_grad():
            m.gate_w.copy_(t["gate.weight"])
            m.e_score_correction_bias.copy_(t["e_score_correction_bias"])
            m.experts_gate.copy_(t["experts.gate_proj"])
            m.experts_up.copy_(t["experts.up_proj"])
            m.experts_down.copy_(t["experts.down_proj"])
            m.shared_gate_w.copy_(t["shared_expert.gate_proj.weight"])
            m.shared_up_w.copy_(t["shared_expert.up_proj.weight"])
            m.shared_down_w.copy_(t["shared_expert.down_proj.weight"])
        return m

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, T, HIDDEN) or (T, HIDDEN)
        *,
        return_intermediates: bool = False,
    ):
        squeeze = hidden_states.dim() == 2
        if squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        b, t, d = hidden_states.shape
        flat = hidden_states.reshape(-1, d)

        logits, scores, selected, weights = route(
            flat, self.gate_w, self.e_score_correction_bias)
        slab = experts_dense_slab(
            flat, self.experts_gate, self.experts_up, self.experts_down)
        routed = combine_onehot(slab, selected, weights)
        shared = shared_expert_forward(
            flat, self.shared_gate_w, self.shared_up_w, self.shared_down_w)
        out = (routed.to(flat.dtype) + shared).reshape(b, t, d)
        if squeeze:
            out = out.squeeze(0)
        if return_intermediates:
            return out, MoeIntermediates(
                logits, scores, selected, weights, slab, routed, shared)
        return out


def self_check(seed: int = 0, tokens: int = 51, atol: float = 1e-5) -> dict:
    """CPU cross-check: one-hot combine vs gather combine must agree.

    Two independently-written combines agreeing over random routing is
    the correctness bar for the Stage-2 engine combine kernel.
    """
    g = torch.Generator().manual_seed(seed)
    flat = torch.randn(tokens, HIDDEN, generator=g)
    gate_w = torch.randn(NUM_EXPERTS, HIDDEN, generator=g) * 0.02
    bias = torch.randn(NUM_EXPERTS, generator=g) * 0.1
    eg = torch.randn(NUM_EXPERTS, MOE_H, HIDDEN, generator=g) * 0.02
    eu = torch.randn(NUM_EXPERTS, MOE_H, HIDDEN, generator=g) * 0.02
    ed = torch.randn(NUM_EXPERTS, HIDDEN, MOE_H, generator=g) * 0.02
    _, _, selected, weights = route(flat, gate_w, bias)
    slab = experts_dense_slab(flat, eg, eu, ed)
    a = combine_onehot(slab, selected, weights)
    b = combine_gather(slab, selected, weights)
    max_d = (a - b).abs().max().item()
    assert max_d <= atol, f"combine mismatch: {max_d}"
    return {"max_abs_diff": max_d, "tokens": tokens,
            "unique_experts_hit": int(selected.unique().numel())}


__all__ = [
    "NUM_EXPERTS", "TOP_K", "MOE_H", "SHARED_H", "HIDDEN", "ROUTED_SCALING",
    "route", "experts_dense_slab", "combine_onehot", "combine_gather",
    "shared_expert_forward", "Vla2TokenMoeRef", "MoeIntermediates",
    "self_check",
]

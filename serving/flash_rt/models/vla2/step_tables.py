"""AdaRMS step tables for the LingBot-VLA-V2-6B action expert.

The V2 expert's time conditioning is byte-for-byte the same machinery
as V1/vla4b, verified against the V2 source + checkpoint:

* ``AdaRMSNorm`` (modeling_lingbot_vla.py :212-240, imported by the V2
  model file): ``rmsnorm(x) * weight``, then
  ``(1 + gamma(cond)) * · + beta(cond)`` — gamma/beta are Linear(768→768)
  (weights confirmed in the ckpt:
  ``...input_layernorm.{gamma,beta}.weight [768, 768]``).
* cond is the RAW sinusoidal time embedding
  (``embed_suffix`` returns ``time_emb_ori``; ``separate_time_proj``
  is False in the shipped config), dimension 768, min_period 4e-3,
  max_period 4.0 — identical to vla4b.
* ``adanorm_time=true``, ``final_norm_adanorm=false`` in the shipped
  ``lingbotvla_cli.yaml`` → both per-layer norms are AdaRMS, the final
  ``model.norm`` is a plain FixQwen2RMSNorm — same as vla4b.
* ``action_time_mlp_in`` is [768, 1536] (action half | time half) with
  bias — identical split → ``t_contrib`` precompute carries over.

FORK vs vla4b (V2 parity divergence, found during M1 bring-up
2026-08-25): the vla4b builder evaluates the sinusoid at the EXACT
Euler times t_s = 1 - s/10.  The stock V2 loop
(``FlowMatchingV2.sample_actions``, modeling_..._v2.py :971-1000) runs
``time``/``dt`` as **bf16 scalars** when served in bf16 (dtype =
state.dtype), so the times the deployed model actually conditions on
are the bf16-accumulated values

    1.0, 0.898438, 0.796875, 0.695312, 0.59375,
    0.494141, 0.394531, 0.294922, 0.195312, 0.095215

— up to 6.25e-3 away from 1 - s/10.  With min_period 4e-3 the fastest
sinusoid channels see phase errors up to ~9.8 rad, so tables built at
the exact times diverge from the stock conditioning.  MEASURED (M1
run 2 schedule ablation, matched inputs, 8 obs, stock-matched rope
tables): torch-ref loop max action delta vs stock e2e = 0.281 with
the exact schedule vs 0.094 with the stock bf16 schedule (fp32 KV) —
3x worse, outside the final 16-draw null envelope max 0.3125's
useful margin and dominated by the schedule term.  See
/home/ubuntu/iwm_distill/thor_t2v2/m1_gate.json (schedule_ablation).  This
module therefore owns its builder with an explicit ``t_values``
schedule instead of re-exporting vla4b's.  The fold math itself is
unchanged (single source of truth for the sinusoid stays
``vla4b.step_tables.time_embedding``); the engine adarms kernels still
compute ``rmsnorm(x) * (1 + scale) + shift`` with
``scale = weight * (1 + gamma(t)) - 1`` and ``shift = beta(t)``.

Do NOT edit the vla4b module — V1's stock baseline is its own gate.
"""
from __future__ import annotations

import torch

from flash_rt.models.vla4b.step_tables import (  # noqa: F401
    EXP_D,
    SUF,
    time_embedding,
)

STEPS = 10


def stock_bf16_time_schedule(steps: int = STEPS) -> list[float]:
    """The Euler times the STOCK V2 bf16 serving loop actually uses.

    Replicates ``sample_actions``: ``time``/``dt`` are bf16 tensors,
    ``time += dt`` per step, loop while ``time >= -dt/2``.
    """
    t = torch.tensor(1.0, dtype=torch.bfloat16)
    dt = torch.tensor(-1.0 / steps, dtype=torch.bfloat16)
    out: list[float] = []
    while float(t) >= -float(dt) / 2:
        out.append(float(t))
        t = t + dt
    assert len(out) == steps, (len(out), steps)
    return out


def exact_time_schedule(steps: int = STEPS) -> list[float]:
    """t_s = 1 - s/steps (the vla4b/V1 builder's schedule)."""
    return [1.0 - s / steps for s in range(steps)]


def build_step_tables(
    ada_params: list,
    atm_t_w32: torch.Tensor,
    atm_in_b32: torch.Tensor,
    *,
    steps: int = STEPS,
    suf: int = SUF,
    t_values: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (style_attn, style_ffn, t_contrib) on CPU.

    Identical fold to ``vla4b.step_tables.build_step_tables`` except the
    time schedule is explicit.  Default: the stock V2 bf16 schedule
    (M1-measured; the exact schedule fails the parity envelope).

    Args:
        ada_params: per-layer dicts with fp32 tensors
            in_w, in_gw, in_gb, in_bw, in_bb,
            po_w, po_gw, po_gb, po_bw, po_bb
        atm_t_w32: [768, 768] time half of action_time_mlp_in.weight
        atm_in_b32: [768] action_time_mlp_in.bias
        t_values: the ``steps`` Euler times to condition on.

    Returns:
        style_attn, style_ffn: (steps*L*suf, 3*768) fp16
        t_contrib: (steps, 768) fp16
    """
    if t_values is None:
        t_values = stock_bf16_time_schedule(steps)
    assert len(t_values) == steps, (len(t_values), steps)
    L = len(ada_params)
    D = EXP_D
    style_a = torch.empty(steps, L, suf, 3 * D, dtype=torch.float16)
    style_f = torch.empty(steps, L, suf, 3 * D, dtype=torch.float16)
    t_contrib = torch.empty(steps, D, dtype=torch.float32)
    ones = torch.ones(D)
    for s in range(steps):
        te = time_embedding(float(t_values[s]))
        t_contrib[s] = atm_in_b32 + atm_t_w32 @ te
        for l in range(L):
            a = ada_params[l]
            for tab, wkey, gw, gb, bw, bb in (
                (style_a, "in_w", "in_gw", "in_gb", "in_bw", "in_bb"),
                (style_f, "po_w", "po_gw", "po_gb", "po_bw", "po_bb"),
            ):
                gamma = a[gw] @ te + a[gb]
                beta = a[bw] @ te + a[bb]
                scale = a[wkey] * (1.0 + gamma) - 1.0
                row = torch.cat([scale, beta, ones]).to(torch.float16)
                tab[s, l] = row.unsqueeze(0).expand(suf, -1)
    return (style_a.reshape(-1, 3 * D).contiguous(),
            style_f.reshape(-1, 3 * D).contiguous(),
            t_contrib.to(torch.float16))


__all__ = [
    "time_embedding",
    "build_step_tables",
    "stock_bf16_time_schedule",
    "exact_time_schedule",
    "EXP_D",
    "SUF",
    "STEPS",
]

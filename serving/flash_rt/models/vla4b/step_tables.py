"""Host-side precomputed step tables for the VLA-4B action expert.

Pure torch (no flash_rt_kernels import) — shared by the Thor frontend
and the H100 torch-twin parity harness so the table math has a single
source of truth.

Tables per denoise step s (t_s = 1 - s/steps):
    time_emb_s = sinusoid(t_s, 768, 4e-3, 4.0)          (fp32)
    style[s, l] rows (51, 3*768) fp16, per norm:
        scale = adaW * (1 + gamma_lin(time_emb_s)) - 1
        shift = beta_lin(time_emb_s)
        gate  = 1
      (the adarms kernels compute rmsnorm(x)*(1+scale)+shift, so the
      AdaRMSNorm.weight is folded into scale)
    t_contrib[s] = action_time_mlp_in.bias
                   + W_t @ time_emb_s      (time half of the mlp_in)

The AdaRMS conditioning input is the RAW sinusoidal embedding
(separate_time_proj = false in the shipped config) — verified against
modeling_lingbot_vla.embed_suffix.
"""
from __future__ import annotations

import math

import torch

EXP_D = 768
SUF = 51


def time_embedding(t_val: float) -> torch.Tensor:
    """create_sinusoidal_pos_embedding(t, 768, 4e-3, 4.0) in fp32."""
    frac = torch.linspace(0.0, 1.0, EXP_D // 2, dtype=torch.float32)
    period = 4e-3 * (4.0 / 4e-3) ** frac
    x = (1.0 / period) * 2 * math.pi * t_val
    return torch.cat([torch.sin(x), torch.cos(x)])


def build_step_tables(
    ada_params: list,
    atm_t_w32: torch.Tensor,
    atm_in_b32: torch.Tensor,
    *,
    steps: int = 10,
    suf: int = SUF,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (style_attn, style_ffn, t_contrib) on CPU.

    Args:
        ada_params: per-layer dicts with fp32 tensors
            in_w, in_gw, in_gb, in_bw, in_bb,
            po_w, po_gw, po_gb, po_bw, po_bb
        atm_t_w32: [768, 768] time half of action_time_mlp_in.weight
        atm_in_b32: [768] action_time_mlp_in.bias

    Returns:
        style_attn, style_ffn: (steps*L*suf, 3*768) fp16
        t_contrib: (steps, 768) fp16
    """
    L = len(ada_params)
    D = EXP_D
    style_a = torch.empty(steps, L, suf, 3 * D, dtype=torch.float16)
    style_f = torch.empty(steps, L, suf, 3 * D, dtype=torch.float16)
    t_contrib = torch.empty(steps, D, dtype=torch.float32)
    ones = torch.ones(D)
    for s in range(steps):
        te = time_embedding(1.0 - s / steps)
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


__all__ = ["time_embedding", "build_step_tables", "EXP_D", "SUF"]

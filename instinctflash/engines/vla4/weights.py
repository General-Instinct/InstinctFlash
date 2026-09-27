"""Prepare a LingBot-VLA-4B checkpoint (Qwen2.5-VL-3B text model + Qwen2 action expert).

Follows the upstream ``LingBotVlaInferencePolicy`` math:

* LM layers: weight-form RMSNorm folded into Q/K/V and gate/up (``W * w``), SiLU MLP,
  Q/K/V bias. Layer 35 only contributes keys/values to the expert.
* Expert layers: AdaRMSNorm ``(1 + gamma(c)) * (w * xhat) + beta(c)`` where ``c`` is the raw
  flow-time sinusoid, precomputed per Euler step as (scale, shift) with
  ``scale = w * (1 + gamma) - 1``. No residual gating; plain weight-form final norm.
* Suffix: ``state_proj`` for the state token; action tokens
  ``mlp_out(silu(mlp_in([action_in_proj(x), tau])))`` with ``action_in_proj`` and the action
  half of ``mlp_in`` folded into one matrix and the time half precomputed per step.
* Schedule: upstream accumulates ``t`` in BF16 from 1.0 with ``dt = bf16(-0.1)``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from instinctflash.engines.pi05.weights import DecoderLayer
from instinctflash.kernels.fp8 import quantize_weight

LM = "model.qwenvl_with_expert.qwenvl.model."
EXPERT = "model.qwenvl_with_expert.qwen_expert.model."
LAYERS = 36


@dataclass
class Vla4Weights:
    lm: list
    expert: list
    embed: torch.Tensor          # CPU [vocab, 2048] FP32 token table
    state_w: torch.Tensor        # [768, 75]
    state_b: torch.Tensor
    act_w: torch.Tensor          # folded action_in_proj -> mlp_in (action half): [768, 75]
    act_c: torch.Tensor          # [steps, 768]: time half of mlp_in + folded biases
    mlp_out_w: torch.Tensor      # [768, 768]
    mlp_out_b: torch.Tensor
    out_w: torch.Tensor          # action_out_proj [75, 768]
    out_b: torch.Tensor
    final_norm: torch.Tensor     # [768]
    mod_attn: torch.Tensor       # [steps, L, 2, 768] (scale, shift)
    mod_ffn: torch.Tensor
    times: list
    dt: float
    config: dict = field(default_factory=dict)


def bf16_schedule(steps: int):
    """Upstream's in-place BF16 time accumulation: t values and dt."""
    dt = torch.tensor(-1.0 / steps, dtype=torch.bfloat16)
    t = torch.tensor(1.0, dtype=torch.bfloat16)
    times = []
    while float(t) >= -float(dt) / 2:
        times.append(float(t))
        t = t + dt
    return times, float(dt)


def sinusoid(times, dim=768, min_period=4e-3, max_period=4.0):
    t = torch.tensor(times, dtype=torch.float32)
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    angle = (2 * math.pi / period)[None, :] * t[:, None]
    # Upstream casts the embedding to BF16 before the conditioning linears.
    return torch.cat([torch.sin(angle), torch.cos(angle)], 1).to(torch.bfloat16).float()


def load_vla4_weights(checkpoint_dir, device, steps: int = 10) -> Vla4Weights:
    from safetensors import safe_open
    path = Path(checkpoint_dir)
    dev = torch.device(device)
    times, dt = bf16_schedule(steps)
    with safe_open(str(path / "model.safetensors"), "pt") as f:
        def get(name):
            return f.get_tensor(name).to(device=dev, dtype=torch.float32)

        lm = []
        for i in range(LAYERS):
            p = f"{LM}layers.{i}."
            g_in = get(p + "input_layernorm.weight")
            qkv = torch.cat([get(p + f"self_attn.{n}_proj.weight") for n in "qkv"]) * g_in[None, :]
            qkv_b = torch.cat([get(p + f"self_attn.{n}_proj.bias") for n in "qkv"])
            if i == LAYERS - 1:
                lm.append(DecoderLayer(quantize_weight(qkv, qkv_b), None, None, None))
                continue
            g_post = get(p + "post_attention_layernorm.weight")
            gate_up = torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")]) * g_post[None, :]
            lm.append(DecoderLayer(quantize_weight(qkv, qkv_b), quantize_weight(get(p + "self_attn.o_proj.weight")),
                                   quantize_weight(gate_up), quantize_weight(get(p + "mlp.down_proj.weight"))))
            del qkv, gate_up

        cond = sinusoid(times).to(dev)                     # [steps, 768]
        expert, mod_attn, mod_ffn = [], [], []
        for i in range(LAYERS):
            p = f"{EXPERT}layers.{i}."
            qkv = torch.cat([get(p + f"self_attn.{n}_proj.weight") for n in "qkv"])
            qkv_b = torch.cat([get(p + f"self_attn.{n}_proj.bias") for n in "qkv"])
            gate_up = torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")])
            expert.append(DecoderLayer(quantize_weight(qkv, qkv_b), quantize_weight(get(p + "self_attn.o_proj.weight")),
                                       quantize_weight(gate_up), quantize_weight(get(p + "mlp.down_proj.weight"))))
            for norm, table in (("input_layernorm", mod_attn), ("post_attention_layernorm", mod_ffn)):
                w = get(p + f"{norm}.weight")
                gamma = cond @ get(p + f"{norm}.gamma.weight").t() + get(p + f"{norm}.gamma.bias")
                beta = cond @ get(p + f"{norm}.beta.weight").t() + get(p + f"{norm}.beta.bias")
                table.append(torch.stack([w * (1 + gamma) - 1, beta], 1))   # [steps, 2, 768]
        mlp_in_w, mlp_in_b = get("model.action_time_mlp_in.weight"), get("model.action_time_mlp_in.bias")
        w_a, w_t = mlp_in_w[:, :768], mlp_in_w[:, 768:]
        ain_w, ain_b = get("model.action_in_proj.weight"), get("model.action_in_proj.bias")
        act_c = cond @ w_t.t() + (w_a @ ain_b + mlp_in_b)[None, :]
        weights = Vla4Weights(
            lm=lm, expert=expert,
            embed=f.get_tensor(LM + "embed_tokens.weight").float(),
            state_w=get("model.state_proj.weight"), state_b=get("model.state_proj.bias"),
            act_w=(w_a @ ain_w).contiguous(), act_c=act_c.contiguous(),
            mlp_out_w=get("model.action_time_mlp_out.weight"), mlp_out_b=get("model.action_time_mlp_out.bias"),
            out_w=get("model.action_out_proj.weight"), out_b=get("model.action_out_proj.bias"),
            final_norm=get(EXPERT + "norm.weight"),
            mod_attn=torch.stack(mod_attn, 1).contiguous(), mod_ffn=torch.stack(mod_ffn, 1).contiguous(),
            times=times, dt=dt, config=json.loads((path / "config.json").read_text()))
    torch.cuda.empty_cache()
    return weights

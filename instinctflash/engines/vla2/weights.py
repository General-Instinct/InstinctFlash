"""Prepare a LingBot-VLA-V2 checkpoint (Qwen3-VL-4B text model + MoE flow-matching expert).

Follows upstream ``LingbotVLAv2`` inference:

* LM (36 layers, 2560 wide, 32/8 heads x 128): weight-form RMSNorm folded into the
  following projections, per-head ``q_norm``/``k_norm``, SiLU MLP, no attention bias.
* Expert (36 layers, 768 wide, same head geometry, Q/K/V bias): AdaRMSNorm from the raw
  time sinusoid (precomputed per step) and a 32-expert top-4 sigmoid-routed MoE with an
  ungated shared expert. The MoE is stored as two dense FP8 matrices over concatenated
  experts: gate/up rows ``[gate_e0..gate_e31, gate_shared, up_e0..up_e31, up_shared]``
  and down columns in the same order.
* Constant prefix pieces: vision start/end token embeddings and the 16 align tokens.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from instinctflash.engines.vla4.weights import bf16_schedule, sinusoid
from instinctflash.kernels.fp8 import quantize_weight

LM = "model.qwenvl_with_expert.qwenvl.model.language_model."
EXPERT = "model.qwenvl_with_expert.qwen_expert.model."
LAYERS, EXPERTS, EXPERT_FF, SHARED_FF, TOP_K, ROUTE_SCALE = 36, 32, 512, 704, 4, 4.0
VISION_START, VISION_END = 151652, 151653


@dataclass
class LmLayer:
    qkv: object
    o: object
    gate_up: object
    down: object
    q_norm: torch.Tensor
    k_norm: torch.Tensor


@dataclass
class MoeLayer:
    qkv: object
    o: object
    gate_up: object              # [2 * (E*512 + 704), 768]
    down: object                 # [768, E*512 + 704]
    router: torch.Tensor         # [E, 768] FP32
    router_bias: torch.Tensor    # [E] selection-only correction


@dataclass
class Vla2Weights:
    lm: list
    expert: list
    embed: torch.Tensor          # CPU [vocab, 2560] FP32
    vision_start: torch.Tensor   # [2560]
    vision_end: torch.Tensor
    align: torch.Tensor          # [16, 2560]
    state_w: torch.Tensor
    state_b: torch.Tensor
    act_w: torch.Tensor
    act_c: torch.Tensor
    mlp_out_w: torch.Tensor
    mlp_out_b: torch.Tensor
    out_w: torch.Tensor
    out_b: torch.Tensor
    final_norm: torch.Tensor
    mod_attn: torch.Tensor       # [steps, L, 2, 768]
    mod_ffn: torch.Tensor
    inv_freq: torch.Tensor       # BF16-rounded Qwen3-VL rotary frequencies
    sections: torch.Tensor       # [64] int32: 0 = t, 1 = h, 2 = w
    times: list
    dt: float
    config: dict = field(default_factory=dict)


class _Shards:
    def __init__(self, root: Path):
        from safetensors import safe_open
        index = root / "model.safetensors.index.json"
        self.map = json.loads(index.read_text())["weight_map"]
        self.files = {f: safe_open(str(root / f), "pt") for f in set(self.map.values())}

    def get(self, key):
        return self.files[self.map[key]].get_tensor(key)


def mrope_sections(head_dim=128, sections=(24, 20, 20)):
    """Qwen3-VL interleaved M-RoPE: frequency i uses h if i%3==1, w if i%3==2 (within the
    h/w section budget), otherwise t."""
    half = head_dim // 2
    out = torch.zeros(half, dtype=torch.int32)
    limit = 3 * min(sections[1], sections[2])
    for i in range(half):
        if i < limit and i % 3 == 1:
            out[i] = 1
        elif i < limit and i % 3 == 2:
            out[i] = 2
    return out


def load_vla2_weights(checkpoint_dir, device, steps: int = 10, rope_theta: float = 5e6) -> Vla2Weights:
    root = Path(checkpoint_dir)
    shards = _Shards(root)
    dev = torch.device(device)
    times, dt = bf16_schedule(steps)

    def get(name):
        return shards.get(name).to(device=dev, dtype=torch.float32)

    lm = []
    for i in range(LAYERS):
        p = f"{LM}layers.{i}."
        g_in = get(p + "input_layernorm.weight")
        qkv = torch.cat([get(p + f"self_attn.{n}_proj.weight") for n in "qkv"]) * g_in[None, :]
        qn, kn = get(p + "self_attn.q_norm.weight"), get(p + "self_attn.k_norm.weight")
        if i == LAYERS - 1:
            lm.append(LmLayer(quantize_weight(qkv), None, None, None, qn, kn))
            continue
        g_post = get(p + "post_attention_layernorm.weight")
        gate_up = torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")]) * g_post[None, :]
        lm.append(LmLayer(quantize_weight(qkv), quantize_weight(get(p + "self_attn.o_proj.weight")),
                          quantize_weight(gate_up), quantize_weight(get(p + "mlp.down_proj.weight")), qn, kn))
        del qkv, gate_up

    cond = sinusoid(times).to(dev)
    expert, mod_attn, mod_ffn = [], [], []
    for i in range(LAYERS):
        p = f"{EXPERT}layers.{i}."
        qkv = torch.cat([get(p + f"self_attn.{n}_proj.weight") for n in "qkv"])
        qkv_b = torch.cat([get(p + f"self_attn.{n}_proj.bias") for n in "qkv"])
        eg = get(p + "mlp.experts.gate_proj").reshape(EXPERTS * EXPERT_FF, -1)
        eu = get(p + "mlp.experts.up_proj").reshape(EXPERTS * EXPERT_FF, -1)
        ed = get(p + "mlp.experts.down_proj").permute(1, 0, 2).reshape(-1, EXPERTS * EXPERT_FF)
        sg, su = get(p + "mlp.shared_expert.gate_proj.weight"), get(p + "mlp.shared_expert.up_proj.weight")
        sd = get(p + "mlp.shared_expert.down_proj.weight")
        gate_up = torch.cat([eg, sg, eu, su])
        down = torch.cat([ed, sd], 1)
        expert.append(MoeLayer(quantize_weight(qkv, qkv_b), quantize_weight(get(p + "self_attn.o_proj.weight")),
                               quantize_weight(gate_up), quantize_weight(down),
                               get(p + "mlp.gate.weight").contiguous(),
                               get(p + "mlp.e_score_correction_bias").contiguous()))
        del gate_up, down, eg, eu, ed
        for norm, table in (("input_layernorm", mod_attn), ("post_attention_layernorm", mod_ffn)):
            w = get(p + f"{norm}.weight")
            gamma = cond @ get(p + f"{norm}.gamma.weight").t() + get(p + f"{norm}.gamma.bias")
            beta = cond @ get(p + f"{norm}.beta.weight").t() + get(p + f"{norm}.beta.bias")
            table.append(torch.stack([w * (1 + gamma) - 1, beta], 1))

    def mean8(x):
        return x.view(8, -1, x.shape[-1]).mean(1)

    def align(proj, a, b):
        return get(f"model.{proj}.weight") @ torch.cat([mean8(get(f"model.{a}")), mean8(get(f"model.{b}"))], 1).t()

    a_cur = align("current_shared_task_proj", "depth_align_embs", "current_video_align_embs").t() \
        + get("model.current_shared_task_proj.bias")
    a_fut = align("future_shared_task_proj", "future_depth_align_embs", "future_video_align_embs").t() \
        + get("model.future_shared_task_proj.bias")
    embed = shards.get(LM + "embed_tokens.weight").float()
    mlp_in_w, mlp_in_b = get("model.action_time_mlp_in.weight"), get("model.action_time_mlp_in.bias")
    w_a, w_t = mlp_in_w[:, :768], mlp_in_w[:, 768:]
    ain_w, ain_b = get("model.action_in_proj.weight"), get("model.action_in_proj.bias")
    exponent = torch.arange(0, 128, 2, dtype=torch.int64).float() / 128
    inv_freq = (1.0 / (rope_theta ** exponent)).to(torch.bfloat16).float().to(dev)
    config = json.loads((root / "config.json").read_text()) if (root / "config.json").exists() else {}
    weights = Vla2Weights(
        lm=lm, expert=expert, embed=embed,
        vision_start=embed[VISION_START].to(dev), vision_end=embed[VISION_END].to(dev),
        align=torch.cat([a_cur, a_fut]).contiguous(),
        state_w=get("model.state_proj.weight"), state_b=get("model.state_proj.bias"),
        act_w=(w_a @ ain_w).contiguous(),
        act_c=(cond @ w_t.t() + (w_a @ ain_b + mlp_in_b)[None, :]).contiguous(),
        mlp_out_w=get("model.action_time_mlp_out.weight"), mlp_out_b=get("model.action_time_mlp_out.bias"),
        out_w=get("model.action_out_proj.weight"), out_b=get("model.action_out_proj.bias"),
        final_norm=get(EXPERT + "norm.weight"),
        mod_attn=torch.stack(mod_attn, 1).contiguous(), mod_ffn=torch.stack(mod_ffn, 1).contiguous(),
        inv_freq=inv_freq, sections=mrope_sections().to(dev), times=times, dt=dt, config=config)
    torch.cuda.empty_cache()
    return weights

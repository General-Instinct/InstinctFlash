"""Prepare a LingBot-VA (Wan2.2-style) transformer for the GI engine.

Per block: fused self-attention Q/K/V (with full-width ``norm_q``/``norm_k``), output
projection, affine ``norm2``, cross-attention Q (+``norm_q``) and text K/V projections
(+``norm_k``), and the gelu-tanh FFN, all as per-channel FP8. The flow-time path
(sinusoid -> time MLP -> ``time_proj``) runs in BF16 exactly like the upstream module
and is tabulated per timestep value; tables are extended lazily for unseen values.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from instinctflash.kernels.fp8 import quantize_weight

D, H, HD, FF, LAYERS = 3072, 24, 128, 14336, 30


@dataclass
class WanBlock:
    sst: torch.Tensor        # [6, D] FP32 (shift, scale, gate, c_shift, c_scale, c_gate)
    qkv: object
    norm_q: torch.Tensor
    norm_k: torch.Tensor
    o: object
    norm2_w: torch.Tensor
    norm2_b: torch.Tensor
    cq: object
    cnorm_q: torch.Tensor
    ckv: object              # [2D, D] text K | V
    cnorm_k: torch.Tensor
    co: object
    ff1: object
    ff2: object


@dataclass
class TimeEmbedder:
    lin1_w: torch.Tensor     # BF16, as upstream computes the time path
    lin1_b: torch.Tensor
    lin2_w: torch.Tensor
    lin2_b: torch.Tensor
    proj_w: torch.Tensor
    proj_b: torch.Tensor


@dataclass
class WanWeights:
    blocks: list
    patch_w: torch.Tensor    # [D, 192] FP32
    patch_b: torch.Tensor
    action_w: torch.Tensor   # [D, 30]
    action_b: torch.Tensor
    text_w1: torch.Tensor
    text_b1: torch.Tensor
    text_w2: torch.Tensor
    text_b2: torch.Tensor
    time: dict               # "video" / "action" -> TimeEmbedder
    head_sst: torch.Tensor   # [2, D] BF16
    proj_out_w: torch.Tensor
    proj_out_b: torch.Tensor
    action_out_w: torch.Tensor
    action_out_b: torch.Tensor
    inv_freq: torch.Tensor   # [64]
    sections: torch.Tensor   # [64] int32 (0 = frame, 1 = h, 2 = w)
    config: dict = field(default_factory=dict)


def rope_tables(head_dim=HD, theta=10000.0, device=None):
    f_dim = head_dim - 2 * (head_dim // 3)
    hw = head_dim // 3
    parts, secs = [], []
    for s, d in enumerate((f_dim, hw, hw)):
        base = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64)[: d // 2] / d))
        parts.append(base)
        secs.append(torch.full((d // 2,), s, dtype=torch.int32))
    return torch.cat(parts).float().to(device), torch.cat(secs).to(device)


def load_wan_weights(transformer_dir, device) -> WanWeights:
    from safetensors import safe_open
    root = Path(transformer_dir)
    index = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    files = {f: safe_open(str(root / f), "pt") for f in set(index.values())}
    dev = torch.device(device)

    def raw(name):
        return files[index[name]].get_tensor(name).to(device=dev)

    def get(name):
        return raw(name).float()

    blocks = []
    for i in range(LAYERS):
        p = f"blocks.{i}."
        qkv = torch.cat([get(p + f"attn1.to_{n}.weight") for n in "qkv"])
        qkv_b = torch.cat([get(p + f"attn1.to_{n}.bias") for n in "qkv"])
        ckv = torch.cat([get(p + "attn2.to_k.weight"), get(p + "attn2.to_v.weight")])
        ckv_b = torch.cat([get(p + "attn2.to_k.bias"), get(p + "attn2.to_v.bias")])
        blocks.append(WanBlock(
            sst=get(p + "scale_shift_table")[0].contiguous(),
            qkv=quantize_weight(qkv, qkv_b), norm_q=get(p + "attn1.norm_q.weight"),
            norm_k=get(p + "attn1.norm_k.weight"),
            o=quantize_weight(get(p + "attn1.to_out.0.weight"), get(p + "attn1.to_out.0.bias")),
            norm2_w=get(p + "norm2.weight"), norm2_b=get(p + "norm2.bias"),
            cq=quantize_weight(get(p + "attn2.to_q.weight"), get(p + "attn2.to_q.bias")),
            cnorm_q=get(p + "attn2.norm_q.weight"), ckv=quantize_weight(ckv, ckv_b),
            cnorm_k=get(p + "attn2.norm_k.weight"),
            co=quantize_weight(get(p + "attn2.to_out.0.weight"), get(p + "attn2.to_out.0.bias")),
            ff1=quantize_weight(get(p + "ffn.net.0.proj.weight"), get(p + "ffn.net.0.proj.bias")),
            ff2=quantize_weight(get(p + "ffn.net.2.weight"), get(p + "ffn.net.2.bias"))))
        del qkv, ckv

    def time_embedder(prefix):
        b16 = torch.bfloat16
        return TimeEmbedder(raw(prefix + "time_embedder.linear_1.weight").to(b16),
                            raw(prefix + "time_embedder.linear_1.bias").to(b16),
                            raw(prefix + "time_embedder.linear_2.weight").to(b16),
                            raw(prefix + "time_embedder.linear_2.bias").to(b16),
                            raw(prefix + "time_proj.weight").to(b16), raw(prefix + "time_proj.bias").to(b16))

    inv_freq, sections = rope_tables(device=dev)
    config = json.loads((root / "config.json").read_text())
    weights = WanWeights(
        blocks=blocks, patch_w=get("patch_embedding_mlp.weight"), patch_b=get("patch_embedding_mlp.bias"),
        action_w=get("action_embedder.weight"), action_b=get("action_embedder.bias"),
        text_w1=get("condition_embedder.text_embedder.linear_1.weight"),
        text_b1=get("condition_embedder.text_embedder.linear_1.bias"),
        text_w2=get("condition_embedder.text_embedder.linear_2.weight"),
        text_b2=get("condition_embedder.text_embedder.linear_2.bias"),
        time={"video": time_embedder("condition_embedder."), "action": time_embedder("condition_embedder_action.")},
        head_sst=raw("scale_shift_table")[0].to(torch.bfloat16).contiguous(),
        proj_out_w=get("proj_out.weight"), proj_out_b=get("proj_out.bias"),
        action_out_w=get("action_proj_out.weight"), action_out_b=get("action_proj_out.bias"),
        inv_freq=inv_freq, sections=sections, config=config)
    torch.cuda.empty_cache()
    return weights


def time_rows(emb: TimeEmbedder, t: torch.Tensor, freq_dim=256):
    """(temb [n, D] BF16, timestep_proj [n, 6, D] BF16) for timestep values ``t`` [n] (FP32)."""
    half = freq_dim // 2
    exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    a = t.float()[:, None] * torch.exp(exponent)[None, :]
    sinus = torch.cat([torch.cos(a), torch.sin(a)], 1).to(torch.bfloat16)   # flip_sin_to_cos, shift 0
    h = torch.nn.functional.silu(torch.nn.functional.linear(sinus, emb.lin1_w, emb.lin1_b))
    temb = torch.nn.functional.linear(h, emb.lin2_w, emb.lin2_b)
    proj = torch.nn.functional.linear(torch.nn.functional.silu(temb), emb.proj_w, emb.proj_b)
    return temb, proj.view(-1, 6, D)

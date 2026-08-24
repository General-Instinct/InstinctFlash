#!/usr/bin/env python3
"""Convert Pi0-FAST JAX/Orbax checkpoint to PyTorch safetensors format.

Key mapping follows the same convention as openpi's convert_jax_model_to_pytorch.py
but without the action expert (Pi0-FAST is a single Gemma 2B, no decoder expert).

Usage:
    python examples/convert_pi0fast_orbax_to_safetensors.py \
        --orbax_dir /path/to/pi0_fast_base \
        --output_dir /path/to/pi0_fast_base_converted

"""

import argparse
import json
import os
import pathlib
import shutil
import sys
import time

import numpy as np

# Key prefix for safetensors (matches Pi0 convention for PaliGemma)
VP = "paligemma.model.vision_tower.vision_model"
EP = "paligemma.model.language_model.layers"


def convert(orbax_dir: str, output_dir: str, precision: str = "bfloat16"):
    """Convert Orbax checkpoint to safetensors."""
    import torch
    from safetensors.torch import save_file

    # Load Orbax params
    print(f"Loading Orbax checkpoint from {orbax_dir}...", flush=True)
    t0 = time.time()

    # openpi must be on PYTHONPATH (e.g. ``pip install openpi`` or point at
    # a local checkout). Users running from a container where openpi lives
    # at a non-standard location can prepend it to sys.path before invoking
    # this script.
    from openpi.models.model import restore_params
    import flax.traverse_util as tu

    raw = restore_params(f"{orbax_dir}/params", restore_type=np.ndarray)
    flat = tu.flatten_dict(raw, sep=".")
    print(f"Loaded {len(flat)} tensors, {time.time()-t0:.1f}s", flush=True)

    # Determine suffix (NNX convention may add /value)
    sfx = ""
    if "PaliGemma.img.embedding.kernel.value" in flat:
        sfx = ".value"

    def g(k):
        return flat[k + sfx]

    state_dict = {}

    # ═══════════════════════════════════════════
    # Vision (SigLIP, 27 layers)
    # ═══════════════════════════════════════════
    state_dict[f"{VP}.embeddings.patch_embedding.weight"] = (
        g("PaliGemma.img.embedding.kernel").transpose(3, 2, 0, 1))
    state_dict[f"{VP}.embeddings.patch_embedding.bias"] = g("PaliGemma.img.embedding.bias")
    state_dict[f"{VP}.embeddings.position_embedding.weight"] = (
        g("PaliGemma.img.pos_embedding").reshape(-1, 1152))

    # Vision layers (stacked [27, ...] → per-layer)
    ln0_s = g("PaliGemma.img.Transformer.encoderblock.LayerNorm_0.scale")
    ln0_b = g("PaliGemma.img.Transformer.encoderblock.LayerNorm_0.bias")
    ln1_s = g("PaliGemma.img.Transformer.encoderblock.LayerNorm_1.scale")
    ln1_b = g("PaliGemma.img.Transformer.encoderblock.LayerNorm_1.bias")
    q_k = g("PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.kernel")
    q_b = g("PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.bias")
    k_k = g("PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.kernel")
    k_b = g("PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.bias")
    v_k = g("PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.kernel")
    v_b = g("PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.bias")
    o_k = g("PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.kernel")
    o_b = g("PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.bias")
    ff0_k = g("PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.kernel")
    ff0_b = g("PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.bias")
    ff1_k = g("PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.kernel")
    ff1_b = g("PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.bias")

    for i in range(27):
        lp = f"{VP}.encoder.layers.{i}"
        state_dict[f"{lp}.layer_norm1.weight"] = ln0_s[i]
        state_dict[f"{lp}.layer_norm1.bias"] = ln0_b[i]
        state_dict[f"{lp}.layer_norm2.weight"] = ln1_s[i]
        state_dict[f"{lp}.layer_norm2.bias"] = ln1_b[i]
        state_dict[f"{lp}.self_attn.q_proj.weight"] = q_k[i].reshape(-1, 1152).T
        state_dict[f"{lp}.self_attn.q_proj.bias"] = q_b[i].reshape(-1)
        state_dict[f"{lp}.self_attn.k_proj.weight"] = k_k[i].reshape(-1, 1152).T
        state_dict[f"{lp}.self_attn.k_proj.bias"] = k_b[i].reshape(-1)
        state_dict[f"{lp}.self_attn.v_proj.weight"] = v_k[i].reshape(-1, 1152).T
        state_dict[f"{lp}.self_attn.v_proj.bias"] = v_b[i].reshape(-1)
        state_dict[f"{lp}.self_attn.out_proj.weight"] = o_k[i].reshape(-1, 1152).T
        state_dict[f"{lp}.self_attn.out_proj.bias"] = o_b[i].reshape(-1)
        state_dict[f"{lp}.mlp.fc1.weight"] = ff0_k[i].T
        state_dict[f"{lp}.mlp.fc1.bias"] = ff0_b[i]
        state_dict[f"{lp}.mlp.fc2.weight"] = ff1_k[i].T
        state_dict[f"{lp}.mlp.fc2.bias"] = ff1_b[i]

    state_dict[f"{VP}.post_layernorm.weight"] = g("PaliGemma.img.Transformer.encoder_norm.scale")
    state_dict[f"{VP}.post_layernorm.bias"] = g("PaliGemma.img.Transformer.encoder_norm.bias")

    # Projector
    state_dict["paligemma.model.multi_modal_projector.linear.weight"] = (
        g("PaliGemma.img.head.kernel").T)
    state_dict["paligemma.model.multi_modal_projector.linear.bias"] = (
        g("PaliGemma.img.head.bias"))

    # ═══════════════════════════════════════════
    # LLM (Gemma 2B, 18 layers)
    # ═══════════════════════════════════════════
    # Embedding (shared for input + output)
    emb = g("PaliGemma.llm.embedder.input_embedding")
    state_dict["paligemma.lm_head.weight"] = emb

    # LLM layers (stacked [18, ...])
    q_einsum = g("PaliGemma.llm.layers.attn.q_einsum.w")      # [18, 8, 2048, 256]
    kv_einsum = g("PaliGemma.llm.layers.attn.kv_einsum.w")    # [18, 2, 1, 2048, 256]
    o_einsum = g("PaliGemma.llm.layers.attn.attn_vec_einsum.w")  # [18, 8, 256, 2048]
    gate_einsum = g("PaliGemma.llm.layers.mlp.gating_einsum")  # [18, 2, 2048, 16384]
    down_w = g("PaliGemma.llm.layers.mlp.linear")              # [18, 16384, 2048]
    attn_norm = g("PaliGemma.llm.layers.pre_attention_norm.scale")  # [18, 2048]
    ffn_norm = g("PaliGemma.llm.layers.pre_ffw_norm.scale")     # [18, 2048]

    for i in range(18):
        lp = f"{EP}.{i}"
        # Q: [8, 2048, 256] → transpose(0,2,1) → [8, 256, 2048] → reshape → [2048, 2048]
        state_dict[f"{lp}.self_attn.q_proj.weight"] = (
            q_einsum[i].transpose(0, 2, 1).reshape(-1, 2048))
        # K: [1, 2048, 256] → [256, 2048]
        state_dict[f"{lp}.self_attn.k_proj.weight"] = kv_einsum[i, 0, 0].T
        # V: [1, 2048, 256] → [256, 2048]
        state_dict[f"{lp}.self_attn.v_proj.weight"] = kv_einsum[i, 1, 0].T
        # O: [8, 256, 2048] → transpose(2,0,1) → reshape → [2048, 2048]
        state_dict[f"{lp}.self_attn.o_proj.weight"] = (
            o_einsum[i].transpose(2, 0, 1).reshape(2048, -1))
        # Gate + Up
        state_dict[f"{lp}.mlp.gate_proj.weight"] = gate_einsum[i, 0].T
        state_dict[f"{lp}.mlp.up_proj.weight"] = gate_einsum[i, 1].T
        state_dict[f"{lp}.mlp.down_proj.weight"] = down_w[i].T
        # Norms
        state_dict[f"{lp}.input_layernorm.weight"] = attn_norm[i]
        state_dict[f"{lp}.post_attention_layernorm.weight"] = ffn_norm[i]

    # Final norm
    state_dict["paligemma.model.language_model.norm.weight"] = (
        g("PaliGemma.llm.final_norm.scale"))

    # ═══════════════════════════════════════════
    # Convert to torch tensors + target precision
    # ═══════════════════════════════════════════
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    target_dtype = dtype_map.get(precision, torch.bfloat16)

    torch_dict = {}
    for k, v in state_dict.items():
        t = torch.from_numpy(np.ascontiguousarray(np.asarray(v, dtype=np.float32)))
        torch_dict[k] = t.to(target_dtype).contiguous()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.safetensors")
    save_file(torch_dict, output_path)
    print(f"Saved {len(torch_dict)} tensors to {output_path}", flush=True)

    # Copy assets
    assets_src = pathlib.Path(orbax_dir) / "assets"
    if assets_src.exists():
        assets_dst = pathlib.Path(output_dir) / "assets"
        if assets_dst.exists():
            shutil.rmtree(assets_dst)
        shutil.copytree(assets_src, assets_dst)
        print(f"Copied assets to {assets_dst}", flush=True)

    # Save config
    config = {"model": "pi0_fast", "precision": precision,
              "source": str(orbax_dir), "num_llm_layers": 18, "num_vision_layers": 27}
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Conversion complete: {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orbax_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--precision", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()
    convert(args.orbax_dir, args.output_dir, args.precision)

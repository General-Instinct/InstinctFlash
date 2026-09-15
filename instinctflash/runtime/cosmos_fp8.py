"""Explicit Cosmos MoT Q/K/V recipe, installed after native model loading.

Not a Runtime registration: public support requires a verified native service
integration. In particular, output projections use direct weight slices in the
replicated-attention path and must retain their native modules.
"""
from __future__ import annotations

from .torch_fp8_linear import ThorFP8Linear


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj",
    "q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen",
)


def install_cosmos_fp8(model):
    """Pack all audited MoT attention projections, then install atomically.

    Call before the first prediction or graph capture, after all model.to()
    calls. Exact class matching prevents silently quantizing custom forwards,
    LoRA wrappers or unrelated visual-encoder projections with similar names.
    """
    from cosmos_framework.model.generator.mot.unified_mot import PackedAttentionMoT

    targets = []
    for path, module in model.named_modules():
        if type(module) is PackedAttentionMoT:
            for name in PROJECTIONS:
                source = getattr(module, name)
                if (source._forward_hooks or source._forward_pre_hooks
                        or source._backward_hooks):
                    raise ValueError(f"Cannot replace hooked Cosmos projection {path}.{name}")
                targets.append((path, module, name, source))
    if not targets:
        raise ValueError("No audited PackedAttentionMoT projections found; FP8 was not installed")

    # Pack before changing model state: bad dtype, shape, weights or allocation
    # failure must not leave a partially converted serving model.
    packed = [ThorFP8Linear(source) for _, _, _, source in targets]
    receipt = {
        "recipe": "cosmos_mot_qkv_" + ThorFP8Linear.recipe,
        "attention_layers": len(targets) // len(PROJECTIONS),
        "projections": [
            {"path": f"{path}.{name}".lstrip("."),
             "in_features": source.in_features,
             "out_features": source.out_features,
             "bias": source.bias is not None}
            for path, _, name, source in targets
        ],
        "scope": "Q/K/V in both MoT towers only; native attention, output projections and MLP",
    }
    for (_, parent, name, _), replacement in zip(targets, packed):
        setattr(parent, name, replacement)
    return receipt


def install_cosmos_dense_mlp_fp8(model):
    """Experimental extension for audited dense MoT MLPs, not enabled by default.

    Invoke after model loading and before capture. This is a distinct numerical
    recipe requiring its own model performance and quality measurements. Sparse
    experts, routers and visual-encoder MLPs are deliberately not converted.
    """
    from cosmos_framework.model.generator.mot.unified_mot import (
        MoTDecoderLayer, Qwen3VLTextMLP, Nemotron3DenseVLMLP,
    )

    layouts = {
        Qwen3VLTextMLP: ("gate_proj", "up_proj", "down_proj"),
        Nemotron3DenseVLMLP: ("up_proj", "down_proj"),
    }
    targets = []
    mlp_count = 0
    for path, layer in model.named_modules():
        if type(layer) is not MoTDecoderLayer:
            continue
        for tower in ("mlp", "mlp_moe_gen"):
            module = getattr(layer, tower)
            names = layouts.get(type(module))
            if names is None:
                raise ValueError(f"Unsupported Cosmos dense MLP type at {path}.{tower}")
            if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
                raise ValueError(f"Cannot convert hooked Cosmos MLP {path}.{tower}")
            mlp_count += 1
            for name in names:
                source = getattr(module, name)
                if source._forward_hooks or source._forward_pre_hooks or source._backward_hooks:
                    raise ValueError(f"Cannot replace hooked Cosmos MLP projection {path}.{tower}.{name}")
                targets.append((f"{path}.{tower}".lstrip("."), module, name, source))
    if not targets:
        raise ValueError("No audited dense MoT MLPs found; FP8 MLP was not installed")
    packed = [ThorFP8Linear(source) for _, _, _, source in targets]
    receipt = {
        "recipe": "cosmos_mot_dense_mlp_" + ThorFP8Linear.recipe,
        "mlp_modules": mlp_count,
        "projections": [
            {"path": f"{path}.{name}", "in_features": source.in_features,
             "out_features": source.out_features, "bias": source.bias is not None}
            for path, _, name, source in targets
        ],
        "scope": "Experimental dense MoT MLP extension; native activation and routing preserved",
    }
    for (_, parent, name, _), replacement in zip(targets, packed):
        setattr(parent, name, replacement)
    return receipt

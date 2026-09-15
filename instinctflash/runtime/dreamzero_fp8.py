"""Explicit DreamZero causal-attention Q/K/V FP8 installation.

Apply to the loaded action head after post_initialize and LoRA merging. Native
T5/CLIP/VAE, attention, KV history and scheduler remain owned by the original
policy. Prediction reuse is selected independently by the Runtime cache policy.
"""
from __future__ import annotations

import os

from .torch_fp8_linear import ThorFP8Linear


def install_dreamzero_fp8(action_head, *, include_ffn=False):
    if getattr(action_head, "cpu_offload", False):
        raise ValueError("Pack DreamZero FP8 only after selecting GPU-resident native components")
    # Native post_initialize may load this through LOAD_TRT_ENGINE even when
    # ENABLE_TENSORRT is false. Such a head bypasses model projections during
    # denoising, so packing them cannot describe the arithmetic being executed.
    if getattr(action_head, "trt_engine", None) is not None:
        raise ValueError("Native PyTorch FP8 projections cannot be installed behind a loaded TensorRT engine")
    if os.environ.get("ENABLE_TENSORRT", "false").lower() == "true":
        raise ValueError("Native PyTorch FP8 projections cannot be installed behind a TensorRT execution path")
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanSelfAttention

    model = action_head.model
    targets = []
    for path, module in model.named_modules():
        if type(module) is CausalWanSelfAttention:
            for name in ("q", "k", "v"):
                source = getattr(module, name)
                if source._forward_hooks or source._forward_pre_hooks or source._backward_hooks:
                    raise ValueError(f"Cannot replace hooked DreamZero projection {path}.{name}")
                targets.append((path, module, name, source))
    if not targets:
        raise ValueError("No audited DreamZero causal Q/K/V projections found")
    attention_layers = len(targets) // 3
    if include_ffn:
        from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanAttentionBlock
        for path, module in model.named_modules():
            if type(module) is CausalWanAttentionBlock:
                # Only plain Linear calls in the audited sequential FFN. Keep
                # cross-attention, modulation, normalization and history native.
                for name in ("0", "2"):
                    source = module.ffn[int(name)]
                    if source._forward_hooks or source._forward_pre_hooks or source._backward_hooks:
                        raise ValueError(f"Cannot replace hooked DreamZero FFN {path}.ffn.{name}")
                    targets.append((path + ".ffn", module.ffn, name, source))
        if len(targets) != attention_layers * 5:
            raise ValueError("DreamZero FFN layout does not match causal attention blocks")
    packed = [ThorFP8Linear(source) for _, _, _, source in targets]
    receipt = {
        "recipe": ("dreamzero_causal_qkv_ffn_" if include_ffn else "dreamzero_causal_qkv_") + ThorFP8Linear.recipe,
        "attention_layers": attention_layers,
        "projections": [{"path": f"{path}.{name}".lstrip("."),
                         "in_features": source.in_features,
                         "out_features": source.out_features,
                         "bias": source.bias is not None}
                        for path, _, name, source in targets],
        # lazy_joint_forward_causal sets both video/action schedulers from this
        # field; num_inference_timesteps is a different config field (often 4).
        "scheduler_steps": int(action_head.num_inference_steps),
        "dit_step_mask": list(action_head.dit_step_mask),
        "dynamic_cache_schedule": bool(getattr(action_head, "dynamic_cache_schedule", False)),
        "scope": "Causal self-attention Q/K/V" + (" and block FFN" if include_ffn else " only") + "; other computation and history remain native",
        "quality_certificate": None if include_ffn else "checkpoint-specific; see evaluation receipts",
    }
    for (_, parent, name, _), replacement in zip(targets, packed):
        setattr(parent, name, replacement)
    return receipt

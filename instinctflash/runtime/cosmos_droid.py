"""Checkpoint-native DROID service construction for native/FP8 qualification.

Uses NVIDIA's RoboLab action/state/gripper and prompt processing. This bridge is
not yet registered in the public Runtime; historical RoboTwin-service timings
are not evidence for this recipe.
"""
from __future__ import annotations

import json
from pathlib import Path


def build_droid_service(checkpoint, *, precision, format_prompt_as_json,
                        steps=4, guidance=3.0, seed=0, shift=5.0,
                        image_height=540, image_width=640, policy_config=None,
                        model_finalizer=None):
    if precision not in ("native", "fp8"):
        raise ValueError("DROID precision must be native or fp8")
    if not isinstance(format_prompt_as_json, bool):
        raise ValueError("Declare the checkpoint's prompt format explicitly")
    root = Path(checkpoint)
    metadata = json.loads((root / "checkpoint.json").read_text())
    # Older Nano exports contain an empty checkpoint.json. Require an explicit
    # serving declaration for them instead of borrowing another model's values.
    policy = dict(policy_config or {})
    for name, value in metadata.get("policy", {}).items():
        if name in policy and policy[name] != value:
            raise ValueError(f"DROID declaration conflicts with checkpoint metadata: {name}")
        policy[name] = value
    missing = {"domain_name", "action_chunk_size", "conditioning_fps"} - policy.keys()
    if missing:
        raise ValueError(f"DROID requires checkpoint metadata or an explicit policy_config: {sorted(missing)}")
    if policy["domain_name"] != "droid_lerobot":
        raise ValueError("This service bridge requires a DROID checkpoint")
    chunk = policy["action_chunk_size"]
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("Invalid checkpoint action_chunk_size")

    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabPolicyService, RobolabServerArgs,
    )

    class DROIDService(RobolabPolicyService):
        def _build_setup_args(self, args):
            setup = super()._build_setup_args(args)
            # Qualification starts with the eager native model. Compile can be
            # admitted separately after changed-prompt and FP8 graph validation.
            return setup.model_copy(update={"guardrails": False,
                                            "use_torch_compile": False})

    service = DROIDService(RobolabServerArgs(
        checkpoint_path=str(root), domain_name=policy["domain_name"],
        action_chunk_size=chunk, conditioning_fps=policy["conditioning_fps"],
        action_dim=8, num_steps=steps, guidance=guidance, shift=shift, seed=seed,
        image_height=image_height, image_width=image_width,
        format_prompt_as_json=format_prompt_as_json,
    ))
    receipt = model_finalizer(service) if model_finalizer is not None else None
    if precision == "fp8" and model_finalizer is None:
        from .cosmos_fp8 import install_cosmos_fp8, install_cosmos_dense_mlp_fp8
        receipt = install_cosmos_fp8(service.model)
        import torch
        if torch.cuda.get_device_capability() == (11, 0):
            mlp = install_cosmos_dense_mlp_fp8(service.model)
            receipt["recipe"] = "cosmos_mot_qkv_dense_mlp_" + mlp["recipe"].split("cosmos_mot_dense_mlp_", 1)[1]
            receipt["projections"].extend(mlp["projections"])
            receipt["mlp_modules"] = mlp["mlp_modules"]
            receipt["scope"] = "Both MoT towers: Q/K/V and dense MLP; native vision, attention and output projections"
            receipt["quality_certificate"] = None
    return service, receipt


class CosmosDROIDLoop:
    """Shared native/FP8 input mapping; the owned native service decodes actions."""

    def __init__(self, service, fp8_receipt=None):
        self._service = service
        self._fp8_receipt = fp8_receipt
        self._prompt = ""

    def _ensure_open(self):
        if self._service is None:
            raise RuntimeError("Cosmos DROID loop is closed")
        return self._service

    def reset(self, **conditioning):
        import numpy as np

        service = self._ensure_open()
        self._prompt = str(conditioning.get("prompt") or "")
        # Native RoboLab has a per-request seed stream, no persistent model KV.
        # Reset its stream without changing the checkpoint's deterministic mode.
        with service._lock:
            service._rng = np.random.default_rng(service.cfg.seed)

    def predict(self, observation, *, executed_action=None):
        # Stateless across requests: actual robot state/history comes from the
        # next observation. The common Runtime executed-action hook is ignored.
        import base64
        import io
        import numpy as np
        from PIL import Image

        service = self._ensure_open()
        obs = dict(observation)
        obs["prompt"] = str(obs.get("prompt") or obs.get("task") or self._prompt)
        if not obs["prompt"]:
            raise ValueError("Cosmos DROID requires a prompt")
        if "observation/image" not in obs and "image" in obs:
            image = obs["image"]
            if isinstance(image, str):
                with Image.open(io.BytesIO(base64.b64decode(image, validate=True))) as decoded:
                    image = np.asarray(decoded.convert("RGB"))
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("Cosmos DROID image must be HxWx3 uint8 RGB")
            obs["observation/image"] = image
        if "state" in obs:
            if "observation/joint_position" in obs or "observation/gripper_position" in obs:
                raise ValueError("Supply state or native DROID joint/gripper fields, not both")
            state = np.asarray(obs["state"], dtype=np.float32)
            if state.ndim not in (1, 2) or state.shape[-1] != 8 or not np.isfinite(state).all():
                raise ValueError("DROID state must have shape (8,) or (history, 8), with finite values")
            obs["observation/joint_position"] = state[..., :7]
            obs["observation/gripper_position"] = state[..., 7:]
        return service.infer(obs)

    def backend_stats(self):
        service = self._ensure_open()
        cache = getattr(service, "_ifl_conditioning_cache", None)
        cache_stats = cache.report() if cache is not None else None
        return {"precision": "fp8" if self._fp8_receipt else "native",
                "fp8_recipe": self._fp8_receipt,
                "action_chunk_size": service.cfg.action_chunk_size,
                "action_steps": service.cfg.num_steps,
                "guidance": service.cfg.guidance,
                "conditioning_fps": service.cfg.conditioning_fps,
                "native_optimizations": getattr(service, "_ifl_native_optimizations", {}),
                "exact_pointwise": getattr(service, "_ifl_exact_pointwise", None),
                "layer_graphs": getattr(service, "_ifl_layer_graphs", None),
                "conditioning_cache_status": getattr(service, "_ifl_conditioning_cache_status", None),
                "conditioning_cache": cache_stats,
                "timestep_cache": (service._ifl_timestep_cache.report()
                    if getattr(service, "_ifl_timestep_cache", None) is not None else None),
                "generation_regions": (service._ifl_generation_regions.report()
                    if getattr(service, "_ifl_generation_regions", None) is not None else None),
                "numeric_attention": (service._ifl_numeric_attention.report()
                    if getattr(service, "_ifl_numeric_attention", None) is not None else None),
                "residency": (service._ifl_sm89_residency.report()
                    if getattr(service, "_ifl_sm89_residency", None) is not None else None),
                "captured": bool(getattr(service, "_ifl_layer_graphs", {}).get("replays", 0)
                    or (cache_stats and any(g['replays'] for g in cache_stats['graph_stats'])))}

    def declaration(self):
        service = self._ensure_open()
        return {"backbone": "cosmos3_policy",
                "frontend": "instinctflash/runtime/cosmos_droid.py",
                "steps": {"prefix": 1, "action": service.cfg.num_steps},
                "guidance": {"action": ("cfg", service.cfg.guidance)},
                "precision": "fp8" if self._fp8_receipt else "native",
                "action_chunk_size": service.cfg.action_chunk_size,
                "evidence": "Native DROID service configuration with explicit MoT Q/K/V FP8 recipe"}

    def close(self):
        residency = getattr(self._service, "_ifl_sm89_residency", None)
        if residency is not None:
            with self._service._lock:
                residency.close()
        timestep_cache = getattr(self._service, "_ifl_timestep_cache", None)
        if timestep_cache is not None:
            timestep_cache.close()
        regions = getattr(self._service, '_ifl_generation_regions', None)
        if regions is not None:
            regions.close()
        cache = getattr(self._service, "_ifl_conditioning_cache", None)
        if cache is not None:
            cache.close()
        attention = getattr(self._service, "_ifl_numeric_attention", None)
        if attention is not None:
            attention.close()
        self._service = None
        self._prompt = ""

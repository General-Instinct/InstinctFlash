"""Build the VA engine at the checkpoint's resolved geometry and schedule."""
from __future__ import annotations

import copy
import os
from pathlib import Path


def resolve_native_config(adapter, checkpoint, server_module, *, nfe=None):
    from instinctflash.adapters.lingbot_va import (
        apply_declared_guidance, resolve_observation_geometry,
    )

    execution = checkpoint.execution
    extra = dict(execution.extra or {})
    name = extra.get("va_config") or os.environ.get("IFL_CFG")
    if name not in server_module.VA_CONFIGS:
        raise ValueError("VA FP8 requires a declared native va_config or IFL_CFG")
    cfg = copy.deepcopy(server_module.VA_CONFIGS[name])
    geometry, _ = resolve_observation_geometry(execution, va_configs=server_module.VA_CONFIGS)
    for key, value in geometry.items():
        setattr(cfg, key, value)
    if "frame_chunk_size" in extra:
        frame_chunk = extra["frame_chunk_size"]
        if type(frame_chunk) is not int or frame_chunk < 2 or frame_chunk != cfg.frame_chunk_size:
            raise ValueError("Declared VA frame chunk must be an integer matching the native config")
    steps = dict(execution.nfe or {})
    steps.update(nfe or {})
    unknown = set(steps) - {"video", "action", "kv_refresh"}
    if unknown or steps.get("kv_refresh", 2) != 2:
        raise ValueError(f"Unsupported VA schedule phases: {steps}")
    for stream, attr in (("video", "num_inference_steps"),
                         ("action", "action_num_inference_steps")):
        value = steps.get(stream, getattr(cfg, attr))
        if type(value) is not int or value < 1:
            raise ValueError(f"VA {stream} steps must be a positive integer")
        setattr(cfg, attr, value)
    apply_declared_guidance(cfg, execution.guidance)
    cfg.wan22_pretrained_model_name_or_path = adapter.materialize(checkpoint)
    return cfg


def build_wan_va_engine_loop(adapter, checkpoint, *, device=None, nfe=None):
    import torch
    from flash_rt.frontends.torch.wan_va_thor import WanVaTorchFrontendThor
    from flash_rt.models.wan_va.geometry import WanVaGeometry
    from flash_rt.models.wan_va.operating_point import WanVaOperatingPoint
    from instinctflash.runtime.lingbot_install import import_lingbot_server
    from instinctflash.runtime.wan_va_engine import (
        WanVaEngineLoop, WanVaEngineServer, build_native_conditioning,
    )

    dev = torch.device(device or "cuda")
    if dev.type != "cuda" or torch.cuda.get_device_capability(dev) != (11, 0):
        raise ValueError("VA FP8 requires a Thor SM110 CUDA device")
    source = import_lingbot_server(adapter.lingbot_root)
    cfg = resolve_native_config(adapter, checkpoint, source, nfe=nfe)
    geometry = WanVaGeometry.from_job_config(cfg)
    point = WanVaOperatingPoint.from_job_config(cfg)
    if cfg.video_exec_step != -1 or cfg.action_guidance_scale > 1:
        raise ValueError("VA FP8 requires a complete video schedule and positive-only action guidance")
    if cfg.param_dtype != torch.bfloat16:
        raise ValueError("VA native conditioning must declare BF16 for this FP8 recipe")
    cfg.rank = 0
    cfg.local_rank = dev.index if dev.index is not None else torch.cuda.current_device()
    cfg.world_size = 1
    cfg.enable_offload = False
    cfg.save_root = os.environ.get("IFL_SAVE_ROOT", "/tmp/iwm_runtime")
    with torch.cuda.device(dev):
        native = build_native_conditioning(source, cfg)
        frontend = WanVaTorchFrontendThor(
            Path(cfg.wan22_pretrained_model_name_or_path) / "transformer",
            point=point, geometry=geometry, device=str(native.device),
            pool_slots=geometry.pool_slots(cfg.attn_window),
            used_action_channels=cfg.used_action_channel_ids,
            precision="fp8", use_cuda_graph=True,
        )
        return WanVaEngineLoop(WanVaEngineServer(native, frontend))

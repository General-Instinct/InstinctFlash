"""Cosmos DROID adapter using checkpoint-native RoboLab processing.

Historical RoboTwin-service benchmark scripts retain their original protocol;
they do not certify this corrected 32-action DROID interface.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from instinctflash import AdapterSpec, GuidanceRule, PhaseSpec
from instinctflash.adapters.base import GuidanceMode, ObservationField, ObservationSpec

BACKBONE = "cosmos3_policy"
MODEL_ID = "nvidia/Cosmos3-Edge-Policy-DROID"
NANO_MODEL_ID = "nvidia/Cosmos3-Nano-Policy-DROID"
SERVER_MODULE = "cosmos_framework.scripts.action_policy_server_robolab"

#: The declaration keys the serving config is built from. Each is a fact about how the
#: published rows were measured; a checkpoint that omits one is refused, not guessed at.
REQUIRED_SERVING_KEYS = (
    "domain_name", "action_dim", "action_chunk_size", "image_height", "image_width",
    "conditioning_fps", "format_prompt_as_json",
)


class Cosmos3PolicyAdapter:
    """Two-tower MoT action policy: one packed prefill, four UniPC denoise steps, no
    persistent KV — every request rebuilds its state, so shapes repeat across cycles."""

    CUDA_GRAPHS_ENV = "IFL_COSMOS3_CUDA_GRAPHS"
    PROMPT_KV_CACHE_ENV = "IFL_COSMOS3_PROMPT_KV_CACHE"
    NANO_ACTION_ONLY_ENV = "IFL_COSMOS3_NANO_ACTION_ONLY"

    def spec(self) -> AdapterSpec:
        return AdapterSpec(
            model_id=MODEL_ID,
            param_bytes=7_574_066_016,       # Edge; the Nano declaration carries 31_499_049_824
            # No KV pool at all: the SequencePack is rebuilt per request and nothing is carried
            # between control cycles (upstream's notify_next_episode says so in as many words).
            streams=(),
            phases=(
                PhaseSpec("prefix", nfe=1, writes=frozenset()),
                PhaseSpec("action", nfe=4, truncatable=True, min_nfe=1,
                          depends_on=("prefix",)),
            ),
            # Native released DROID serving profile; declarations may override guidance.
            guidance={"action": GuidanceRule(mode=GuidanceMode.CFG, scale=3.0)},
            observation=ObservationSpec(
                fields=(
                    ObservationField("image", (540, 640, 3), "uint8"),
                    ObservationField("state", (8,), "float32"),
                ),
                history=1,
                batched=False,
                conditioning=("prompt",),
            ),
            notes={
                "backbone": "cosmos3_policy",
                "family": "action_policy",
                "action_reply": "(chunk, action_dim) = (32, 8) at the declared serving config",
                "sampler": "unipc, shift 5.0",
                "numeric_tier": "Current DROID quality qualification pending",
                "cuda_graphs": "Native Thor: checked decoder-layer graphs with a shared pool and owned outputs",
                "capture_supported": False,
                "capture_unavailable_reason": "The DROID adapter does not install generic whole-forward graphs; native Thor uses checked decoder-layer graphs",
            },
        )

    def observation_contract(self, checkpoint):
        """The request geometry FOR THIS CHECKPOINT, from its declaration.

        The image height/width guard is enforced by the service itself (a mismatched mosaic is
        refused at request time), so the contract shown to a caller must come from the same
        declared numbers — printing another embodiment's geometry would teach a wrong request.
        """
        import dataclasses

        extra = dict(checkpoint.execution.extra or {})
        # "FILL_ME" is a scaffold sentinel (descriptors/scaffold.py), not a value: an unfilled
        # scaffolded declaration gets the same loud missing-serving-config message.
        missing = [k for k in REQUIRED_SERVING_KEYS if extra.get(k) in (None, "FILL_ME")]
        if missing:
            raise RuntimeError(_missing_serving_config_message(checkpoint, missing))
        fields = (
            ObservationField("image",
                             (int(extra["image_height"]), int(extra["image_width"]), 3),
                             "uint8"),
            ObservationField("state", (int(extra["action_dim"]),), "float32"),
        )
        return dataclasses.replace(self.spec().observation, fields=fields), \
            "the checkpoint's declared serving config (image_height/width, action_dim)"

    def can_host_in_process(self):
        from instinctflash.runtime.execution import imports_available

        ok, reason = imports_available(("torch", "numpy", "PIL", "pydantic"))
        if not ok:
            return ok, reason
        try:
            spec_found = importlib.util.find_spec(SERVER_MODULE)
        except ModuleNotFoundError:
            spec_found = None
        if spec_found is None:
            return False, (
                f"{SERVER_MODULE} is not importable. This adapter needs the native "
                f"cosmos-framework checkout containing the RoboLab "
                f"policy server. Use the pinned `scripts/bootstrap_vendor.py install` "
                f"workflow in INSTALL.rst for `edge` or `nano`, select the matching "
                f"`--target` and Python 3.13, then activate the generated environment.")
        return True, "the model stack imports and the patched cosmos-framework server is present"

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
        return self._build_droid(checkpoint, device=device, nfe=nfe, precision="native", plan=plan)

    def build_fp8(self, checkpoint, *, device=None, nfe=None):
        return self._build_droid(checkpoint, device=device, nfe=nfe, precision="fp8")

    def build_sm89_fp8(self, checkpoint, plan, *, device=None, nfe=None):
        from instinctflash.runtime.sm89_fp8 import requested
        if not requested(plan):
            raise ValueError("Cosmos SM89 FP8 requires its explicit executor plan")
        return self._build_droid(checkpoint, device=device, nfe=nfe, precision="fp8", plan=plan)

    def _build_droid(self, checkpoint, *, device, nfe, precision, plan=None):
        import torch
        from instinctflash.runtime.cosmos_droid import build_droid_service, CosmosDROIDLoop
        from instinctflash.runtime.engine_backend import requested_operating_point

        timestep_cache = _env_flag("IFL_COSMOS3_TIMESTEP_CACHE", default=False)
        if timestep_cache and precision != "native":
            raise ValueError("Native timestep cache currently requires native precision")
        fused_linear = _env_flag("IFL_BF16_LINEAR_RELU2", default=False)
        generation_regions = _env_flag("IFL_COSMOS3_GEN_REGIONS", default=False)
        split_prefill = _env_flag("IFL_COSMOS3_SPLIT_PREFILL", default=False)
        contiguous_kv = _env_flag("IFL_COSMOS3_CONTIGUOUS_KV", default=False)
        if fused_linear and not generation_regions:
            raise ValueError("BF16 linear fusion requires Cosmos GEN regions")
        if contiguous_kv and not generation_regions:
            raise ValueError("Contiguous K/V requires Cosmos GEN regions")
        if split_prefill and not generation_regions:
            raise ValueError("Split prefill requires Cosmos GEN regions")
        if generation_regions:
            if precision != "native":
                raise ValueError("Cosmos GEN regions currently require native precision")
            from instinctflash.runtime.precision import require_transform_permission
            from instinctflash.planners.planner import Tier
            require_transform_permission(plan, Tier.NUMERIC, "Cosmos generation regions")
        if not torch.cuda.is_available():
            raise RuntimeError("Cosmos3 policy inference requires CUDA")
        dev = str(device or "cuda")
        if ":" in dev and dev.rsplit(":", 1)[1] not in ("", "0"):
            raise RuntimeError("Select the Cosmos GPU with CUDA_VISIBLE_DEVICES")
        extra = dict(checkpoint.execution.extra or {})
        missing = [k for k in REQUIRED_SERVING_KEYS if extra.get(k) in (None, "FILL_ME")]
        if missing:
            raise RuntimeError(_missing_serving_config_message(checkpoint, missing))
        if int(extra["action_dim"]) != 8:
            raise ValueError("Native DROID joint-position actions require action_dim=8")
        if _env_flag(self.CUDA_GRAPHS_ENV, default=bool(extra.get("cuda_graphs", False))):
            raise ValueError("CUDA graphs require new DROID changed-prompt qualification")
        steps, guidance, _ = requested_operating_point(self, checkpoint, nfe)
        if steps.get("prefix") != 1 or steps.get("action", 0) < 1:
            raise ValueError("DROID requires one prefix pass and positive action NFE")
        mode, scale = guidance["action"]
        if mode not in ("cfg", "none", "positive_only"):
            raise ValueError(f"Unsupported DROID guidance mode: {mode}")
        capability = torch.cuda.get_device_capability()
        sm120_native = precision == "native" and capability == (12, 0)
        declared_id = str(getattr(checkpoint.execution, "model_id", "") or checkpoint.model_id)
        sm89_residency = capability == (8, 9) and declared_id in (MODEL_ID, NANO_MODEL_ID)
        if capability == (8, 9) and precision == "fp8":
            from instinctflash.runtime.sm89_fp8 import requested
            if not sm89_residency or not requested(plan):
                raise ValueError("Cosmos SM89 FP8 requires a released Edge/Nano declaration and recipe")
        thor_native = (precision == "native" and capability == (11, 0)
                       and declared_id in (MODEL_ID, NANO_MODEL_ID))
        nano_action_only = _env_flag(
            self.NANO_ACTION_ONLY_ENV,
            default=(sm120_native or thor_native or sm89_residency) and declared_id == NANO_MODEL_ID)
        prompt_kv_cache = _env_flag(
            self.PROMPT_KV_CACHE_ENV,
            default=sm120_native and (scale if mode == "cfg" else 1.0) == 1.0)
        numeric_attention = _numeric_attention_requested(precision, plan, thor_native)
        conditioning_cache = _env_flag("IFL_COSMOS3_CONDITIONING_CACHE", default=numeric_attention)
        if precision != "native" and (prompt_kv_cache or conditioning_cache
                                     or (nano_action_only and not sm89_residency)):
            raise ValueError("Native residency/KV options require precision='native'")
        if nano_action_only and declared_id != NANO_MODEL_ID:
            raise ValueError("Nano action-only residency requires the declared Nano checkpoint")
        from .nano_action_only import nano_action_only_construction, verify_nano_action_only_model
        from .sm89_residency import cpu_construction, finalize_service
        with nano_action_only_construction(enabled=nano_action_only), \
                cpu_construction(enabled=sm89_residency) as construction:
            finalizer = (lambda service: finalize_service(
                service, construction, precision=precision, device=dev)) if sm89_residency else None
            service, receipt = build_droid_service(
                _resolve_model_path(checkpoint), precision=precision,
                format_prompt_as_json=extra["format_prompt_as_json"],
                steps=steps["action"], guidance=scale if mode == "cfg" else 1.0,
                seed=int(extra.get("seed", 0)), shift=float(extra.get("shift", 5.0)),
                image_height=int(extra["image_height"]), image_width=int(extra["image_width"]),
                policy_config={k: extra[k] for k in
                               ("domain_name", "action_chunk_size", "conditioning_fps")},
                **({"model_finalizer": finalizer} if finalizer is not None else {}),
            )
            elided_bytes = verify_nano_action_only_model(service.model) if nano_action_only else 0
        if numeric_attention:
            from .numeric_attention import install
            from instinctflash.planners.planner import PassResult, Tier
            install(service)
            plan.results.append(PassResult(
                "cosmos3_cudnn_attention", True, Tier.NUMERIC,
                "Explicit native BF16 numeric attention; task quality is not certified",
                params={"backend": "cudnn", "quality_status": "screen"}))
        if _env_flag("IFL_COSMOS3_EXACT_POINTWISE", default=thor_native):
            from .exact_pointwise import install
            install(service)
        if _env_flag("IFL_COSMOS3_LAYER_GRAPHS", default=thor_native):
            from .thor_graphs import install
            install(service)
        if prompt_kv_cache:
            from .persistent_text_kv import install_persistent_text_kv
            install_persistent_text_kv(service)
        if conditioning_cache:
            if thor_native:
                from .conditioning_cache import install
                install(service)
            else:
                service._ifl_conditioning_cache_status = {
                    "admitted": False, "reason": "Qualified native Thor Edge/Nano declarations only"}
        if generation_regions:
            from .generation_region import install
            from instinctflash.planners.planner import PassResult, Tier
            install(service, plan, split_prefill=split_prefill, contiguous_kv=contiguous_kv,
                    fused_linear=fused_linear)
            plan.results.append(PassResult(
                "cosmos3_generation_regions", True, Tier.NUMERIC,
                "Experimental tensor-only GEN compilation; task quality pending",
                params={"quality_status": "screen", "fullgraph": True, "split_prefill": split_prefill,
                        "contiguous_kv": contiguous_kv, "fused_linear": fused_linear}))
        if timestep_cache:
            from .timestep_cache import install
            install(service)
        service._ifl_native_optimizations = {
            "timestep_cache_requested": timestep_cache,
            "persistent_text_kv": prompt_kv_cache,
            "elided_lm_head_bytes": elided_bytes,
            "conditioning_cache_requested": conditioning_cache,
        }
        if prompt_kv_cache or nano_action_only:
            print(f"InstinctFlash Cosmos native options: {service._ifl_native_optimizations}")
        return CosmosDROIDLoop(service, receipt)

# Compatibility name for callers importing the former adapter-local loop.
from instinctflash.runtime.cosmos_droid import CosmosDROIDLoop as _Cosmos3PolicyLoop


def _encode_png_b64(image) -> str:
    """Lossless PNG round-trip — the same bytes-in-bytes-out channel every arm was measured
    over, so wrapping in-process changes nothing about what reaches the transform."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"'image' must be HxWx3 RGB, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"'image' must be uint8 pixels, got {array.dtype}")
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resolve_model_path(checkpoint) -> Path:
    """The consolidated HF-layout checkpoint dir (config.json + model.safetensors.index.json).
    cosmos-framework wants a LOCAL ABSOLUTE path, so a Hub pointer is snapshotted first."""
    root = Path(checkpoint.path)
    if _is_cosmos_checkpoint(root):
        return root.resolve()
    pointer = (checkpoint.execution.extra or {}).get("base_weights")
    if pointer and Path(str(pointer)).exists():
        base = Path(str(pointer))
    elif pointer:
        from huggingface_hub import snapshot_download

        base = Path(snapshot_download(str(pointer)))
    else:
        raise RuntimeError(f"{checkpoint.model_id}: no local weights and no base_weights pointer")
    if not _is_cosmos_checkpoint(base):
        raise RuntimeError(
            f"Cosmos3 checkpoint not found under {base}: expected the released HF layout "
            f"(config.json + model.safetensors.index.json + checkpoint.json).")
    return base.resolve()


def _is_cosmos_checkpoint(path: Path) -> bool:
    return ((path / "config.json").exists()
            and ((path / "model.safetensors.index.json").exists()
                 or next(path.glob("*.safetensors"), None) is not None))


def _missing_serving_config_message(checkpoint, missing) -> str:
    name = getattr(checkpoint.execution, "model_id", "") or getattr(checkpoint, "path", "")
    return (
        f"{name}: the declaration is missing "
        f"{missing} from its execution block. The Cosmos3 serving config is a set of "
        f"measured facts, not defaults — the published DROID rows used domain_name="
        f"'droid_lerobot', action_dim=8, action_chunk_size=16, image 540x640, num_steps 4, "
        f"guidance 1.0 — and a guessed value silently skews serve-time preprocessing away "
        f"from training. Declare them in the checkpoint's instinctflash.json.")


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean flag, got {value!r}")


def _numeric_attention_requested(precision, plan, thor_native):
    from instinctflash.planners.planner import Tier
    from instinctflash.runtime.precision import require_transform_permission

    choice = os.environ.get("IFL_COSMOS3_ATTENTION", "auto").lower()
    if choice not in ("auto", "native", "cudnn"):
        raise ValueError("IFL_COSMOS3_ATTENTION must be auto, native or cudnn")
    requested = choice == "cudnn" or (
        choice == "auto" and thor_native and getattr(plan, "tier_ceiling", Tier.BITEXACT) >= Tier.NUMERIC)
    if requested:
        if precision != "native" or not thor_native:
            raise ValueError("Cosmos cuDNN attention requires native Thor Edge/Nano")
        require_transform_permission(plan, Tier.NUMERIC, "Cosmos cuDNN attention")
    return requested

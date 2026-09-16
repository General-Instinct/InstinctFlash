"""Runtime adapter for ``GEAR-Dreams/DreamZero-DROID`` causal video-action WAM.

The serving wrapper's Wan225B name does not determine checkpoint architecture;
the released DROID checkpoint and historical 5B variant have different geometry.

Wraps the official serving wrapper in-process — ``DreamZeroWan225BPolicy`` over
``GrootSimPolicy`` from the GEAR-Dreams checkout (``eval_utils/serve_dreamzero_wan22.py``) —
the same stack the H100 row was measured through. Inference is causal with a KV cache carried
ACROSS control cycles within an episode: the first call of a session warms the cache with one
frame per camera, later calls append four; ``reset()`` clears the buffers and the action head's
``current_start_frame``. That is why this family declares a WINDOW-lifetime stream and why
whole-cycle graph capture correctly does not apply.

DYNAMIC_CACHE_SCHEDULE exposes upstream's velocity-cosine step skipper with explicit
BEHAVIORAL permission. The pinned vLLM-Omni backend implements the same decision rule:
video similarity controls reuse of both video and conditional action predictions.
This changes computation, is off by default and carries no task-quality certificate.
Native and FP8 share the explicitly selected controller. See
INSTALL.rst for the API and
eval/dynamic_step_cache_integration_2026-09-14/audit.json for the bounded
integration screen; historical H100 timing is not a Thor claim.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from instinctflash import AdapterSpec, GuidanceRule, KVLifetime, KVStreamSpec, PhaseSpec
from instinctflash.adapters.base import GuidanceMode, ObservationField, ObservationSpec

BACKBONE = "dreamzero"
MODEL_ID = "GEAR-Dreams/DreamZero-DROID"
#: Where the GEAR-Dreams checkout is looked for when DREAMZERO_ROOT is unset.
SOURCE_ROOT_CANDIDATES = (
    Path.home() / "dreamzero-repo",
    Path.home() / "dreamzero",
)
#: Upstream's own env var, read by the action head at construction. SCREEN-tier: see module doc.
DYNAMIC_CACHE_ENV = "DYNAMIC_CACHE_SCHEDULE"
SHIPPED_DIT_MASK = (True, True, True, False, False, False, True, False,
                    False, False, True, False, False, True, True, True)
_FIXED_DIT_MASKS = {
    5: (True, True, True, False, False, False, False, True,
        False, False, False, False, True, False, False, False),
    6: (True, True, False, False, False, True, False, False,
        False, False, True, False, False, False, True, True),
    7: (True, True, True, False, False, False, True, False,
        False, False, True, False, False, False, True, True),
    8: SHIPPED_DIT_MASK,
}


class DreamZeroAdapter:
    """Causal video-action WAM: 16-step CFG diffusion per chunk (the shipped mask computes 8),
    KV committed and carried across chunks within an episode."""

    # The declaration fixes the schedule; the checkpoint config fixes actual DiT geometry.
    # Hub preflight downloads only these metadata files, pinned to the declaration revision.
    PLANNING_FILES = ("config.json",)

    def spec(self) -> AdapterSpec:
        return AdapterSpec(
            model_id=MODEL_ID,
            param_bytes=45_848_344_232,
            # 160x320 through the Wan2.2 VAE38 -> 50 tokens per latent frame (upstream's
            # frame_seqlen). The stream OUTLIVES a control cycle — appended per chunk, reset at
            # the episode boundary — which is what makes per-cycle shapes non-static here.
            streams=(KVStreamSpec("video", tokens_per_frame=50, lifetime=KVLifetime.WINDOW),),
            phases=(
                # 16 scheduler steps at CFG 5.0. The SHIPPED configuration computes 8 of the 16
                # DiT forwards (upstream's fixed dit_step_mask, NUM_DIT_STEPS=8) — the baseline
                # is itself already a skipper, which is why nfe declares the scheduler grid and
                # the notes carry the computed count.
                PhaseSpec("video_action", nfe=16, reads=frozenset({"video"}),
                          truncatable=False),
                PhaseSpec("kv_commit", nfe=1, writes=frozenset({"video"}),
                          commit_steps=frozenset({0}), depends_on=("video_action",)),
            ),
            guidance={"video_action": GuidanceRule(mode=GuidanceMode.CFG, scale=5.0,
                                                   batchable=False)},
            observation=ObservationSpec(
                fields=(
                    ObservationField("observation/exterior_image_0_left",
                                     (4, 160, 320, 3), "uint8"),
                    ObservationField("observation/exterior_image_1_left",
                                     (4, 160, 320, 3), "uint8"),
                    ObservationField("observation/wrist_image_left",
                                     (4, 160, 320, 3), "uint8"),
                    ObservationField("observation/joint_position", (7,), "float32"),
                    ObservationField("observation/gripper_position", (1,), "float32"),
                ),
                history=1,
                batched=False,
                conditioning=("prompt",),
            ),
            notes={
                "backbone": "dreamzero",
                "family": "wam",
                "action_reply": "(24, 8): 7 joints + 1 gripper",
                "computed_dit_steps": "8 of 16 (upstream's shipped fixed mask)",
                "first_call": "one frame per camera warms the causal cache; later calls take 4",
                "dynamic_cache_schedule": (
                    "SCREEN evidence, default OFF; requires BEHAVIORAL permission. "
                    "Upstream's video-velocity-cosine step skipper "
                    "(DYNAMIC_CACHE_SCHEDULE=true or execution.dynamic_cache_schedule); "
                    "reuses video and action predictions; no task-quality certificate."),
            },
        )

    def spec_for_checkpoint(self, checkpoint) -> AdapterSpec:
        """Read causal token geometry from the actual model, not wrapper naming.

        The released DROID snapshot has a 40-layer, width-5120 DiT with 880
        tokens/frame; the Wan2.2 5B variant used by older examples has 50/55.
        The native service supports these through its own checkpoint processors.
        """
        import dataclasses
        import json

        config_path = Path(checkpoint.path) / "config.json"
        config = json.loads(config_path.read_text())
        head = config.get("action_head_cfg", {}).get("config", {})
        dit = head.get("diffusion_model_cfg", {})
        tokens = dit.get("frame_seqlen")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("DreamZero checkpoint must declare a positive diffusion_model_cfg.frame_seqlen")
        spec = self.spec()
        notes = dict(spec.notes)
        notes.update({"tokens_per_frame": tokens,
                      "dit_width": dit.get("dim"), "dit_layers": dit.get("num_layers"),
                      "geometry_source": str(config_path),
                      "image_processing": "Native checkpoint eval transforms and service resizing"})
        return dataclasses.replace(spec,
            streams=tuple(dataclasses.replace(stream, tokens_per_frame=tokens)
                          if stream.name == "video" else stream for stream in spec.streams),
            notes=notes)

    def can_host_in_process(self):
        from instinctflash.runtime.execution import imports_available

        ok, reason = imports_available(("torch", "numpy", "cv2", "tianshou", "openpi_client"))
        if not ok:
            return ok, reason
        root = _source_root(required=False)
        if root is None or not (root / "eval_utils" / "serve_dreamzero_wan22.py").exists():
            return False, (f"DreamZero source not found "
                           f"({'at ' + str(root) if root else 'no candidate exists'}); "
                           f"set DREAMZERO_ROOT to the GEAR-Dreams checkout")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from groot.vla.data.dataset import ModalityConfig  # noqa: F401
            from groot.vla.data.schema import EmbodimentTag  # noqa: F401
        except Exception as error:  # noqa: BLE001 - import compatibility IS the host check
            return False, f"GEAR-Dreams cannot import from {root}: {type(error).__name__}: {error}"
        return True, f"the model stack imports and the GEAR-Dreams source is at {root}"

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None, step_cache=None):
        return self._build_native(checkpoint, plan, device=device, nfe=nfe,
                                  step_cache=step_cache)

    def build_sm89_fp8(self, checkpoint, plan, *, device=None, nfe=None, step_cache=None):
        """Pack explicit SM89 projections before bounded single-device residency."""
        from instinctflash.runtime.precision import require_dreamzero_fp8_environment
        from instinctflash.runtime.sm89_fp8 import _require_requested_recipe

        require_dreamzero_fp8_environment()
        _require_requested_recipe(plan, BACKBONE)
        return self._build_native(checkpoint, plan, device=device, nfe=nfe,
                                  step_cache=step_cache, desktop_fp8=True)

    def build_sm120_fp8(self, checkpoint, plan, *, device=None, nfe=None, step_cache=None):
        """Bind the SM120 recipe before native CPU loading and residency."""
        from instinctflash.runtime.precision import require_dreamzero_fp8_environment
        from instinctflash.runtime.sm120_fp8 import _require_requested_recipe

        require_dreamzero_fp8_environment()
        _require_requested_recipe(plan, BACKBONE)
        return self._build_native(checkpoint, plan, device=device, nfe=nfe,
                                  step_cache=step_cache, desktop_fp8=True)

    def _build_native(self, checkpoint, plan, *, device=None, nfe=None, step_cache=None,
                      desktop_fp8=False):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("DreamZero inference requires CUDA")
        dev = str(device or "cuda")
        if ":" in dev and dev.rsplit(":", 1)[1] not in ("", "0"):
            raise RuntimeError(
                f"the GEAR-Dreams stack pins itself to the process's first visible GPU "
                f"(its distributed init calls torch.cuda.set_device(0)); got device={dev!r}. "
                f"Select the GPU with CUDA_VISIBLE_DEVICES instead.")

        schedule = {**dict(checkpoint.execution.nfe or {}), **dict(nfe or {})}
        if int(schedule.get("kv_commit", 1)) != 1:
            raise RuntimeError("DreamZero's native KV commit schedule is fixed at one step")
        declared_steps = int(schedule.get("video_action", 16))
        if declared_steps != 16:
            raise RuntimeError(
                f"DreamZero's scheduler grid is fixed at 16 steps (the shipped mask computes 8 "
                f"of them); nfe['video_action']={declared_steps} is not a servable knob here. "
                f"Fewer computed steps go through upstream's own knobs — NUM_DIT_STEPS (fixed "
                f"masks for 5-8) or DYNAMIC_CACHE_SCHEDULE — and BOTH change outputs, so they "
                f"are SCREEN-tier: closed-loop gate before serving, never a latency flag.")

        from instinctflash.runtime.step_cache_policy import (
            ResolvedStepCache,
            annotate_step_cache_plan,
            resolve_step_cache,
        )
        if step_cache is None:
            ceiling = getattr(getattr(plan, "tier_ceiling", None), "name", "BITEXACT").lower()
            step_cache = resolve_step_cache(checkpoint, tier_ceiling=ceiling, family="dreamzero")
        if not isinstance(step_cache, ResolvedStepCache):
            raise TypeError("DreamZero build requires a resolved step-cache selection")
        if desktop_fp8:
            from instinctflash.runtime.precision import require_dreamzero_fp8_schedule
            require_dreamzero_fp8_schedule(step_cache)
        if plan is not None:
            annotate_step_cache_plan(plan, step_cache)
        elif step_cache.dynamic or step_cache.fixed_steps != 8:
            raise ValueError("DreamZero altered step schedule requires an explicit behavioral plan")
        extra = dict(checkpoint.execution.extra or {})
        if step_cache.dynamic:
            print(
                "InstinctFlash DreamZero: dynamic step cache enabled — OPERATING-POINT, "
                "SCREEN evidence. Video similarity controls video/action prediction reuse; "
                "realized DiT counts vary. No task-quality certificate.")

        root = _source_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from eval_utils.serve_dreamzero_wan22 import (
            DreamZeroWan225BPolicy,
            _get_expected_video_resolution,
            _maybe_init_distributed,
        )

        # Native policy construction resolves this after loading its weights.
        # Check it first: missing video/data dependencies must fail before a
        # multi-gigabyte model is constructed.
        from groot.vla.data.dataset import ModalityConfig  # noqa: F401
        from groot.vla.data.schema import EmbodimentTag
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        from torch.distributed.device_mesh import init_device_mesh

        _maybe_init_distributed()
        mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))
        tag = str(extra.get("embodiment_tag") or "oxe_droid")
        model_path = _resolve_model_path(checkpoint)
        from .residency import use_desktop_residency
        low_memory = use_desktop_residency(dev)
        if desktop_fp8 and not low_memory:
            raise ValueError("DreamZero desktop FP8 requires the single-device residency path")
        return _build_owned_native_loop(
            model_path,
            lambda path: GrootSimPolicy(
                embodiment_tag=EmbodimentTag(tag), model_path=str(path),
                tokenizer_path_override=None, device="cuda", device_mesh=mesh,
                **({"lazy_load": True} if low_memory else {})),
            lambda policy: DreamZeroWan225BPolicy(
                groot_policy=policy,
                image_height=_get_expected_video_resolution(policy)[0],
                image_width=_get_expected_video_resolution(policy)[1], embodiment_tag=tag),
            step_cache=step_cache,
            residency_precision=("fp8" if desktop_fp8 else "native") if low_memory else None,
        )

    def build_fp8(self, checkpoint, *, device=None, nfe=None, plan=None, step_cache=None):
        """Build the native policy, then install the audited causal Q/K/V recipe.

        Arithmetic and dynamic prediction reuse require independent permission.
        The native loop and shared cache remain outside projection installation.
        """
        # The native loader honors LOAD_TRT_ENGINE independently of the enable
        # flag, including an empty value. Refuse before loading model weights.
        from instinctflash.runtime.precision import (
            require_dreamzero_fp8_environment,
            require_dreamzero_fp8_schedule,
        )
        require_dreamzero_fp8_environment()
        from instinctflash.runtime.step_cache_policy import (
            ResolvedStepCache,
            resolve_step_cache,
        )
        if step_cache is None:
            ceiling = getattr(getattr(plan, "tier_ceiling", None), "name", "BITEXACT").lower()
            step_cache = resolve_step_cache(checkpoint, tier_ceiling=ceiling, family="dreamzero")
        if not isinstance(step_cache, ResolvedStepCache):
            raise TypeError("DreamZero build requires a resolved step-cache selection")
        require_dreamzero_fp8_schedule(step_cache)
        from instinctflash.runtime.dreamzero_fp8 import install_dreamzero_fp8

        loop = self.build_in_process(checkpoint, plan, device=device, nfe=nfe,
                                     step_cache=step_cache)
        try:
            head = loop._wrapper._policy.trained_model.action_head
            if head.num_inference_steps != 16 or tuple(head.dit_step_mask) != SHIPPED_DIT_MASK:
                raise ValueError("DreamZero FP8 requires the native fixed 8-of-16 mask or its dynamic profile")
            if loop._dynamic_cache:
                from instinctflash.planners.planner import Tier
                from instinctflash.runtime.precision import require_transform_permission
                require_transform_permission(plan, Tier.BEHAVIORAL, "DreamZero FP8 dynamic step cache")
                if loop._step_cache_hook is None:
                    raise ValueError("DreamZero FP8 dynamic cache requires the installed shared controller")
            import torch
            loop._fp8_recipe = install_dreamzero_fp8(
                head, include_ffn=torch.cuda.get_device_capability() == (11, 0))
        except Exception:
            loop.close()
            raise
        return loop


class _DreamZeroLoop:
    """One control cycle = one causal chunk. The KV cache lives ACROSS cycles, so episode
    boundaries matter: reset() clears the upstream buffers and starts a new session id."""

    def __init__(self, wrapper, *, dynamic_cache: bool, build_declaration=None,
                 checkpoint_view=None, loading_receipt=None, step_cache_hook=None,
                 residency=None):
        self._wrapper = wrapper
        self._dynamic_cache = bool(dynamic_cache)
        self._prompt = ""
        self._session = 0
        self._fp8_recipe = None
        self._build_declaration = build_declaration
        self._checkpoint_view = checkpoint_view
        self._loading_receipt = loading_receipt
        self._step_cache_hook = step_cache_hook
        self._residency = residency
        self._call_lock = threading.Lock()

    @contextmanager
    def _call_scope(self):
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("DreamZero requires serial predict/reset/close calls")
        try:
            yield
        finally:
            self._call_lock.release()

    def reset(self, **conditioning) -> None:
        with self._call_scope():
            self._reset(conditioning)

    def _reset(self, conditioning):
        if self._wrapper is None:
            raise RuntimeError("DreamZero loop is closed")
        if self._step_cache_hook is not None:
            self._step_cache_hook.reset()
        self._prompt = str(conditioning.get("prompt") or "")
        self._session += 1
        self._wrapper.reset({})

    def validate_executed_action(self, executed_action):
        if executed_action is not None:
            raise ValueError("DreamZero's native wrapper does not accept executed-action overrides")

    def predict(self, observation, *, executed_action=None):
        with self._call_scope():
            return self._predict(observation, executed_action=executed_action)

    def _predict(self, observation, *, executed_action=None):
        import numpy as np

        if self._wrapper is None:
            raise RuntimeError("DreamZero loop is closed")
        self.validate_executed_action(executed_action)
        obs = dict(observation)
        prompt = str(obs.get("prompt") or obs.get("task") or self._prompt)
        if not prompt:
            raise ValueError("DreamZero requires a prompt (in reset() or predict())")
        obs["prompt"] = prompt
        # the wrapper resets itself on a session change; ride our episode counter on its logic
        obs.setdefault("session_id", f"instinctflash-{self._session}")
        action = self._wrapper.infer(obs)
        return {"action": np.asarray(action, dtype=np.float32)}

    @property
    def backend_stats(self) -> dict:
        import copy
        declared = copy.deepcopy(self._build_declaration or {})
        mask = declared.get("dit_step_mask")
        changed_mask = mask is not None and tuple(mask) != SHIPPED_DIT_MASK
        return {
            "dynamic_cache_schedule": self._dynamic_cache,
            "tier": "SCREEN" if self._dynamic_cache or changed_mask else "upstream shipped mask",
            "precision": "fp8" if self._fp8_recipe else "native",
            "fp8_recipe": self._fp8_recipe,
            "build": declared,
            "loading": copy.deepcopy(self._loading_receipt),
            "step_cache": self._step_cache_hook.report() if self._step_cache_hook else None,
            "residency": self._residency.report() if self._residency else None,
        }

    def declaration(self):
        if self._build_declaration is None:
            raise RuntimeError("DreamZero build has no verified native-head declaration")
        import copy
        result = copy.deepcopy(self._build_declaration)
        result["precision"] = "fp8" if self._fp8_recipe else "native"
        return result

    def close(self) -> None:
        with self._call_scope():
            self._close()

    def _close(self):
        hook = self._step_cache_hook
        released = hook is None
        try:
            if hook is not None:
                hook.close()
                released = True
        finally:
            # A foreign wrapper conflict is reported after owned hooks/storage
            # are released. Active-generation refusal must retain the model.
            if released or getattr(hook, "closed", False) is True:
                self._step_cache_hook = None
                if self._residency is not None:
                    self._residency.close()
                    self._residency = None
                self._wrapper = None
                if self._checkpoint_view is not None:
                    self._checkpoint_view.cleanup()
                    self._checkpoint_view = None


def _build_owned_native_loop(model_path, policy_factory, wrapper_factory, *, step_cache=None,
                             residency_precision=None):
    """Keep the checkpoint view alive for the policy, releasing it on all failures.

    Full native DiT values load directly as BF16, avoiding the discarded FP32
    allocation on Thor. Other architectures retain their original native loader.
    This preserves checkpoint values, not random-initialization RNG consumption.
    """
    import json

    from instinctflash.runtime.dreamzero_checkpoint import (
        DIT_TARGET,
        prepare_full_checkpoint,
    )

    config = json.loads((Path(model_path) / "config.json").read_text())
    head_config = config.get("action_head_cfg", {}).get("config", {})
    dit_config = head_config.get("diffusion_model_cfg", {})
    view, receipt, hook, residency = None, None, None, None
    try:
        if residency_precision not in (None, "native", "fp8"):
            raise ValueError("Unknown DreamZero residency precision")
        if residency_precision is not None and step_cache is None:
            raise ValueError("DreamZero residency requires an explicitly resolved native schedule")
        if (head_config.get("train_architecture") == "full"
                and dit_config.get("_target_") == DIT_TARGET):
            from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
                CausalWanModel,
            )
            view, receipt = prepare_full_checkpoint(model_path, CausalWanModel, direct_bf16=True)
            receipt = {key: value for key, value in receipt.items() if key != "expected_shapes"}
            receipt["initialization_rng_equivalent"] = False
        if step_cache is not None:
            # Native __init__ parses NUM_DIT_STEPS before the owned-field check.
            # An instance-local factory freezes these reads without global env
            # mutation, including when the ambient value changed or is invalid.
            import tempfile
            if view is None:
                view = tempfile.TemporaryDirectory(prefix="instinctflash-dreamzero-schedule-")
                for item in Path(model_path).iterdir():
                    if item.name != "config.json":
                        (Path(view.name) / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
                (Path(view.name) / "config.json").write_text(json.dumps(config))
            config_path = Path(view.name) / "config.json"
            owned_config = json.loads(config_path.read_text())
            native_target = "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead"
            if owned_config["action_head_cfg"].get("_target_") != native_target:
                raise ValueError("DreamZero native head target changed; re-audit schedule construction")
            owned_config["action_head_cfg"].update(
                _target_="dreamzero_iwm.schedule.build_head",
                ifl_dynamic_cache_schedule=step_cache.dynamic,
                ifl_fixed_dit_steps=step_cache.fixed_steps)
            if residency_precision is not None:
                owned_config["action_head_cfg"].update(
                    _target_="dreamzero_iwm.residency.build_head",
                    ifl_residency_precision=residency_precision)
            config_path.write_text(json.dumps(owned_config, indent=2) + "\n")
        selected_path = Path(view.name) if view is not None else Path(model_path)
        if residency_precision is not None:
            from .residency import construction_scope
            with construction_scope() as owners:
                policy = policy_factory(selected_path)
                residency = owners[0] if owners else None
        else:
            policy = policy_factory(selected_path)
        head = policy.trained_model.action_head
        if residency_precision is not None:
            residency = getattr(head, "_ifl_residency", None)
            if residency is None or not residency.initialized:
                raise RuntimeError("DreamZero native post-initialize did not install owned residency")
        if step_cache is not None:
            # Resolve once in preflight and bind to this owned instance. Native
            # construction's legacy environment reads cannot select serving math.
            head.dynamic_cache_schedule = step_cache.dynamic
            head.dit_step_mask = list(_FIXED_DIT_MASKS[step_cache.fixed_steps])
        wrapper = wrapper_factory(policy)
        if step_cache is not None and step_cache.dynamic:
            from .dynamic_cache import install
            hook = install(head)
        declaration = _head_declaration(head)
        if step_cache is not None:
            declaration["step_cache_selection"] = step_cache.to_dict()
        loop = _DreamZeroLoop(wrapper, dynamic_cache=bool(head.dynamic_cache_schedule),
                             build_declaration=declaration, checkpoint_view=view,
                             loading_receipt=receipt, step_cache_hook=hook, residency=residency)
        if residency is not None:
            loop._fp8_recipe = residency.fp8_recipe
        return loop
    except Exception:
        if hook is not None:
            hook.close()
        if residency is not None:
            residency.close()
        if view is not None:
            view.cleanup()
        raise


def _head_declaration(head):
    """Record the loaded native head, including mask changes from upstream knobs."""
    return {
        "frontend": "dreamzero_iwm.adapter.DreamZeroAdapter",
        "steps": {"video_action": int(head.num_inference_steps), "kv_commit": 1},
        "guidance": {"video_action": ("cfg", float(head.cfg_scale))},
        "dit_step_mask": [bool(step) for step in head.dit_step_mask],
        "dynamic_cache_schedule": bool(head.dynamic_cache_schedule),
        "evidence": "Loaded native DreamZero action head; not a task-quality certificate",
    }


def _source_root(*, required: bool = True) -> "Path | None":
    env = os.environ.get("DREAMZERO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in SOURCE_ROOT_CANDIDATES:
        if (candidate / "eval_utils" / "serve_dreamzero_wan22.py").exists():
            return candidate.resolve()
    if required:
        raise RuntimeError(
            "DreamZero upstream source not found. Set DREAMZERO_ROOT to the GEAR-Dreams "
            f"checkout (searched: {[str(c) for c in SOURCE_ROOT_CANDIDATES]}).")
    return None


def _is_dreamzero_checkpoint(path: Path) -> bool:
    return ((path / "config.json").exists()
            and ((path / "model.safetensors.index.json").exists()
                 or next(path.glob("*.safetensors"), None) is not None))


def _resolve_model_path(checkpoint) -> Path:
    root = Path(checkpoint.path)
    if _is_dreamzero_checkpoint(root):
        return root.resolve()
    pointer = (checkpoint.execution.extra or {}).get("base_weights")
    if pointer and Path(str(pointer)).exists():
        base = Path(str(pointer))
    elif pointer:
        from huggingface_hub import snapshot_download

        base = Path(snapshot_download(str(pointer)))
    else:
        raise RuntimeError(f"{checkpoint.model_id}: no local weights and no base_weights pointer")
    if not _is_dreamzero_checkpoint(base):
        raise RuntimeError(
            f"DreamZero checkpoint not found under {base}: expected the released layout "
            f"(config.json + sharded model.safetensors + experiment_cfg/). Note the Wan "
            f"components (umt5 text encoder, CLIP, Wan2.2 VAE) resolve separately through "
            f"upstream's ensure_file and must be reachable in the HF cache.")
    return base.resolve()


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

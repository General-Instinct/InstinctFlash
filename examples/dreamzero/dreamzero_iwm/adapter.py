"""Runtime adapter for ``GEAR-Dreams/DreamZero-DROID`` (Wan2.2-TI2V-5B causal video-action WAM).

Wraps the official serving wrapper in-process — ``DreamZeroWan225BPolicy`` over
``GrootSimPolicy`` from the GEAR-Dreams checkout (``eval_utils/serve_dreamzero_wan22.py``) —
the same stack the H100 row was measured through. Inference is causal with a KV cache carried
ACROSS control cycles within an episode: the first call of a session warms the cache with one
frame per camera, later calls append four; ``reset()`` clears the buffers and the action head's
``current_start_frame``. That is why this family declares a WINDOW-lifetime stream and why
whole-cycle graph capture correctly does not apply.

DYNAMIC_CACHE_SCHEDULE — upstream's own velocity-cosine step skipper (the exact algorithm
vLLM-Omni's "stepcache" vendored file-for-file) — is surfaced as a DECLARED option and is
SCREEN-tier: it changes outputs by construction (measured max |Δaction| 0.288 on identical
request streams), so it is never default-on and a closed-loop success-rate gate is mandatory
before anyone ships it. H100 pair with it ON: 3226.7 -> 1843.1 ms (1.75x).
"""

from __future__ import annotations

import os
import sys
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


class DreamZeroAdapter:
    """Causal video-action WAM: 16-step CFG diffusion per chunk (the shipped mask computes 8),
    KV committed and carried across chunks within an episode."""

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
                "family": "wam",
                "action_reply": "(24, 8): 7 joints + 1 gripper",
                "computed_dit_steps": "8 of 16 (upstream's shipped fixed mask)",
                "first_call": "one frame per camera warms the causal cache; later calls take 4",
                "dynamic_cache_schedule": (
                    "SCREEN-tier option, default OFF. Upstream's velocity-cosine step skipper "
                    "(DYNAMIC_CACHE_SCHEDULE=true or execution.dynamic_cache_schedule); "
                    "changes outputs by construction (measured max |dA| 0.288) — a closed-loop "
                    "gate is mandatory before it ships as anyone's default."),
            },
        )

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
            from groot.vla.data.schema import EmbodimentTag  # noqa: F401
        except Exception as error:  # noqa: BLE001 - import compatibility IS the host check
            return False, f"GEAR-Dreams cannot import from {root}: {type(error).__name__}: {error}"
        return True, f"the model stack imports and the GEAR-Dreams source is at {root}"

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
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
        declared_steps = int(schedule.get("video_action", 16))
        if declared_steps != 16:
            raise RuntimeError(
                f"DreamZero's scheduler grid is fixed at 16 steps (the shipped mask computes 8 "
                f"of them); nfe['video_action']={declared_steps} is not a servable knob here. "
                f"Fewer computed steps go through upstream's own knobs — NUM_DIT_STEPS (fixed "
                f"masks for 5-8) or DYNAMIC_CACHE_SCHEDULE — and BOTH change outputs, so they "
                f"are SCREEN-tier: closed-loop gate before serving, never a latency flag.")

        extra = dict(checkpoint.execution.extra or {})
        dynamic = _env_flag(DYNAMIC_CACHE_ENV,
                            default=bool(extra.get("dynamic_cache_schedule", False)))
        # Upstream reads the env var at action-head construction, so it must be set BEFORE the
        # policy is built, and it must be set explicitly either way — an inherited stale value
        # from the parent shell would silently change tiers.
        os.environ[DYNAMIC_CACHE_ENV] = "true" if dynamic else "false"
        if dynamic:
            print(
                "InstinctFlash DreamZero: DYNAMIC_CACHE_SCHEDULE is ON — SCREEN TIER. This is "
                "upstream's velocity-cosine step skipper; it changes actions by construction "
                "(measured max |dA| 0.288 vs the shipped mask) and carries no closed-loop "
                "certificate. Do not report benchmark numbers from this arm as the default "
                "configuration. (H100 latency reference: 3226.7 -> 1843.1 ms, 1.75x.)")

        root = _source_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from eval_utils.serve_dreamzero_wan22 import (
            DreamZeroWan225BPolicy, _get_expected_video_resolution, _maybe_init_distributed,
        )
        from groot.vla.data.schema import EmbodimentTag
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        from torch.distributed.device_mesh import init_device_mesh

        _maybe_init_distributed()
        mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))
        tag = str(extra.get("embodiment_tag") or "oxe_droid")
        model_path = _resolve_model_path(checkpoint)
        policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag(tag),
            model_path=str(model_path),
            tokenizer_path_override=None,
            device="cuda",
            device_mesh=mesh,
        )
        height, width = _get_expected_video_resolution(policy)
        wrapper = DreamZeroWan225BPolicy(
            groot_policy=policy, image_height=height, image_width=width, embodiment_tag=tag,
        )
        return _DreamZeroLoop(wrapper, dynamic_cache=dynamic)


class _DreamZeroLoop:
    """One control cycle = one causal chunk. The KV cache lives ACROSS cycles, so episode
    boundaries matter: reset() clears the upstream buffers and starts a new session id."""

    def __init__(self, wrapper, *, dynamic_cache: bool):
        self._wrapper = wrapper
        self._dynamic_cache = bool(dynamic_cache)
        self._prompt = ""
        self._session = 0

    def reset(self, **conditioning) -> None:
        self._prompt = str(conditioning.get("prompt") or "")
        self._session += 1
        self._wrapper.reset({})

    def predict(self, observation):
        import numpy as np

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
        return {
            "dynamic_cache_schedule": self._dynamic_cache,
            "tier": "SCREEN" if self._dynamic_cache else "upstream shipped mask",
        }

    def close(self) -> None:
        self._wrapper = None


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

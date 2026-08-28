"""Runtime adapter for ``nvidia/GR00T-N1.7-3B``.

The initial backend is a correctness-first upstream BF16 baseline.  GR00T N1.7
has no persistent KV stream: each control cycle computes backbone features once
and reuses them across the four flow-matching steps inside upstream ``get_action``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from instinctflash import AdapterSpec, GuidanceRule, KVLifetime, PhaseSpec, PurityKey
from instinctflash.adapters.base import GuidanceMode, ObservationField, ObservationSpec

BACKBONE = "groot_n17"
MODEL_ID = "nvidia/GR00T-N1.7-3B"
#: Looked for when GR00T_ROOT is unset; the env var is the contract, these are conventions.
SOURCE_ROOT_CANDIDATES = (
    Path.home() / "Isaac-GR00T",
    Path.home() / "Code" / "Isaac-GR00T",
)
DEFAULT_EMBODIMENT = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"


class GR00TN17Adapter:
    """N1.7: one vision/language backbone pass plus four action-flow steps."""

    HOST_REQUIRES = ("torch", "transformers", "safetensors", "numpy")
    #: RETIRED opt-in (the old release policy predating the startup self-check): "1" is a
    #: no-op with a notice — capture is the default now; "0" is honored as an explicit
    #: opt-out with a notice naming the kill-switch that replaces it.
    STATIC_CAPTURE_ENV = "IFL_GROOT_STATIC_CAPTURE"
    #: The kill-switch for the family's DEFAULT DiT graph capture. Family-scoped, the
    #: IFL_PI05_NO_CAPTURE convention. Honored by `install`, recorded on the plan, printed.
    CAPTURE_KILL_SWITCH = "IFL_GROOT_NO_CAPTURE"
    CPU_THREADS_ENV = "IFL_GROOT_CPU_THREADS"
    FAST_DECODE_ENV = "IFL_GROOT_FAST_DECODE"
    BACKBONE_FASTPATH_ENV = "IFL_GROOT_BACKBONE_FASTPATH"

    def spec(self) -> AdapterSpec:
        return AdapterSpec(
            model_id=MODEL_ID,
            param_bytes=6_910_499_416,
            streams=(),
            phases=(
                PhaseSpec("backbone", nfe=1),
                PhaseSpec(
                    "action", nfe=4, truncatable=True, min_nfe=1,
                    depends_on=("backbone",),
                ),
            ),
            guidance={"action": GuidanceRule(mode=GuidanceMode.NONE)},
            purity=(
                PurityKey(
                    "backbone_features",
                    ("video", "state", "prompt", "embodiment"),
                    KVLifetime.CHUNK,
                    already_hoisted=True,
                ),
            ),
            observation=ObservationSpec(
                fields=(
                    ObservationField(
                        "video.exterior_image_1_left", (2, 180, 320, 3), "uint8"
                    ),
                    ObservationField(
                        "video.wrist_image_left", (2, 180, 320, 3), "uint8"
                    ),
                    # position(3) + rotation-6D(6). The zero vector is a DEGENERATE rotation —
                    # upstream's SVD orthonormalization diverges on it — so the declared smoke
                    # example is the identity frame's first two basis vectors, same construction
                    # as verify_fastpaths.py's synth_obs.
                    ObservationField("state.eef_9d", (1, 9), "float32",
                                     example=(0, 0, 0, 1, 0, 0, 0, 1, 0)),
                    ObservationField("state.gripper_position", (1, 1), "float32"),
                    ObservationField("state.joint_position", (1, 7), "float32"),
                ),
                history=1,
                batched=False,
                conditioning=("prompt",),
            ),
            notes={
                "family": "vla",
                "action_horizon": "40",
                "default_embodiment": DEFAULT_EMBODIMENT,
                "backend": "upstream BF16; optional bitexact DiT CUDA Graph",
                "numeric_tier": "upstream BF16",
            },
        )

    def can_host_in_process(self):
        from instinctflash.runtime.execution import imports_available

        root = _source_root()
        if not (root / "gr00t" / "policy" / "gr00t_policy.py").exists():
            return False, f"Isaac-GR00T source not found at {root}; set GR00T_ROOT"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        ok, reason = imports_available(self.HOST_REQUIRES)
        if not ok:
            return ok, reason
        try:
            from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: F401
        except Exception as error:  # noqa: BLE001 - import compatibility is the host check
            return False, f"Isaac-GR00T cannot import: {type(error).__name__}: {error}"
        return True, f"the model stack imports and Isaac-GR00T source is at {root}"

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GR00T N1.7 inference requires CUDA")
        dev = str(device or f"cuda:{torch.cuda.current_device()}")
        if not dev.startswith("cuda"):
            raise RuntimeError(f"GR00T N1.7 only supports a CUDA device, got {dev!r}")
        if ":" in dev:
            torch.cuda.set_device(int(dev.rsplit(":", 1)[1]))

        extra = dict(checkpoint.execution.extra or {})
        # CPU thread pinning is HOST configuration, opt-in via the env var ONLY. A checkpoint
        # manifest must not mutate process-global torch/cv2 thread pools: the win is
        # host-specific (~165 ms on a 240-CPU box vs ~2 ms measured on our 208-CPU H100),
        # a co-hosted engine inherits the cap, and on <16-core hosts a fixed value RAISES
        # rather than caps threads.
        cpu_threads = _configure_preprocessing_threads(os.environ.get(self.CPU_THREADS_ENV))
        root = _source_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        # transformers 4.57.3 _patch_mistral_regex calls the Hub API even in offline mode (and
        # 401s on the gated backbone repo when online without credentials); the checkpoint's
        # Qwen3 tokenizer is not a mistral model, skip it. Same workaround as
        # verify_fastpaths.py / verify_fast_decode.py — the adapter must load from a warm cache.
        try:
            import transformers.tokenization_utils_base as tub

            def _no_mistral_patch(cls, tokenizer, *args, **kwargs):
                return tokenizer

            tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(_no_mistral_patch)
        except Exception:  # noqa: BLE001 - a transformers without the probe needs no patch
            pass
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        embodiment = str(extra.get("embodiment_tag") or DEFAULT_EMBODIMENT)
        model_path = _resolve_model_path(checkpoint)
        policy = Gr00tPolicy(
            embodiment_tag=embodiment,
            model_path=str(model_path),
            device=dev,
            strict=True,
        )
        schedule = {**dict(checkpoint.execution.nfe or {}), **dict(nfe or {})}
        action_nfe = int(schedule.get("action", 4))
        if action_nfe < 1:
            raise ValueError(f"GR00T N1.7 action NFE must be positive, got {action_nfe}")
        policy.model.action_head.num_inference_timesteps = action_nfe
        if hasattr(policy.model, "config"):
            policy.model.config.num_inference_timesteps = action_nfe
        fast_decoder = None
        if _env_flag(
            self.FAST_DECODE_ENV, default=bool(extra.get("fast_decode", False))
        ):
            from .fast_decode import install_fast_decode

            fast_decoder = install_fast_decode(policy)
        backbone_fastpath = None
        if _env_flag(
            self.BACKBONE_FASTPATH_ENV,
            default=bool(extra.get("backbone_fastpath", False)),
        ):
            from .backbone_fastpath import install_backbone_fastpath

            backbone_fastpath = install_backbone_fastpath(policy.model)
        self.install(policy, plan, device=dev)
        driver = getattr(policy, "_instinctflash_static_capture", None)
        return _GR00TN17Loop(
            policy,
            model_path=model_path,
            action_nfe=action_nfe,
            driver=driver,
            cpu_threads=cpu_threads,
            fast_decode=fast_decoder is not None,
            backbone_fastpath=backbone_fastpath,
        )

    @classmethod
    def install(cls, policy, plan, *, device=None) -> list[str]:
        """Act on the plan: the DiT graph is the FAMILY DEFAULT, gated by the self-check.

        This SUPERSEDES the old release policy that kept capture opt-in through
        ``IFL_GROOT_STATIC_CAPTURE=1``. That policy predates the startup self-check: the
        standalone gate merged in ``3142eee`` was bitexact, but it was a one-time measurement
        on OTHER checkpoints, so defaulting the graph on for a fresh fine-tune had nothing
        per-process to stand on. Now it does — the first capture of every signature is
        compared against the upstream eager DiT forward on staged inputs it was not captured
        from (exact equality, seconds of startup, once per process), and a mismatch releases
        the graphs and falls back to eager loudly while serving continues.

        ``IFL_GROOT_NO_CAPTURE=1`` is the kill-switch (recorded on the plan, printed). The
        retired opt-in stays recognized: "1" is a no-op with a notice, an explicit "0" is
        honored as an opt-out with a notice naming the kill-switch.
        """
        import os

        wanted = {
            getattr(result, "name", "")
            for result in getattr(plan, "results", ())
            if getattr(result, "applies", False)
        }
        if "graph_capture" not in wanted:
            return []
        import torch

        if not (device and str(device).startswith("cuda") and torch.cuda.is_available()):
            print("InstinctFlash GR00T N1.7: CUDA Graph needs CUDA; running eager.")
            return []
        capture = next(r for r in plan.results if r.name == "graph_capture" and r.applies)
        legacy = os.environ.get(cls.STATIC_CAPTURE_ENV)
        legacy_off = legacy is not None and not _env_flag(cls.STATIC_CAPTURE_ENV, default=True)
        if os.environ.get(cls.CAPTURE_KILL_SWITCH) == "1" or legacy_off:
            cause = (f"{cls.CAPTURE_KILL_SWITCH}=1" if not legacy_off
                     else f"{cls.STATIC_CAPTURE_ENV}={legacy} (the retired opt-in's explicit "
                          f"opt-out, honored; use {cls.CAPTURE_KILL_SWITCH}=1)")
            note = (f"{cause} — the default DiT graph capture is disabled by the caller; "
                    f"running eager (upstream's arithmetic exactly)")
            capture.params["decision"] = tuple(capture.params.get("decision", ())) + (note,)
            print(f"InstinctFlash GR00T N1.7: {note}.")
            return []
        if legacy is not None:
            print(f"InstinctFlash GR00T N1.7: {cls.STATIC_CAPTURE_ENV}={legacy} is a no-op — "
                  f"the DiT graph is the default for N1.7-class checkpoints on capture-capable "
                  f"devices now ({cls.CAPTURE_KILL_SWITCH}=1 disables it).")
        from instinctflash.runtime.capture_self_check import record_self_check_on_plan

        from .static_capture import install_static_capture

        driver = install_static_capture(
            policy.model,
            on_self_check=record_self_check_on_plan(capture, "GR00T N1.7"))
        policy._instinctflash_static_capture = driver
        print("InstinctFlash GR00T N1.7: DiT CUDA Graph installed — the family default on "
              "capture-capable devices, superseding the retired IFL_GROOT_STATIC_CAPTURE "
              "opt-in. Each captured signature is gated by a bit-exact self-check (replay vs "
              "upstream eager on staged inputs it was not captured from, exact equality); a "
              "mismatch releases the graphs and falls back to eager, loudly. Kill-switch: "
              f"{cls.CAPTURE_KILL_SWITCH}=1.")
        return ["graph_capture"]


class _GR00TN17Loop:
    def __init__(
        self,
        policy,
        *,
        model_path: Path,
        action_nfe: int,
        driver=None,
        cpu_threads: "_ThreadPin | None" = None,
        fast_decode: bool = False,
        backbone_fastpath=None,
    ):
        self._policy = policy
        self._model_path = model_path
        self._prompt = ""
        self._action_nfe = int(action_nfe)
        self._driver = driver
        self._cpu_threads = cpu_threads
        self._fast_decode = bool(fast_decode)
        self._backbone_fastpath = backbone_fastpath
        self._state_dims = _state_dimensions(
            model_path,
            policy.embodiment_tag.value,
            policy.modality_configs["state"].modality_keys,
        )

    def reset(self, **conditioning) -> None:
        self._prompt = str(conditioning.get("prompt") or "").strip()
        self._policy.reset()

    def predict(self, observation):
        nested = _normalise_observation(
            self._policy,
            dict(observation),
            prompt=self._prompt,
            state_dims=self._state_dims,
        )
        action_dict, info = self._policy.get_action(nested)
        action, split = _format_actions(self._policy, action_dict)
        return {"action": action, "actions": split, "info": info}

    @property
    def backend_stats(self) -> dict[str, Any]:
        return {
            "backend": (
                "upstream_bf16_cuda_graph" if self._driver else "upstream_bf16_eager"
            ),
            "precision": "bfloat16",
            "captured": bool(self._driver and self._driver.captured),
            "graph_captures": int(self._driver.captures if self._driver else 0),
            "graph_replays": int(self._driver.replays if self._driver else 0),
            "action_nfe": self._action_nfe,
            "cpu_threads": (self._cpu_threads.target if self._cpu_threads else None),
            "fast_decode": self._fast_decode,
            "backbone_fastpath": self._backbone_fastpath is not None,
            "backbone_cache_hits": int(
                self._backbone_fastpath.hits if self._backbone_fastpath else 0
            ),
            "backbone_cache_misses": int(
                self._backbone_fastpath.misses if self._backbone_fastpath else 0
            ),
        }

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
        if self._backbone_fastpath is not None:
            self._backbone_fastpath.close()
        if self._cpu_threads is not None:
            # The pin is process-global; a closed model must not leave its cap on the process.
            self._cpu_threads.restore()
        self._driver = None
        self._backbone_fastpath = None
        self._cpu_threads = None
        self._policy = None


def _normalise_observation(policy, observation, *, prompt, state_dims):
    import numpy as np

    cfg = policy.modality_configs
    video_keys = tuple(cfg["video"].modality_keys)
    state_keys = tuple(cfg["state"].modality_keys)
    video_horizon = len(cfg["video"].delta_indices)
    state_horizon = len(cfg["state"].delta_indices)

    source_video = observation.get("video")
    video_values = dict(source_video) if isinstance(source_video, dict) else {}
    for key in video_keys:
        if key not in video_values and f"video.{key}" in observation:
            video_values[key] = observation[f"video.{key}"]
    compact_images = observation.get("images")
    if compact_images is None:
        compact_images = [
            observation[key]
            for key in ("image", "wrist_image", "wrist_image_right")
            if key in observation
        ]
    if isinstance(compact_images, np.ndarray):
        compact_images = [compact_images]
    if compact_images is not None:
        for index, value in enumerate(list(compact_images)[:len(video_keys)]):
            video_values.setdefault(video_keys[index], value)
    missing = [key for key in video_keys if key not in video_values]
    if missing:
        raise ValueError(f"GR00T N1.7 is missing video streams {missing}")
    video = {
        key: _video_batch(video_values[key], video_horizon, key) for key in video_keys
    }

    source_state = observation.get("state")
    state_values = dict(source_state) if isinstance(source_state, dict) else {}
    for key in state_keys:
        if key not in state_values and f"state.{key}" in observation:
            state_values[key] = observation[f"state.{key}"]
    if source_state is not None and not isinstance(source_state, dict):
        for key, value in _split_state(source_state, state_keys, state_dims).items():
            state_values.setdefault(key, value)
    missing = [key for key in state_keys if key not in state_values]
    if missing:
        raise ValueError(f"GR00T N1.7 is missing state streams {missing}")
    state = {
        key: _state_batch(state_values[key], state_horizon, state_dims[key], key)
        for key in state_keys
    }

    batch = next(iter(video.values())).shape[0]
    if any(value.shape[0] != batch for value in (*video.values(), *state.values())):
        raise ValueError("GR00T N1.7 video/state batch dimensions must match")
    language_keys = tuple(cfg["language"].modality_keys)
    source_language = observation.get("language")
    language = dict(source_language) if isinstance(source_language, dict) else {}
    selected_prompt = observation.get("prompt") or observation.get("task") or prompt
    for key in language_keys:
        if key not in language:
            if not selected_prompt:
                raise ValueError("GR00T N1.7 requires a prompt in reset() or predict()")
            language[key] = [[str(selected_prompt)] for _ in range(batch)]
    return {"video": video, "state": state, "language": language}


def _format_actions(policy, actions):
    import numpy as np

    keys = tuple(policy.modality_configs["action"].modality_keys)
    arrays = {key: np.asarray(actions[key], dtype=np.float32) for key in keys}
    joined = np.concatenate([arrays[key] for key in keys], axis=-1)
    if joined.shape[0] == 1:
        return joined[0], {key: value[0] for key, value in arrays.items()}
    return joined, arrays


def _video_batch(value, horizon, key):
    import numpy as np

    array = np.asarray(value)
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"video {key!r} must be uint8 or floating point")
        scale = 255.0 if array.size and float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    if array.ndim == 3:
        array = np.repeat(array[None], horizon, axis=0)[None]
    elif array.ndim == 4:
        if array.shape[0] == 1 and horizon > 1:
            array = np.repeat(array, horizon, axis=0)
        if array.shape[0] != horizon:
            raise ValueError(f"video {key!r} requires T={horizon}, got {array.shape}")
        array = array[None]
    elif array.ndim == 5:
        if array.shape[1] == 1 and horizon > 1:
            array = np.repeat(array, horizon, axis=1)
        if array.shape[1] != horizon:
            raise ValueError(f"video {key!r} requires T={horizon}, got {array.shape}")
    else:
        raise ValueError(f"video {key!r} must be HWC, THWC, or BTHWC, got {array.shape}")
    if array.shape[-1] != 3:
        raise ValueError(f"video {key!r} must have three RGB channels")
    return np.ascontiguousarray(array)


def _state_batch(value, horizon, width, key):
    import numpy as np

    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, None]
    elif array.ndim == 2:
        array = array[None]
    elif array.ndim != 3:
        raise ValueError(f"state {key!r} must be D, TD, or BTD, got {array.shape}")
    if array.shape[1:] != (horizon, width):
        raise ValueError(
            f"state {key!r} requires shape (*, {horizon}, {width}), got {array.shape}"
        )
    return np.ascontiguousarray(array)


def _split_state(value, keys, dimensions):
    import numpy as np

    array = np.asarray(value, dtype=np.float32)
    expected = sum(dimensions[key] for key in keys)
    if array.ndim not in (1, 2, 3) or array.shape[-1] != expected:
        raise ValueError(f"flat GR00T N1.7 state must end in {expected} values, got {array.shape}")
    stops = np.cumsum([dimensions[key] for key in keys])[:-1]
    return dict(zip(keys, np.split(array, stops, axis=-1), strict=True))


def _state_dimensions(checkpoint: Path, embodiment: str, keys) -> dict[str, int]:
    path = checkpoint / "statistics.json"
    statistics = json.loads(path.read_text())
    try:
        state = statistics[embodiment]["state"]
        return {key: len(state[key]["mean"]) for key in keys}
    except KeyError as error:
        raise ValueError(f"no state statistics for embodiment {embodiment!r}") from error


def _source_root() -> Path:
    env = os.environ.get("GR00T_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in SOURCE_ROOT_CANDIDATES:
        if (candidate / "gr00t" / "policy" / "gr00t_policy.py").exists():
            return candidate.resolve()
    # Return the first convention so can_host_in_process can name a concrete missing path.
    return SOURCE_ROOT_CANDIDATES[0]


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


class _ThreadPin:
    """A process-global torch/cv2 thread cap that remembers what it replaced.

    torch.set_num_threads / cv2.setNumThreads mutate the WHOLE process — a co-hosted engine
    inherits the cap — so the pin is (a) opt-in via IFL_GROOT_CPU_THREADS only, never a manifest
    or frontend default, and (b) restored on close(). The win is host-specific: ~165 ms on a
    240-logical-CPU box, ~2 ms measured on our 208-CPU H100 (bitexact either way).
    """

    def __init__(self, target: int, prev_torch: int, prev_cv2: int | None):
        self.target = int(target)
        self._prev_torch = int(prev_torch)
        self._prev_cv2 = prev_cv2
        self.restored = False

    def restore(self) -> None:
        if self.restored:
            return
        import torch

        torch.set_num_threads(self._prev_torch)
        if self._prev_cv2 is not None:
            try:
                import cv2
            except ImportError:
                pass
            else:
                cv2.setNumThreads(self._prev_cv2)
        self.restored = True


def _configure_preprocessing_threads(value) -> _ThreadPin | None:
    """Bound tiny image-batch CPU pools without changing numerical kernels. Returns the pin
    handle (carrying the previous values for restore-on-close), or None when not requested."""
    if value is None or str(value).strip().lower() in {"", "0", "off", "false", "none"}:
        return None
    text = str(value).strip().lower()
    # `auto` CAPS at min(16, cores): a fixed 16 on a <16-core host would RAISE thread counts.
    target = min(16, os.cpu_count() or 16) if text == "auto" else int(text)
    if target < 1:
        raise ValueError(f"GR00T N1.7 CPU threads must be positive, got {target}")

    import torch

    prev_torch = torch.get_num_threads()
    torch.set_num_threads(target)
    prev_cv2 = None
    try:
        import cv2
    except ImportError:
        pass
    else:
        prev_cv2 = cv2.getNumThreads()
        cv2.setNumThreads(target)
    return _ThreadPin(target, prev_torch, prev_cv2)


def _is_checkpoint(path: Path) -> bool:
    return (
        ((path / "model.safetensors.index.json").exists() or any(path.glob("*.safetensors")))
        and (path / "processor_config.json").exists()
    )


def _resolve_model_path(checkpoint) -> Path:
    root = Path(checkpoint.path).expanduser().resolve()
    candidates = [root]
    extra = dict(checkpoint.execution.extra or {})
    pointer = extra.get("base_weights")
    env_checkpoint = os.environ.get("GR00T_N17_CHECKPOINT")
    if env_checkpoint:
        candidates.append(Path(env_checkpoint).expanduser().resolve())
    # Convenience only for this official pointer package. A different
    # pointer-only N1.7 package must never be shadowed by a workstation's
    # unrelated base checkpoint.
    if checkpoint.model_id == MODEL_ID and pointer == MODEL_ID:
        candidates.append(_source_root() / "checkpoints" / "GR00T-N1.7-3B")
    for candidate in candidates:
        if _is_checkpoint(candidate):
            return candidate

    if pointer and Path(str(pointer)).expanduser().exists():
        candidate = Path(str(pointer)).expanduser().resolve()
    elif pointer:
        from huggingface_hub import snapshot_download

        candidate = Path(snapshot_download(str(pointer)))
    else:
        raise RuntimeError(f"{checkpoint.model_id}: no local weights and no base_weights pointer")
    if not _is_checkpoint(candidate):
        raise RuntimeError(f"GR00T N1.7 weights or processor files not found under {candidate}")
    return candidate

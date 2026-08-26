"""GR00T N1.7 upstream-BF16 frontend for A100 and H100.

This is the correctness baseline for CUDA architectures that are not served by
the experimental Thor implementation.  It intentionally uses NVIDIA's
``Gr00tPolicy`` end to end: processor, BF16 model, flow loop, and action
denormalisation all remain upstream code. The verified companion adapter can
replace only the four repeated DiT calls with replay-safe CUDA Graphs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EMBODIMENT = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
#: Looked for when GR00T_ROOT is unset; the env var / source_root= argument is the contract.
SOURCE_ROOT_CANDIDATES = (
    Path.home() / "Isaac-GR00T",
    Path.home() / "Code" / "Isaac-GR00T",
)


class GrootN17TorchFrontendCuda:
    """Run the upstream GR00T N1.7 BF16 policy on SM80/SM90 GPUs."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        num_views: int = 2,
        embodiment_tag: str = DEFAULT_EMBODIMENT,
        source_root: str | None = None,
        action_horizon: int | None = None,
        use_cuda_graph: bool = True,
        device: str | None = None,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GR00T N1.7 inference requires CUDA")
        self.device = str(device or f"cuda:{torch.cuda.current_device()}")
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        _require_checkpoint(Path(self.checkpoint_path))

        root = _source_root(source_root)
        if not (root / "gr00t" / "policy" / "gr00t_policy.py").exists():
            raise FileNotFoundError(
                f"Isaac-GR00T source not found at {root}; pass source_root=... "
                "or set GR00T_ROOT"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self._policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=self.checkpoint_path,
            device=self.device,
            strict=True,
        )
        expected_views = len(self._policy.modality_configs["video"].modality_keys)
        if int(num_views) != expected_views:
            raise ValueError(
                f"embodiment {self._policy.embodiment_tag.value!r} requires "
                f"{expected_views} views, got num_views={num_views}"
            )
        self.num_views = expected_views
        self.embodiment_tag = self._policy.embodiment_tag.value
        self._prompt: str | None = None
        self._state_dims = _state_dimensions(
            Path(self.checkpoint_path), self.embodiment_tag,
            self._policy.modality_configs["state"].modality_keys,
        )
        native_horizon = len(self._policy.modality_configs["action"].delta_indices)
        horizon = native_horizon if action_horizon is None else int(action_horizon)
        if not 1 <= horizon <= native_horizon:
            raise ValueError(
                f"action_horizon must be in [1, {native_horizon}], got {horizon}"
            )
        # This trims the returned chunk only. It does not claim to reduce N1.7's
        # fixed 40-token DiT compute.
        self.action_horizon = horizon
        self._last_action_dict: dict[str, np.ndarray] | None = None
        self._graph_driver = None
        if use_cuda_graph:
            try:
                from groot_n17_iwm.static_capture import install_static_capture
            except ImportError as error:
                raise RuntimeError(
                    "GR00T N1.7 CUDA Graph needs the companion adapter package. "
                    "Install `examples/groot_n17` into this environment, or pass "
                    "use_cuda_graph=False for the upstream eager baseline."
                ) from error
            self._graph_driver = install_static_capture(self._policy.model)

    def set_prompt(self, prompt: str, embodiment_tag: str | None = None) -> None:
        if embodiment_tag is not None:
            requested = str(embodiment_tag).lower()
            active = {
                str(self.embodiment_tag).lower(),
                str(self._policy.embodiment_tag.name).lower(),
            }
            if requested not in active:
                raise ValueError(
                    "changing a GR00T N1.7 embodiment requires reloading the model; "
                    f"active={self.embodiment_tag!r}, requested={embodiment_tag!r}"
                )
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("GR00T N1.7 requires a non-empty prompt")
        self._prompt = prompt

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        nested = normalise_observation(
            self._policy,
            observation,
            prompt=self._prompt,
            state_dims=self._state_dims,
        )
        action_dict, info = self._policy.get_action(nested)
        actions, unbatched = format_actions(
            self._policy, action_dict, horizon=self.action_horizon
        )
        self._last_action_dict = unbatched
        return {"actions": actions, "action_dict": unbatched, "info": info}

    predict = infer

    @property
    def backend_stats(self) -> dict[str, Any]:
        driver = self._graph_driver
        return {
            "backend": "upstream_bf16_cuda_graph" if driver else "upstream_bf16_eager",
            "captured": bool(driver and driver.captured),
            "cuda_graph": bool(driver),
            "graph_captures": int(driver.captures if driver else 0),
            "graph_replays": int(driver.replays if driver else 0),
            "precision": "bfloat16",
            "action_horizon": self.action_horizon,
            "embodiment_tag": self.embodiment_tag,
        }

    @property
    def last_action_dict(self) -> dict[str, np.ndarray] | None:
        return self._last_action_dict

    def close(self) -> None:
        if self._graph_driver is not None:
            self._graph_driver.close()
        self._graph_driver = None
        self._last_action_dict = None
        self._policy = None


def normalise_observation(
    policy,
    observation: dict[str, Any],
    *,
    prompt: str | None,
    state_dims: dict[str, int],
) -> dict[str, Any]:
    """Accept upstream nested observations or FlashRT's compact public form."""
    if not isinstance(observation, dict):
        raise TypeError("GR00T N1.7 observation must be a dict")
    cfg = policy.modality_configs
    video_keys = tuple(cfg["video"].modality_keys)
    state_keys = tuple(cfg["state"].modality_keys)
    video_horizon = len(cfg["video"].delta_indices)
    state_horizon = len(cfg["state"].delta_indices)

    nested_video = observation.get("video")
    video_values = dict(nested_video) if isinstance(nested_video, dict) else {}
    for key in video_keys:
        flat_key = f"video.{key}"
        if key not in video_values and flat_key in observation:
            video_values[key] = observation[flat_key]

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
        compact_images = list(compact_images)
        for index, key in enumerate(video_keys):
            if key not in video_values and index < len(compact_images):
                video_values[key] = compact_images[index]
    missing_video = [key for key in video_keys if key not in video_values]
    if missing_video:
        raise ValueError(
            f"GR00T N1.7 is missing video streams {missing_video}; expected {list(video_keys)}"
        )
    video = {
        key: _video_batch(video_values[key], video_horizon, key)
        for key in video_keys
    }

    nested_state = observation.get("state")
    state_values = dict(nested_state) if isinstance(nested_state, dict) else {}
    for key in state_keys:
        flat_key = f"state.{key}"
        if key not in state_values and flat_key in observation:
            state_values[key] = observation[flat_key]
    if not isinstance(nested_state, dict) and nested_state is not None:
        split = _split_state(nested_state, state_keys, state_dims)
        for key, value in split.items():
            state_values.setdefault(key, value)
    missing_state = [key for key in state_keys if key not in state_values]
    if missing_state:
        raise ValueError(
            f"GR00T N1.7 is missing state streams {missing_state}; expected {list(state_keys)}"
        )
    state = {
        key: _state_batch(state_values[key], state_horizon, state_dims[key], key)
        for key in state_keys
    }

    batch = next(iter(video.values())).shape[0]
    if any(value.shape[0] != batch for value in (*video.values(), *state.values())):
        raise ValueError("GR00T N1.7 video/state batch dimensions must match")

    language_keys = tuple(cfg["language"].modality_keys)
    nested_language = observation.get("language")
    language = dict(nested_language) if isinstance(nested_language, dict) else {}
    selected_prompt = observation.get("prompt") or observation.get("task") or prompt
    for key in language_keys:
        if key not in language:
            if not selected_prompt:
                raise ValueError(
                    "GR00T N1.7 requires language in the observation or a prompt via set_prompt()"
                )
            language[key] = [[str(selected_prompt)] for _ in range(batch)]
    return {"video": video, "state": state, "language": language}


def format_actions(policy, actions: dict[str, Any], *, horizon: int) -> tuple[np.ndarray, dict]:
    keys = tuple(policy.modality_configs["action"].modality_keys)
    arrays = {key: np.asarray(actions[key], dtype=np.float32)[:, :horizon] for key in keys}
    joined = np.concatenate([arrays[key] for key in keys], axis=-1)
    if joined.shape[0] == 1:
        return joined[0], {key: value[0] for key, value in arrays.items()}
    return joined, arrays


def _video_batch(value: Any, horizon: int, key: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"video {key!r} must be uint8 or floating point, got {array.dtype}")
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
        raise ValueError(f"video {key!r} must have 3 RGB channels, got {array.shape}")
    return np.ascontiguousarray(array)


def _state_batch(value: Any, horizon: int, width: int, key: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, None]
    elif array.ndim == 2:
        array = array[None]
    elif array.ndim != 3:
        raise ValueError(f"state {key!r} must be D, TD, or BTD, got {array.shape}")
    if array.shape[1] != horizon or array.shape[-1] != width:
        raise ValueError(
            f"state {key!r} requires shape (*, {horizon}, {width}), got {array.shape}"
        )
    return np.ascontiguousarray(array)


def _split_state(
    value: Any, keys: tuple[str, ...], dimensions: dict[str, int]
) -> dict[str, np.ndarray]:
    array = np.asarray(value, dtype=np.float32)
    expected = sum(dimensions[key] for key in keys)
    if array.ndim not in (1, 2, 3) or array.shape[-1] != expected:
        raise ValueError(
            f"flat GR00T N1.7 state must end in {expected} values, got {array.shape}"
        )
    stops = np.cumsum([dimensions[key] for key in keys])[:-1]
    chunks = np.split(array, stops, axis=-1)
    return dict(zip(keys, chunks, strict=True))


def _state_dimensions(checkpoint: Path, embodiment: str, keys) -> dict[str, int]:
    path = checkpoint / "statistics.json"
    if not path.exists():
        raise FileNotFoundError(f"GR00T N1.7 statistics not found: {path}")
    statistics = json.loads(path.read_text())
    try:
        state = statistics[embodiment]["state"]
        return {key: len(state[key]["mean"]) for key in keys}
    except KeyError as error:
        raise ValueError(
            f"statistics.json has no state dimensions for embodiment {embodiment!r}"
        ) from error


def _source_root(source_root: str | None = None) -> Path:
    explicit = source_root or os.environ.get("GR00T_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    for candidate in SOURCE_ROOT_CANDIDATES:
        if (candidate / "gr00t" / "policy" / "gr00t_policy.py").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Isaac-GR00T source not found; pass source_root= or set GR00T_ROOT "
        f"(searched: {[str(c) for c in SOURCE_ROOT_CANDIDATES]})")


def _require_checkpoint(path: Path) -> None:
    has_weights = (path / "model.safetensors.index.json").exists() or any(
        path.glob("*.safetensors")
    )
    if not has_weights or not (path / "processor_config.json").exists():
        raise FileNotFoundError(
            f"{path} is not a complete GR00T N1.7 checkpoint "
            "(weights and processor_config.json are required)"
        )

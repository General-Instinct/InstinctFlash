"""Cross-dataset protocols for the RECAP / ACP training pipeline.

The RECAP driver (:func:`training.rl.train_recap.train_recap_policy`)
is dataset-agnostic by design: any dataset that exposes the small
contract below can plug in. LIBERO ships as the first concrete
implementation (``training.rl.lerobot_libero.LeRobotLiberoDataset``),
but nothing in the driver hardcodes LIBERO-specific assumptions.

Two protocols here, separated by who consumes them:

* :class:`RecapPolicyDataset` — what the policy training driver
  needs (chunk-start indexing, per-frame fetch, optional ACP
  indicator column).

* :class:`RecapMetadataDataset` — what the value-function /
  RECAP-iteration pipeline (``training.rl.recap_iter``) needs on
  top of the policy contract: per-episode metadata + flat
  per-frame indices used for N-step advantage and per-task
  thresholds.

A dataset wanting to participate in BOTH pipelines satisfies both
protocols. ``LeRobotLiberoDataset`` does.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RecapFrame(Protocol):
    """One sampled frame, deferred image-bytes form.

    Mirrors the fields of
    :class:`training.rl.lerobot_libero.LeRobotFrame`. Anything else
    on a concrete frame class is permitted but unused by the driver.
    """

    state: np.ndarray
    action: np.ndarray
    image_bytes: dict[str, bytes]
    episode_index: int
    frame_index: int
    task_index: int
    task_name: str
    success: bool


@runtime_checkable
class RecapPolicyDataset(Protocol):
    """Contract :func:`train_recap_policy` expects from a dataset.

    Any dataset that supplies these five entry points can drive the
    generic policy training loop. The async data loader
    (``training.rl.async_loader.make_step_dataloader``) consumes
    ``get_frame`` and ``get_action_chunk`` from worker processes,
    so both must be process-safe (no shared CUDA state, no torch
    tensors).
    """

    def build_chunk_starts(self, action_horizon: int) -> np.ndarray:
        """Return the ``(N,)`` int64 array of valid chunk-start indices.

        A chunk-start index ``s`` is valid iff
        ``[s, s+action_horizon)`` is contained inside one episode of
        the dataset. The driver samples uniformly from this array
        each step.
        """
        ...

    def has_acp_column(self) -> bool:
        """True iff the dataset already ships per-frame
        ``acp_indicator`` values (paper §V-B). When False the driver
        derives them via one RECAP iteration before training, if
        ``derive_acp_if_missing`` is set."""
        ...

    def ensure_acp_indicators(self) -> np.ndarray:
        """Return the ``(num_frames,)`` int64 ``acp_indicator`` array."""
        ...

    def get_frame(self, global_frame_idx: int) -> RecapFrame:
        """Fetch one frame by its global flat index. Used by the
        async loader's worker processes — must be picklable / safe
        to call from a forked subprocess."""
        ...

    def get_action_chunk(
        self, global_frame_idx: int, action_horizon: int,
    ) -> np.ndarray:
        """Return ``(action_horizon, action_dim)`` action target."""
        ...


@runtime_checkable
class RecapMetadataDataset(RecapPolicyDataset, Protocol):
    """Extends :class:`RecapPolicyDataset` with per-episode metadata.

    Required by the RECAP-iteration pipeline
    (:mod:`training.rl.recap_iter`) which builds per-frame value
    targets and per-task thresholds.
    """

    @property
    def num_frames(self) -> int: ...

    @property
    def state_dim(self) -> int: ...

    @property
    def episodes(self) -> list: ...
    """List of objects with ``length: int``, ``success: bool``,
    ``task_index: int`` attributes (one per episode)."""

    @property
    def episode_indices(self) -> np.ndarray: ...
    """``(num_frames,)`` int64 — per-frame episode index."""

    @property
    def frame_indices(self) -> np.ndarray: ...
    """``(num_frames,)`` int64 — per-frame index within its episode."""

    @property
    def task_indices(self) -> np.ndarray: ...
    """``(num_frames,)`` int64 — per-frame task index."""

    @property
    def task_max_lengths(self) -> dict[int, int]: ...
    """``{task_index: max_episode_length}`` for value-target
    normalisation."""

    def ensure_state_action(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(states[N, state_dim], actions[N, action_dim])``."""
        ...

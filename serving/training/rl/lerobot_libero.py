"""LeRobot v3 dataset loader for a local LIBERO parquet snapshot.

Reads the dataset in-place — no downloads, no copying. The dataset
is expected to already exist on the user's filesystem; this module
is a thin random-access view over the LeRobot v3 layout:

::

    <root>/
      meta/info.json
      meta/tasks.parquet
      meta/episodes/chunk-{c:03d}/file-{f:03d}.parquet
      data/chunk-{c:03d}/file-{f:03d}.parquet  ← per-frame data,
                                                  including image bytes

Image columns (``observation.image``, ``observation.wrist_image``)
carry per-row ``{"bytes": <PNG/JPEG>, "path": ...}`` dicts; the
caller is expected to decode them with PIL or cv2 when needed
(decoding is deferred so the loader stays cheap).

For RECAP we also need a per-episode ``success`` flag. The local
dataset is curated rollouts (200 episodes, 76 k frames, 2 tasks)
without an explicit success column. The current default
(``assume_all_success=True``) treats every episode as a success —
this matches the way iter2 RECAP rollouts are typically filtered
upstream of the dataset. Override via ``success_predicate`` for
mixed-success datasets.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def _resolve_libero_root(root: str | Path | None) -> Path:
    """Resolve the LIBERO dataset root from arg or ``FLASHVLA_LIBERO_ROOT``."""
    if root is not None:
        return Path(root)
    env = os.environ.get("FLASHVLA_LIBERO_ROOT")
    if env:
        return Path(env)
    raise FileNotFoundError(
        "Pass an explicit `root` or set FLASHVLA_LIBERO_ROOT to a "
        "LeRobot v3 LIBERO dataset directory."
    )


@dataclass(frozen=True)
class LeRobotEpisodeInfo:
    """Per-episode metadata extracted from ``meta/episodes/...``."""

    episode_index: int
    task_index: int
    task_name: str
    length: int
    success: bool
    data_chunk_index: int
    data_file_index: int
    dataset_from_index: int
    dataset_to_index: int


@dataclass
class LeRobotFrame:
    """One sampled frame — kept lean (image bytes are deferred)."""

    state: np.ndarray            # (state_dim,) float32
    action: np.ndarray           # (action_dim,) float32
    image_bytes: dict[str, bytes]  # camera_name → encoded image bytes
    episode_index: int
    frame_index: int
    task_index: int
    task_name: str
    success: bool


class LeRobotLiberoDataset:
    """Random-access wrapper over the local LIBERO parquet snapshot.

    Attributes:
        root: Dataset root directory.
        info: Parsed ``meta/info.json``.
        tasks: ``{task_index: task_name}``.
        episodes: ``list[LeRobotEpisodeInfo]`` sorted by episode_index.
        num_frames / num_episodes / num_tasks: counts.
        state_dim / action_dim: from info.json features.
        image_keys: image columns present in the parquet
            (``["observation.image", "observation.wrist_image"]`` for LIBERO).
        episode_indices / frame_indices / task_indices / success:
            per-frame ``np.ndarray`` arrays — built lazily from
            episode metadata, no parquet read required for these.
        states / actions: per-frame state/action arrays — read
            once on first access (lazy via ``ensure_state_action``).
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        success_predicate: Callable[[LeRobotEpisodeInfo], bool] | None = None,
        assume_all_success: bool = True,
    ):
        self.root = _resolve_libero_root(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"LeRobot LIBERO dataset not found at {self.root}; "
                "provide an explicit `root` or set FLASHVLA_LIBERO_ROOT. "
                "No download fallback."
            )
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.data_path_tmpl = self.info["data_path"]

        self.tasks = self._load_tasks()
        self.episodes = self._load_episodes(
            success_predicate=success_predicate,
            assume_all_success=assume_all_success,
        )

        self.num_frames = int(self.info["total_frames"])
        self.num_episodes = int(self.info["total_episodes"])
        self.num_tasks = int(self.info["total_tasks"])

        feats = self.info["features"]
        self.state_dim = int(feats["observation.state"]["shape"][0])
        self.action_dim = int(feats["action"]["shape"][0])
        self.image_keys = [k for k in feats if feats[k].get("dtype") == "image"]

        # Per-frame metadata arrays from episode info (no parquet read).
        self._build_frame_metadata()

        # Optional per-frame state/action caches (lazy).
        self._states: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        # Optional pre-computed acp_indicator column from parquet
        # (only present in the libero10_recap_lerobot snapshot).
        self._acp_indicators: np.ndarray | None = None
        self._has_acp_column: bool | None = None
        # Action-chunking starts (set by ``build_chunk_starts``).
        self._chunk_starts: np.ndarray | None = None
        self._chunk_horizon: int | None = None

    # ── Construction helpers ────────────────────────────────────────

    def _load_tasks(self) -> dict[int, str]:
        path = self.root / "meta" / "tasks.parquet"
        df = pq.read_table(path).to_pandas().reset_index()
        # Two upstream conventions: a column literally named "task"
        # (legacy) or the task string in the index.
        if "task" in df.columns:
            tasks = dict(zip(df["task_index"], df["task"], strict=True))
        else:
            tasks = dict(zip(df["task_index"], df["index"], strict=True))
        return {int(k): str(v) for k, v in tasks.items()}

    def _load_episodes(
        self,
        *,
        success_predicate: Callable[[LeRobotEpisodeInfo], bool] | None,
        assume_all_success: bool,
    ) -> list[LeRobotEpisodeInfo]:
        ep_dir = self.root / "meta" / "episodes"
        episodes: list[LeRobotEpisodeInfo] = []
        for parquet in sorted(ep_dir.rglob("*.parquet")):
            cols = [
                "episode_index", "tasks", "length",
                "data/chunk_index", "data/file_index",
                "dataset_from_index", "dataset_to_index",
            ]
            df = pq.read_table(parquet, columns=cols).to_pandas()
            for _, row in df.iterrows():
                task_name = (
                    str(row["tasks"][0])
                    if hasattr(row["tasks"], "__len__") and len(row["tasks"]) > 0
                    else ""
                )
                # Map task_name → task_index via the tasks dict.
                task_index = self._task_name_to_index(task_name)
                ep = LeRobotEpisodeInfo(
                    episode_index=int(row["episode_index"]),
                    task_index=task_index,
                    task_name=task_name,
                    length=int(row["length"]),
                    success=True,  # placeholder — overridden below
                    data_chunk_index=int(row["data/chunk_index"]),
                    data_file_index=int(row["data/file_index"]),
                    dataset_from_index=int(row["dataset_from_index"]),
                    dataset_to_index=int(row["dataset_to_index"]),
                )
                episodes.append(ep)

        episodes.sort(key=lambda e: e.episode_index)

        # Apply success policy.
        if success_predicate is not None:
            episodes = [
                LeRobotEpisodeInfo(
                    **{**ep.__dict__, "success": bool(success_predicate(ep))}
                )
                for ep in episodes
            ]
        elif assume_all_success:
            pass  # already True
        else:
            episodes = [
                LeRobotEpisodeInfo(**{**ep.__dict__, "success": False})
                for ep in episodes
            ]
        return episodes

    def _task_name_to_index(self, task_name: str) -> int:
        for idx, name in self.tasks.items():
            if name == task_name:
                return idx
        # Unknown task — register a fresh index.
        new_idx = max(self.tasks) + 1 if self.tasks else 0
        self.tasks[new_idx] = task_name
        return new_idx

    def _build_frame_metadata(self) -> None:
        n = self.num_frames
        self.episode_indices = np.empty(n, dtype=np.int64)
        self.frame_indices = np.empty(n, dtype=np.int64)
        self.task_indices = np.empty(n, dtype=np.int64)
        self.success = np.empty(n, dtype=bool)
        # Per-frame parquet locator: which (chunk, file) holds the frame.
        self._frame_chunk = np.empty(n, dtype=np.int32)
        self._frame_file = np.empty(n, dtype=np.int32)
        self._frame_local_row = np.empty(n, dtype=np.int64)
        self._task_max_lengths = self._compute_task_max_lengths()

        cursor = 0
        # Group episodes by their data file so we can compute the local
        # row offset for each frame inside that file.
        from collections import defaultdict
        file_to_eps: dict[tuple[int, int], list[LeRobotEpisodeInfo]] = defaultdict(list)
        for ep in self.episodes:
            file_to_eps[(ep.data_chunk_index, ep.data_file_index)].append(ep)

        for (chunk, file_idx), eps in file_to_eps.items():
            eps.sort(key=lambda e: e.dataset_from_index)
            local_row_cursor = 0
            for ep in eps:
                length = ep.length
                start = cursor
                end = cursor + length
                self.episode_indices[start:end] = ep.episode_index
                self.frame_indices[start:end] = np.arange(length, dtype=np.int64)
                self.task_indices[start:end] = ep.task_index
                self.success[start:end] = ep.success
                self._frame_chunk[start:end] = chunk
                self._frame_file[start:end] = file_idx
                self._frame_local_row[start:end] = (
                    local_row_cursor + np.arange(length, dtype=np.int64)
                )
                local_row_cursor += length
                cursor = end

        if cursor != n:
            raise RuntimeError(
                f"Frame count mismatch: cumulated {cursor} but info reports {n}"
            )

    def _compute_task_max_lengths(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for ep in self.episodes:
            cur = out.get(ep.task_index, 0)
            if ep.length > cur:
                out[ep.task_index] = ep.length
        return out

    @property
    def task_max_lengths(self) -> dict[int, int]:
        return dict(self._task_max_lengths)

    # ── Lazy parquet readers ────────────────────────────────────────

    @functools.lru_cache(maxsize=4)
    def _read_data_file(self, chunk: int, file: int) -> pd.DataFrame:
        rel = self.data_path_tmpl.format(chunk_index=chunk, file_index=file)
        path = self.root / rel
        return pq.read_table(path).to_pandas()

    def get_frame(self, global_frame_idx: int) -> LeRobotFrame:
        """Random access — reads the frame's parquet (cached LRU)."""
        if not 0 <= global_frame_idx < self.num_frames:
            raise IndexError(global_frame_idx)
        chunk = int(self._frame_chunk[global_frame_idx])
        file_idx = int(self._frame_file[global_frame_idx])
        local_row = int(self._frame_local_row[global_frame_idx])
        df = self._read_data_file(chunk, file_idx)
        row = df.iloc[local_row]

        ep_idx = int(self.episode_indices[global_frame_idx])
        ep = self.episodes[ep_idx]
        return LeRobotFrame(
            state=np.asarray(row["observation.state"], dtype=np.float32),
            action=np.asarray(row["action"], dtype=np.float32),
            image_bytes={
                key: row[key]["bytes"] if isinstance(row[key], dict) else row[key]
                for key in self.image_keys
            },
            episode_index=ep_idx,
            frame_index=int(self.frame_indices[global_frame_idx]),
            task_index=ep.task_index,
            task_name=ep.task_name,
            success=ep.success,
        )

    # ── Action chunking + ACP annotation ────────────────────────────

    def build_chunk_starts(self, action_horizon: int) -> np.ndarray:
        """Precompute valid global frame indices that have ``action_horizon``
        consecutive frames remaining inside the same episode.

        Mirrors openpi's ``train_jax_lora_recap.ACPDataset._valid_indices``:
        the policy training loop samples a chunk-start ``i`` from the
        returned array, then reads ``actions[i : i+action_horizon]``
        as one ground-truth chunk for the flow-matching loss. Episodes
        shorter than ``action_horizon`` contribute zero starts.

        Cached per ``action_horizon`` value; calling with a different
        horizon recomputes.
        """
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be > 0, got {action_horizon}")
        if self._chunk_starts is not None and self._chunk_horizon == action_horizon:
            return self._chunk_starts

        starts: list[int] = []
        cursor = 0
        for ep in self.episodes:
            if ep.length >= action_horizon:
                starts.extend(range(cursor, cursor + ep.length - action_horizon + 1))
            cursor += ep.length
        if cursor != self.num_frames:
            raise RuntimeError(
                f"chunk-start cursor mismatch: {cursor} vs {self.num_frames}"
            )
        self._chunk_starts = np.asarray(starts, dtype=np.int64)
        self._chunk_horizon = action_horizon
        return self._chunk_starts

    def get_action_chunk(self, global_frame_idx: int, action_horizon: int) -> np.ndarray:
        """Read ``actions[i : i+action_horizon]`` as ``(action_horizon, action_dim)``.

        Caller must have validated ``global_frame_idx`` via
        :meth:`build_chunk_starts` first.
        """
        _, actions = self.ensure_state_action()
        end = global_frame_idx + action_horizon
        if end > self.num_frames:
            raise IndexError(
                f"chunk [{global_frame_idx}:{end}] exceeds dataset "
                f"length {self.num_frames}"
            )
        ep_at_start = self.episode_indices[global_frame_idx]
        ep_at_end = self.episode_indices[end - 1]
        if ep_at_start != ep_at_end:
            raise ValueError(
                f"chunk crosses episode boundary "
                f"({ep_at_start} → {ep_at_end}); "
                "use build_chunk_starts() to filter valid starts."
            )
        return actions[global_frame_idx:end].copy()

    def ensure_acp_indicators(self) -> np.ndarray:
        """Load the ``complementary_info.acp_indicator`` column if present.

        The ``libero10_recap_lerobot`` snapshot ships pre-computed
        indicators (compatible with the JAX baseline); the
        ``libero10_iter2_lerobot`` snapshot does not. Returns
        ``None``-equivalent missing-flag handling: raises
        :class:`ValueError` if the column is absent — caller is
        expected to compute indicators via :func:`value_infer` instead.
        """
        if self._acp_indicators is not None:
            return self._acp_indicators

        if self._has_acp_column is False:
            raise ValueError(
                "Dataset has no complementary_info.acp_indicator column; "
                "run value_infer to derive indicators on the fly."
            )

        # Probe one parquet to decide if the column is present.
        first_chunk = int(self._frame_chunk[0])
        first_file = int(self._frame_file[0])
        rel = self.data_path_tmpl.format(chunk_index=first_chunk, file_index=first_file)
        sample = pq.read_table(self.root / rel).schema
        if "complementary_info.acp_indicator" not in sample.names:
            self._has_acp_column = False
            raise ValueError(
                "Dataset has no complementary_info.acp_indicator column; "
                "run value_infer to derive indicators on the fly."
            )
        self._has_acp_column = True

        out = np.empty(self.num_frames, dtype=np.int64)
        from collections import defaultdict
        by_file: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for global_idx in range(self.num_frames):
            by_file[
                (int(self._frame_chunk[global_idx]),
                 int(self._frame_file[global_idx]))
            ].append((global_idx, int(self._frame_local_row[global_idx])))

        for (chunk, file_idx), rows in by_file.items():
            rel = self.data_path_tmpl.format(chunk_index=chunk, file_index=file_idx)
            df = pq.read_table(
                self.root / rel,
                columns=["complementary_info.acp_indicator"],
            ).to_pandas()
            for global_idx, local_row in rows:
                out[global_idx] = int(
                    df["complementary_info.acp_indicator"].iloc[local_row]
                )

        if not set(np.unique(out).tolist()).issubset({0, 1}):
            raise ValueError(
                f"acp_indicator must be 0/1, got {set(np.unique(out).tolist())}"
            )
        self._acp_indicators = out
        return out

    def has_acp_column(self) -> bool:
        """``True`` if the parquet ships a ``complementary_info.acp_indicator``."""
        if self._has_acp_column is None:
            try:
                self.ensure_acp_indicators()
            except ValueError:
                pass
        return bool(self._has_acp_column)

    def ensure_state_action(self) -> tuple[np.ndarray, np.ndarray]:
        """Load all states + actions into contiguous arrays.

        Skips image columns (cheap). Useful when training a VF that
        uses state only — avoids decoding 76 k images for nothing.
        """
        if self._states is not None and self._actions is not None:
            return self._states, self._actions

        states = np.empty((self.num_frames, self.state_dim), dtype=np.float32)
        actions = np.empty((self.num_frames, self.action_dim), dtype=np.float32)
        from collections import defaultdict
        by_file: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for global_idx in range(self.num_frames):
            by_file[
                (int(self._frame_chunk[global_idx]),
                 int(self._frame_file[global_idx]))
            ].append((global_idx, int(self._frame_local_row[global_idx])))

        for (chunk, file_idx), rows in by_file.items():
            rel = self.data_path_tmpl.format(chunk_index=chunk, file_index=file_idx)
            df = pq.read_table(
                self.root / rel,
                columns=["observation.state", "action"],
            ).to_pandas()
            for global_idx, local_row in rows:
                states[global_idx] = np.asarray(
                    df["observation.state"].iloc[local_row], dtype=np.float32
                )
                actions[global_idx] = np.asarray(
                    df["action"].iloc[local_row], dtype=np.float32
                )

        self._states = states
        self._actions = actions
        return states, actions

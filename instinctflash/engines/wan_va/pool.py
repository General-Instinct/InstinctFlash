"""Host-side bookkeeping for the Wan self-attention KV pool.

Mirrors upstream ``WanAttention``'s per-layer cache exactly (every layer performs the
same operations, so one bookkeeper serves all layers and both CFG branches):

* append: take the lowest-index free slots; if too few are free, evict the oldest live
  slots by id (stable order), then take the lowest free ones. All tokens of one forward
  share a new id ``max(live ids) + 1``.
* a transient forward (``update_cache == 0``) frees its slots after attention but keeps
  their stale ids; ``update_cache == 1`` marks slots as predictions.
* ``clear_pred`` frees every prediction slot.

Attention keys are the live slots in ascending slot order.
"""
from __future__ import annotations

import numpy as np


class SlotPool:
    def __init__(self, size: int):
        self.size = int(size)
        self.reset()

    def reset(self):
        self.mask = np.zeros(self.size, dtype=bool)
        self.ids = np.full(self.size, -1, dtype=np.int64)
        self.is_pred = np.zeros(self.size, dtype=bool)

    def append(self, count: int, *, is_pred: bool) -> np.ndarray:
        free = np.flatnonzero(~self.mask)
        if free.size < count:
            used = np.flatnonzero(self.mask)
            order = np.argsort(self.ids[used], kind="stable")
            evict = used[order[: count - free.size]]
            self.mask[evict] = False
            self.ids[evict] = -1
            free = np.flatnonzero(~self.mask)
        if free.size < count:
            raise RuntimeError(f"KV pool of {self.size} slots cannot hold {count} tokens")
        slots = free[:count]
        new_id = self.ids[self.mask].max() + 1 if self.mask.any() else 0
        self.mask[slots] = True
        self.ids[slots] = new_id
        self.is_pred[slots] = is_pred
        return slots

    def restore(self, slots: np.ndarray):
        self.mask[slots] = False

    def clear_pred(self):
        self.mask[self.is_pred] = False

    def live(self) -> np.ndarray:
        return np.flatnonzero(self.mask)

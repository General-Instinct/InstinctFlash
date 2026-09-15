"""LingBot-VA KV allocator references — stock ring vs engine linear slab.

Stock ground truth (`wan_va/modules/model.py`):
  * pool per layer: [B=2, 9792, 24, 128], 9792 = 36 * (240 + 32)
    (server:398-411; ``attn_window`` 72 halves for video/action chunks);
  * ``allocate_slots`` (:366-385): lowest free slots; when short, frees the
    OLDEST ids first (argsort over ids — assumed stable within a group,
    the assumption the shipped ring_kv pass validated 800/800 against the
    real allocator);
  * ``update_cache`` (:396-409): one new id per commit; ``is_pred`` set iff
    update_cache == 1;
  * transient appends (update_cache == 0) run attention with the appended
    rows then ``restore_cache`` frees them (:457-459) — but any eviction the
    append CAUSED is permanent: at saturation, transient/pred appends
    destroy the oldest history. Certified stock behavior; must reproduce.
  * ``clear_pred_cache`` (:333-338) drops every is_pred slot — fired once
    per kv message (server:574) covering BOTH pred commits of the previous
    cycle (video last step + action last step);
  * presentation order: ascending slot index (:451-453) — at most two
    ascending segments (slice-or-cat; instinctwm/passes/lingbot/ring_kv.py).

Engine model — the LINEAR EPISODE SLAB (mapping_memo §A.3): pred and
transient rows are always the chronologically newest, so every stock
mutation is head/tail motion on a chronological log:
    append k   : evict head += max(0, live + k - CAP) first; rows written
                 at [tail, tail+k) (restored transient rows are physically
                 overwritten by later appends); tail += k
    restore k  : tail -= k
    clear_pred : tail -= (pred suffix length)  [assert pred IS the suffix]
A slab of SLAB_ROWS = 15360 rows (> worst RoboTwin episode: 1700 steps ~=
53.1 cycles ~= 392 + 53*272 = 14,808 committed rows + transient headroom)
keeps the live window [head, tail) contiguous in memory — zero gathers, FA2
gets base pointer + length. Presentation is chronological: identical SET to
stock always (proven here), identical ORDER only pre-wrap (post-wrap stock's
ascending-slot order interleaves — precision-tier difference, gate ladder
M1b/M2 rung in repack_calib_plan.md).

Fail-loud rules implemented here and mirrored by the frontend:
  * committed rows exceeding the slab -> RuntimeError (episode too long);
  * clear_pred when the pred rows are not the newest suffix -> RuntimeError;
  * restore is only ever of the newest rows (asserted).

``self_check()`` replays the deployed 2V/4A message pattern 53 cycles (wrap
at ~36; the worst real episode length) with SHARED token ids and asserts
live-token-SET equality between the two models at every event, plus the
<=2-ascending-segments stock invariant, plus the overflow fail-loud.
"""
from __future__ import annotations

import torch

POOL_SLOTS = 9792
SLAB_ROWS = 15360
VIDEO_TOKENS = 240
ACTION_TOKENS = 32
CYCLE0_VIDEO_COMMIT = 360        # init keyframe latent concat: 3 frames


class StockAllocatorRef:
    """Mask/id/is_pred bookkeeping, mirroring WanAttention (CPU, no weights).

    ``update`` takes explicit per-token identity labels so live SETS can be
    compared against a differently-addressed model.
    """

    def __init__(self, capacity: int = POOL_SLOTS):
        self.cap = capacity
        self.mask = torch.zeros(capacity, dtype=torch.bool)
        self.ids = torch.full((capacity,), -1, dtype=torch.long)
        self.is_pred = torch.zeros(capacity, dtype=torch.bool)
        self.tok = torch.full((capacity,), -1, dtype=torch.long)

    # model.py:366-385 — stable argsort documented (ring_kv-validated)
    def _allocate(self, k: int) -> torch.Tensor:
        free = (~self.mask).nonzero(as_tuple=False).squeeze(-1)
        if free.numel() < k:
            used = self.mask.nonzero(as_tuple=False).squeeze(-1)
            order = torch.argsort(self.ids[used], stable=True)
            need = k - free.numel()
            to_free = used[order[:need]]
            self.mask[to_free] = False
            self.ids[to_free] = -1
            free = (~self.mask).nonzero(as_tuple=False).squeeze(-1)
        assert free.numel() >= k
        return free[:k]

    def _next_id(self) -> int:
        return int(self.ids[self.mask].max()) + 1 if bool(self.mask.any()) else 0

    def update(self, token_ids: torch.Tensor, is_pred: bool) -> torch.Tensor:
        k = token_ids.numel()
        slots = self._allocate(k)
        new_id = self._next_id()
        self.mask[slots] = True
        self.ids[slots] = new_id
        self.is_pred[slots] = is_pred     # model:408 — always assigned
        self.tok[slots] = token_ids
        return slots

    def restore(self, slots: torch.Tensor):
        self.mask[slots] = False          # model:411-412 (mask only)

    def clear_pred(self):
        self.mask[self.is_pred] = False   # model:333-338 (stale flags stay)

    def live_tokens(self) -> set:
        return set(self.tok[self.mask].tolist())

    def num_segments(self) -> int:
        """Count of contiguous ascending-slot segments in the live set."""
        idx = self.mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return 0
        return int((idx[1:] - idx[:-1] > 1).sum()) + 1


class LinearSlabRef:
    """Engine linear-episode-slab bookkeeping (head/tail on a token log)."""

    def __init__(self, capacity: int = POOL_SLOTS, slab_rows: int = SLAB_ROWS):
        self.cap = capacity          # LOGICAL pool capacity (stock semantics)
        self.slab_rows = slab_rows   # physical rows available
        self.head = 0                # oldest live row
        self.tail = 0                # one past newest live row
        self.log: list = []          # row -> token id (rows are overwritten
        #                              on transient reuse, like the device)
        self.pred_marks: list = []   # [(start_row, count)] oldest-first

    @property
    def live(self) -> int:
        return self.tail - self.head

    def append(self, token_ids, kind: str) -> None:
        k = len(token_ids)
        evict = max(0, self.live + k - self.cap)
        self.head += evict
        self.pred_marks = [(s, c) for (s, c) in self.pred_marks
                           if s + c > self.head]
        if self.tail + k > self.slab_rows:
            raise RuntimeError(
                f"episode exceeds the linear slab ({self.tail + k} > "
                f"{self.slab_rows}) — fail-loud per mapping_memo §A.3")
        if len(self.log) < self.tail + k:
            self.log.extend([-1] * (self.tail + k - len(self.log)))
        self.log[self.tail:self.tail + k] = list(token_ids)
        if kind == "pred":
            self.pred_marks.append((self.tail, k))
        self.tail += k

    def restore(self, k: int) -> None:
        # transient rows are by construction the newest
        self.tail -= k
        assert self.tail >= self.head

    def clear_pred(self) -> None:
        if not self.pred_marks:
            return
        start = max(min(s for (s, c) in self.pred_marks), self.head)
        pred_live = sum(c - max(0, self.head - s)
                        for (s, c) in self.pred_marks)
        if self.tail - start != pred_live:
            raise RuntimeError(
                "clear_pred: pred rows are not the newest suffix — the wire "
                "protocol should make this impossible; refusing")
        self.tail = start
        self.pred_marks = []

    def live_tokens(self) -> set:
        return set(self.log[self.head:self.tail])


# ────────────────────────────────────────────────────────────────────
# Self-check: replay the deployed 2V/4A message pattern
# ────────────────────────────────────────────────────────────────────

def _cycle_events(cycle: int, video_steps: int = 2, action_steps: int = 4):
    """(kind, tokens) events of one cycle at V video / A action steps (server:489-604):
    infer: video V transient + 1 pred; action A transient + 1 pred;
    kv msg: clear_pred, commit video (360 at cycle 0 else 240), commit action.
    The certified point is (2, 4); the Stage-2 second point is (2, 2) — the eviction
    pattern is identical (only committed rows persist), which is why the slab bookkeeping
    is point-independent and only the transient count changes.
    """
    ev = []
    for _ in range(video_steps):
        ev.append(("transient", VIDEO_TOKENS))
    ev.append(("pred", VIDEO_TOKENS))
    for _ in range(action_steps):
        ev.append(("transient", ACTION_TOKENS))
    ev.append(("pred", ACTION_TOKENS))
    ev.append(("clear_pred", 0))
    ev.append(("commit", CYCLE0_VIDEO_COMMIT if cycle == 0 else VIDEO_TOKENS))
    ev.append(("commit", ACTION_TOKENS))
    return ev


def self_check(cycles: int = 53, video_steps: int = 2, action_steps: int = 4) -> dict:
    """53 cycles == the worst real RoboTwin episode (1700 steps); the pool
    wraps at ~36, so ~17 cycles run in the eviction regime. Run for every
    Stage-2 operating point: (2, 4) certified and (2, 2) guidance-off."""
    stock = StockAllocatorRef()
    slab = LinearSlabRef()
    next_tok = 0
    checks = 0
    max_segments = 0
    evicting_appends = 0

    def fresh(k):
        nonlocal next_tok
        t = torch.arange(next_tok, next_tok + k)
        next_tok += k
        return t

    for c in range(cycles):
        for kind, k in _cycle_events(c, video_steps, action_steps):
            if kind == "transient":
                toks = fresh(k)
                if int(stock.mask.sum()) + k > stock.cap:
                    evicting_appends += 1
                slots = stock.update(toks, is_pred=False)
                slab.append(toks.tolist(), "transient")
                assert stock.live_tokens() == slab.live_tokens(), (c, kind)
                stock.restore(slots)
                slab.restore(k)
            elif kind == "pred":
                toks = fresh(k)
                stock.update(toks, is_pred=True)
                slab.append(toks.tolist(), "pred")
            elif kind == "clear_pred":
                stock.clear_pred()
                slab.clear_pred()
            elif kind == "commit":
                toks = fresh(k)
                stock.update(toks, is_pred=False)
                slab.append(toks.tolist(), "commit")
            assert stock.live_tokens() == slab.live_tokens(), (c, kind)
            max_segments = max(max_segments, stock.num_segments())
            checks += 1

    assert evicting_appends > 0, \
        "self-check never exercised the wrap/eviction path"
    assert max_segments <= 2, (
        "stock live set exceeded 2 ascending segments — the slice-or-cat "
        f"invariant (ring_kv.py) is broken: {max_segments}")

    # fail-loud check: an over-long episode must raise
    over = LinearSlabRef(capacity=10**9, slab_rows=100)
    over.append(list(range(60)), "commit")
    try:
        over.append(list(range(60, 120)), "commit")
        raise AssertionError("slab overflow did not raise")
    except RuntimeError:
        pass

    return {"events_checked": checks, "cycles": cycles,
            "video_steps": video_steps, "action_steps": action_steps,
            "evicting_appends": evicting_appends,
            "max_stock_segments": max_segments,
            "final_live": slab.live, "final_head": slab.head,
            "final_tail": slab.tail,
            "slab_headroom_rows": SLAB_ROWS - slab.tail}


__all__ = ["StockAllocatorRef", "LinearSlabRef", "self_check",
           "POOL_SLOTS", "SLAB_ROWS"]

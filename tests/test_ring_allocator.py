#!/usr/bin/env python3
"""Unit-test the KV allocator across multiple full wraparounds of the ring.

Why this exists as a unit test rather than an end-to-end run: the pool holds 9792 slots and a
LingBot-VA control cycle commits 272 tokens, so the first wrap is at cycle ~36 and a second at
~72. Reaching that with the real 10 GB model costs minutes per arm and (as we found) OOMs when
`empty_cache` is disabled. The allocator is pure bookkeeping over small tensors, so it can be
exercised directly against the REAL stock implementation, thousands of cycles deep, in seconds.

The reference is `WanAttention.allocate_slots` / `update_cache` / `clear_pred_cache` themselves,
constructed via `__new__` so no weights are loaded. Whatever those do IS the specification; the
ring model has to reproduce it exactly, including after eviction begins.

Two message-order facts the model has to carry, both added after they were found missing:

  * TRANSIENT forwards. Every denoise step but the last runs with update_cache=0 -- the write lands,
    attention reads the pool INCLUDING it, then `restore_cache` rolls the mask back (model.py:457-459).
    The ring model reads `count + key_size` for exactly this reason (ring_kv.py:219), and a model
    without the transient could not have caught a rollback bug.
  * The ELISION bookkeeping variant (`passes/lingbot/action_terminal_elision.py`): the action
    pred-commit forward is skipped and only its allocation is replayed. `test_elision_bookkeeping`
    runs that against the real stock allocator, ON arm vs ELIDED arm, payload-tracked, through more
    than two ring wraps -- a naive skip fails it at the first wrap (cycle 36), which is the
    2026-08-09 refutation reproduced at allocator level.

Run:  python tests/test_ring_allocator.py            (CYCLES=... ELISION_CYCLES=... NFE_VIDEO=... NFE_ACTION=...)
"""
from __future__ import annotations

import os
import sys

if __name__ != "__main__":
    import pytest
    pytest.importorskip("diffusers", reason="Native allocator oracle requires the LingBot vendor environment")
    pytest.importorskip("transformers", reason="Native allocator oracle requires the LingBot vendor environment")

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LINGBOT = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
sys.path.insert(0, os.path.join(LINGBOT, "wan_va"))

# The model module imports flash_attn at module scope even though the served path never calls it.
sys.path.insert(0, "/home/ubuntu/iwm_shims")

from modules.model import WanAttention  # noqa: E402

TOTAL = 9792          # (attn_window//2)*240 + (attn_window//2)*32
VIDEO_TOK = 240       # frame_chunk_size(2) * latent_h(24) * latent_w(20) / (patch 2*2)
ACTION_TOK = 32       # frame_chunk_size(2) * action_per_frame(16)
HEADS, DIM = 2, 4     # tiny; the allocator does not care about the payload width
NFE_VIDEO = int(os.environ.get("NFE_VIDEO", "2"))    # forwards per video phase (shipped 2V/4A point)
NFE_ACTION = int(os.environ.get("NFE_ACTION", "4"))  # the last of each phase is the update_cache=1 write


def bare_attn(device="cpu"):
    a = WanAttention.__new__(WanAttention)
    a.attn_caches = {}
    a.init_kv_cache("pos", TOTAL, HEADS, DIM, device, torch.float32, 1)
    return a


def kv(n, device="cpu"):
    return (torch.zeros(1, n, HEADS, DIM, device=device),
            torch.zeros(1, n, HEADS, DIM, device=device))


def live_set(a):
    """The exact key ordering stock presents to attention: ascending slot index."""
    return a.attn_caches["pos"]["mask"].nonzero(as_tuple=False).squeeze(-1)


def transient(a, n_tok, record, tag):
    """An update_cache=0 forward: write, read the pool with the write in it, roll the mask back.

    model.py:444-459 -- `update_cache` then `restore_cache(slots)`. The `id` of the rolled-back slots
    is NOT reset by stock; only the mask is. Recorded before the rollback, because that is the state
    attention read.
    """
    k, v = kv(n_tok)
    slots = a.update_cache("pos", k, v, is_pred=False)
    record(a, tag, slots)
    a.restore_cache("pos", slots)


def cycle(a, record):
    """One control cycle, in the real message order.

    Per `wan_va_server.py`: each denoise loop runs NFE-1 transient forwards (update_cache=0) and then
    writes provisionally (update_cache=1) at its padded last step -- video (:504), action (:544);
    `_compute_kv_cache` then calls clear_pred_cache (:574) and commits the observed frames with
    update_cache=2 (:595, :600).
    """
    for _ in range(NFE_VIDEO - 1):
        transient(a, VIDEO_TOK, record, "video_transient")
    k, v = kv(VIDEO_TOK)
    record(a, "video_prov", a.update_cache("pos", k, v, is_pred=True))
    for _ in range(NFE_ACTION - 1):
        transient(a, ACTION_TOK, record, "action_transient")
    k, v = kv(ACTION_TOK)
    record(a, "action_prov", a.update_cache("pos", k, v, is_pred=True))

    a.clear_pred_cache("pos")
    record(a, "after_clear_pred", None)

    k, v = kv(VIDEO_TOK)
    record(a, "video_commit", a.update_cache("pos", k, v, is_pred=False))
    k, v = kv(ACTION_TOK)
    record(a, "action_commit", a.update_cache("pos", k, v, is_pred=False))


def main() -> int:
    n_cycles = int(os.environ.get("CYCLES", "120"))   # ~3.3 full wraps
    a = bare_attn()

    obs = []

    def record(att, tag, slots):
        lv = live_set(att)
        contig = lv.numel() <= 1 or bool(((lv[1:] - lv[:-1]) == 1).all())
        s_contig = slots is None or slots.numel() <= 1 or bool(((slots[1:] - slots[:-1]) == 1).all())
        obs.append({
            "tag": tag, "live": int(lv.numel()),
            "lo": int(lv[0]) if lv.numel() else -1,
            "hi": int(lv[-1]) if lv.numel() else -1,
            "live_contig": contig,
            "slot0": int(slots[0]) if slots is not None and slots.numel() else -1,
            "slots_contig": s_contig,
        })

    for _ in range(n_cycles):
        cycle(a, record)

    committed = VIDEO_TOK + ACTION_TOK
    print(f"pool={TOTAL}  committed/cycle={committed}  first wrap ~cycle {TOTAL // committed}  "
          f"forwards/cycle={NFE_VIDEO + NFE_ACTION + 2} ({NFE_VIDEO - 1}+{NFE_ACTION - 1} transient)")
    print(f"cycles simulated={n_cycles}  observations={len(obs)}")

    nc_live = [o for o in obs if not o["live_contig"]]
    nc_slots = [o for o in obs if not o["slots_contig"]]
    print(f"\nlive set NON-contiguous in ascending index order : {len(nc_live)}")
    print(f"returned slots NON-contiguous                    : {len(nc_slots)}")
    print(f"max live                                         : {max(o['live'] for o in obs)}")

    if nc_live:
        print("\nfirst 5 non-contiguous live sets (this is what breaks a naive ring):")
        for o in nc_live[:5]:
            print(f"   {o['tag']:18s} live={o['live']:5d} lo={o['lo']:5d} hi={o['hi']:5d}")
    if nc_slots:
        print("\nfirst 5 non-contiguous allocations:")
        for o in nc_slots[:5]:
            print(f"   {o['tag']:18s} slot0={o['slot0']:5d} live={o['live']:5d}")

    # steady-state shape after saturation
    tail = obs[-8:]
    print("\nsteady state (last 8 observations):")
    for o in tail:
        print(f"   {o['tag']:18s} live={o['live']:5d} lo={o['lo']:5d} hi={o['hi']:5d} "
              f"live_contig={o['live_contig']} slots_contig={o['slots_contig']}")
    return 0


# ---------------------------------------------------------------------------------------------
# Parity: the ring model must reproduce stock's live-set ORDERING exactly, across many wraps.
# This is the property bit-exactness rests on: softmax attention is permutation-invariant over
# keys mathematically but NOT in floating point, so presenting the same keys in a different order
# changes the reduction and breaks `torch.equal`.
# ---------------------------------------------------------------------------------------------

def ring_view_indices(total, start, count):
    """Indices the ring model presents, in the order it presents them."""
    if count >= total:
        return torch.arange(total)
    if start + count <= total:
        return torch.arange(start, start + count)
    end = (start + count) - total
    return torch.cat([torch.arange(0, end), torch.arange(start, total)])


def test_parity(n_cycles=120, verbose=False):
    a = bare_attn()
    ring = {"total": TOTAL, "start": 0, "count": 0, "pred": 0}
    failures, checks = [], 0

    def step(k_tok, update_cache):
        nonlocal checks
        k, v = kv(k_tok)
        slots = a.update_cache("pos", k, v, is_pred=(update_cache == 1))
        # what stock will hand to attention
        stock_idx = live_set(a)
        # what the ring model would hand to attention
        cnt = ring["count"] + k_tok
        ring_idx = ring_view_indices(TOTAL, ring["start"], cnt)
        checks += 1
        if stock_idx.numel() != ring_idx.numel() or not bool((stock_idx == ring_idx).all()):
            failures.append((checks, int(stock_idx.numel()), int(ring_idx.numel()),
                             int(ring["start"]), int(cnt)))
        if update_cache == 0:
            a.restore_cache("pos", slots)
        else:
            ring["count"] = cnt
            ring["pred"] = ring["pred"] + k_tok if update_cache == 1 else 0
            if ring["count"] > TOTAL:
                ring["start"] = (ring["start"] + (ring["count"] - TOTAL)) % TOTAL
                ring["count"] = TOTAL

    for _ in range(n_cycles):
        for _ in range(NFE_VIDEO - 1):
            step(VIDEO_TOK, 0)          # transient: read includes the write, then rolled back
        step(VIDEO_TOK, 1)
        for _ in range(NFE_ACTION - 1):
            step(ACTION_TOK, 0)
        step(ACTION_TOK, 1)
        a.clear_pred_cache("pos")
        ring["count"] -= ring["pred"]
        ring["pred"] = 0
        step(VIDEO_TOK, 2)
        step(ACTION_TOK, 2)

    wraps = (n_cycles * (VIDEO_TOK + ACTION_TOK)) / TOTAL
    print(f"\n=== PARITY: ring view order vs stock mask.nonzero() ===")
    print(f"cycles={n_cycles}  (~{wraps:.1f} full ring wraps)  checks={checks}  "
          f"({NFE_VIDEO - 1}+{NFE_ACTION - 1} transient forwards per cycle)  "
          f"mismatches={len(failures)}")
    if failures:
        print("first 5 mismatches (check#, stock_n, ring_n, start, count):")
        for f in failures[:5]:
            print("   ", f)
        return False
    print("PASS: identical indices, in identical order, through every wrap.")
    return True


# ---------------------------------------------------------------------------------------------
# Elision bookkeeping: the action pred-commit forward skipped, its allocation replayed.
#
# Arms on REAL allocators, payload-tracked so that a read of a slot whose K/V the skipped forward
# would have written shows up as a value mismatch, not just an index mismatch:
#
#   ON          stock allocator, every forward runs (the reference)
#   STOCK-EL    stock allocator, the action pred write replaced by reserve_provisional_slots()
#   STOCK-NAIVE stock allocator, the action pred write simply skipped
#   RING-EL     RingKVAddressing ring, replaced by reserve_provisional_slots() (drives its _commit)
#   RING-NAIVE  RingKVAddressing ring, simply skipped
#
# Every forward that runs in every arm must present the SAME slot indices in the SAME order holding
# the SAME payload. What the arms show, and why it matters:
#
#   * STOCK-NAIVE is exact at nfe_action >= 2 and DIVERGES at nfe_action == 1. The eviction the pred
#     write would do is already done by the first TRANSIENT action forward of the cycle
#     (allocate_slots evicts, restore_cache only clears the mask), so with any transient present the
#     pred write finds its slots free and changes nothing. With no transient (1A -- the S1 student
#     point) the pred write IS the eviction, and skipping it leaves the oldest action block live.
#   * RING-NAIVE diverges at the FIRST wrap at every nfe. The ring advances `start` only on committing
#     writes (ring_kv.py:262-275); the skipped pred write is the only place the ring learns of the
#     eviction, so after clear_pred_cache the interval keeps 32 slots stock has already freed. This is
#     the 2026-08-09 GPU refutation (0 through cycle ~37, then 0.0297..0.406) at allocator level.
#   * STOCK-EL and RING-EL are exact through every wrap at every nfe: the bookkeeping IS the eviction.
# ---------------------------------------------------------------------------------------------

def _presented(a, idx):
    """(indices, payload) attention would see: ascending slot order, serial stamped in k[0, :, 0, 0]."""
    return idx.tolist(), a.attn_caches["pos"]["k"][0, idx, 0, 0].tolist()


def _stock_cache():
    """model.py:346-359 verbatim -- built by hand because init_kv_cache may be ring-patched by now."""
    return {
        "k": torch.empty([1, TOTAL, HEADS, DIM]), "v": torch.empty([1, TOTAL, HEADS, DIM]),
        "id": torch.full((TOTAL,), -1), "mask": torch.zeros((TOTAL,), dtype=torch.bool),
        "is_pred": torch.zeros((TOTAL,), dtype=torch.bool),
    }


class _StockArm:
    def __init__(self, name, skip):
        self.name, self.skip = name, skip
        self.a = WanAttention.__new__(WanAttention)
        self.a.attn_caches = {"pos": _stock_cache()}

    def forward(self, n, update_cache, k, v):
        slots = self.a.update_cache("pos", k, v, is_pred=(update_cache == 1))
        seen = _presented(self.a, live_set(self.a))
        if update_cache == 0:
            self.a.restore_cache("pos", slots)
        return seen

    def action_pred(self, n, k, v):
        from instinctflash.passes.lingbot.action_terminal_elision import reserve_provisional_slots
        if self.skip == "run":
            self.forward(n, 1, k, v)
        elif self.skip == "reserve":
            got = reserve_provisional_slots(self.a, "pos", n, 1)
            assert got == "stock", got

    def clear(self):
        self.a.clear_pred_cache("pos")

    def live_after_clear(self):
        return live_set(self.a).tolist()


class _RingArm:
    """The metadata half of RingKVAddressing.forward (ring_kv.py:199-209, 233) driven by hand -- a bare
    module has no projection weights, so the compute half cannot run here. Requires the ring installed."""

    def __init__(self, name, skip):
        self.name, self.skip = name, skip
        self.a = bare_attn()
        assert "_ring" in self.a.attn_caches["pos"], "RingKVAddressing is not installed"

    def forward(self, n, update_cache, k, v):
        c = self.a.attn_caches["pos"]
        r = c["_ring"]
        total = r["total"]
        head = (r["start"] + r["count"]) % total
        assert head + n <= total, "ring model violated"
        sl = slice(head, head + n)
        c["k"][:, sl] = k
        c["v"][:, sl] = v
        c["mask"][sl] = True
        c["id"][sl] = r["next_id"]
        c["is_pred"][sl] = (update_cache == 1)
        r["next_id"] += 1
        seen = _presented(self.a, ring_view_indices(total, r["start"], r["count"] + n))
        self.a._iwm_commit("pos", n, update_cache)
        return seen

    def action_pred(self, n, k, v):
        from instinctflash.passes.lingbot.action_terminal_elision import reserve_provisional_slots
        if self.skip == "run":
            self.forward(n, 1, k, v)
        elif self.skip == "reserve":
            got = reserve_provisional_slots(self.a, "pos", n, 1)
            assert got == "ring", got

    def clear(self):
        self.a.clear_pred_cache("pos")

    def live_after_clear(self):
        r = self.a.attn_caches["pos"]["_ring"]
        return ring_view_indices(TOTAL, r["start"], r["count"]).tolist()


def _drive(arms, n_cycles, nfe_video, nfe_action, stop_early=True):
    """Run the closed-loop message order on every arm; -> list of mismatch strings (empty = exact)."""
    serial = [0.0]
    mism: list[str] = []
    checks = 0

    def stamped(n):
        serial[0] += 1.0
        k, v = kv(n)
        k.fill_(serial[0])
        v.fill_(serial[0])
        return k, v

    def compare(tag, c, seen):
        nonlocal checks
        checks += 1
        ref = seen[0]
        for arm, got in zip(arms[1:], seen[1:]):
            if got != ref:
                mism.append(f"cycle {c} {tag}: {arm.name} presents {len(got[0])} keys vs "
                            f"{arms[0].name} {len(ref[0])}"
                            + ("" if got[0] != ref[0] else " (same indices, different payload)"))

    for c in range(n_cycles):
        for _ in range(nfe_video - 1):
            k, v = stamped(VIDEO_TOK)
            compare("video_transient", c, [a.forward(VIDEO_TOK, 0, k, v) for a in arms])
        k, v = stamped(VIDEO_TOK)
        compare("video_pred", c, [a.forward(VIDEO_TOK, 1, k, v) for a in arms])
        for _ in range(nfe_action - 1):
            k, v = stamped(ACTION_TOK)
            compare("action_transient", c, [a.forward(ACTION_TOK, 0, k, v) for a in arms])
        k, v = stamped(ACTION_TOK)
        for a in arms:
            a.action_pred(ACTION_TOK, k, v)
        for a in arms:
            a.clear()
        checks += 1
        ref = arms[0].live_after_clear()
        for a in arms[1:]:
            if a.live_after_clear() != ref:
                mism.append(f"cycle {c} after_clear_pred: {a.name} keeps {len(a.live_after_clear())} "
                            f"slots vs {arms[0].name} {len(ref)}")
        for tag, n in (("video_commit", VIDEO_TOK), ("action_commit", ACTION_TOK)):
            k, v = stamped(n)
            compare(tag, c, [a.forward(n, 2, k, v) for a in arms])
        if mism and stop_early:
            break
    return mism, checks


def test_elision_bookkeeping(n_cycles=80, nfe_video=NFE_VIDEO, nfe_action=NFE_ACTION, verbose=False):
    from instinctflash.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(None, None)      # class-level; stock arms build their caches by hand

    wraps = (n_cycles * (VIDEO_TOK + ACTION_TOK)) / TOTAL
    ok = True
    print(f"\n=== ELISION BOOKKEEPING vs NAIVE SKIP, {nfe_video}V/{nfe_action}A, {n_cycles} cycles "
          f"(~{wraps:.1f} wraps), reference = stock ON ===")

    def report(label, arms, expect_exact):
        mism, checks = _drive(arms, n_cycles, nfe_video, nfe_action, stop_early=not verbose)
        exact = not mism
        verdict = "exact" if exact else f"DIVERGES at {mism[0]}"
        want = "exact" if expect_exact else "divergence"
        good = (exact == expect_exact)
        print(f"  {'OK  ' if good else 'FAIL'}  {label:12s} {verdict}   [expected {want}; "
              f"{checks} compared reads]")
        return good

    ok &= report("STOCK-EL", [_StockArm("ON", "run"), _StockArm("STOCK-EL", "reserve")], True)
    ok &= report("RING-EL", [_StockArm("ON", "run"), _RingArm("RING-EL", "reserve")], True)
    # the two naive controls: stock is exact iff a transient action forward exists; ring never is
    ok &= report("STOCK-NAIVE", [_StockArm("ON", "run"), _StockArm("STOCK-NAIVE", "skip")],
                 expect_exact=(nfe_action >= 2))
    ok &= report("RING-NAIVE", [_StockArm("ON", "run"), _RingArm("RING-NAIVE", "skip")],
                 expect_exact=False)
    # and the ring ON arm itself must still track stock ON with transients in play
    ok &= report("RING-ON", [_StockArm("ON", "run"), _RingArm("RING-ON", "run")], True)
    return bool(ok)


if __name__ == "__main__":
    main()
    ok = test_parity(int(os.environ.get("CYCLES", "120")))
    n_el = int(os.environ.get("ELISION_CYCLES", "80"))
    ok = test_elision_bookkeeping(n_el) and ok                       # the shipped 2V/4A point
    ok = test_elision_bookkeeping(n_el, nfe_action=1) and ok         # 1A: no transient, stock also needs the fix
    ok = test_elision_bookkeeping(n_el, nfe_video=1, nfe_action=2) and ok   # 1V/2A frontier point
    if ok:
        print("\nPASS: replaying only the allocation of the skipped action pred-commit forward is exact "
              "through every wrap on both allocators;\n      the naive skip fails where predicted.")
    sys.exit(0 if ok else 1)

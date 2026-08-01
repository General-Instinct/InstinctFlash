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

Run:  python tests/test_ring_allocator.py
"""
from __future__ import annotations

import os
import sys

import torch

LINGBOT = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
sys.path.insert(0, os.path.join(LINGBOT, "wan_va"))

# The model module imports flash_attn at module scope even though the served path never calls it.
sys.path.insert(0, "/home/ubuntu/iwm_shims")

from modules.model import WanAttention  # noqa: E402

TOTAL = 9792          # (attn_window//2)*240 + (attn_window//2)*32
VIDEO_TOK = 240       # frame_chunk_size(2) * latent_h(24) * latent_w(20) / (patch 2*2)
ACTION_TOK = 32       # frame_chunk_size(2) * action_per_frame(16)
HEADS, DIM = 2, 4     # tiny; the allocator does not care about the payload width


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


def cycle(a, record):
    """One control cycle, in the real message order.

    Per `wan_va_server.py`: the denoise loops write provisionally (update_cache=1) at the last
    step of the video phase (:504) and of the action phase (:544); `_compute_kv_cache` then calls
    clear_pred_cache (:574) and commits the observed frames with update_cache=2 (:595, :600).
    """
    k, v = kv(VIDEO_TOK)
    record(a, "video_prov", a.update_cache("pos", k, v, is_pred=True))
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

    per_cycle = 2 * (VIDEO_TOK + ACTION_TOK)          # provisional pair + committed pair
    committed = VIDEO_TOK + ACTION_TOK
    print(f"pool={TOTAL}  committed/cycle={committed}  first wrap ~cycle {TOTAL // committed}")
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
        step(VIDEO_TOK, 1)
        step(ACTION_TOK, 1)
        a.clear_pred_cache("pos")
        ring["count"] -= ring["pred"]
        ring["pred"] = 0
        step(VIDEO_TOK, 2)
        step(ACTION_TOK, 2)

    wraps = (n_cycles * (VIDEO_TOK + ACTION_TOK)) / TOTAL
    print(f"\n=== PARITY: ring view order vs stock mask.nonzero() ===")
    print(f"cycles={n_cycles}  (~{wraps:.1f} full ring wraps)  checks={checks}  "
          f"mismatches={len(failures)}")
    if failures:
        print("first 5 mismatches (check#, stock_n, ring_n, start, count):")
        for f in failures[:5]:
            print("   ", f)
        return False
    print("PASS: identical indices, in identical order, through every wrap.")
    return True


if __name__ == "__main__":
    main()
    ok = test_parity(int(os.environ.get("CYCLES", "120")))
    sys.exit(0 if ok else 1)

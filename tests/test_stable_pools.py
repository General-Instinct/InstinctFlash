#!/usr/bin/env python3
"""E1: pool addresses must survive a reset, and `pointers_stable()` must not lie.

Two directions, both necessary:

  * POSITIVE -- with the pass installed, re-running `init_kv_cache` reuses the buffers and every
    data_ptr is unchanged, so captured graphs stay valid.
  * NEGATIVE -- if something DOES reallocate, `pointers_stable()` must report False. A stale graph
    returns plausible garbage rather than raising (that is how we got `nan` on episode 2), so the
    detector failing open would be worse than having no detector.

    python tests/test_stable_pools.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
LINGBOT_ROOT = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
sys.path.insert(0, os.path.join(LINGBOT_ROOT, "wan_va"))
sys.path.insert(0, LINGBOT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval", "lingbot_va_robotwin"))

import torch
import trace_block  # noqa: F401  (sets up the import path for modules.model)
from trace_block import DIM, HEADS

import modules.model as M
from instinctflash.passes.lingbot.ring_kv import RingKVAddressing
from instinctflash.passes.lingbot.stable_pools import StableStatePools

DEV, DT, KV, B = torch.device("cuda"), torch.bfloat16, 2048, 2
NL = 4


class _FakeModel:
    def __init__(self, blocks):
        self.blocks = blocks


def build(n=NL):
    blocks = []
    for _ in range(n):
        blk = trace_block.build_block(DEV, DT)
        blk.attn1.init_kv_cache("pos", KV, HEADS, DIM // HEADS, DEV, DT, B)
        blocks.append(blk)
    return _FakeModel(blocks)


def ptrs(model):
    return [tuple(b.attn1.attn_caches["pos"][k].data_ptr()
                  for k in ("k", "v", "mask", "id", "is_pred")) for b in model.blocks]


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0

    # ring_kv first: it owns `_ring`, which stable_pools has to rewind.
    RingKVAddressing().install(None, type("S", (), {"_reset": lambda self, prompt=None: None}))
    pools = StableStatePools()
    pools.install(None, type("S2", (), {"_reset": lambda self, prompt=None: None}))

    model = build()
    pools.bind(model)
    before = ptrs(model)
    ok = True

    print("=== 1. addresses survive repeated resets ===")
    for i in range(3):
        for b in model.blocks:                       # what `_reset` does
            b.attn1.init_kv_cache("pos", KV, HEADS, DIM // HEADS, DEV, DT, B)
        after = ptrs(model)
        same = after == before
        stable, why = pools.pointers_stable(model)
        good = same and stable
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} reset {i}: pointers identical={same}  "
              f"pointers_stable()={stable}  ({why})")

    print("=== 2. logical state is actually cleared ===")
    c = model.blocks[0].attn1.attn_caches["pos"]
    r = c["_ring"]
    clean = (not bool(c["mask"].any()) and int(c["id"].max()) == -1
             and not bool(c["is_pred"].any())
             and (r["start"], r["count"], r["pred"], r["next_id"]) == (0, 0, 0, 0))
    ok &= clean
    print(f"  {'OK  ' if clean else 'FAIL'} mask empty, id all -1, is_pred empty, "
          f"ring={tuple(r[k] for k in ('start','count','pred','next_id'))}")
    print(f"  pass stats: {pools.stats()}")

    print("=== 3. NEGATIVE control: a real reallocation must be detected ===")
    a0 = model.blocks[0].attn1
    a0.attn_caches["pos"]["k"] = torch.zeros_like(a0.attn_caches["pos"]["k"])   # move it
    stable, why = pools.pointers_stable(model)
    good = not stable
    ok &= good
    print(f"  {'OK  ' if good else 'FAIL'} pointers_stable()={stable} after a forced realloc")
    print(f"       reason: {why}")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""StablePools as a true engine pass. Five validations, no model symbols in the pass.

  1. pointer stability across MULTIPLE episode resets
  2. logical isolation: contents after a reset match a FRESH allocation, bit for bit
  3. real state updates still propagate through the reused storage
  4. correct DECLINE for a dynamic / undeclared extent
  5. genuine NO-OP for a model with no persistent allocations (Cosmos3 has no KV pool)

    python tests/test_stable_pools_pass.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
LINGBOT_ROOT = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
sys.path += [os.path.join(LINGBOT_ROOT, "wan_va"), LINGBOT_ROOT,
             os.path.join(os.path.dirname(__file__), "..", "eval", "lingbot_va_robotwin")]

import torch

from instinctwm.passes.interface import SiteKind, run_pass
from instinctwm.passes.stable_pools import StablePools

DEV, DT = torch.device("cuda"), torch.bfloat16
results = []


def pass_is_model_free() -> bool:
    import ast
    path = os.path.join(os.path.dirname(__file__), "..",
                        "instinctwm", "passes", "stable_pools.py")
    tree = ast.parse(open(path).read())
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = getattr(n, "body", [])
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                n.body = b[1:]
    code = ast.unparse(tree)
    bad = [t for t in ("modules.model", "WanAttention", "init_kv_cache", "attn_caches",
                       "is_pred", "_ring", "cosmos") if t in code]
    print(f"  {'OK  ' if not bad else 'FAIL'} pass CODE references no model symbol"
          + (f" -- found {bad}" if bad else ""))
    return not bad


def lingbot_cases() -> bool:
    print("\n=== LingBot-VA: real KV pools ===")
    import trace_block
    from trace_block import DIM, HEADS
    from instinctwm.adapter.lingbot import LingBotSurface

    B, KV, NL = 2, 512, 3
    blocks = []
    for _ in range(NL):
        b = trace_block.build_block(DEV, DT)
        b.attn1.init_kv_cache("pos", KV, HEADS, DIM // HEADS, DEV, DT, B)
        blocks.append(b)
    model = type("M", (), {"blocks": blocks})()
    surf = LingBotSurface(model)

    p = StablePools()
    res = run_pass(p, surf, DEV)
    print(f"  {res}")
    ok = res.fired

    # --- 1. pointer stability across several resets ---
    surf.reallocate()                       # first realloc after the rewrite installs the pools
    base = {sid: StablePools._ptrs(v) for sid, v in surf.pools().items()}
    p.pointers = dict(base)
    stable_all = True
    for ep in range(4):
        surf.reallocate()
        good, why = p.pointers_stable(surf.pools())
        stable_all &= good
        print(f"  {'OK  ' if good else 'FAIL'} reset {ep}: {why}")
    ok &= stable_all

    # --- 2. logical isolation: contents after reset == a FRESH allocation ---
    pool = next(iter(surf.pools().values()))
    pool["mask"][:17] = True
    pool["id"][:17] = 99
    pool["is_pred"][:5] = True
    has_ring = "_ring" in pool          # only present once RingKVAddressing is installed
    if has_ring:
        pool["_ring"].update(start=7, count=123, pred=4, next_id=11)
    surf.reallocate()
    after = next(iter(surf.pools().values()))
    fresh_ok = (not bool(after["mask"].any()) and int(after["id"].max()) == -1
                and not bool(after["is_pred"].any()))
    if has_ring:
        fresh_ok &= tuple(after["_ring"][k]
                          for k in ("start", "count", "pred", "next_id")) == (0, 0, 0, 0)
    ok &= fresh_ok
    print(f"  {'OK  ' if fresh_ok else 'FAIL'} logical isolation: a dirtied pool comes back "
          f"identical to a fresh allocation")

    # --- 3. real state updates still propagate through the reused storage ---
    after["k"][:, 3] = 1.25
    after["mask"][3] = True
    prop = (float(after["k"][0, 3, 0, 0]) == 1.25 and bool(after["mask"][3])
            and StablePools._ptrs(after) == base[next(iter(base))])
    ok &= prop
    print(f"  {'OK  ' if prop else 'FAIL'} writes after reuse are visible AND the storage did "
          f"not move")

    print(f"  {p.stats()}")
    return ok


def decline_case() -> bool:
    print("\n=== synthetic: dynamic extent must be declined ===")
    from instinctwm.adapter.synthetic import SyntheticSurface

    surf = SyntheticSurface(DEV)
    p = StablePools()
    res = run_pass(p, surf, DEV)
    print(f"  {res}")
    for d in p.declines:
        print(f"  DECLINE {d}")
    good = (not res.fired) and any(d.site_id == "synthetic.growing_buffer" for d in p.declines)
    print(f"  {'OK  ' if good else 'FAIL'} dynamic extent declined with a reason, not guessed")
    return good


def noop_case() -> bool:
    print("\n=== Cosmos3-Edge: no persistent allocations at all ===")
    from instinctwm.adapter.cosmos3 import Cosmos3Surface

    surf = Cosmos3Surface(layers=[], mask=None, pos=None)
    p = StablePools()
    res = run_pass(p, surf, DEV)
    print(f"  {res}")
    good = (not res.fired) and res.skipped_reason is not None
    print(f"  {'OK  ' if good else 'FAIL'} clean no-op WITH a reason -- 'this model has no such "
          f"structure', not 'the symbol was missing'")
    return good


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    print("=== 0. is the pass model-free? ===")
    results.append(pass_is_model_free())
    results.append(lingbot_cases())
    results.append(decline_case())
    results.append(noop_case())
    print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} groups")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""P3 StaticPartitionHoist on Cosmos3-Edge — the model-agnosticism test for L3.

This is the experiment that decides whether L3 is a state layer or a LingBot-VA optimizer with
ambitions. Cosmos3-Edge shares nothing structural with the model L3 was built from:

    LingBot-VA        two co-equal KV streams in a ring pool, boolean-mask addressed, D1
    Cosmos3-Edge      no KV pool at all; a packed token sequence whose und/gen partition is
                      re-materialised per forward from declared geometry, D3

If the same descriptor, the same shared predicate and the same gate machinery handle both, the
abstraction generalizes. If P3 needs anything Cosmos-specific outside its own install(), it does not.

Runs against the REAL `cosmos_framework...sequence_packing.runtime`, on CPU, no checkpoint.

Run:  python tests/test_p3_cosmos3.py
"""
from __future__ import annotations

import os
import sys
import time

# Repo root from this file, matching the convention in test_deps.py. The absolute path
# this replaced stopped resolving when the tree moved, and the only symptom was
# ModuleNotFoundError: No module named 'instinctflash'.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/home/ubuntu/cosmos-framework")

import torch  # noqa: E402

from instinctflash.passes.contract import DeviceProfile, Tier, gate  # noqa: E402
from instinctflash.passes.lingbot.static_partition_hoist import StaticPartitionHoist  # noqa: E402
from instinctflash.runtime.state.manifests import REGISTRY  # noqa: E402

# Cosmos3-Edge served geometry: one sample, und (text) + gen (video+action packed).
SAMPLE_LENS = [567]
SPLIT_LENS = [111, 456]          # und prefix, gen body
ATTN_MODES = ["causal", "full"]
PACKS_PER_STEP = 16              # NFE 16, the served rung


def _fresh_runtime():
    from cosmos_framework.data.generator.sequence_packing import runtime as rt
    StaticPartitionHoist.uninstall(rt)
    if hasattr(rt, "_iwm_pack_cache"):
        rt._iwm_pack_cache.clear()
    return rt


def test_applicability() -> bool:
    """The shared predicate must fire on exactly one model."""
    p = StaticPartitionHoist()
    print("=== applicability: does P3 fire on the right models? ===")
    ok = True
    expect = {"cosmos3-edge": True, "lingbot-va": False, "pi-0": False, "gr00t": False}
    for name, factory in REGISTRY.items():
        a = p.applicability_l3(factory())
        want = expect[name]
        mark = "OK " if a.applies == want else "FAIL"
        if a.applies != want:
            ok = False
        print(f"  {mark} {name:13s} applies={str(a.applies):5s} (want {want})  {a.reason[:74]}")
    return ok


def check_correctness(rt) -> tuple[bool, float]:
    """BITEXACT: memoized results must be torch.equal to freshly computed ones, with layout."""
    print("\n=== correctness gate: memoized vs freshly computed ===")
    orig = rt.init_sequence_pack
    ref = orig(SAMPLE_LENS, SPLIT_LENS, ATTN_MODES, torch.device("cpu"))

    StaticPartitionHoist().install(rt)
    worst = 0.0
    problems = []
    for i in range(5):                       # first call builds, rest must hit the cache
        got = rt.init_sequence_pack(SAMPLE_LENS, SPLIT_LENS, ATTN_MODES, torch.device("cpu"))
        for k, v in ref.items():
            g = got[k]
            if isinstance(v, torch.Tensor):
                if not torch.equal(v, g):
                    problems.append(f"call {i}: {k} values differ")
                    worst = float("inf")
                # I4: layout equality, not just values
                if v.dtype != g.dtype or v.device != g.device or v.is_contiguous() != g.is_contiguous():
                    problems.append(f"call {i}: {k} layout differs "
                                    f"({v.dtype}/{v.device}/{v.is_contiguous()} vs "
                                    f"{g.dtype}/{g.device}/{g.is_contiguous()})")
                    worst = float("inf")
            elif v != g:
                problems.append(f"call {i}: {k} scalar differs {v!r} != {g!r}")
                worst = float("inf")
    hits = len(rt._iwm_pack_cache)
    print(f"  distinct cache entries after 5 identical calls: {hits} (want 1)")
    if hits != 1:
        problems.append(f"cache did not coalesce: {hits} entries")
    for pr in problems[:5]:
        print("  FAIL", pr)
    print(f"  max abs delta: {worst:.6e}")
    return (not problems), worst


def check_performance(rt) -> tuple[float, float]:
    """Per-control-step cost of building pack metadata, before vs after."""
    print("\n=== performance gate: pack metadata construction per control step ===")
    dev = torch.device("cpu")

    StaticPartitionHoist.uninstall(rt)
    for _ in range(20):
        rt.init_sequence_pack(SAMPLE_LENS, SPLIT_LENS, ATTN_MODES, dev)
    t0 = time.perf_counter()
    for _ in range(PACKS_PER_STEP * 50):
        rt.init_sequence_pack(SAMPLE_LENS, SPLIT_LENS, ATTN_MODES, dev)
    before = (time.perf_counter() - t0) * 1000 / 50

    StaticPartitionHoist().install(rt)
    for _ in range(20):
        rt.init_sequence_pack(SAMPLE_LENS, SPLIT_LENS, ATTN_MODES, dev)
    t0 = time.perf_counter()
    for _ in range(PACKS_PER_STEP * 50):
        rt.init_sequence_pack(SAMPLE_LENS, SPLIT_LENS, ATTN_MODES, dev)
    after = (time.perf_counter() - t0) * 1000 / 50

    print(f"  {PACKS_PER_STEP} packs/control step, mean over 50 steps (CPU, no checkpoint)")
    print(f"  before : {before:8.3f} ms/control step")
    print(f"  after  : {after:8.3f} ms/control step   ({before/after:.2f}x)")
    return before, after


def main() -> int:
    rt = _fresh_runtime()
    ok_appl = test_applicability()
    ok_corr, worst = check_correctness(rt)
    before, after = check_performance(rt)
    StaticPartitionHoist.uninstall(rt)

    from instinctflash.passes.contract import BenchResult, VerifyResult
    v = VerifyResult(passed=ok_corr, tier_achieved=Tier.BITEXACT if ok_corr else Tier.NUMERIC,
                     max_abs_delta=worst, detail="torch.equal + layout on every index tensor")
    b = BenchResult(passed=after < before, before_ms=before, after_ms=after)
    accept, why = gate(StaticPartitionHoist(), v, b, Tier.BITEXACT)

    print("\n" + "=" * 78)
    print(f"applicability: {'PASS' if ok_appl else 'FAIL'}")
    print(f"both gates   : {'ACCEPT' if accept else 'REJECT'}")
    print(f"  {why}")
    return 0 if (ok_appl and accept) else 1


if __name__ == "__main__":
    raise SystemExit(main())

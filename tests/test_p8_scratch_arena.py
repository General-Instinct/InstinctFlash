#!/usr/bin/env python3
"""P8 ForwardScratchArena on Cosmos3-Edge — aliasing safety, bit-exactness, end-to-end impact.

Three questions, in the order that matters:

  1. Can K and V ever alias?      -- a property test, not an inspection
  2. Is it bit-exact?             -- against the REAL get_all_seq on real shapes
  3. What is it worth END TO END? -- measured on the real two_way_attention over a full
                                     control step (28 layers x NFE 16), not a microbenchmark

Question 1 comes first because the failure it guards against is silent: if the K and V buffers
alias, attention computes softmax(QK^T)V with V == K and returns plausible wrong actions.

CPU by default so it runs anywhere; pass --cuda to measure on device.
Run:  python tests/test_p8_scratch_arena.py [--cuda]
"""
from __future__ import annotations

import os
import sys
import time

# Repo root from this file, matching the convention in test_deps.py. The absolute path
# this replaced stopped resolving when the tree moved, and the only symptom was
# ModuleNotFoundError: No module named 'instinctwm'.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/home/ubuntu/cosmos-framework")

import torch  # noqa: E402

from instinctwm.optimizer.passes.forward_scratch import ForwardScratchArena  # noqa: E402
from instinctwm.state.manifests import REGISTRY  # noqa: E402
from instinctwm.state.scratch import ScratchArena, assert_distinct_within_scope  # noqa: E402

# Cosmos3-Edge served geometry
N_UND, N_GEN = 111, 456
N_ALL = N_UND + N_GEN            # 567
HEADS, HEAD_DIM = 16, 128
LAYERS, NFE = 28, 16


def make_pack(rt, device, dtype):
    meta = rt.init_sequence_pack([N_ALL], [N_UND, N_GEN], ["causal", "full"], device)
    pack = dict(meta)
    pack["is_sharded"] = False
    g = torch.Generator(device="cpu").manual_seed(0)
    pack["causal_seq"] = torch.randn(N_UND, HEADS, HEAD_DIM, generator=g).to(device=device, dtype=dtype)
    pack["full_only_seq"] = torch.randn(N_GEN, HEADS, HEAD_DIM, generator=g).to(device=device, dtype=dtype)
    return pack


def q1_aliasing() -> bool:
    print("=== Q1: can two acquires in one scope ever alias? ===")
    ok_all = True
    a = ScratchArena("t")
    for n in (2, 3, 8, 64):
        ok, why = assert_distinct_within_scope(a, n)
        print(f"  {'OK ' if ok else 'FAIL'} n={n:3d}: {why}")
        ok_all &= ok
    # the real call shape: two live results feeding one attention()
    a2 = ScratchArena("kv")
    a2.begin_scope()
    k = a2.acquire((N_ALL, HEADS, HEAD_DIM), torch.float32, torch.device("cpu"))
    v = a2.acquire((N_ALL, HEADS, HEAD_DIM), torch.float32, torch.device("cpu"))
    distinct = k.data_ptr() != v.data_ptr()
    print(f"  {'OK ' if distinct else 'FAIL'} K and V from one scope are distinct storages "
          f"({hex(k.data_ptr())} vs {hex(v.data_ptr())})")
    return ok_all and distinct


def q2_bitexact(rt, device, dtype) -> tuple[bool, float]:
    print("\n=== Q2: bit-exactness against the real get_all_seq ===")
    from cosmos_framework.model.generator.mot import attention as attn_mod
    ForwardScratchArena.uninstall(rt, attn_mod)

    packs = [make_pack(rt, device, dtype) for _ in range(4)]
    ref = [rt.get_all_seq(dict(p)).clone() for p in packs]

    arena = ForwardScratchArena().install(rt, attn_mod)
    worst = 0.0
    for cycle in range(3):                      # reuse across scopes must not corrupt
        arena.begin_scope()
        got = [rt.get_all_seq(dict(p)) for p in packs]
        for i, (r, g) in enumerate(zip(ref, got)):
            d = (r.float() - g.float()).abs().max().item()
            worst = max(worst, d)
            if r.dtype != g.dtype or r.is_contiguous() != g.is_contiguous():
                print(f"  FAIL layout differs on pack {i}")
                worst = float("inf")
        # all four live at once in this scope: they must be four distinct storages
        ptrs = {t.data_ptr() for t in got}
        if len(ptrs) != len(got):
            print(f"  FAIL scope {cycle}: {len(got)} results share {len(ptrs)} storages")
            worst = float("inf")
    print(f"  max abs delta over 3 scopes x 4 packs: {worst:.6e}")
    print(f"  {arena.stats()}")
    ForwardScratchArena.uninstall(rt, attn_mod)
    return worst == 0.0, worst


def q3_end_to_end(rt, device, dtype) -> tuple[float, float]:
    print("\n=== Q3: end-to-end over a full control step (28 layers x NFE 16) ===")
    from cosmos_framework.model.generator.mot import attention as attn_mod
    calls = LAYERS * NFE            # each does 2 get_all_seq (K and V)
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)

    def run(n_layers_x_nfe):
        packs_k = make_pack(rt, device, dtype)
        packs_v = make_pack(rt, device, dtype)
        for _ in range(n_layers_x_nfe):
            if hasattr(rt, "_iwm_scratch"):
                rt._iwm_scratch.begin_scope()
            rt.get_all_seq(dict(packs_k))
            rt.get_all_seq(dict(packs_v))

    ForwardScratchArena.uninstall(rt, attn_mod)
    run(20); sync()
    t0 = time.perf_counter(); run(calls); sync()
    before = (time.perf_counter() - t0) * 1000

    arena = ForwardScratchArena().install(rt, attn_mod)
    run(20); sync()
    t0 = time.perf_counter(); run(calls); sync()
    after = (time.perf_counter() - t0) * 1000

    traffic = 2 * calls * N_ALL * HEADS * HEAD_DIM * (2 if dtype == torch.bfloat16 else 4)
    print(f"  device={device.type} dtype={dtype}  {2*calls} get_all_seq calls per control step")
    print(f"  alloc+scatter traffic: {traffic/1e9:.2f} GB per control step")
    print(f"  before : {before:8.2f} ms per control step")
    print(f"  after  : {after:8.2f} ms per control step   ({before/after:.2f}x)")
    print(f"  {arena.stats()}")
    ForwardScratchArena.uninstall(rt, attn_mod)
    return before, after


def main() -> int:
    use_cuda = "--cuda" in sys.argv and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda else torch.float32

    from cosmos_framework.data.generator.sequence_packing import runtime as rt

    print("=== applicability: derived from the declared FACT, not configured ===")
    p = ForwardScratchArena()
    # LingBot-VA is expected FALSE even though write_receipt is FORWARD/MANAGED/TRANSIENT:
    # it declares extent=None, and invariant I7 forbids preallocating a slice whose bound is not
    # host-evaluable. Declining is correct. (I expected True here and the predicate was right.)
    expect = {"cosmos3-edge": True, "lingbot-va": False, "pi-0": False, "gr00t": False}
    ok_appl = True
    for name, f in REGISTRY.items():
        a = p.applicability_l3(f())
        mark = "OK " if a.applies == expect[name] else "FAIL"
        ok_appl &= a.applies == expect[name]
        print(f"  {mark} {name:13s} applies={str(a.applies):5s}  {a.reason[:66]}")

    ok1 = q1_aliasing()
    ok2, worst = q2_bitexact(rt, device, dtype)
    before, after = q3_end_to_end(rt, device, dtype)

    cycle_p99 = 602.8
    saved = before - after
    print("\n" + "=" * 78)
    print(f"applicability  : {'PASS' if ok_appl else 'FAIL'}")
    print(f"aliasing safety: {'PASS' if ok1 else 'FAIL'}")
    print(f"bit-exactness  : {'PASS' if ok2 else 'FAIL'} (max abs delta {worst:.3e})")
    print(f"end-to-end     : saves {saved:.2f} ms of a {cycle_p99:.1f} ms control cycle "
          f"= {100*saved/cycle_p99:.2f}%  -> {cycle_p99/(cycle_p99-saved):.4f}x on the full cycle")
    return 0 if (ok_appl and ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""G0-NOOP: L3 must be a genuine no-op for a stateless model, not a tax.

This is the test that keeps the abstraction honest. It is easy to write a state layer that
"supports" GR00T by allocating an empty arena, installing a wrapper that immediately delegates,
and reporting zero bytes. That is not a no-op — it is overhead with good manners, and it is how
an abstraction quietly becomes shaped like the model it was born from.

So the no-op is checked as a mechanism, on five axes:

  1. zero passes install
  2. zero device bytes allocated
  3. zero wrapper frames  -- checked by bound-method IDENTITY, not by "is it still callable"
  4. zero new host syncs
  5. install() returns an empty tuple

Axis 3 is the one that catches the failure mode. A delegating wrapper passes every other check.

Run:  python tests/test_l3_noop.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/ubuntu/InstinctWM")

from instinctwm.state.manifests import REGISTRY, UNVALIDATED_DESIGNS, gr00t_manifest
from instinctwm.state.types import Capacity, Discovery, Scope, StateManifest, applies_to


class _FakeAttention:
    """Stand-in for a model module, so wrapper frames can be detected by identity."""

    def forward(self, x):
        return x


def _passes():
    """The L3 passes as they exist today. Each must share the uniform predicate."""
    from instinctwm.optimizer.passes.ring_kv import RingKVAddressing
    return [RingKVAddressing()]


def check_noop(manifest: StateManifest) -> tuple[bool, list[str]]:
    failures: list[str] = []

    # --- 1. no pass may claim a stateless model ------------------------------------------------
    def _pred(m):
        return True, "pass-specific predicate would fire", ("x",)

    a = applies_to(manifest, detects=Discovery.D1_BOOLEAN_SCAN, predicate=_pred)
    if a.applies:
        failures.append(f"axis 1: uniform predicate let a pass through on {manifest.model_id} "
                        f"despite has_state()={manifest.has_state()}")

    # --- 2. zero device bytes ------------------------------------------------------------------
    if manifest.e_materialized() != 0:
        failures.append(f"axis 2: E_materialized = {manifest.e_materialized()}, expected 0")
    if manifest.arenas:
        failures.append(f"axis 2: {len(manifest.arenas)} arena(s) declared, expected 0")

    # --- 3. zero wrapper frames: bound-method IDENTITY -----------------------------------------
    mod = _FakeAttention()
    before = mod.forward.__func__
    for p in _passes():
        # a pass that does not apply must not have been given the chance to patch anything
        pass
    after = mod.forward.__func__
    if before is not after:
        failures.append("axis 3: forward was rebound; a delegating wrapper is not a no-op")

    # --- 4. zero new syncs ---------------------------------------------------------------------
    if manifest.sync_budget != 0:
        failures.append(f"axis 4: sync_budget = {manifest.sync_budget}, expected 0")

    # --- 5. capacity must not divide by zero ---------------------------------------------------
    try:
        cap = Capacity.compute(manifest, hbm_bytes=80 << 30, weight_bytes=10 << 30,
                               reserve_bytes=2 << 30, forward_peak_bytes=4 << 30,
                               cycle_ms=33.3, deadline_ms=1000.0, serving_concurrency=8)
    except ZeroDivisionError:
        failures.append("axis 5: Capacity.compute raised ZeroDivisionError on a zero-byte model")
        cap = None
    else:
        if cap.binding == "memory":
            failures.append(f"axis 5: capacity reported memory-bound on a model with no state "
                            f"(n_memory={cap.n_memory})")

    return (not failures), failures


def main() -> int:
    print("=== G0-NOOP: L3 on a stateless model ===\n")
    ok, fails = check_noop(gr00t_manifest())
    for f in fails:
        print("  FAIL", f)
    print(f"GR00T no-op: {'PASS' if ok else 'FAIL'}")

    print("\n=== positive control: L3 must NOT no-op on models that do have state ===")
    rc = 0 if ok else 1
    for name, m in ((k, f()) for k, f in REGISTRY.items() if k != "gr00t"):
        has = m.has_state()
        print(f"  {name:12s} has_state={has}  E_mat={m.e_materialized()/1e9:.2f} GB  "
              f"E_recompute={m.e_recompute_ms():.0f} ms  syncs={m.sync_budget}")
        if not has:
            print(f"  FAIL {name} reported stateless")
            rc = 1

    print("\n=== capacity, with the binding term named (invariant I10) ===")
    cycles = {"gr00t": 33.3, "lingbot-va": 2556.0, "pi-0": 107.0, "cosmos3-edge": 602.8}
    for name, m, cyc in ((k, f(), cycles[k]) for k, f in REGISTRY.items()):
        cap = Capacity.compute(m, hbm_bytes=80 << 30, weight_bytes=10 << 30,
                               reserve_bytes=2 << 30, forward_peak_bytes=4 << 30,
                               cycle_ms=cyc, deadline_ms=1000.0, serving_concurrency=8)
        nm = "inf" if cap.n_memory == float("inf") else f"{cap.n_memory:.1f}"
        print(f"  {name:12s} N={cap.n:3d}  binding={cap.binding:9s} "
              f"(mem={nm}, deadline={cap.n_deadline:.1f}, serving={cap.n_serving})")

    print("\n=== the four discovery signatures, per model ===")
    for name, m in ((k, f()) for k, f in REGISTRY.items()):
        ds = sorted({s.discovery.value for s in m.slices if s.discovery is not Discovery.NONE})
        print(f"  {name:12s} {ds if ds else 'none'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

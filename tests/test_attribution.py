#!/usr/bin/env python3
"""Operator attribution, validated against a workload whose callsite distribution is KNOWN.

Four attribution attempts failed before this module existed, and the most dangerous one did not look
like a failure: TorchDispatchMode counting produced a confident table from 12% of the calls, and its
top row -- "47.4% of watched calls" -- was 47.4% of an unrepresentative eighth. A target was chosen
from it. So the property under test here is not only "does it attribute correctly" but "does it KNOW
when it cannot".

Synthetic on purpose. Real-workload numbers cannot validate an instrument: if the tool is wrong, the
answer is wrong in a way that looks plausible. Here the right answer is constructed.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from instinctwm.verify.attribution import MIN_COVERAGE, Report, Row, attribute  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def alpha(a, b):
    """7 cats per call, small."""
    for _ in range(7):
        torch.cat([a, b], dim=0)


def beta(a, b):
    """2 cats per call, but each moves ~16x the bytes of alpha's."""
    big_a, big_b = a.repeat(16, 1), b.repeat(16, 1)
    for _ in range(2):
        torch.cat([big_a, big_b], dim=0)


def workload():
    a = torch.ones(64, 256, device=DEV)
    b = torch.ones(64, 256, device=DEV)
    alpha(a, b)
    beta(a, b)


def test_callsites_are_separated():
    print("\n=== 1. two callsites for one operator are reported separately ===")
    rep = attribute(workload, watch=("cat",), repeats=2)
    rows = rep.by_operator().get("cat", [])
    check(len(rows) >= 2, f"cat resolved to {len(rows)} distinct callsites", "expected >= 2")
    sites = {r.callsite.split()[-1]: r for r in rows}
    check("alpha" in sites and "beta" in sites,
          "both source functions are named", ", ".join(sorted(sites)))
    if "alpha" in sites and "beta" in sites:
        check(sites["alpha"].calls / rep.cycles == 7,
              "alpha's call count is exact", f"{sites['alpha'].calls / rep.cycles:.0f} of 7")
        check(sites["beta"].calls / rep.cycles == 2,
              "beta's call count is exact", f"{sites['beta'].calls / rep.cycles:.0f} of 2")
        print("       => calls alone would rank alpha 3.5x above beta.")


def test_bytes_separate_frequency_from_volume():
    print("\n=== 2. bytes distinguish 'called often' from 'moves data' ===")
    rep = attribute(workload, watch=("cat",), repeats=2)
    sites = {r.callsite.split()[-1]: r for r in rep.by_operator().get("cat", [])}
    if "alpha" in sites and "beta" in sites:
        ra, rb = sites["alpha"], sites["beta"]
        check(rb.nbytes > ra.nbytes,
              "beta moves more bytes despite 3.5x fewer calls",
              f"alpha {ra.mib():.2f} MiB / {ra.calls} calls, beta {rb.mib():.2f} MiB / {rb.calls}")
        print("       => this is exactly the copy_ (1.9 us, launch-bound) vs cat (114 us,")
        print("          bandwidth-bound) distinction that operator totals hid.")


def test_shapes_are_recorded():
    print("\n=== 3. shapes per callsite, because a fusion target needs a stable shape ===")
    rep = attribute(workload, watch=("cat",), repeats=2)
    rows = rep.by_operator().get("cat", [])
    check(all(r.n_shapes() >= 1 for r in rows), "every row carries at least one shape")
    check(any(r.dominant_shape() for r in rows), "and a dominant shape",
          str(rows[0].dominant_shape()) if rows else "")


def test_coverage_is_computed_and_gates_ranking():
    print("\n=== 4. THE ONE THAT MATTERS: it knows when it cannot rank ===")
    rep = attribute(workload, watch=("cat",), repeats=2)
    cov = rep.coverage("cat")
    check(0.0 < cov <= 1.0 + 1e-9, f"coverage computed against the profiler's own count: {cov:.0%}",
          f"{rep.true_calls.get('cat', 0)} calls seen by the profiler")

    # Force the failure mode of attempt 4: a table built from a small unrepresentative sample.
    partial = Report(
        rows=[Row(operator="copy_", callsite="[lingbot] a.py:1 f", calls=900, nbytes=1 << 20,
                  exclusive_us=1000.0)],
        true_calls={"copy_": 34710}, true_us={"copy_": 66420.0})
    check(partial.coverage("copy_") < MIN_COVERAGE,
          f"a 900-of-34,710 sample reports {partial.coverage('copy_'):.1%} coverage")
    w = partial.coverage_warnings()
    check("not rankable" in w.lower(), "and is explicitly declared not rankable")
    check(not partial.rankable("copy_"), "rankable() agrees")

    # And the symmetric failure, which the first version of this module did NOT catch: coverage
    # ABOVE 100%. It cannot happen for a stationary workload, so it means the operator's call count
    # depends on state that advanced between the two passes. `cat` measured 121% for exactly that
    # reason -- the ring-wrap branch fires only during the wrap transition.
    nonstat = Report(
        rows=[Row(operator="cat", callsite="[iwm] ring_kv.py:198 forward", calls=60,
                  nbytes=1 << 20, exclusive_us=10210.0)],
        true_calls={"cat": 50}, true_us={"cat": 21850.0})
    check(not nonstat.rankable("cat"), f"{nonstat.coverage('cat'):.0%} coverage is NOT rankable",
          "over-attribution is a non-stationarity alarm, not a rounding artefact")
    check("NON-STATIONARY" in nonstat.coverage_warnings(), "and is labelled non-stationary")
    check("NON-STATIONARY, NOT RANKABLE" in nonstat.format_table(),
          "with the flag in the table too")
    check("PARTIAL, NOT RANKABLE" in partial.format_table(),
          "the table itself carries the flag, so a reader cannot miss it")
    print("       => attempt 4 would have printed '47.4% of watched calls' with no such flag.")
    print("          That is the specific mistake this check exists to prevent.")


def test_unwatched_operators_are_untouched():
    print("\n=== 5. watching is opt-in; nothing else is intercepted ===")
    rep = attribute(workload, watch=("cat",), repeats=1)
    check(set(rep.by_operator()) <= {"cat"}, "only the watched operator is reported",
          str(sorted(rep.by_operator())))


def main() -> int:
    print(f"device: {DEV}")
    test_callsites_are_separated()
    test_bytes_separate_frequency_from_volume()
    test_shapes_are_recorded()
    test_coverage_is_computed_and_gates_ranking()
    test_unwatched_operators_are_untouched()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: attribution is correct on a known distribution, and reports its own coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

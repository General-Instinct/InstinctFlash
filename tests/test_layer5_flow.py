#!/usr/bin/env python3
"""The Layer 5 flow is enforced, not requested.

LAYER5.md documents planner -> backend -> verification and says the tier's evidence must match. A
document cannot stop the next backend from claiming BITEXACT with no certificate, or from being
selected without a measurement. These checks can.

The specific failure being prevented: a lossy pass inheriting the credibility of six bit-exact ones.
P007 is the first non-BITEXACT release in the project, so this is the first moment the distinction has
teeth.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instinctwm.passes.contract import Tier  # noqa: E402
from instinctwm.verify.released import RELEASED, summary  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def test_every_release_has_evidence_matching_its_tier():
    print("\n=== 1. a non-BITEXACT release without a certificate is NOT verified ===")
    for r in RELEASED:
        if r.tier is Tier.BITEXACT:
            check(bool(r.gates), f"{r.pid} {r.name}: bit-exactness gates recorded")
        else:
            check(bool(r.certificate),
                  f"{r.pid} {r.name}: NUMERIC/BEHAVIORAL, so a paired certificate is REQUIRED",
                  f"{len(r.certificate)} chars")
            check(r.evidence_kind() == "paired non-inferiority",
                  f"{r.pid}: evidence kind is reported as paired non-inferiority")
    check(all(r.is_verified() for r in RELEASED), "every release is verified")


def test_the_refusal_actually_fires():
    print("\n=== 2. the refusal can fail -- otherwise it is decoration ===")
    from dataclasses import replace
    lossy = next((r for r in RELEASED if r.tier is not Tier.BITEXACT), None)
    check(lossy is not None, "there is at least one non-BITEXACT release to test with",
          lossy.pid if lossy else "none")
    if lossy is not None:
        stripped = replace(lossy, certificate="")
        check(not stripped.is_verified(),
              "stripping the certificate makes it UNVERIFIED",
              "a NUMERIC pass cannot be verified by bit-exactness gates it never ran")
        bitexact = next(r for r in RELEASED if r.tier is Tier.BITEXACT)
        check(replace(bitexact, certificate="").is_verified(),
              "while a BITEXACT pass needs no certificate", "its gates are the right evidence")


def test_summary_does_not_hide_the_tier():
    print("\n=== 3. the chain's tier is stated, not left to be inferred ===")
    s = summary()
    lossy = [r.pid for r in RELEASED if r.tier is not Tier.BITEXACT]
    if lossy:
        check("NOT bit-exact end to end" in s,
              "summary() warns the chain is no longer bit-exact end to end")
        check(all(p in s for p in lossy), "and names the pass responsible", ", ".join(lossy))
    check("paired non-inferiority" in s or not lossy,
          "evidence kind appears beside each pass, so NUMERIC cannot read as BITEXACT")


def test_conv_selection_requires_measurement_and_consent():
    print("\n=== 4. the planner will not leave BITEXACT on its own ===")
    from instinctwm.backends.conv import ConvBackendRegistry, ConvShape, MemoryLayout as L
    from instinctwm.backends.conv import register_declared
    from instinctwm.backends.conv.semantics import ConvSemantics as C
    r = ConvBackendRegistry()
    register_declared(r)
    kw = dict(semantics=C.CAUSAL_TIME,
              shape=ConvShape(160, 160, (3, 3, 3), spatial=(8, 128, 160)),
              have_layout=L.NCDHW, subgraph_size=62)
    check(r.select(**kw).tier is Tier.BITEXACT, "no measurement -> incumbent")
    m = {("torch_fallback", L.NCDHW): 175.72, ("cudnn_conv3d", L.NDHWC): 17.00}
    check(r.select(measured=m, **kw).tier is Tier.BITEXACT,
          "measurement showing a 10x win, default ceiling -> STILL the incumbent")
    p = r.select(measured=m, prefer_bitexact=False, **kw)
    check(p.tier is Tier.NUMERIC and p.convert_subgraph,
          "only explicit consent selects the NUMERIC pair", f"{p.backend_name}/{p.use_layout.value}")


def main() -> int:
    test_every_release_has_evidence_matching_its_tier()
    test_the_refusal_actually_fires()
    test_summary_does_not_hide_the_tier()
    test_conv_selection_requires_measurement_and_consent()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the Layer 5 flow is enforced -- evidence must match the tier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

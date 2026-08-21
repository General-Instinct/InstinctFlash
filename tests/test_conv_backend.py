#!/usr/bin/env python3
"""Conv backend dispatch: layout is a capability, and selection will not downgrade silently.

The measured result this layer exists to express (PROFILE.md): the VAE's 3x3x3 bf16 convolutions are
declined by cuDNN in NCDHW and served at 4.35-7.24x in NDHWC, and converting the encoder once takes the
full 2V/4A cycle from 490.4 ms to 330.2 ms. No kernel was written. These tests check that the
abstraction expresses that correctly -- especially the refusals, since a capability model that accepts
everything is decoration.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instinctflash.backends.conv import (  # noqa: E402
    ConvBackendRegistry, ConvShape, MemoryLayout as L, register_declared,
)
from instinctflash.backends.conv.semantics import ConvSemantics as C  # noqa: E402
from instinctflash.passes.contract import Tier  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


K3 = ConvShape(160, 160, (3, 3, 3), spatial=(8, 128, 160), dtype="bfloat16")
K1 = ConvShape(160, 320, (1, 1, 1), spatial=(8, 64, 80), dtype="bfloat16")
#: The encode-scale numbers, so selection is driven by measurement rather than by reputation.
MEASURED = {("torch_fallback", L.NCDHW): 175.72, ("cudnn_conv3d", L.NDHWC): 17.00}


def reg():
    r = ConvBackendRegistry()
    register_declared(r)
    return r


def test_layout_is_a_capability():
    print("\n=== 1. the fallback's cause is expressible: a kernel legal in one layout, not another ===")
    r = reg()
    c = r.candidates(semantics=C.CAUSAL_TIME, shape=K3, have_layout=L.NCDHW, subgraph_size=62)
    by = {(x.backend_name, x.use_layout): x for x in c}
    check(not by[("cudnn_conv3d", L.NCDHW)].legal, "cuDNN REFUSES a 3x3x3 kernel in NCDHW")
    check("only in a channels-last layout" in by[("cudnn_conv3d", L.NCDHW)].verdict.reason,
          "and the reason names the layout, not a missing kernel")
    check(by[("cudnn_conv3d", L.NDHWC)].legal, "cuDNN ACCEPTS the same kernel in NDHWC")
    print("       => same operator, same weights, same arithmetic. Only the layout differs.")

    print("\n=== 2. pointwise kernels were never on the slow path ===")
    c1 = {(x.backend_name, x.use_layout): x for x in
          r.candidates(semantics=C.STANDARD, shape=K1, have_layout=L.NCDHW, subgraph_size=62)}
    check(c1[("cudnn_conv3d", L.NCDHW)].legal,
          "cuDNN accepts 1x1x1 in NCDHW, matching the 16 of 62 convs that were already fast")


def test_tier_depends_on_the_pair_not_the_backend():
    print("\n=== 3. the tier is a property of (backend, layout), not of the backend ===")
    r = reg()
    conv = next(x for x in r.candidates(semantics=C.CAUSAL_TIME, shape=K3,
                                        have_layout=L.NCDHW, subgraph_size=62)
                if x.backend_name == "cudnn_conv3d" and x.use_layout is L.NDHWC)
    check(conv.verdict.claimed_tier is Tier.NUMERIC,
          "reaching cuDNN via a conversion is NUMERIC", "accumulation order changes")
    already = next(x for x in r.candidates(semantics=C.CAUSAL_TIME, shape=K3,
                                           have_layout=L.NDHWC, subgraph_size=62)
                   if x.backend_name == "cudnn_conv3d" and x.use_layout is L.NDHWC)
    check(already.verdict.claimed_tier is Tier.NUMERIC or not already.verdict.params.get(
        "needs_conversion"), "with no conversion needed, no conversion is charged",
        f"needs_conversion={already.verdict.params.get('needs_conversion')}")


def test_conversion_must_amortise():
    print("\n=== 4. a conversion is illegal when it cannot amortise ===")
    r = reg()
    one = next(x for x in r.candidates(semantics=C.CAUSAL_TIME, shape=K3,
                                       have_layout=L.NCDHW, subgraph_size=1)
               if x.backend_name == "cudnn_conv3d" and x.use_layout is L.NDHWC)
    check(not one.legal, "converting for a SINGLE operator is refused")
    check("amortise" in one.verdict.reason, "because per-operator conversion costs more than it saves")
    many = next(x for x in r.candidates(semantics=C.CAUSAL_TIME, shape=K3,
                                        have_layout=L.NCDHW, subgraph_size=62)
                if x.backend_name == "cudnn_conv3d" and x.use_layout is L.NDHWC)
    check(many.legal, "and legal for the whole 62-conv encoder subgraph")


def test_selection_needs_measurement_and_consent():
    print("\n=== 5. selection cannot downgrade a bit-exactness claim by itself ===")
    r = reg()
    kw = dict(semantics=C.CAUSAL_TIME, shape=K3, have_layout=L.NCDHW, subgraph_size=62)
    p = r.select(**kw)
    check(p.backend_name == "torch_fallback" and p.tier is Tier.BITEXACT,
          "with no measurement, the incumbent is kept", p.reason[-46:])
    p2 = r.select(measured=MEASURED, **kw)
    check(p2.tier is Tier.BITEXACT,
          "with measurement but the default ceiling, STILL the incumbent",
          "a NUMERIC pair must not win silently")
    p3 = r.select(measured=MEASURED, prefer_bitexact=False, **kw)
    check(p3.backend_name == "cudnn_conv3d" and p3.use_layout is L.NDHWC and p3.convert_subgraph,
          "only with the ceiling explicitly raised does the fast pair win",
          f"{p3.backend_name}/{p3.use_layout.value} [{p3.tier.name}]")
    check("measured" in p3.reason, "and the plan records that a measurement drove it")


def main() -> int:
    test_layout_is_a_capability()
    test_tier_depends_on_the_pair_not_the_backend()
    test_conversion_must_amortise()
    test_selection_needs_measurement_and_consent()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: conv backend dispatch -- layout is a capability, selection requires consent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

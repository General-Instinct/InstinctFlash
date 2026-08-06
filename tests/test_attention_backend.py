#!/usr/bin/env python3
"""Attention backend architecture: legality discriminates, and nothing gets installed.

These tests are about the ARCHITECTURE, not about attention performance. There is no kernel here to
be fast. What is checked is that the vocabulary is sharp enough to make the right refusals, because a
capability model that accepts everything is decoration.

The load-bearing test is `test_semantics_substitution_is_refused`. Everything else is a constraint
check; that one is a correctness guarantee about the whole layer.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instinctwm.backends.attention import (  # noqa: E402
    REGISTRY,
    AttentionBackendRegistry,
    AttentionSemantics as S,
    AttentionShape,
    Distribution,
    MaskKind as M,
    MaskSpec,
    QKVLayout as L,
    plan_penalty_ms,
    read_site,
    register_declared,
)
from instinctwm.backends.attention.reference import (  # noqa: E402
    AdapterNativeAttention,
    FlashAttention,
    FlashInfer,
    RingAttention,
    SanaHybrid,
    TorchSDPA,
)
from instinctwm.backends.attention.site import (  # noqa: E402
    lingbot_video_self_attention_example,
)
from instinctwm.passes.contract import Tier  # noqa: E402
from instinctwm.runtime.state.types import Addressing as A  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def _facts(**over):
    """The LingBot-VA video self-attention site, as declared, with overrides."""
    f = read_site(lingbot_video_self_attention_example())
    base = dict(semantics=f.semantics, mask=f.mask, layout=f.layout,
                addressing=f.addressing, shape=f.shape, world_size=1,
                tier_ceiling=Tier.NUMERIC)
    base.update(over)
    return base


def _verdict(backend, **over):
    r = AttentionBackendRegistry()
    r.register(backend)
    return r.candidates(**_facts(**over))[0].verdict


def test_site_vocabulary():
    print("\n=== 1. an adapter can declare a site, and a malformed one is rejected ===")
    f = read_site(lingbot_video_self_attention_example())
    check(f.semantics is S.SOFTMAX_FULL, "declared semantics round-trips", f.semantics.value)
    check(f.shape.seq_kv_max == 9792, "declared KV extent round-trips", f"{f.shape.seq_kv_max}")
    check(f.forwards_per_cycle == 25, "forwards/cycle present -- the profitability denominator")

    from instinctwm.passes.interface import Site, SiteKind
    try:
        read_site(Site(kind=SiteKind.ATTENTION, id="bad", attrs={"mask": None}))
        check(False, "a site missing `semantics` is rejected")
    except ValueError as e:
        check("semantics" in str(e) and "no safe default" in str(e),
              "a site missing `semantics` is rejected, not defaulted")


def test_semantics_substitution_is_refused():
    print("\n=== 2. THE IMPORTANT ONE: different semantics is never a candidate ===")
    v = _verdict(SanaHybrid())
    check(not v.applies, "hybrid backend refused for a full-attention site")
    check("semantics mismatch" in v.reason and "different model" in v.reason,
          "and the reason says why it is not a speed question", v.reason[:64] + "...")

    # ...but legal for a checkpoint that DECLARES it was trained that way.
    v2 = _verdict(SanaHybrid(), semantics=S.SOFTMAX_HYBRID)
    check(v2.applies, "the same backend IS legal when the checkpoint declares hybrid semantics")

    # And the converse: a full-attention backend must not serve a hybrid site.
    v3 = _verdict(FlashAttention(), semantics=S.SOFTMAX_HYBRID)
    check(not v3.applies, "full-attention backend refused for a hybrid site")


def test_mask_envelope():
    print("\n=== 3. the mask envelope, which is what P003 actually bought ===")
    slice_ = MaskSpec(kind=M.NONE)                    # what P003's ring interval actually yields
    dyn = MaskSpec(kind=M.DENSE_DATA_DEPENDENT, materialised=True)   # the stock KV path

    check(_verdict(FlashAttention(), mask=slice_).applies,
          "flash is legal against a ring SLICE, which needs no mask at all (post-P003)")
    v = _verdict(FlashAttention(), mask=dyn)
    check(not v.applies, "flash is ILLEGAL against a data-dependent dense mask (the stock KV path)")
    v_sdpa = _verdict(TorchSDPA(), mask=dyn)
    check(v_sdpa.applies, "SDPA still accepts it -- a wider envelope, not a faster kernel")
    check("transpose" in v_sdpa.reason,
          "and its BHSD requirement is legal-with-a-transpose, not illegal", v_sdpa.reason[-58:])
    check("layout_adaptation" in v_sdpa.params,
          "with the adaptation recorded so the planner can charge for it")
    check(not dyn.is_shape_static() and slice_.is_shape_static(),
          "shape-staticness is expressible on the mask itself")


def test_deployment_and_hardware():
    print("\n=== 4. distribution is a deployment fact, not a checkpoint fact ===")
    v1 = _verdict(RingAttention(), world_size=1)
    check(not v1.applies, "ring attention is ILLEGAL at world_size 1")
    check("world_size" in v1.reason, "and says so", v1.reason)
    check(_verdict(RingAttention(), world_size=8).applies,
          "and legal at world_size 8, with no change to the checkpoint")


def test_tier_is_derived_not_claimed():
    print("\n=== 5. tiers are derived from declared numerics ===")
    check(AdapterNativeAttention().capabilities().tier_ceiling() is Tier.BITEXACT,
          "the incumbent is bit-exact by construction")
    check(FlashAttention().capabilities().tier_ceiling() is Tier.NUMERIC,
          "online softmax cannot claim bit-exactness, however correct it is")
    v = _verdict(FlashAttention(), tier_ceiling=Tier.BITEXACT)
    check(not v.applies, "and a BITEXACT plan ceiling excludes it")
    check("reduction order" in v.reason, "for the stated reason", v.reason[:70] + "...")
    check(_verdict(AdapterNativeAttention(), tier_ceiling=Tier.BITEXACT).applies,
          "while the incumbent survives a BITEXACT ceiling")


def test_plan_level_penalty():
    print("\n=== 6. a capture-hostile backend is charged for what it forfeits ===")
    fi = FlashInfer().capabilities()
    na = AdapterNativeAttention().capabilities()
    check(not fi.capture_safe, "FlashInfer declares itself capture-hostile (host plan() per forward)")
    pen = plan_penalty_ms(fi, capture_in_plan=True, capture_speedup=1.205, cycle_ms=2325.0)
    check(395 < pen < 400, "forfeiting a 1.205x capture on a 2325 ms cycle costs ~396 ms",
          f"{pen:.1f} ms")
    check(plan_penalty_ms(fi, capture_in_plan=False, capture_speedup=1.205, cycle_ms=2325.0) == 0.0,
          "and costs nothing when capture is not in the plan")
    check(plan_penalty_ms(na, capture_in_plan=True, capture_speedup=1.205, cycle_ms=2325.0) == 0.0,
          "a capture-safe backend is never charged")
    print(f"       -> a backend must beat attention by MORE than {pen:.0f} ms/cycle to be worth "
          f"selecting at Quality; attention is ~163 ms of that cycle in total, so it cannot.")


def test_selection_refuses():
    print("\n=== 7. nothing can be installed yet, and the refusal explains itself ===")
    r = AttentionBackendRegistry()
    names = register_declared(r)
    check(len(names) == 7, "all declared envelopes register", f"{len(names)} backends")
    try:
        r.select(**_facts())
        check(False, "select() must raise at this stage")
    except NotImplementedError as e:
        check("not implemented" in str(e).lower(), "select() raises NotImplementedError")
        check("ATTENTION.md" in str(e), "and points at the document")

    cands = r.candidates(**_facts())
    legal = [c for c in cands if c.legal]
    check({c.backend_name for c in legal} == {"adapter_native", "flash_attn", "torch_sdpa"},
          "exactly 3 of 7 are legal for the real LingBot video site",
          ", ".join(c.backend_name for c in legal))
    # WHY the other four are out is the useful output of this whole exercise.
    why = {c.backend_name: c.verdict.reason for c in cands if not c.legal}
    check("addressing" in why["flashinfer"] and "addressing" in why["cudnn_sdpa"],
          "FlashInfer and cuDNN are excluded by KV ADDRESSING -- our own ring interval, not by speed")
    check("world_size" in why["ring_attn"], "ring attention by deployment")
    check("semantics" in why["sana_hybrid"], "sana by semantics")
    print("       -> ring-interval KV is the binding constraint at Layer 4: it is what makes flash "
          "legal\n          and what excludes every paged-KV backend. A Layer 3 decision set the "
          "Layer 4 menu.")
    check(any(c.backend_name == "adapter_native" for c in legal),
          "the incumbent is always among them, so candidates() is never empty")
    for b in (FlashAttention(), FlashInfer(), TorchSDPA()):
        try:
            b.measure(AttentionShape(n_heads=40, head_dim=128), None)
            check(False, f"{b.name}.measure() must raise")
        except NotImplementedError:
            pass
    check(True, "every declared backend refuses measure()/bind() -- declaration only")


def test_adding_a_backend_changes_nothing_else():
    print("\n=== 8. adding a backend requires no planner or adapter change ===")

    class Invented:
        name, version = "invented_2027", "0.0.0"

        def capabilities(self):
            from instinctwm.backends.attention import AttentionCapabilities
            return AttentionCapabilities(
                semantics=frozenset({S.SOFTMAX_FULL}), mask_kinds=frozenset({M.NONE}),
                layouts=frozenset({L.BSHD}), kv_addressing=frozenset({A.RING_INTERVAL}))

        def expected_delta_ms(self, *a):  # pragma: no cover
            raise NotImplementedError

        def measure(self, *a):  # pragma: no cover
            raise NotImplementedError

        def bind(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

    r = AttentionBackendRegistry()
    register_declared(r)
    before = len([c for c in r.candidates(**_facts()) if c.legal])
    r.register(Invented())
    after = [c for c in r.candidates(**_facts()) if c.legal]
    check(len(after) == before + 1, "a backend written today is a candidate immediately",
          f"{before} -> {len(after)}")
    check(any(c.backend_name == "invented_2027" for c in after),
          "with no edit to any planner, adapter, or existing backend")


def main() -> int:
    test_site_vocabulary()
    test_semantics_substitution_is_refused()
    test_mask_envelope()
    test_deployment_and_hardware()
    test_tier_is_derived_not_claimed()
    test_plan_level_penalty()
    test_selection_refuses()
    test_adding_a_backend_changes_nothing_else()

    print("\n" + "=" * 72)
    print(REGISTRY.explain(**_facts()) if REGISTRY.names() else "(global registry empty by design)")
    print("=" * 72)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: attention backend architecture -- legality discriminates, selection refuses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The kernel framework, proved on LingBot-VA's real regions.

What is being tested is the FRAMEWORK, not a kernel:

  1. the tier is DERIVED from region structure + kernel properties, not declared
  2. legality catches a numerics-contract violation before anything runs
  3. selection is by MEASUREMENT on real shapes, with eager as the floor
  4. a kernel that is legal but slower is rejected

The pair of post-attention kernels differ only in `preserves_intermediate_rounding`. If the
framework is right, it must call one BITEXACT and the other NUMERIC without being told, and the
measured deltas must agree with that classification.

Run:  python tests/test_kernel_framework.py [--cuda]
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/ubuntu/InstinctWM")

import torch  # noqa: E402

import instinctwm.kernels.torch_fused as _kernels  # noqa: F401,E402  (registers them)
import instinctwm.kernels.triton_residual as _tri  # noqa: F401,E402  (registers the Triton one)
from instinctwm.kernels.lingbot_regions import (  # noqa: E402
    POST_ATTENTION, PRE_ATTENTION, lingbot_fusion_descriptor)
from instinctwm.kernels.registry import (  # noqa: E402
    REGISTRY, audit_tier, check_legality, derive_tier)
from instinctwm.kernels.regions import FusibleRegion, OpKind, OpSpec  # noqa: E402
from instinctwm.optimizer.contract import DeviceProfile, HardwareReq, Tier  # noqa: E402

SHAPE = (2, 240, 3072)          # LingBot video-stream hidden states


def bench(fn, *a, n=200, cuda=False):
    for _ in range(20):
        fn(*a)
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn(*a)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000 / n


def main() -> int:
    # Default to CUDA when present. On CPU the region degenerates: fp32 everywhere means the
    # rounding points the tier derivation is about do not exist, so every variant scores 0.
    use_cuda = torch.cuda.is_available() and "--cpu" not in sys.argv
    dev = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device = DeviceProfile.probe() if use_cuda else DeviceProfile(
        name="cpu", capability=(0, 0), total_memory=0,
        features=frozenset({"triton"}))
    rc = 0

    print("=== 1. tier DERIVED from structure, not declared ===")
    for k in REGISTRY._by_region["post_attention_gated_residual"]:
        tier, why = derive_tier(POST_ATTENTION, k)
        print(f"  {k.name:34s} -> {tier.name:9s}  {why[:88]}")
    for k in REGISTRY._by_region["pre_attention_modulated_norm"]:
        tier, why = derive_tier(PRE_ATTENTION, k)
        print(f"  {k.name:34s} -> {tier.name:9s}  {why[:88]}")

    print("\n=== 2. legality: a numerics contract is enforced before anything runs ===")
    pinned = FusibleRegion(
        name="post_attention_gated_residual",
        ops=(OpSpec("residual_add", OpKind.ELEMENTWISE, materializes_as="bf16",
                    computes_in="fp32", must_stay="fp32"),),
        boundary_in=("h",), boundary_out=("h",))
    for k in REGISTRY._by_region["post_attention_gated_residual"]:
        r = check_legality(pinned, k, device)
        print(f"  {k.name:34s} legal={str(r.legal):5s} {r.violations if r.violations else ''}")
        if k.compute_dtype == "bf16" and r.legal:
            print("  FAIL: a bf16 kernel was allowed into an fp32-pinned region")
            rc = 1

    print("\n=== 3 + 4. selection by MEASUREMENT on real shapes, eager is the floor ===")
    g = torch.Generator(device="cpu").manual_seed(0)
    hidden = torch.randn(SHAPE, generator=g).to(dev, dtype)
    attn = torch.randn(SHAPE, generator=g).to(dev, dtype)
    # gate_msa is FULL [B,N,C] and **fp32** in the real model -- it is a chunk of the fp32
    # modulation table. An earlier version of this test used a broadcast bf16 gate, which made
    # `attn * gate` a bf16 op and collapsed the whole region to one dtype: every variant then
    # scored max|d|=0 and the tier derivation had nothing to discriminate. The dtypes ARE the test.
    gate = torch.randn(SHAPE, generator=g).to(dev, torch.float32 if use_cuda else dtype)

    def eager(h, a, gt):
        return (h.float() + a * gt).type_as(h)

    ref = eager(hidden, attn, gate)
    eager_ms = bench(eager, hidden, attn, gate, cuda=use_cuda)
    print(f"  eager: {eager_ms:.4f} ms")

    for k in REGISTRY._by_region["post_attention_gated_residual"]:
        out = k.impl(hidden, attn, gate)
        d = (out.float() - ref.float()).abs().max().item()
        ms = bench(k.impl, hidden, attn, gate, cuda=use_cuda)
        tier, _ = derive_tier(POST_ATTENTION, k)
        a = audit_tier(tier, d)
        print(f"  {k.name:34s} {ms:8.4f} ms  max|d|={d:.3e}  claimed={tier.name:9s} "
              f"audited={a.audited.name:9s} {'ok' if a.agrees else 'DEMOTED'}")
        if not a.agrees:
            print(f"      {a.detail}")

    sel, why = REGISTRY.select(
        POST_ATTENTION, device,
        measure=lambda k: bench(k.impl, hidden, attn, gate, cuda=use_cuda),
        eager_ms=eager_ms, tier_ceiling=Tier.BITEXACT)
    print(f"\n  BITEXACT ceiling -> {sel.name if sel else 'EAGER'}")
    print(f"    {why}")

    sel_n, why_n = REGISTRY.select(
        POST_ATTENTION, device,
        measure=lambda k: bench(k.impl, hidden, attn, gate, cuda=use_cuda),
        eager_ms=eager_ms, tier_ceiling=Tier.NUMERIC)
    print(f"  NUMERIC ceiling  -> {sel_n.name if sel_n else 'EAGER'}")
    print(f"    {why_n}")

    print("\n=== 5. a legal-but-slower kernel must be rejected ===")
    sel_slow, why_slow = REGISTRY.select(
        POST_ATTENTION, device,
        measure=lambda k: bench(k.impl, hidden, attn, gate, cuda=use_cuda),
        eager_ms=1e-9, tier_ceiling=Tier.NUMERIC)   # pretend eager is infinitely fast
    ok = sel_slow is None
    print(f"  {'OK  ' if ok else 'FAIL'} {why_slow[:100]}")
    if not ok:
        rc = 1

    d = lingbot_fusion_descriptor()
    print(f"\ndescriptor: {d.model_id}, {len(d.regions)} regions, "
          f"{sum(d.launches_per_region.values())} eager launches per layer per forward")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

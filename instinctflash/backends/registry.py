"""Kernel registration, legality, and hardware selection.

A custom Triton or CUDA kernel plugs in by registering against a region name. It never names a
model. That is what makes the layer a framework rather than a collection of patches: the same
`pre_attention` kernel serves any adapter whose declared region matches its signature.

Three things happen before a kernel is allowed to run:

  1. LEGALITY   -- structural, numerics, and effect checks against the declared region
  2. TIER       -- DERIVED from the ops and the kernel's own properties, never taken on trust
  3. SELECTION  -- capability filter, then measured cost on the target's real shapes

Selection is measured rather than predicted because a plausible kernel can be slower: on pi-0's
real shapes, swapping eager attention for SDPA while keeping the mask measures 133.5 -> 144-184 us.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from instinctflash.backends.regions import FusibleRegion, OpKind
from instinctflash.passes.contract import DeviceProfile, HardwareReq, Tier


@dataclass(frozen=True)
class KernelVariant:
    """One implementation of one region."""

    name: str
    region: str
    hardware: HardwareReq
    impl: Callable
    #: Does the kernel round every intermediate the eager chain materialised? If False, it skips
    #: roundings and CANNOT be bit-exact, however elementwise it is. See regions.py.
    preserves_intermediate_rounding: bool = False
    #: Does it preserve reduction order exactly (tree shape, accumulation dtype)?
    preserves_reduction_order: bool = False
    #: Does the kernel's FP CONTRACTION match the reference's? A backend that contracts `a + b*c`
    #: into one FMA skips the fp32 rounding of the product. This is invisible in source and is
    #: decided by the compiler, so it must be asserted at the instruction level, not declared.
    #:
    #: The rule is NOT "disable FMA" -- it is "match the reference". Measured on H100 / Triton
    #: 3.5.0, where eager PyTorch is:
    #:
    #:   * SEVERAL kernels (gated residual = mul then add) -> each rounds to fp32 at the kernel
    #:     boundary, so the fused kernel must NOT contract. enable_fp_fusion=False -> bit-exact.
    #:   * ONE kernel (F.layer_norm) -> it contracts internally, so a fused kernel that refuses to
    #:     contract diverges MORE, not less: 490,810 differing elements vs 311,163 with fusion on.
    #:
    #: So contraction policy is a property of the region's eager decomposition, not a global flag.
    matches_reference_contraction: bool = False
    #: Compute dtype, if the kernel widens or narrows relative to eager.
    compute_dtype: str = "fp32"
    note: str = ""


def derive_tier(region: FusibleRegion, k: KernelVariant) -> tuple[Tier, str]:
    """Compute the equivalence tier from structure. Not a declaration to be trusted.

    The rules, in order of severity:

      * an EFFECTFUL op in the region means fusion reorders a side effect -> BEHAVIORAL
      * a REDUCTION whose order the kernel does not preserve -> NUMERIC
      * materialisation points the kernel does not reproduce -> NUMERIC (the rounding argument)
      * FP CONTRACTION that does not match the reference -> NUMERIC
      * otherwise -> BITEXACT

    The contraction rule was added after a kernel passed every other check and was still wrong: a
    silent `fma.rn.f32` cost one fp32 ULP on 19% of elements, surviving the bf16 round on 33 of
    1.47M. Structure alone cannot detect it -- the source is identical either way -- so the flag
    must be backed by a PTX assertion. See `tests/test_triton_residual.py:test_ptx`.
    """
    if region.has_effects():
        return Tier.BEHAVIORAL, (
            f"region {region.name!r} contains an effectful op; fusing reorders a side effect, "
            f"which changes behaviour rather than only numerics")
    if region.has_reduction() and not k.preserves_reduction_order:
        return Tier.NUMERIC, (
            f"region contains a reduction and {k.name!r} does not declare "
            f"preserves_reduction_order; a different tree shape changes the sum")
    if not k.matches_reference_contraction:
        return Tier.NUMERIC, (
            f"{k.name!r} does not declare matches_reference_contraction; the backend is free to "
            f"contract a multiply-add and skip an intermediate fp32 rounding the reference "
            f"performs. Assert on emitted PTX, then set the flag")
    rp = region.rounding_points()
    if rp and not k.preserves_intermediate_rounding:
        return Tier.NUMERIC, (
            f"eager materialises {len(rp)} intermediate(s) {list(rp)[:4]}, each of which ROUNDS to "
            f"storage dtype; {k.name!r} does not reproduce those roundings, so it computes a "
            f"different (usually better) answer")
    return Tier.BITEXACT, (
        f"elementwise chain with {len(rp)} rounding point(s), all reproduced; no reduction "
        f"reordering; no effects")


@dataclass
class TierAudit:
    """A DERIVED tier checked against a MEASURED delta. This is not optional.

    `preserves_intermediate_rounding` is a claim, and a kernel author cannot make it true by
    writing careful Python. Measured on this box: the expression

        prod = (attn_out * gate).to(hidden.dtype)
        return (hidden.float() + prod.float()).type_as(hidden)

    is bit-exact against eager when run eagerly, and NOT bit-exact when run under
    `torch.compile` -- inductor elides the `.to(bf16).float()` round-trip as a redundant cast
    pair, which is algebraically true and numerically false in bf16 (max|d| = 6.25e-02).

    So every kernel claiming BITEXACT is audited against a real delta before the tier is trusted,
    and a claim that fails is DEMOTED rather than reported.
    """

    claimed: Tier
    measured_delta: float
    audited: Tier
    agrees: bool
    detail: str


def audit_tier(claimed: Tier, measured_delta: float) -> TierAudit:
    if claimed is Tier.BITEXACT and measured_delta != 0.0:
        return TierAudit(claimed, measured_delta, Tier.NUMERIC, False,
                         f"claimed BITEXACT but measured max|delta| = {measured_delta:.3e}; "
                         f"DEMOTED to NUMERIC. A rounding the kernel claims to reproduce is being "
                         f"elided -- check whether a compiler removed a cast round-trip.")
    if claimed is not Tier.BITEXACT and measured_delta == 0.0:
        return TierAudit(claimed, measured_delta, claimed, True,
                         f"claimed {claimed.name} and measured 0; kept at {claimed.name} "
                         f"(exactness on one input is not a proof of exactness)")
    return TierAudit(claimed, measured_delta, claimed, True,
                     f"claimed {claimed.name}, measured max|delta| = {measured_delta:.3e}")


@dataclass
class LegalityResult:
    legal: bool
    tier: Tier
    reason: str
    violations: list[str] = field(default_factory=list)


def check_legality(region: FusibleRegion, k: KernelVariant,
                   device: DeviceProfile) -> LegalityResult:
    v: list[str] = []

    if k.region != region.name:
        v.append(f"kernel registered for region {k.region!r}, not {region.name!r}")

    ok_hw, why_hw = k.hardware.satisfied_by(device)
    if not ok_hw:
        v.append(f"hardware: {why_hw}")

    # A pinned dtype is a contract the kernel must honour. pi-0's fp32 keep-list is the live
    # example, and getting it wrong produces silently wrong actions rather than a crash.
    pinned = region.pinned_dtypes()
    for op_name, dt in pinned.items():
        if k.compute_dtype != dt:
            v.append(f"op {op_name!r} must stay {dt}, kernel computes in {k.compute_dtype}")

    tier, why = derive_tier(region, k)
    return LegalityResult(legal=not v, tier=tier, reason=why, violations=v)


class KernelRegistry:
    """Region name -> candidate kernels. Populated by @register_kernel."""

    def __init__(self):
        self._by_region: dict[str, list[KernelVariant]] = {}

    def add(self, k: KernelVariant) -> None:
        self._by_region.setdefault(k.region, []).append(k)

    def candidates(self, region: FusibleRegion, device: DeviceProfile,
                   tier_ceiling: Tier = Tier.BITEXACT) -> list[tuple[KernelVariant, LegalityResult]]:
        """Legal kernels for this region on this device, at or below the tier ceiling.

        Deliberately returns a LIST, ordered but not chosen: the choice is made by MEASUREMENT
        (`select`), because a legal kernel can still be slower than eager.
        """
        out = []
        for k in self._by_region.get(region.name, []):
            r = check_legality(region, k, device)
            if r.legal and r.tier <= tier_ceiling:
                out.append((k, r))
        return out

    def select(self, region: FusibleRegion, device: DeviceProfile,
               measure: Callable[[KernelVariant], float],
               eager_ms: float, tier_ceiling: Tier = Tier.BITEXACT
               ) -> tuple[KernelVariant | None, str]:
        """Pick by measured time on the target's real shapes. Eager is always the fallback."""
        cands = self.candidates(region, device, tier_ceiling)
        if not cands:
            return None, f"no legal kernel for {region.name!r} at tier <= {tier_ceiling.name}"
        scored = []
        for k, r in cands:
            try:
                ms = measure(k)
            except Exception as e:                     # a kernel that cannot run loses, loudly
                scored.append((float("inf"), k, r, f"raised {type(e).__name__}: {e}"))
                continue
            scored.append((ms, k, r, f"{ms:.3f} ms"))
        scored.sort(key=lambda t: t[0])
        best_ms, best, res, detail = scored[0]
        if best_ms >= eager_ms:
            return None, (f"best candidate {best.name!r} at {best_ms:.3f} ms does not beat eager "
                          f"at {eager_ms:.3f} ms; keeping eager")
        return best, (f"{best.name} [{res.tier.name}] {detail} vs eager {eager_ms:.3f} ms "
                      f"({eager_ms/best_ms:.2f}x)")


REGISTRY = KernelRegistry()


def register_kernel(*, region: str, hardware: HardwareReq | None = None,
                    preserves_intermediate_rounding: bool = False,
                    preserves_reduction_order: bool = False,
                    matches_reference_contraction: bool = False,
                    compute_dtype: str = "fp32", note: str = ""):
    """Decorator. A kernel names a REGION, never a model."""

    def deco(fn):
        REGISTRY.add(KernelVariant(
            name=fn.__name__, region=region, hardware=hardware or HardwareReq(), impl=fn,
            preserves_intermediate_rounding=preserves_intermediate_rounding,
            preserves_reduction_order=preserves_reduction_order,
            matches_reference_contraction=matches_reference_contraction,
            compute_dtype=compute_dtype, note=note))
        return fn

    return deco

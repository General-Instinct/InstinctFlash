"""What a backend declares about itself, and the pure predicate that checks it against a site.

A backend states capabilities the way an adapter states facts: things it can defend, with no
reference to any model. `legality()` is then a total function of (site, capabilities, deployment,
device) with no GPU access and no imports of the backend's implementation -- so "which backends could
serve this checkpoint" is answerable on a laptop, before any weights exist.

WHY LEGALITY IS SEPARATE FROM PROFITABILITY, again

Because they fail independently, and the attention layer is where that bites hardest. A backend can
be perfectly legal and still a regression: on pi-0's real shapes, swapping eager attention for SDPA
while keeping the mask measures 133.5 -> 144-184 us. Legality is a predicate and is cheap;
profitability is a measurement and must be paid for. Anything that ranks backends by reputation is
guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from instinctwm.backends.attention.semantics import (
    AttentionSemantics,
    AttentionShape,
    Distribution,
    MaskKind,
    MaskSpec,
    QKVLayout,
)
from instinctwm.passes.contract import Applicability, CostTerm, Discovery, HardwareReq, Tier
from instinctwm.runtime.state.types import Addressing


@dataclass(frozen=True)
class AttentionCapabilities:
    """One backend's declared envelope.

    Every field exists because some real backend would be misselected without it. Fields are added
    here only when a legality check reads one -- the same discipline `DeploymentSpec` follows.
    """

    #: WHICH FUNCTIONS this backend implements. Selection intersects this with the site's declared
    #: semantics; an empty intersection is an immediate, structural refusal.
    semantics: frozenset[AttentionSemantics]

    mask_kinds: frozenset[MaskKind]
    layouts: frozenset[QKVLayout]
    kv_addressing: frozenset[Addressing]
    dtypes: frozenset[str] = frozenset({"bfloat16", "float16"})

    #: Head dim envelope. FlashAttention-family kernels are compiled per head dim and typically cap
    #: at 256 with multiple-of-8 strides; declaring it beats discovering it in a kernel assert.
    head_dim_max: int = 256
    head_dim_multiple_of: int = 1

    supports_gqa: bool = False
    supports_varlen: bool = False
    supports_bias: bool = False
    #: Will it accept Q/K/V in a layout it does not natively want, given a transpose first?
    #: Nearly always yes between BSHD and BHSD -- they differ by one permute. Modelling a layout
    #: mismatch as ILLEGAL would have excluded torch SDPA (BHSD) from every BSHD site, which is
    #: wrong; modelling it as free would hide a per-call copy of three tensors. So it is legal with a
    #: cost, and the cost is declared here rather than discovered in a profile.
    accepts_layout_adaptation: bool = True
    layout_adaptation_us: float = 0.0

    hardware: HardwareReq = field(default_factory=HardwareReq)
    distribution: frozenset[Distribution] = frozenset({Distribution.LOCAL})
    #: RING/ULYSSES need ranks to distribute over; below this world_size they are illegal, not slow.
    min_world_size: int = 1

    # ---- numerics, which DERIVE the tier rather than claiming it -------------------------------
    #: Does it accumulate in the same order as the reference? FlashAttention's online softmax does
    #: not: the answer is correct and the bits differ. Bit-exactness is therefore unavailable, and a
    #: plan containing this backend cannot claim BITEXACT no matter what sits beside it.
    preserves_reduction_order: bool = False
    #: Same kernel as the adapter's own path, so substitution is the identity.
    is_identity_substitution: bool = False
    #: Run-to-run reproducible on fixed inputs. A non-deterministic backend cannot be gated by
    #: `max|delta| = 0` at all, so it needs the paired non-inferiority regime instead.
    deterministic: bool = True

    # ---- cross-layer interaction, which is where the real traps are ---------------------------
    #: Safe to call inside a captured CUDA graph: no host synchronisation, no data-dependent shapes,
    #: no allocation. A backend that is capture-hostile does not merely fail to help -- it forfeits
    #: graph capture, measured at 1.205x on the Quality profile. See `PLAN_INTERACTIONS` in
    #: ATTENTION.md.
    capture_safe: bool = True
    #: Per-forward host-side setup cost (FlashInfer's `plan()`, mask building). Charged to the FIXED
    #: term, so it is what makes a fast kernel lose at a low forward count.
    host_setup_us: float = 0.0
    #: Extra device memory as a multiple of the KV footprint (workspace, paged block tables).
    workspace_bytes: int = 0

    def tier_ceiling(self) -> Tier:
        """The best correctness claim this backend can support. DERIVED, never taken on trust.

        Mirrors `backends/registry.derive_tier`: bit-exactness requires that nothing about the
        arithmetic changed, and a different reduction order is a change even when the result is
        better.
        """
        if self.is_identity_substitution:
            return Tier.BITEXACT
        if not self.deterministic:
            return Tier.BEHAVIORAL
        if self.preserves_reduction_order:
            return Tier.BITEXACT
        return Tier.NUMERIC


def legality(
    *,
    caps: AttentionCapabilities,
    semantics: AttentionSemantics,
    mask: MaskSpec,
    layout: QKVLayout,
    addressing: Addressing,
    shape: AttentionShape,
    world_size: int = 1,
    device=None,
    tier_ceiling: Tier = Tier.NUMERIC,
) -> Applicability:
    """Is this backend allowed to serve this site? Pure, no GPU, no torch.

    Returns the first failing reason rather than a list, because the first failure is the one worth
    reporting and a plan explanation that prints six reasons is read by nobody.
    """
    # 1. SEMANTICS. The check that makes this layer safe rather than merely flexible.
    if semantics not in caps.semantics:
        return Applicability(
            False,
            f"semantics mismatch: site computes {semantics.value}, backend implements "
            f"{sorted(s.value for s in caps.semantics)}. This is not a performance question -- "
            f"substituting a different function would silently serve a different model.",
            discovery=Discovery.DECLARED)

    # 2. TIER CEILING. Checked before anything expensive: if the caller demanded BITEXACT, a backend
    #    with a different reduction order is out regardless of how legal the rest of it is.
    achievable = caps.tier_ceiling()
    if achievable > tier_ceiling:
        return Applicability(
            False,
            f"tier: backend can only support {achievable.name}, plan ceiling is "
            f"{tier_ceiling.name}"
            + ("" if caps.preserves_reduction_order else
               " (its reduction order differs from the reference, so max|delta| = 0 is unavailable)"),
            discovery=Discovery.DECLARED)

    # 3. STRUCTURE.
    if mask.kind not in caps.mask_kinds:
        return Applicability(False, f"mask {mask.kind.value} unsupported "
                                    f"(backend takes {sorted(m.value for m in caps.mask_kinds)})")
    if mask.kind is MaskKind.SLIDING and mask.window is None:
        return Applicability(False, "site declares a sliding mask without a window width")
    if layout is QKVLayout.PACKED_VARLEN and not caps.supports_varlen:
        return Applicability(False, "packed layout requires varlen support")
    adaptation = None
    if layout not in caps.layouts:
        # PACKED_VARLEN is not reachable by a permute -- it needs cu_seqlens the site may not have --
        # so adaptation is only offered between the two strided layouts.
        strided = {QKVLayout.BSHD, QKVLayout.BHSD}
        target = next((l for l in caps.layouts if l in strided), None)
        if not (caps.accepts_layout_adaptation and layout in strided and target is not None):
            return Applicability(False, f"layout {layout.value} unsupported and not adaptable "
                                        f"(backend takes {sorted(l.value for l in caps.layouts)})")
        adaptation = (layout, target)
    if addressing not in caps.kv_addressing:
        return Applicability(False, f"KV addressing {addressing.value} unsupported "
                                    f"(backend takes {sorted(a.value for a in caps.kv_addressing)})")

    # 4. SHAPES.
    if shape.dtype not in caps.dtypes:
        return Applicability(False, f"dtype {shape.dtype} unsupported")
    if shape.head_dim > caps.head_dim_max:
        return Applicability(False, f"head_dim {shape.head_dim} > {caps.head_dim_max}")
    if shape.head_dim % caps.head_dim_multiple_of:
        return Applicability(False, f"head_dim {shape.head_dim} is not a multiple of "
                                    f"{caps.head_dim_multiple_of}")
    if shape.is_gqa() and not caps.supports_gqa:
        return Applicability(False, f"site is GQA ({shape.n_kv_heads} KV heads for "
                                    f"{shape.n_heads} heads) and backend has no GQA path")

    # 5. DEPLOYMENT. A distributed backend below its rank floor is illegal, not slow -- the same
    #    shape of guard as FSDPElision reading world_size.
    if Distribution.LOCAL not in caps.distribution and world_size < caps.min_world_size:
        return Applicability(False, f"needs world_size >= {caps.min_world_size}, have {world_size}")

    # 6. HARDWARE.
    if device is not None:
        ok, why = caps.hardware.satisfied_by(device)
        if not ok:
            return Applicability(False, f"hardware: {why}")

    note = ""
    if adaptation is not None:
        note = (f"; needs a {adaptation[0].value}->{adaptation[1].value} transpose of Q/K/V, "
                f"{caps.layout_adaptation_us:.0f} us/call as declared")
    return Applicability(
        True,
        f"legal at {achievable.name}" + note
        + ("" if caps.capture_safe else "; NOT capture-safe, so the plan must charge it the cost of "
                                       "forfeiting graph capture"),
        discovery=Discovery.DECLARED,
        cost_term=CostTerm.PER_STEP,
        claimed_tier=achievable,
        params={} if adaptation is None else {"layout_adaptation": adaptation},
    )

"""What a conv backend declares, and the pure predicate that selects over (backend x layout).

THE DIFFERENCE FROM THE ATTENTION LAYER, and it is the point of this module.

An attention backend either accepts a site's layout or needs a transpose worth a few microseconds.
A conv backend's layout requirement IS the optimization: the operator is legal for cuDNN in NDHWC and
illegal for it in NCDHW, so the interesting question is not "which backend" but "which (backend,
layout) pair, and is the conversion worth it". `legality()` therefore returns a verdict per pair, and
`best_pair()` ranks them with the conversion charged explicitly.

WHY A CONVERSION CAN BE FREE, AND WHY IT USUALLY IS NOT. A layout conversion is a copy. Converting per
operator costs more than the faster kernel saves — that is the default outcome and the reason this
looks like a bad idea on paper. It pays only when the layout PROPAGATES: convert the weights and the
subgraph input once, and every intermediate stays converted, so one copy is amortised over all 62
convolutions in the VAE encoder. Measured that way the encode goes 175.72 -> 17.00 ms. Measured
per-operator it would lose. `amortises_over` is how a backend states which regime it needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from instinctwm.backends.conv.semantics import ConvSemantics, ConvShape, MemoryLayout
from instinctwm.passes.contract import Applicability, CostTerm, Discovery, HardwareReq, Tier


@dataclass(frozen=True)
class ConvCapabilities:
    """One conv backend's declared envelope."""

    #: WHICH functions this backend implements.
    semantics: frozenset[ConvSemantics]
    #: WHICH LAYOUTS it accepts. The axis this whole layer exists to expose.
    layouts: frozenset[MemoryLayout]
    dtypes: frozenset[str] = frozenset({"bfloat16", "float16", "float32"})

    spatial_ranks: frozenset[int] = frozenset({2, 3})
    supports_dilation: bool = True
    supports_groups: bool = True
    #: Kernel extents this backend will take. Empty means "any".
    kernels: frozenset[tuple[int, ...]] = frozenset()
    #: True when the backend declines non-pointwise kernels in a layout it does not prefer. This is
    #: the observed cuDNN behaviour on 3D bf16: 1x1x1 is served in either layout, 3x3x3 only in NDHWC.
    pointwise_only_off_preferred_layout: bool = False

    hardware: HardwareReq = field(default_factory=HardwareReq)

    # ---- numerics, which DERIVE the tier ------------------------------------------------------
    #: Same kernel and same accumulation order as the reference path.
    is_reference_path: bool = False
    #: Does serving this backend in a non-native layout change the accumulation order? For cuDNN
    #: NDHWC vs the NCDHW fallback the answer is yes, measured: max|delta| 1.25e-01 on the VAE
    #: encoder output, relative 6.67e-03, ~1.7x bf16 resolution at that magnitude.
    layout_changes_reduction_order: bool = True
    deterministic: bool = True

    # ---- cost structure ----------------------------------------------------------------------
    #: Over how many operators a single layout conversion must amortise before it pays. 1 means the
    #: conversion is cheap enough to do per call; a large number means the backend is only viable if
    #: an entire subgraph is converted together. The VAE encoder has 62 convolutions.
    amortises_over: int = 1
    #: Measured cost of one conversion of a typical activation, in ms. Declared, then verified.
    conversion_ms: float = 0.0
    capture_safe: bool = True

    def tier_ceiling(self, *, needs_conversion: bool) -> Tier:
        """Best claim this backend supports, given whether a conversion is required.

        DERIVED, and it depends on the pair rather than on the backend alone: the same cuDNN backend is
        BITEXACT when the tensor already arrives in a layout it accepts, and NUMERIC when reaching it
        required a layout change. That is exactly why layout belongs in the capability model — a
        backend-only tier would have been wrong in one of the two cases.
        """
        if not self.deterministic:
            return Tier.BEHAVIORAL
        if self.is_reference_path and not needs_conversion:
            return Tier.BITEXACT
        if needs_conversion and self.layout_changes_reduction_order:
            return Tier.NUMERIC
        return Tier.BITEXACT if self.is_reference_path else Tier.NUMERIC


def legality(*, caps: ConvCapabilities, semantics: ConvSemantics, shape: ConvShape,
             have_layout: MemoryLayout, use_layout: MemoryLayout,
             device=None, tier_ceiling: Tier = Tier.NUMERIC,
             subgraph_size: int = 1) -> Applicability:
    """Is this (backend, layout) pair legal for this operator? Pure, no GPU, no torch.

    `have_layout` is what the caller holds; `use_layout` is what we would serve in. They differ exactly
    when a conversion is proposed, and that difference is what drives both the tier and the cost.
    """
    if semantics not in caps.semantics:
        return Applicability(
            False, f"semantics mismatch: operator is {semantics.value}, backend implements "
                   f"{sorted(s.value for s in caps.semantics)}",
            discovery=Discovery.DECLARED)

    if use_layout not in caps.layouts:
        return Applicability(False, f"backend does not accept {use_layout.value} "
                                    f"(takes {sorted(l.value for l in caps.layouts)})")
    if use_layout.rank() != shape.spatial_rank() + 2:
        return Applicability(False, f"{use_layout.value} is rank {use_layout.rank()}, operator is "
                                    f"{shape.spatial_rank()}D")
    if shape.spatial_rank() not in caps.spatial_ranks:
        return Applicability(False, f"{shape.spatial_rank()}D unsupported")
    if shape.dtype not in caps.dtypes:
        return Applicability(False, f"dtype {shape.dtype} unsupported")
    if shape.is_dilated() and not caps.supports_dilation:
        return Applicability(False, f"dilation {shape.dilation} unsupported")
    if shape.groups != 1 and not caps.supports_groups:
        return Applicability(False, f"groups={shape.groups} unsupported")
    if caps.kernels and tuple(shape.kernel) not in caps.kernels:
        return Applicability(False, f"kernel {tuple(shape.kernel)} unsupported")

    # THE OBSERVED cuDNN 3D BEHAVIOUR, declared rather than discovered at runtime: pointwise kernels
    # are served in either layout, non-pointwise ones only in the preferred (channels-last) layout.
    # This is what made 16 of 62 VAE convolutions fast already while 46 fell back.
    needs_conversion = have_layout is not use_layout
    if (caps.pointwise_only_off_preferred_layout and not use_layout.is_channels_last()
            and not shape.is_pointwise()):
        return Applicability(
            False,
            f"backend declines a {tuple(shape.kernel)} kernel in {use_layout.value}; it serves "
            f"non-pointwise kernels only in a channels-last layout. This is the fallback's actual "
            f"cause: nothing about the model needs to change, only the layout it is called in.")

    achievable = caps.tier_ceiling(needs_conversion=needs_conversion)
    if achievable > tier_ceiling:
        return Applicability(
            False,
            f"tier: pair supports {achievable.name}, plan ceiling is {tier_ceiling.name}"
            + (" (a layout conversion changes the convolution's accumulation order, so max|delta| = 0 "
               "is unavailable)" if needs_conversion else ""),
            discovery=Discovery.DECLARED)

    if device is not None:
        ok, why = caps.hardware.satisfied_by(device)
        if not ok:
            return Applicability(False, f"hardware: {why}")

    if needs_conversion and subgraph_size < caps.amortises_over:
        return Applicability(
            False,
            f"conversion needs to amortise over >= {caps.amortises_over} operators, but only "
            f"{subgraph_size} would be converted. Converting per operator costs more than the faster "
            f"kernel saves -- this pair is legal only for a whole converted subgraph.")

    note = ""
    if needs_conversion:
        note = (f"; requires {have_layout.value} -> {use_layout.value} conversion, amortised over "
                f"{subgraph_size} operators")
    return Applicability(
        True, f"legal at {achievable.name}" + note,
        discovery=Discovery.DECLARED, cost_term=CostTerm.PER_STEP, claimed_tier=achievable,
        params={"use_layout": use_layout, "needs_conversion": needs_conversion,
                "subgraph_size": subgraph_size},
    )

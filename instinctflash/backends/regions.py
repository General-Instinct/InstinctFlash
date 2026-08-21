"""L1 kernel layer — how a Backend Adapter describes a fusible region.

The fact this design is built around
------------------------------------
**Fusing elementwise ops is NOT bit-exact by default.** This is the thing everyone gets wrong.

In an unfused bf16 chain, every intermediate is *materialised* — and materialising in bf16 rounds
it. `y = (a + b) * c` executed as two kernels computes `round_bf16(a + b)` and then multiplies. A
fused kernel that keeps `a + b` in an fp32 register and multiplies from there skips a rounding and
produces a *different* answer. It is usually a better answer. It is not the same answer.

So a fused kernel is bit-exact only if it **reproduces every intermediate rounding the eager chain
performed**. That is a real cost — you must round back to the storage dtype at each original
materialisation point — and it must be a deliberate, declared property rather than an assumption.
`KernelVariant.preserves_intermediate_rounding` is that declaration, and the tier is derived from
it rather than trusted.

Why regions are DECLARED and then VERIFIED, not discovered
-----------------------------------------------------------
Pattern-matching a module tree by name is how Unsloth does it, and it does not survive this corpus.
Cosmos3's RMSNorm *looks* like a fusable RMSNorm and is deliberately not one: `transformer_cosmos3.py:49-59`
overrides `forward_cuda` back to the native implementation to preserve fp32 parity. A name-matcher
fuses it and silently breaks the model. So the adapter declares the region as a fact, and the
optimizer verifies the declaration against a trace before acting on it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class OpKind(enum.Enum):
    """What an op does, which is what decides whether fusing it can be bit-exact."""

    ELEMENTWISE = "elementwise"   # per-element; fusable, rounding-sensitive
    REDUCTION = "reduction"       # sum/mean/max; fusing CHANGES reduction order
    GEMM = "gemm"                 # matmul; fusable at the epilogue only
    RESHAPE = "reshape"           # view/permute/unflatten; free, no numerics
    ATTENTION = "attention"       # opaque; has its own reduction order
    EFFECTFUL = "effectful"       # writes state, draws RNG, syncs -- reordering is illegal


@dataclass(frozen=True)
class OpSpec:
    """One op in a region, in execution order."""

    name: str
    kind: OpKind
    #: storage dtype the eager path materialises this op's OUTPUT in. `None` means the op does not
    #: materialise (it is already fused into its consumer). This is the field that decides whether
    #: a fused kernel has a rounding to reproduce.
    materializes_as: str | None = "bf16"
    #: dtype the eager path COMPUTES in, which may be wider than storage
    computes_in: str = "fp32"
    must_stay: str | None = None      # a hard numerics constraint, e.g. pi-0's fp32 keep-list


@dataclass(frozen=True)
class FusibleRegion:
    """A contiguous op sequence an adapter offers for fusion.

    The adapter states the sequence and its boundary. It does not say "fuse this" and it does not
    name a kernel — the optimizer decides whether any registered kernel is legal and profitable.
    """

    name: str
    ops: tuple[OpSpec, ...]
    boundary_in: tuple[str, ...]
    boundary_out: tuple[str, ...]
    #: phases this region occurs in; used to compute how often fusing it pays
    phases: tuple[str, ...] = ()
    occurrences_per_forward: int = 1
    note: str = ""

    def has_effects(self) -> bool:
        return any(o.kind is OpKind.EFFECTFUL for o in self.ops)

    def has_reduction(self) -> bool:
        return any(o.kind is OpKind.REDUCTION for o in self.ops)

    def rounding_points(self) -> tuple[str, ...]:
        """Ops whose output the eager path materialises, i.e. roundings a bit-exact kernel must
        reproduce. If this is empty the region is rounding-free and fusion cannot change values."""
        return tuple(o.name for o in self.ops if o.materializes_as is not None)

    def pinned_dtypes(self) -> dict[str, str]:
        return {o.name: o.must_stay for o in self.ops if o.must_stay}


@dataclass(frozen=True)
class FusionDescriptor:
    """The L1 half of a Backend Adapter's declarations."""

    model_id: str
    regions: tuple[FusibleRegion, ...] = ()
    #: measured, per model: how many kernel launches the eager region costs. Lets the planner rank
    #: fusion against other passes in launches rather than in vibes.
    launches_per_region: dict[str, int] = field(default_factory=dict)

    def region(self, name: str) -> FusibleRegion:
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(name)

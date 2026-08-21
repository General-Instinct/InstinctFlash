"""Conv backend registration and selection over (backend x layout).

Selection here ranges over PAIRS, not backends, which is the structural difference from the attention
layer: a conv backend's answer depends on the layout it is offered, and the layout is ours to choose.
`candidates()` therefore enumerates the product and retains every refusal with its reason.

`select()` is implemented — unlike the attention layer's — because it chooses between backends that
already exist rather than kernels that do not, and because the decision it makes is measured. It still
refuses to install anything: it returns a `ConvPlan` describing what to do, and applying it is a pass's
job.
"""

from __future__ import annotations

from dataclasses import dataclass

from instinctflash.backends.conv.capabilities import ConvCapabilities, legality
from instinctflash.backends.conv.semantics import ConvSemantics, ConvShape, MemoryLayout
from instinctflash.passes.contract import Applicability, Tier


@dataclass(frozen=True)
class Candidate:
    backend_name: str
    caps: ConvCapabilities
    use_layout: MemoryLayout
    verdict: Applicability

    @property
    def legal(self) -> bool:
        return self.verdict.applies


@dataclass(frozen=True)
class ConvPlan:
    """What to do, without doing it."""

    backend_name: str
    use_layout: MemoryLayout
    convert_subgraph: bool
    tier: Tier
    reason: str


class ConvBackendRegistry:
    def __init__(self) -> None:
        self._b: dict = {}

    def register(self, backend) -> None:
        if backend.name in self._b:
            raise ValueError(f"conv backend {backend.name!r} already registered")
        self._b[backend.name] = backend

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._b))

    def get(self, name):
        return self._b[name]

    def candidates(self, *, semantics: ConvSemantics, shape: ConvShape,
                   have_layout: MemoryLayout, device=None,
                   tier_ceiling: Tier = Tier.NUMERIC,
                   subgraph_size: int = 1) -> tuple[Candidate, ...]:
        """Every (backend, layout) pair with a verdict. Refusals retained, with reasons."""
        out = []
        for name in self.names():
            caps = self._b[name].capabilities()
            for use in sorted(caps.layouts, key=lambda l: l.value):
                if use.rank() != shape.spatial_rank() + 2:
                    continue
                out.append(Candidate(name, caps, use, legality(
                    caps=caps, semantics=semantics, shape=shape, have_layout=have_layout,
                    use_layout=use, device=device, tier_ceiling=tier_ceiling,
                    subgraph_size=subgraph_size)))
        return tuple(out)

    def explain(self, **kw) -> str:
        cands = self.candidates(**kw)
        rows = [f"conv backends: {sum(1 for c in cands if c.legal)} legal of {len(cands)} pairs"]
        for c in cands:
            rows.append(f"  {'LEGAL ' if c.legal else 'REFUSE'} {c.backend_name:<16} "
                        f"{c.use_layout.value:<6} [{c.verdict.claimed_tier.name:<10}] "
                        f"{c.verdict.reason}")
        return "\n".join(rows)

    def select(self, *, prefer_bitexact: bool = True,
               measured: dict | None = None, **kw) -> ConvPlan:
        """Pick a (backend, layout) pair. Returns a PLAN; installing it is a pass's job.

        `measured` maps (backend_name, MemoryLayout) -> ms on the target's real shapes, INCLUDING any
        conversion. Supply it and the fastest legal pair wins; omit it and the incumbent wins.

        THAT ASYMMETRY IS THE DESIGN, and the first version of this function got it wrong. Written with
        no speed term at all, it could only ever return the incumbent, which made it useless: the
        cuDNN/NDHWC pair was correctly ranked legal and then never chosen. Written with a GUESSED speed
        term it would have been the reputation-ranking this project keeps rejecting. So speed enters
        only as measurement, and absent measurement the answer is "change nothing" -- which is the
        right default for a function that can silently downgrade a bit-exactness claim.

        Ranking:
          1. legality
          2. tier, if `prefer_bitexact` -- Layers 2-3 are gated at max|delta| = 0, and a NUMERIC pair
             winning silently would invalidate that. Raising the ceiling is the caller's explicit act.
          3. measured time, when supplied
          4. no-conversion over conversion, then the incumbent, on ties
        """
        legal = [c for c in self.candidates(**kw) if c.legal]
        if not legal:
            raise RuntimeError("no legal conv backend/layout pair; the fallback should always be one")

        def key(c: Candidate):
            m = (measured or {}).get((c.backend_name, c.use_layout))
            return (c.verdict.claimed_tier.value if prefer_bitexact else 0,
                    m if m is not None else float("inf"),
                    1 if c.verdict.params.get("needs_conversion") else 0,
                    0 if c.caps.is_reference_path else 1)

        best = min(legal, key=key)
        m = (measured or {}).get((best.backend_name, best.use_layout))
        why = best.verdict.reason + (f"; measured {m:.2f} ms" if m is not None else
                                     "; NO MEASUREMENT SUPPLIED, so the incumbent was kept")
        return ConvPlan(
            backend_name=best.backend_name, use_layout=best.use_layout,
            convert_subgraph=bool(best.verdict.params.get("needs_conversion")),
            tier=best.verdict.claimed_tier, reason=why)


REGISTRY = ConvBackendRegistry()

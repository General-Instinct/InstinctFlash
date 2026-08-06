"""Registration and candidate filtering. Selection is DELIBERATELY NOT IMPLEMENTED.

WHAT WORKS HERE, TODAY

    register(backend)                    add a backend; nothing else in the tree changes
    candidates(site, ...)                the legal set, with a reason for every rejection
    explain(site, ...)                   that set as text, for a plan explanation

WHAT DOES NOT, ON PURPOSE

    select(...)                          raises NotImplementedError

Selection needs measured numbers on real shapes on an idle fleet, and installing a chosen backend
changes runtime behaviour. Both are out of scope for the architecture step: the point of this module
is that when selection is built, it will be a ranking function over `candidates()` and nothing above
it will move. `select()` exists as a raising stub so that the signature is reviewable now and so
nothing quietly starts choosing backends in the meantime.

WHY THE ADAPTER'S OWN ATTENTION IS ALWAYS A CANDIDATE

There is always exactly one backend that is trivially legal and trivially bit-exact: the one the
model already uses. Keeping it in the candidate set means selection never has to invent a fallback,
`candidates()` is never empty, and "we measured four backends and kept the original" is an expressible
outcome rather than a bug. It is also the only honest baseline for `max_abs_delta`.
"""

from __future__ import annotations

from dataclasses import dataclass

from instinctwm.backends.attention.backend import AttentionBackend
from instinctwm.backends.attention.capabilities import AttentionCapabilities, legality
from instinctwm.backends.attention.semantics import (
    AttentionSemantics,
    AttentionShape,
    MaskSpec,
    QKVLayout,
)
from instinctwm.passes.contract import Applicability, Tier
from instinctwm.runtime.state.types import Addressing


@dataclass(frozen=True)
class Candidate:
    """One backend's verdict for one site."""

    backend_name: str
    caps: AttentionCapabilities
    verdict: Applicability

    @property
    def legal(self) -> bool:
        return self.verdict.applies


class AttentionBackendRegistry:
    """Backends by name. Adding one must not require editing a planner or an adapter."""

    def __init__(self) -> None:
        self._backends: dict[str, AttentionBackend] = {}

    def register(self, backend: AttentionBackend) -> None:
        if backend.name in self._backends:
            raise ValueError(f"attention backend {backend.name!r} already registered")
        self._backends[backend.name] = backend

    def get(self, name: str) -> AttentionBackend:
        return self._backends[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))

    def candidates(
        self,
        *,
        semantics: AttentionSemantics,
        mask: MaskSpec,
        layout: QKVLayout,
        addressing: Addressing,
        shape: AttentionShape,
        world_size: int = 1,
        device=None,
        tier_ceiling: Tier = Tier.NUMERIC,
    ) -> tuple[Candidate, ...]:
        """Every registered backend with a verdict. Illegal ones are RETAINED, with their reason.

        Returning the rejections is the point: `plan.explain()` reports the passes it declined and why,
        and a backend silently missing from a list is indistinguishable from a backend that was never
        written. Order is by name so the output is stable and diffable.
        """
        out = []
        for name in self.names():
            caps = self._backends[name].capabilities()
            out.append(Candidate(name, caps, legality(
                caps=caps, semantics=semantics, mask=mask, layout=layout, addressing=addressing,
                shape=shape, world_size=world_size, device=device, tier_ceiling=tier_ceiling)))
        return tuple(out)

    def explain(self, **kw) -> str:
        """The candidate set as text, in the shape `plan.explain()` already uses."""
        rows = []
        for c in self.candidates(**kw):
            verb = "LEGAL " if c.legal else "REFUSE"
            rows.append(f"  {verb} {c.backend_name:<24} [{c.verdict.claimed_tier.name:<10}] "
                        f"{c.verdict.reason}")
        legal_n = sum(1 for c in self.candidates(**kw) if c.legal)
        head = (f"attention backends: {legal_n} legal of {len(self.names())} registered "
                f"(selection NOT IMPLEMENTED -- no backend will be installed)")
        return "\n".join([head, *rows])

    def select(self, **kw):
        """Rank the legal candidates and return one. NOT IMPLEMENTED.

        When built, it is a pure ranking over `candidates()`:

            legal = [c for c in candidates(**kw) if c.legal]
            for c in legal:
                win = backend.expected_delta_ms(shape, forwards_per_cycle, device)
                win -= plan_penalty_ms(c.caps, capture_in_plan=..., capture_speedup=...,
                                       cycle_ms=...)
            # then MEASURE the top few rather than trusting the prediction, discard any
            # measurement that reports not_evaluated, and keep the adapter-native backend
            # unless a challenger wins by more than the measured spread.

        Two rules that must survive into the implementation:
          * a predicted win is a filter for what to measure, never a reason to install
          * ties go to the incumbent, because a swap that is not measurably better is a change
            with no upside and a numerics story to defend
        """
        raise NotImplementedError(
            "attention backend selection is not implemented. This is the architecture step: "
            "candidates() and legality() are complete and testable, ranking needs measured numbers "
            "on an idle fleet, and installing a backend would change runtime behaviour. "
            "See ATTENTION.md, 'What is deliberately not built'.")


#: Process-wide registry. Backends register at import time, as kernels do in `backends/registry.py`.
REGISTRY = AttentionBackendRegistry()

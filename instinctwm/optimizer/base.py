"""Optimizer/Compiler — the layer that turns declarations into optimizations.

A pass never asks "did the user enable me?". It asks "do this model's declarations make me
legal, and profitable?" and answers from `AdapterSpec` alone. That is what makes the framework
model-aware rather than model-specific: the same `CFGBranchElision` pass fires on any adapter
that declares a `POSITIVE_ONLY` stream, whether that is LingBot-VA today or a model nobody has
written yet.

Accuracy tiers do NOT compose upward. A plan containing one BEHAVIORAL pass is BEHAVIORAL, no
matter how many BITEXACT passes sit beside it. This is enforced in `Plan.tier()` rather than
left to discipline, because the failure mode — quoting a bit-exactness claim for a plan that
contains a lossy pass — is exactly the kind of thing that survives review and then invalidates
a benchmark.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from instinctwm.adapter.base import AdapterSpec


class Tier(enum.IntEnum):
    """Ordered weakest-claim-last so `max()` gives the tier of a composed plan."""

    BITEXACT = 0    # torch.equal on per-step latents and committed K/V, production kernels
    NUMERIC = 1     # bounded ||delta|| justified by a NAMED structural invariant
    BEHAVIORAL = 2  # changes outputs; needs a paired non-inferiority run vs a measured floor


@dataclass(frozen=True)
class PassResult:
    """What a pass decided, including why it declined."""

    name: str
    applies: bool
    tier: Tier
    reason: str
    #: free-form knobs the runtime layer consumes when installing this pass
    params: dict = field(default_factory=dict)
    expected_win: str = "unknown"


class OptimizationPass(Protocol):
    name: str

    def evaluate(self, spec: AdapterSpec) -> PassResult:
        """Decide, from declarations alone, whether this pass is legal and profitable."""
        ...


@dataclass
class Plan:
    """An ordered set of passes the optimizer chose for one model."""

    model_id: str
    results: list[PassResult]

    @property
    def applied(self) -> list[PassResult]:
        return [r for r in self.results if r.applies]

    def tier(self) -> Tier:
        """The weakest claim in the plan. A plan is only BITEXACT if EVERY applied pass is."""
        if not self.applied:
            return Tier.BITEXACT
        return max(r.tier for r in self.applied)

    def bitexact_subset(self) -> "Plan":
        """The largest sub-plan that can still be claimed bit-exact.

        Useful in practice: it is the configuration you can ship without buying a paired
        non-inferiority run, which costs roughly 10x the GPU time of measuring the speedup.
        """
        return Plan(self.model_id, [r for r in self.results if r.tier == Tier.BITEXACT])

    def explain(self) -> str:
        out = [f"InstinctWM plan for {self.model_id}", f"  plan tier: {self.tier().name}", ""]
        for r in self.results:
            mark = "APPLY " if r.applies else "skip  "
            out.append(f"  {mark} {r.name:26s} [{r.tier.name:10s}] {r.reason}")
            if r.applies and r.expected_win != "unknown":
                out.append(f"         expected: {r.expected_win}")
        if self.tier() > Tier.BITEXACT:
            lossy = [r.name for r in self.applied if r.tier > Tier.BITEXACT]
            out += [
                "",
                f"  NOTE: plan is {self.tier().name} because of {lossy}.",
                "        Any accuracy-neutrality claim for this plan requires a paired",
                "        non-inferiority run. `plan.bitexact_subset()` is the largest",
                "        configuration that does not.",
            ]
        return "\n".join(out)


class Optimizer:
    """Runs every registered pass against a model's declarations and produces a Plan."""

    def __init__(self, passes: Sequence[OptimizationPass], tier_ceiling: Tier = Tier.BITEXACT):
        #: passes are evaluated in registration order; ordering matters where one pass is a
        #: precondition for another (sync elimination gates graph capture, for instance).
        self._passes = list(passes)
        self._ceiling = tier_ceiling

    def compile(self, spec: AdapterSpec) -> Plan:
        results: list[PassResult] = []
        for p in self._passes:
            r = p.evaluate(spec)
            if r.applies and r.tier > self._ceiling:
                r = PassResult(
                    name=r.name, applies=False, tier=r.tier,
                    reason=f"legal but tier {r.tier.name} exceeds ceiling "
                           f"{self._ceiling.name}: {r.reason}",
                    params=r.params, expected_win=r.expected_win,
                )
            results.append(r)
        return Plan(spec.model_id, results)

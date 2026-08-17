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

from instinctwm.adapters.base import AdapterSpec
from instinctwm.descriptors.deployment import DeploymentSpec


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

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        """Decide whether this pass is legal and profitable.

        Both arguments are facts, never requests: `spec` is what the model declared about
        itself, `deployment` is how this particular server is running it. A pass reads them
        and decides; it never asks whether the user enabled it.
        """
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

    def without(self, *names: str) -> "Plan":
        """The same plan with the named passes demoted to skipped.

        The named passes stay in `results` with `applies=False` and a reason, so `explain()`
        still shows that they were legal and were dropped by hand. Silently deleting them
        would make the plan indistinguishable from one where they never fired.
        """
        unknown = set(names) - {r.name for r in self.results}
        if unknown:
            raise KeyError(f"no such pass in this plan: {sorted(unknown)}")
        return Plan(self.model_id, [
            PassResult(name=r.name, applies=False, tier=r.tier,
                       reason=f"dropped by caller via Plan.without(): {r.reason}",
                       params=r.params, expected_win=r.expected_win)
            if r.name in names else r
            for r in self.results
        ])

    def serve(self, model, port: int, **kwargs):
        """Install this plan on `model` and start serving it.

        Deliberately thin: the plan knows which passes fired, the backend knows how to apply
        them to its own server, and neither needs to know the other's internals. A backend
        that cannot install an applied pass raises rather than serving a plan it did not
        actually apply — the alternative is a server whose `explain()` output is a lie.
        """
        return model.serve(self, port=port, **kwargs)

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

    def __init__(
        self,
        passes: Sequence[OptimizationPass] | None = None,
        tier_ceiling: Tier = Tier.BITEXACT,
    ):
        #: passes are evaluated in registration order; ordering matters where one pass is a
        #: precondition for another (sync elimination gates graph capture, for instance).
        if passes is None:
            # Imported lazily: the pass modules import this one, so a module-scope import
            # here would be circular.
            from instinctwm.passes.lingbot import default_passes

            passes = default_passes()
        self._passes = list(passes)
        self._ceiling = tier_ceiling

    def compile(self, spec: AdapterSpec, deployment: DeploymentSpec | None = None,
                capabilities: frozenset[str] | None = None) -> Plan:
        """Evaluate every pass against one model's declarations and one server's situation.

        `deployment` defaults to `DeploymentSpec()` — single GPU, actions only — because that
        is the regime this framework targets. Pass one explicitly when it is not true; the
        passes that care will decline on their own.

        `capabilities` is `Checkpoint.capabilities()` — tokens derived from the checkpoint's
        EXECUTION block and nothing else. A pass that declares `requires_capabilities` is skipped
        unless every token it needs is present. Passing `None` means "do not filter", which is the
        behaviour every existing pass has always had: an empty requirement composes with every
        checkpoint, and that is the default on purpose.

        THERE IS NO ARGUMENT HERE THAT CARRIES A TRAINING METHOD, and there is no way to add one
        without changing this signature. `capabilities` cannot smuggle one either: it is built by
        `Checkpoint.capabilities()` from the execution block, which `load_declaration` populates
        without ever parsing provenance. tests/test_checkpoint_platform.py asserts the resulting
        plan is invariant to provenance.
        """
        deployment = deployment if deployment is not None else DeploymentSpec()
        results: list[PassResult] = []
        for p in self._passes:
            # `capabilities=None` is INSPECTION MODE and deliberately does not filter, so that a pass
            # can be reasoned about, and unit-tested, without inventing a checkpoint. It stays that
            # way. But it cannot be silent: planning a VLA this framework had never seen produced a
            # plan reporting APPLY on three passes that rewrite the LingBot-VA server object, and
            # nothing in the output said their applicability had not been checked. So the requirement
            # is annotated instead of dropped -- the flag is unchanged, the reader is not misled.
            need = frozenset(getattr(p, "requires_capabilities", ()) or ())
            if need and capabilities is not None and not need <= capabilities:
                results.append(PassResult(
                    name=getattr(p, "name", type(p).__name__), applies=False, tier=Tier.BITEXACT,
                    reason=f"checkpoint does not declare {sorted(need - capabilities)}; the pass is "
                           f"not applicable to it. This is a CAPABILITY decision, not a recipe one.",
                ))
                continue
            unchecked = need if (need and capabilities is None) else frozenset()
            r = p.evaluate(spec, deployment)
            if r.applies and r.tier > self._ceiling:
                r = PassResult(
                    name=r.name, applies=False, tier=r.tier,
                    reason=f"legal but tier {r.tier.name} exceeds ceiling "
                           f"{self._ceiling.name}: {r.reason}",
                    params=r.params, expected_win=r.expected_win,
                )
            if unchecked and r.applies:
                r = PassResult(
                    name=r.name, applies=r.applies, tier=r.tier,
                    reason=f"APPLICABILITY UNCHECKED -- requires {sorted(unchecked)} and no "
                           f"capabilities were supplied, so this may not be applicable to the model "
                           f"you are planning: {r.reason}",
                    params=r.params, expected_win=r.expected_win,
                )
            results.append(r)
        return Plan(spec.model_id, results)

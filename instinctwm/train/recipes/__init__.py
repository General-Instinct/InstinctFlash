"""Recipe registry. Adding a paper should mean adding a file here and nothing else.

Three of the four entries below are DECLARATION-ONLY: capabilities, descriptor delta and state are
real, `step()` raises. That is not laziness, it is the point being tested. A declaration-only recipe
still proves the two things the framework claims:

  * `sCM` and `rCM` are REJECTED on this box before any GPU work, because they need forward-mode AD
    through attention and there is no flash-attn here. The rejection comes from the capability
    declaration, not from a crash several hours in.
  * `DMD2` needs a fake score network and a discriminator on a two-time-scale update, and it can say
    so through `RecipeState.update_order` without the trainer knowing what a discriminator is.

They raise rather than returning a placeholder loss so no one can mistake them for trained models.
"""

from __future__ import annotations

from typing import Callable, Mapping

from instinctwm.train.recipe import (
    Capabilities, DescriptorDelta, Environment, RecipeState, StepOutput,
)
from instinctwm.train.recipes.pdd import ParallelDecoding

_DECLARED_ONLY = (
    "{name}: capabilities, descriptor delta and state are declared and exercised by tests, but the "
    "objective is not implemented. Raising instead of returning a placeholder loss, so this cannot "
    "be mistaken for a trained model. Implementing it means writing step() in this file only."
)


class _StepReduction:
    """Shared shape for the declared-only step-reduction recipes."""
    name = "override me"
    _caps = Capabilities()
    _note = ""

    def __init__(self, nfe: Mapping[str, int]):
        if not isinstance(nfe, Mapping):
            raise TypeError("nfe must map phase name -> steps, e.g. {'video': 2, 'action': 2}")
        self.nfe = dict(nfe)

    def requires(self) -> Capabilities:
        return self._caps

    def descriptor_delta(self, model) -> DescriptorDelta:
        known = {p.name for p in model.execution.phases}
        unknown = set(self.nfe) - known
        if unknown:
            raise ValueError(f"nfe names phases {sorted(unknown)}; model has {sorted(known)}")
        return DescriptorDelta(nfe=dict(self.nfe), note=self._note)

    def build(self, model, env: Environment) -> RecipeState:
        return RecipeState(update_order=("student",))

    def step(self, batch, teacher, student, state: RecipeState) -> StepOutput:
        raise NotImplementedError(_DECLARED_ONLY.format(name=self.name))


class SimplifiedConsistency(_StepReduction):
    name = "scm"
    _caps = Capabilities(jvp_through_attention=True, teacher_calls_per_step=1)
    _note = "sCM: continuous-time consistency; forward divergence, so expect blur before collapse"


class ScoreRegularizedConsistency(_StepReduction):
    name = "rcm"
    _caps = Capabilities(jvp_through_attention=True, teacher_calls_per_step=2,
                         aux_modules=("fake_score",))
    _note = "rCM: consistency regularised by score distillation; the two failure modes cancel"


class DistributionMatching(_StepReduction):
    """DMD2. Declared-only, but its STATE is real -- it is what exercises alternating updates."""
    name = "dmd2"
    _caps = Capabilities(adversarial=True, teacher_calls_per_step=1,
                         aux_modules=("fake_score", "discriminator"))
    _note = "DMD2: distribution matching with adversarial supervision; mode collapse is the risk"

    def build(self, model, env: Environment) -> RecipeState:
        # Three updates on a two-time-scale schedule. The trainer drives these in order with no
        # knowledge of what any of them are; that is the interface claim being made concrete.
        return RecipeState(modules={}, optimizers={},
                           update_order=("student", "fake_score", "discriminator"))


#: name -> factory taking the per-phase NFE mapping.
REGISTRY: dict[str, Callable[..., object]] = {
    "pdd": ParallelDecoding,
    "scm": SimplifiedConsistency,
    "rcm": ScoreRegularizedConsistency,
    "dmd2": DistributionMatching,
}


def register(name: str, factory: Callable[..., object]) -> None:
    if name in REGISTRY:
        raise KeyError(f"recipe {name!r} already registered")
    REGISTRY[name] = factory


def build(name: str, *args, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown recipe {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](*args, **kwargs)


def available() -> list[str]:
    return sorted(REGISTRY)


def implemented() -> list[str]:
    """Recipes whose objective actually exists. Kept separate from `available()` on purpose."""
    return ["pdd"]


__all__ = ["ParallelDecoding", "REGISTRY", "register", "build", "available", "implemented",
           "SimplifiedConsistency", "ScoreRegularizedConsistency", "DistributionMatching"]

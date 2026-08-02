"""The Recipe interface: a model optimization is an interchangeable component.

Layer 1 is a MODEL OPTIMIZATION framework, not a distillation framework. The platform does not care
why the model changes -- only that a recipe performs

    (teacher, descriptor)  ->  (student, descriptor')  +  certificate

Step reduction, latent compression and quantization-aware training are all that transformation, and
differ only in which part of the descriptor they touch. That is why `descriptor_delta` is the
central method and not an afterthought: it is what lets the runtime execute the result with no glue,
so a recipe is one file and no runtime code.

WHAT THE PLATFORM OWNS: the training loop, distributed execution, checkpointing, teacher caching,
certification, packaging, runtime integration.
WHAT A RECIPE OWNS: the objective, and any auxiliary modules the objective needs.

If adding a new method requires touching the trainer, the harness, the registry or the runtime, this
interface is wrong and should be fixed rather than worked around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class Capabilities:
    """What a recipe needs from the environment. Checked BEFORE training starts.

    This exists because of a concrete failure mode: rCM needs forward-mode AD through attention (a
    FlashAttention-2 JVP kernel), and this box deliberately has no flash-attn installed -- it was
    removed to keep the LingBot-VA baseline environment fixed. Declaring the requirement turns that
    into a startup error instead of a discovery several hours into a job.

    Deliberately the same shape as `HardwareReq.satisfied_by(device)` in the kernel registry, and it
    fails closed for the same reason.
    """
    jvp_through_attention: bool = False     # sCM, rCM
    adversarial: bool = False               # DMD2: discriminator, and its tuning burden
    teacher_calls_per_step: int = 1         # memory and throughput planning
    aux_modules: tuple[str, ...] = ()       # "fake_score", "discriminator"
    min_gpus: int = 1

    def satisfied_by(self, env: "Environment") -> tuple[bool, str]:
        if self.jvp_through_attention and not env.has_jvp_attention:
            return False, ("recipe needs forward-mode AD through attention (a FlashAttention-2 JVP "
                           "kernel); this environment has none")
        if self.adversarial and not env.allows_adversarial:
            return False, ("recipe needs adversarial supervision; this environment disallows it "
                           "(mode collapse in an action head is a behavioural failure, not a "
                           "quality complaint)")
        if env.n_gpus < self.min_gpus:
            return False, f"recipe needs {self.min_gpus} GPUs, environment has {env.n_gpus}"
        return True, "all declared capabilities available"


@dataclass(frozen=True)
class Environment:
    n_gpus: int = 1
    has_jvp_attention: bool = False
    allows_adversarial: bool = True


@dataclass(frozen=True)
class DescriptorDelta:
    """What the student changes about EXECUTION, relative to the teacher.

    The bridge between Layer 1 and Layers 2-6. A recipe that cannot state its delta cannot have its
    output executed by the runtime without hand-written glue, which is the thing this framework
    exists to remove.

    `nfe` is per-phase because Flash-WAM's modality-aware result (1 video step / 2 action steps) is
    only expressible that way -- and `ExecutionDescriptor.phases[].nfe` already is per-phase.
    """
    nfe: Mapping[str, int] | None = None            # phase name -> new NFE
    shapes: Mapping[str, tuple] | None = None       # latent compression
    dtypes: Mapping[str, str] | None = None         # quantization-aware training
    note: str = ""

    def is_empty(self) -> bool:
        return not (self.nfe or self.shapes or self.dtypes)

    def describe(self) -> str:
        bits = []
        if self.nfe:
            bits.append("nfe " + ", ".join(f"{k}={v}" for k, v in sorted(self.nfe.items())))
        if self.shapes:
            bits.append(f"shapes {sorted(self.shapes)}")
        if self.dtypes:
            bits.append(f"dtypes {sorted(self.dtypes)}")
        return "; ".join(bits) or "no execution change"

    def apply_to(self, execution) -> Any:
        """Produce descriptor' from descriptor. Only NFE is wired today.

        Shapes and dtypes are accepted and reported but not yet applied -- a latent-compression or
        QAT recipe can be written against this interface, and the gap is here rather than in the
        recipe. Stated plainly so it is not mistaken for support.
        """
        import dataclasses
        if not self.nfe:
            return execution
        phases = tuple(
            dataclasses.replace(p, nfe=self.nfe.get(p.name, p.nfe)) for p in execution.phases)
        return dataclasses.replace(execution, phases=phases)


@dataclass
class RecipeState:
    """Auxiliary modules and optimizers a recipe owns.

    DMD2 needs a fake score network AND a discriminator on a two-time-scale update, so the trainer
    must support several optimizers and alternating steps. That belongs to the recipe, not to a
    hardcoded loop.
    """
    modules: dict[str, Any] = field(default_factory=dict)
    optimizers: dict[str, Any] = field(default_factory=dict)
    update_order: tuple[str, ...] = ("student",)
    extra: dict[str, Any] = field(default_factory=dict)


class Recipe(Protocol):
    name: str

    def requires(self) -> Capabilities: ...

    def descriptor_delta(self, model) -> DescriptorDelta: ...

    def build(self, model, env: Environment) -> RecipeState: ...

    def step(self, batch, teacher, student, state: RecipeState) -> Mapping[str, float]: ...


class RecipeRejected(RuntimeError):
    """The environment cannot run this recipe. Raised before training, never during."""


def admit(recipe: Recipe, env: Environment) -> tuple[bool, str]:
    ok, why = recipe.requires().satisfied_by(env)
    return ok, why


def prepare(recipe: Recipe, model, env: Environment):
    """Capability check, then state. The check runs first, always."""
    ok, why = admit(recipe, env)
    if not ok:
        raise RecipeRejected(f"{recipe.name}: {why}")
    delta = recipe.descriptor_delta(model)
    if delta.is_empty():
        raise RecipeRejected(
            f"{recipe.name} declares no descriptor delta. A recipe that changes nothing about "
            f"execution cannot be integrated by the runtime; if it truly changes only weights, say "
            f"so explicitly with a note.")
    return recipe.build(model, env), delta

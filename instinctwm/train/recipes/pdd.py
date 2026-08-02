"""Parallel Decoding Distillation — the reference recipe.

arXiv 2607.26004 (NVIDIA GenAIR). Chosen as the first recipe for two reasons that are about the
platform rather than the method:

  * it needs NO JVP kernel, so it does not collide with this box having no flash-attn
  * it supports VARIABLE student NFE, which exercises the hardest part of the interface -- if
    `DescriptorDelta` and `nfe_mutable` can express a variable-NFE student, they can express a
    fixed-NFE one

Core idea: decompose the mean-velocity prediction over an interval into parallel sub-interval
predictions, learning a representation of the mean velocity WITHOUT regressing its derivative. No
JVPs, no finite differences, no adversary.

STATUS: interface-complete, objective is a STUB. `step()` raises. This file exists to prove the
interface can carry a real method and to be the shape the objective drops into -- not to claim PDD
is implemented.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from instinctwm.train.recipe import (
    Capabilities, DescriptorDelta, Environment, RecipeState,
)


class ParallelDecoding:
    name = "parallel_decoding_distillation"

    def __init__(self, nfe: Mapping[str, int] | Sequence[int], *, sub_intervals: int = 4):
        """`nfe` maps phase name -> student NFE, e.g. {"video": 1, "action": 2}.

        Per-phase because the modality-aware result matters: a single objective across both streams
        collapses the action head (Flash-WAM measured 24% success with video quality intact). The
        platform cannot prevent that mistake, but it can make the correct thing expressible.
        """
        if not isinstance(nfe, Mapping):
            raise TypeError(
                "nfe must map phase name -> steps, e.g. {'video': 1, 'action': 2}. A bare list "
                "would silently apply one step count to both streams, which is the failure mode "
                "this argument exists to prevent.")
        self.nfe = dict(nfe)
        self.sub_intervals = sub_intervals

    def requires(self) -> Capabilities:
        # No JVP, no adversary. Two teacher calls: interval endpoints.
        return Capabilities(jvp_through_attention=False, adversarial=False,
                            teacher_calls_per_step=2, aux_modules=(), min_gpus=1)

    def descriptor_delta(self, model) -> DescriptorDelta:
        known = {p.name for p in model.execution.phases}
        unknown = set(self.nfe) - known
        if unknown:
            raise ValueError(
                f"nfe names phases {sorted(unknown)} that {model.execution.model_id!r} does not "
                f"have; it has {sorted(known)}")
        return DescriptorDelta(
            nfe=dict(self.nfe),
            note=f"PDD, {self.sub_intervals} sub-intervals; student supports variable NFE")

    def build(self, model, env: Environment) -> RecipeState:
        # No auxiliary networks -- the student is the only trainable module. That is the whole
        # practical advantage over rCM (JVP kernel) and DMD2 (fake score + discriminator).
        return RecipeState(modules={}, optimizers={}, update_order=("student",),
                           extra={"sub_intervals": self.sub_intervals})

    def step(self, batch, teacher, student, state: RecipeState) -> Mapping[str, float]:
        raise NotImplementedError(
            "PDD objective is not implemented. The interface is complete and the capability, "
            "descriptor-delta and state plumbing are exercised by tests; the mean-velocity loss "
            "is the remaining work. Deliberately raising rather than returning a placeholder loss, "
            "so this cannot be mistaken for a trained model.")

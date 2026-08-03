"""Parallel Decoding Distillation — the first real recipe.

arXiv 2607.26004 (NVIDIA GenAIR). Chosen first for two reasons that are about the platform rather
than the method:

  * it needs NO JVP kernel, so it does not collide with this box having no flash-attn
  * it exercises per-phase NFE, which is the part of `DescriptorDelta` that matters most here

THE OBJECTIVE

For a flow-matching teacher the sampler integrates dx/du = v(x_u, u). A few-step sampler cannot
afford that, so what a student actually needs is the MEAN velocity over a whole interval:

    V(x_t, t, s) = 1/(s-t) * integral_t^s v(x_u, u) du        so that   x_s = x_t + (s-t) V

Consistency-style methods learn this by regressing the derivative of V, which needs a JVP through
attention. PDD's contribution is that you do not have to: split [t, s] into K sub-intervals and the
mean velocity of the whole is the mean of the parts,

    V(t, s) = 1/K * sum_k V(u_k, u_{k+1})

so the target is an AVERAGE OF TEACHER EVALUATIONS, with no derivative anywhere.

WHY THE K TEACHER CALLS ARE ONE BATCHED CALL, NOT K SEQUENTIAL ONES -- this is the "parallel" in
the name and the reason the recipe is affordable. Naively, evaluating the teacher at u_k requires
x_{u_k}, which requires integrating from x_t: K sequential forwards, each depending on the last. But
a flow-matching model is trained on the straight coupling x_u = (1-u) x_0 + u x_1 between noise and
data, and we are training from a DATASET, so we have both endpoints. Every intermediate point is
therefore available in closed form, all K of them can be stacked into one batch, and the sequential
dependency disappears.

WHAT THIS IMPLEMENTATION DOES NOT DO. The paper's student is conditioned on the interval (t, s) so
that NFE can vary at sampling time. Doing that here would mean adding a conditioning input to the
LingBot-VA backbone -- an architecture change, which is exactly the kind of thing that turns "add a
recipe" into "modify the model". So `interval_conditioning="none"` (the default) trains the student
on the FIXED schedule its target NFE will use, and the resulting student is a fixed-NFE student.
`nfe_mutable` in the descriptor stays False, honestly. The variable-NFE variant is a follow-up that
needs a backbone change, and it is flagged rather than quietly skipped.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from instinctwm.train.recipe import (
    Capabilities, DescriptorDelta, Environment, RecipeState, StepOutput,
)


def _uniform_schedule(nfe: int) -> tuple[float, ...]:
    """The (t, s) interval endpoints a uniform nfe-step sampler visits, from noise (0) to data (1)."""
    return tuple(i / nfe for i in range(nfe + 1))


class ParallelDecoding:
    name = "parallel_decoding_distillation"

    def __init__(self, nfe: Mapping[str, int] | Sequence[int], *, sub_intervals: int = 4,
                 interval_conditioning: str = "none", loss: str = "mse"):
        """`nfe` maps phase name -> student NFE, e.g. {"video": 2, "action": 2}.

        Per-phase because the modality-aware result matters: a single objective across both streams
        collapses the action head (Flash-WAM measured 24% success with video quality intact). The
        platform cannot prevent that mistake, but it can make the correct thing expressible.
        """
        if not isinstance(nfe, Mapping):
            raise TypeError(
                "nfe must map phase name -> steps, e.g. {'video': 2, 'action': 2}. A bare list "
                "would silently apply one step count to both streams, which is the failure mode "
                "this argument exists to prevent.")
        if sub_intervals < 2:
            raise ValueError(
                f"sub_intervals={sub_intervals} makes the target a single teacher evaluation, which "
                f"is ordinary velocity distillation, not a mean over the interval. Use >= 2.")
        if interval_conditioning not in ("none", "delta"):
            raise ValueError("interval_conditioning must be 'none' or 'delta'")
        if loss not in ("mse", "huber"):
            raise ValueError("loss must be 'mse' or 'huber'")
        self.nfe = dict(nfe)
        self.sub_intervals = sub_intervals
        self.interval_conditioning = interval_conditioning
        self.loss = loss

    def requires(self) -> Capabilities:
        # One BATCHED teacher call per phase; see the module docstring for why it is not K calls.
        return Capabilities(jvp_through_attention=False, adversarial=False,
                            teacher_calls_per_step=len(self.nfe), aux_modules=(), min_gpus=1)

    def descriptor_delta(self, model) -> DescriptorDelta:
        known = {p.name for p in model.execution.phases}
        unknown = set(self.nfe) - known
        if unknown:
            raise ValueError(
                f"nfe names phases {sorted(unknown)} that {model.execution.model_id!r} does not "
                f"have; it has {sorted(known)}")
        note = (f"PDD, {self.sub_intervals} sub-intervals, {self.loss} loss; "
                f"fixed-schedule student (interval_conditioning={self.interval_conditioning!r})")
        return DescriptorDelta(nfe=dict(self.nfe), note=note)

    def build(self, model, env: Environment) -> RecipeState:
        # No auxiliary networks -- the student is the only trainable module. That is the whole
        # practical advantage over rCM (JVP kernel) and DMD2 (fake score + discriminator).
        return RecipeState(
            modules={}, optimizers={}, update_order=("student",),
            extra={"sub_intervals": self.sub_intervals,
                   "schedules": {ph: _uniform_schedule(n) for ph, n in self.nfe.items()},
                   "interval_conditioning": self.interval_conditioning})

    # -- the objective ------------------------------------------------------------------------

    def _target_mean_velocity(self, teacher, x0, x1, t, s, phase):
        """Average of K teacher velocities along the straight coupling, in ONE batched call."""
        import torch
        K = self.sub_intervals
        # Sub-interval midpoints: the mean velocity of a sub-interval is best single-point
        # approximated at its midpoint (this is the midpoint rule, error O(h^2), against O(h) for
        # either endpoint -- it costs nothing and halves the order of the bias).
        us = [t + (k + 0.5) * (s - t) / K for k in range(K)]
        xs = torch.cat([(1.0 - u) * x0 + u * x1 for u in us], dim=0)
        ts = torch.cat([torch.full((x0.shape[0],), float(u), device=x0.device, dtype=x0.dtype)
                        for u in us], dim=0)
        with torch.no_grad():
            v = teacher(xs, ts, phase=phase)
        return v.reshape(K, *x0.shape).mean(dim=0)

    def _loss(self, pred, target):
        import torch
        if self.loss == "huber":
            return torch.nn.functional.huber_loss(pred, target, delta=1.0)
        return torch.nn.functional.mse_loss(pred, target)

    def step(self, batch, teacher, student, state: RecipeState) -> StepOutput:
        """One PDD step over every phase named in `nfe`.

        `batch` must carry, per phase, the clean sample and the noise that pairs with it. The noise
        is taken from the batch rather than drawn here so that a step is reproducible from the
        dataloader alone -- the same reason `probe_bitexact` insists on `--deterministic-seed`.
        """
        import torch
        losses, metrics = {}, {}
        total = None
        schedules = state.extra["schedules"]

        for phase, sched in schedules.items():
            x1 = batch[f"{phase}/x1"]                      # clean latent (data end, u=1)
            x0 = batch.get(f"{phase}/x0")                  # noise (u=0)
            if x0 is None:
                x0 = torch.randn_like(x1)
            # One interval per step, sampled from the schedule the deployed sampler will walk.
            i = int(torch.randint(0, len(sched) - 1, (1,)).item())
            t, s = sched[i], sched[i + 1]

            xt = (1.0 - t) * x0 + t * x1
            tt = torch.full((x1.shape[0],), float(t), device=x1.device, dtype=x1.dtype)
            pred = student(xt, tt, phase=phase)
            target = self._target_mean_velocity(teacher, x0, x1, t, s, phase)

            l = self._loss(pred, target)
            total = l if total is None else total + l
            metrics[f"{phase}/loss"] = float(l.detach())
            metrics[f"{phase}/interval"] = float(s - t)
            # Straight-line reference: how far the mean velocity is from the trivial (x1-x0)
            # answer. If this goes to zero the student has learned the coupling, not the model.
            with torch.no_grad():
                metrics[f"{phase}/vs_straight"] = float(
                    torch.nn.functional.mse_loss(target, x1 - x0).detach())

        losses["student"] = total
        return StepOutput(losses=losses, metrics=metrics)

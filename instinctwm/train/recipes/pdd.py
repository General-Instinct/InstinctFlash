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


def shifted_sigmas(nfe: int, shift: float,
                   sigma_min: float = 0.003 / 1.002, sigma_max: float = 1.0) -> tuple[float, ...]:
    """The sigma edges an `nfe`-step FlowMatchScheduler actually visits, ending at 0.

    Reproduces `FlowMatchScheduler.set_timesteps` exactly, including the SNR shift, because the
    student has to be trained on the intervals it will be asked to jump at serving time. Training on
    a uniform linspace instead would put the training intervals in the wrong places -- and LingBot-VA
    runs shift=5.0 on video against 1.0 on action, so "uniform" is wrong by a lot on one stream and
    right by accident on the other.

    The trailing 0 mirrors the server's `F.pad(timesteps, (0, 1), value=0)`: the last step lands on
    clean data, and `FlowMatchScheduler.step` uses sigma_=0 there.
    """
    sig = [sigma_max + (sigma_min - sigma_max) * i / max(1, nfe - 1) for i in range(nfe)]
    sig = [shift * s / (1.0 + (shift - 1.0) * s) for s in sig]
    return tuple(sig) + (0.0,)


class ParallelDecoding:
    name = "parallel_decoding_distillation"

    def __init__(self, nfe: Mapping[str, int] | Sequence[int], *, sub_intervals: int = 4,
                 interval_conditioning: str = "none", loss: str = "mse",
                 shifts: Mapping[str, float] | None = None):
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
        #: phase -> FlowMatchScheduler shift. Defaults are LingBot-VA's served values
        #: (va_robotwin_cfg: snr_shift=5.0, action_snr_shift=1.0); `build()` refuses if a
        #: phase is trained without one rather than assuming 1.0.
        self._shifts = dict(shifts) if shifts is not None else {"video": 5.0, "action": 1.0}

    def requires(self) -> Capabilities:
        # K SEQUENTIAL teacher calls per phase: the target integrates the teacher's ODE, and each
        # sub-step depends on the previous one. See `_target_mean_velocity` for why this cannot be
        # collapsed into one batched call, which is what the first version wrongly did.
        return Capabilities(jvp_through_attention=False, adversarial=False,
                            teacher_calls_per_step=self.sub_intervals * len(self.nfe),
                            aux_modules=(), min_gpus=1)

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
        # No auxiliary networks -- the student is the only trainable module, which is what makes
        # this the cheapest recipe to bring up first.
        #
        # Per-phase SNR shift is not cosmetic: LingBot-VA serves video at shift=5.0 and action at
        # 1.0, so the two streams visit genuinely different sigma grids. Training both on one grid
        # would teach the student to jump between points the sampler never visits.
        missing = [ph for ph in self.nfe if ph not in self._shifts]
        if missing:
            raise ValueError(
                f"no SNR shift given for phase(s) {sorted(missing)}. Pass shifts={{'video': 5.0, "
                f"'action': 1.0}} matching the served config's snr_shift / action_snr_shift; "
                f"guessing would silently train on the wrong sigma grid.")
        return RecipeState(
            modules={}, optimizers={}, update_order=("student",),
            extra={"sub_intervals": self.sub_intervals,
                   "schedules": {ph: shifted_sigmas(n, self._shifts[ph])
                                 for ph, n in self.nfe.items()},
                   "shifts": dict(self._shifts),
                   "interval_conditioning": self.interval_conditioning})

    # -- the objective ------------------------------------------------------------------------

    def _target_mean_velocity(self, teacher, x_t, sigma_t, sigma_s, phase, cond=None):
        """Mean velocity of the TEACHER'S OWN ODE trajectory from sigma_t to sigma_s.

        WHY THIS IS SEQUENTIAL, AND WHY THE BATCHED VERSION WAS WRONG.

        The first implementation stacked K points along the straight coupling
        x_u = (1-u) x_data + u x_noise, evaluated the teacher at all of them in ONE batched call,
        and averaged. That is much cheaper, and it is not the right target.

        For a *given* (x_data, noise) pair the flow-matching training target is `noise - x_data`,
        constant in sigma -- so averaging the teacher along that line converges to the straight-line
        answer, which the student can get without a teacher at all. What a few-step student actually
        has to reproduce is the many-step sampler, and the sampler follows the ODE of the LEARNED
        field, which approximates E[noise - x_data | x_sigma] and is therefore curved. The gap
        between the straight line and that curve is precisely why few-step sampling degrades, so a
        target built on the straight line cannot teach the student to close it.

        So: integrate the teacher, K Euler sub-steps, matching the sampler's own update rule
        (`FlowMatchScheduler.step`: x <- x + v * (sigma_next - sigma)). Then the mean velocity over
        the interval is the secant, (x_s - x_t) / (sigma_s - sigma_t), because that is exactly the
        velocity a ONE-step jump would need to land where K steps landed.

        The cost is K sequential teacher calls, not one batched call. That is the honest price of
        the correct target, and `requires()` declares it.
        """
        import torch
        K = self.sub_intervals
        edges = [sigma_t + (sigma_s - sigma_t) * k / K for k in range(K + 1)]
        x = x_t
        with torch.no_grad():
            for k in range(K):
                sig, sig_next = edges[k], edges[k + 1]
                v = teacher(x, self._sigma_to_timestep(sig, x), phase=phase, cond=cond)
                x = x + v * (sig_next - sig)
        return (x - x_t) / (sigma_s - sigma_t)

    @staticmethod
    def _sigma_to_timestep(sigma: float, like):
        """The backbone is conditioned on sigma * num_train_timesteps, not on sigma.

        `FlowMatchScheduler` sets `timesteps = sigmas * num_train_timesteps` (1000), and that scaled
        value is what reaches the transformer. Passing sigma directly would silently condition the
        model on a timestep ~1000x too small -- it would still train, just on the wrong thing.
        """
        import torch
        return torch.full((like.shape[0],), float(sigma) * 1000.0,
                          device=like.device, dtype=like.dtype)

    def _loss(self, pred, target):
        import torch
        if self.loss == "huber":
            return torch.nn.functional.huber_loss(pred, target, delta=1.0)
        return torch.nn.functional.mse_loss(pred, target)

    def step(self, batch, teacher, student, state: RecipeState) -> StepOutput:
        """One PDD step over every phase named in `nfe`.

        `batch` carries, per phase, the clean sample (`<phase>/x1`), optionally the noise that pairs
        with it (`<phase>/x0`), and optionally whatever conditioning the backbone needs
        (`<phase>/cond`, passed through opaquely -- the recipe never interprets it). The noise comes
        from the batch when present so a step is reproducible from the dataloader alone, the same
        reason `probe_bitexact` insists on `--deterministic-seed`.
        """
        import torch
        losses, metrics = {}, {}
        total = None
        schedules = state.extra["schedules"]

        for phase, sched in schedules.items():
            x_data = batch[f"{phase}/x1"]
            noise = batch.get(f"{phase}/x0")
            if noise is None:
                noise = torch.randn_like(x_data)
            cond = batch.get(f"{phase}/cond")

            # One interval per step, drawn from the schedule the deployed sampler will walk.
            i = int(torch.randint(0, len(sched) - 1, (1,)).item())
            sigma_t, sigma_s = sched[i], sched[i + 1]

            # x_sigma = (1 - sigma) * data + sigma * noise, matching FlowMatchScheduler.add_noise.
            x_t = (1.0 - sigma_t) * x_data + sigma_t * noise
            pred = student(x_t, self._sigma_to_timestep(sigma_t, x_t), phase=phase, cond=cond)
            target = self._target_mean_velocity(teacher, x_t, sigma_t, sigma_s, phase, cond=cond)

            l = self._loss(pred, target)
            total = l if total is None else total + l
            metrics[f"{phase}/loss"] = float(l.detach())
            metrics[f"{phase}/d_sigma"] = float(sigma_s - sigma_t)
            # How far the teacher's trajectory bends away from the straight coupling over this
            # interval. This is the quantity the student is actually being taught: if it were ~0 the
            # target would be reachable with no teacher at all and the recipe would be pointless.
            # Logged precisely because the first version of this objective got that wrong.
            with torch.no_grad():
                metrics[f"{phase}/curvature"] = float(
                    torch.nn.functional.mse_loss(target, noise - x_data).detach())

        losses["student"] = total
        return StepOutput(losses=losses, metrics=metrics)

"""PDD as an InstinctWM Recipe: per-stream composition over the backbone-agnostic core.

The algorithm lives in `instinctwm.train.pdd`, which knows nothing about world-action models and is
destined to split out as `instinct-pdd`. THIS file is the orchestration half, and it owns exactly
the three things the algorithm should not:

  * running PDD over SEVERAL streams. LingBot-VA denoises video then action, on different SNR-shifted
    grids. PDD is single-stream; applying it per phase is composition, not method.
  * the `DescriptorDelta`, so the runtime can execute the student with no glue.
  * the data-free rollout state per stream, which is stateful across optimisation steps.

REPRODUCTION SETTINGS. Defaults follow the paper's primary configuration rather than anything tuned
for this box: N = 256 intervals, Euler solver, uniform k within the block, MSE, no EMA, no loss
weighting. The 2-NFE target gives L = 128. These are deliberately not tunable-by-accident -- a
failed reproduction with altered hyperparameters is uninterpretable, so the knobs exist but the
defaults are the paper's.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from instinctwm.train.pdd import Grid, Rollout, pdd_loss
from instinctwm.train.recipe import (
    Capabilities, DescriptorDelta, Environment, RecipeState, StepOutput,
)

#: LingBot-VA's served guidance, from va_robotwin_cfg. PDD folds guidance into the teacher target,
#: so the student learns already-guided trajectories and needs no CFG branch at inference. That
#: fixes the scales at distillation time -- serving-time guidance becomes a property of the
#: checkpoint, not a knob. Stated here because it is a real capability being traded away.
DEFAULT_GUIDANCE = {"video": 5.0, "action": 1.0}

#: FlowMatchScheduler shift per stream, also from va_robotwin_cfg (snr_shift / action_snr_shift).
DEFAULT_SHIFTS = {"video": 5.0, "action": 1.0}


def _for_phase(obj, phase: str, method: str):
    """Resolve the per-stream oracle out of whatever container the caller used.

    Deliberately duck-typed on the protocol method rather than on `isinstance(obj, Mapping)`. The
    natural way to hold per-phase modules in torch is `nn.ModuleDict`, which supports `[]`, `keys()`
    and iteration but is NOT a `collections.abc.Mapping` -- so a Mapping check silently passes the
    whole container through as if it were a single model, and the failure surfaces much later as a
    missing attribute. Checking for the method first makes a single shared model work too, which is
    what a single-stream backbone wants.
    """
    if hasattr(obj, method):
        return obj
    try:
        return obj[phase]
    except (KeyError, TypeError) as e:
        raise TypeError(
            f"cannot resolve the {phase!r} stream: {type(obj).__name__} has no .{method}() and is "
            f"not indexable by phase name. Pass either one oracle exposing .{method}(), or a "
            f"dict/ModuleDict keyed by phase.") from e


class ParallelDecoding:
    name = "parallel_decoding_distillation"

    def __init__(self, nfe: Mapping[str, int] | Sequence[int], *,
                 n_intervals: int = 256,
                 solver: str = "euler",
                 loss: str = "mse",
                 shifts: Mapping[str, float] | None = None,
                 guidance: Mapping[str, float] | None = None,
                 time_scale: float = 1000.0,
                 data_free: bool = True):
        """`nfe` maps phase name -> student NFE, e.g. {"video": 2, "action": 2}.

        Per-phase because the modality split is the one thing a WAM cannot get wrong: a single
        objective across both streams is what collapses the action head while video quality holds.
        The platform cannot prevent that mistake, but it can refuse to make it expressible only by
        accident.
        """
        if not isinstance(nfe, Mapping):
            raise TypeError(
                "nfe must map phase name -> steps, e.g. {'video': 2, 'action': 2}. A bare list "
                "would silently apply one step count to both streams, which is the failure mode "
                "this argument exists to prevent.")
        self.nfe = dict(nfe)
        self.n_intervals = n_intervals
        self.solver = solver
        self.loss = loss
        self.time_scale = time_scale
        self.data_free = data_free
        self._shifts = dict(shifts) if shifts is not None else dict(DEFAULT_SHIFTS)
        self._guidance = dict(guidance) if guidance is not None else dict(DEFAULT_GUIDANCE)

        for ph, k in self.nfe.items():
            if n_intervals % k:
                raise ValueError(
                    f"phase {ph!r}: NFE={k} does not divide N={n_intervals}, so the block size "
                    f"L = N/NFE is not an integer and the last block would run off the grid. "
                    f"The paper's N=256 admits NFE in {{1,2,4,8,...}}.")

    # -- declarations the platform reads ---------------------------------------------------------

    def requires(self) -> Capabilities:
        # No JVP (that is the point of choosing PDD over sCM/rCM on a box with no flash-attn) and no
        # adversary. Teacher cost is one call per supervised interval for Euler, two for midpoint --
        # per stream, per step.
        per_call = 2 if self.solver == "midpoint" else 1
        return Capabilities(jvp_through_attention=False, adversarial=False,
                            teacher_calls_per_step=per_call * len(self.nfe),
                            aux_modules=(), min_gpus=1)

    def descriptor_delta(self, model) -> DescriptorDelta:
        known = {p.name for p in model.execution.phases}
        unknown = set(self.nfe) - known
        if unknown:
            raise ValueError(
                f"nfe names phases {sorted(unknown)} that {model.execution.model_id!r} does not "
                f"have; it has {sorted(known)}")
        note = (f"PDD N={self.n_intervals} solver={self.solver} "
                f"{'data-free' if self.data_free else 'data-based'}; guidance distilled in at "
                + ", ".join(f"{k}={v}" for k, v in sorted(self._guidance.items())))
        return DescriptorDelta(nfe=dict(self.nfe), note=note)

    def build(self, model, env: Environment) -> RecipeState:
        missing = [ph for ph in self.nfe if ph not in self._shifts]
        if missing:
            raise ValueError(
                f"no SNR shift declared for phase(s) {sorted(missing)}. LingBot-VA serves video at "
                f"shift=5.0 and action at 1.0, so the two streams visit different sigma grids; "
                f"guessing 1.0 would train the student to jump between times the sampler never "
                f"visits.")
        grids = {
            ph: Grid.from_shift(self.n_intervals, self.n_intervals // k,
                                shift=self._shifts[ph], scale=self.time_scale)
            for ph, k in self.nfe.items()
        }
        return RecipeState(
            modules={}, optimizers={}, update_order=("student",),
            extra={"grids": grids, "rollouts": {}, "solver": self.solver, "loss": self.loss,
                   "guidance": dict(self._guidance), "data_free": self.data_free})

    # -- the step --------------------------------------------------------------------------------

    def step(self, batch, teacher, student, state: RecipeState) -> StepOutput:
        """One optimisation step: one block per stream.

        `student` and `teacher` must be mappings phase -> velocity oracle, because LingBot-VA's two
        streams are different entry points into one backbone (action_mode toggles which). Resolving
        that here rather than in the core is what keeps the core backbone-agnostic.
        """
        import torch

        grids = state.extra["grids"]
        losses, metrics = {}, {}
        total = None

        for phase, grid in grids.items():
            t_model = _for_phase(teacher, phase, "velocity")
            s_model = _for_phase(student, phase, "heads")
            cond = batch.get(f"{phase}/cond")

            if state.extra["data_free"]:
                roll = state.extra["rollouts"].get(phase)
                if roll is None:
                    shape_src = batch[f"{phase}/noise_like"]
                    roll = Rollout(grid, lambda s=shape_src: torch.randn_like(s))
                    state.extra["rollouts"][phase] = roll
                n, x_n = roll.begin_block()
                k = roll.pick_k(n)
            else:
                # Algorithm 2: X_n from the interpolant against a real sample. Kept because a fixed
                # batch makes the target deterministic, which is what an overfit test needs.
                x_data = batch[f"{phase}/x1"]
                noise = batch.get(f"{phase}/x0")
                if noise is None:
                    noise = torch.randn_like(x_data)
                starts = grid.block_starts()
                n = int(starts[torch.randint(0, len(starts), (1,)).item()])
                sigma = grid.times[n]          # already the ODE variable, unscaled
                x_n = (1.0 - sigma) * x_data + sigma * noise
                k = int(torch.randint(n, min(n + grid.block, grid.n_intervals), (1,)).item())

            # ONE student forward per stream per step. The same head output is regressed against the
            # teacher and used to advance the rollout, exactly as the paper prescribes; computing it
            # twice would double the dominant cost of the step.
            heads = s_model.heads(x_n, grid.cond(n), cond=cond)
            value, m = pdd_loss(s_model, t_model, x_n, grid, n, k,
                                cond=cond, solver=state.extra["solver"],
                                loss=state.extra["loss"], heads=heads)
            total = value if total is None else total + value
            for key, v in m.items():
                metrics[f"{phase}/{key.split('/', 1)[1]}"] = v

            if state.extra["data_free"]:
                state.extra["rollouts"][phase].advance(heads.detach())
                metrics[f"{phase}/traj"] = float(state.extra["rollouts"][phase].trajectories)

        losses["student"] = total
        return StepOutput(losses=losses, metrics=metrics)

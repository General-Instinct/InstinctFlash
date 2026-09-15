"""pi05 few-step adapter — EXPERIMENTAL: declarations landed, trainer seamed.

pi05 is the single-stream instantiation, and it is the family with TWO measured untrained
frontiers whose disagreement is itself the finding:

  * R1 (research log 2026-08-21; LIBERO spatial n=100, nas=10, batch=10 lockstep envs, one
    global RNG stream): nfe=1 98.0 == nfe=10 98.0 (p=1.0), nfe=2 88.0 (−10 pp, p=0.006) —
    the non-monotone cliff.
  * E9 (research log 2026-09-01; the schedule-sweep campaign: 4 LIBERO suites x 10 tasks x
    4 seeds = 160 pairs/point, per-episode seeding, zero-discordance noise floor): the
    frontier is FLAT — every point nfe 1..5 within ±3.5 pp of the 10-step teacher (nfe2
    +0.6 pp [−2.6, +3.9]; nfe1 +1.3 pp [−2.2, +4.7]); the R1 cliff DOES NOT REPLICATE,
    including on spatial itself (+2.5 pp at nfe2). Open-loop, at identical noise, chunk
    deviation from nfe10 is smoothly MONOTONE in step count — no t=0.5 grid-point anomaly
    (Euler step size dominates), and nfe1's far-off chunks (median cos 0.81) still tie on
    task success.

Consequences carried by this adapter:

  * schedule quality is PROTOCOL-COUPLED: the same (checkpoint, nfe) measured −10 pp under
    one RNG-draw structure and +0.6 pp under another. Per-(NFE, protocol) evaluation is
    mandatory; certification never transfers across NFE or across eval protocol;
  * grid placement (H3, the shift override) remains available but is DOWNGRADED from "the
    proven lever" to an open lever: both refutations above say step size, not grid-point
    placement, sets the untrained cost at this family's scale;
  * the matched-NFE control is not optional: the untrained schedule already ties the teacher
    at EVERY budget on the E9 harness, so a trained few-step student must beat a ~0.96-0.97
    standing control (pooled_frontier.json rows), not a strawman.

THE EVIDENCE BASE (`evidence()` -> distill/evidence/pi05_libero_schedule_frontier.json): the E9
frontier registered as data — nfe 5..1 rows with paired deltas, intervals, per-suite success,
t-grids and per-shard latency (174 -> 51 ms at nfe1), the zero-discordance noise floor. The
screen verdict over it is NO DISTILLATION NEEDED at every nfe (deciding lower bound above the
−0.05 margin at 160 pairs); the untrained nfe1 point is what goes to certification
(iwm_distill/fewstep/cert_pi05_nfe1_prereg.md). pi05 has no guidance axis: the family serves no
negative branch, so its operating point is (schedule grid, none@1, batch-1) and the control gate
demands no grid here.

TRAINING SEAM: pi05 has no committed few-step training campaign yet; distillation on this
family currently composes with InstinctCompress (QAD-KD, the verified W4 recipe) rather than
with a step trainer. build_student()/consistency_target() raise until a pi05 few-step campaign
declares its recipe. The distill×quantize ORDERING is an open experiment (RFC §7).
"""

from __future__ import annotations

from typing import Any, Mapping

from instinctflash.distill.adapter import (
    ExperimentalSeam,
    FamilyAdapter,
    ScheduleGrid,
    StreamSpec,
    TrainableSet,
    register_family,
)

_SEAM = (
    "pi05 few-step training has no committed campaign; the declarations exist so screening "
    "(schedule sweeps, H3 grid search) runs today. A training recipe joins by filling this "
    "method. See docs/rfc/fewstep-distillation.md §5."
)


class Pi05FewStep(FamilyAdapter):
    family = "pi05"

    def streams(self) -> tuple[StreamSpec, ...]:
        return (
            StreamSpec(
                name="action",
                scheduler="flow_match_shift",
                shift=1.0,  # linear grid; the lever is the shift override (H3)
                teacher_nfe=10,
                guidance_scale=1.0,
                guidance_mode="none",
                cfg_batched=False,
                commit_forwards=0,
                certified_nfe=None,  # nfe=1 ties the teacher but is screened, not certified
                notes=(
                    "E5 (spatial n=100, batched-RNG protocol): nfe1 == nfe10, nfe2 −10 pp. "
                    "E9 (4 suites, 160 pairs/point, per-episode seeding): FLAT to nfe1, the "
                    "nfe2 cliff does not replicate — schedule quality is protocol-coupled; "
                    "evaluate per (NFE, protocol), never interpolate between measured budgets"
                ),
            ),
        )

    def trainable_set(self, stream: str) -> TrainableSet:
        if stream == "action":
            return TrainableSet(
                stream="action",
                unfreeze=("action_expert",),
                keep_frozen=("paligemma", "vision_tower"),
                note=(
                    "the same seam InstinctCompress's QAD-KD trains through (the unquantized "
                    "remainder against the original checkpoint); a few-step recipe would reuse "
                    "that trainer shape"
                ),
            )
        raise KeyError(f"pi05 has no stream {stream!r}")

    def build_student(self, teacher: Any, schedule: Mapping[str, ScheduleGrid]) -> Any:
        raise ExperimentalSeam(_SEAM)

    def consistency_target(
        self, stream: str, batch: Any, teacher: Any, grid: ScheduleGrid
    ) -> Any:
        raise ExperimentalSeam(_SEAM)


register_family(Pi05FewStep())

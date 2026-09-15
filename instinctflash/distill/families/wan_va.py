"""LingBot-VA (wan_va) few-step adapter — EXPERIMENTAL: declarations landed, trainer seamed.

The declarations below are measured facts, each with its artifact:

  * two streams over one backbone (action_mode toggles the entry point), video shift-5 /
    action shift-1, video CFG g=5 batch-2 on every forward, action negative branch computed
    then DISCARDED (mapping memo §0.5/§0.8; the same constants as
    `train/recipes/pdd.py` DEFAULT_SHIFTS/DEFAULT_GUIDANCE);
  * 2 kv-commit forwards per cycle are protocol, not denoise steps (C5);
  * the certified operating point is UNTRAINED 2V/4A (n=1153 paired, NON-INFERIOR robust,
    `thor_column/va_2v4a_certificate_final.txt`) — so `certified_nfe` is video 2 / action 4,
    and H2's reduced-step teacher is exactly this sampler's trajectory;
  * the paired frontier at the SHIPPED guidance said the cliff is video-shaped: 1V/4A −16.8 pp
    untrained vs 2V/2A −3.1 pp (frontier E4) — which is why the video trainable set unfreezes
    the trunk and the action stream is untouched BY CONSTRUCTION (H1).

THE EVIDENCE BASE (`evidence()` -> distill/evidence/wan_va_robotwin_guidance_frontier.json;
research log 2026-09-02, h1_report.md, gw_autotable.md) — the facts the campaign then measured,
registered as data rather than as this sentence:

  * the 1V cliff was a GUIDANCE artifact, not a step-count artifact: the same 1V/4A schedule
    scores 0.752 at w=5, 0.882 at w=3, 0.885 at w=1 (video positive-only, batch-1), 0.276 at
    w=9 on the same pinned scenes. An operating point is (schedule grid, per-stream guidance
    scale, CFG batching); nfe alone underspecifies it by 60 points (RFC §11);
  * with the video negative branch off the frontier is flat in video steps (2->1 +3.0 pp
    n.s. at 2A) and in action steps down to 2 (4->2 +0.9 pp n.s.), then cliffs at 1 action
    step (−20 pp): the action stream needs >= 2 steps, the video stream 1 once guidance is off;
  * the trained H1 student bought +1.3 pp (p=0.70) at 1V/4A and +1.7 pp (p=0.59) at 1V/2A over
    its execution-matched untrained control, and exactly +0.0 over the best untrained knob
    (1V/4A@w3). Verdict: DO NOT TRAIN; certify UNTRAINED 2V/2A@w1 first (+0.9 pp
    [−0.035, +0.058] vs the baseline arm, 8 batch-1 forwards), 1V/2A@w1 second. The standing
    matched-NFE controls are the evidence rows: per schedule, the best over the guidance grid.

THE TRAINING SEAM (do not quietly fill): build_student()/consistency_target() raise
ExperimentalSeam. The H1 campaign (video-stream-only consistency: VP/LCM parametrization at
shift-5 per Flash-WAM's video half, Huber, EMA 0.995, CFG folded per H4, targets from the
certified 2V/4A sampler per H2; `iwm_distill/fewstep/h1_prereg.md`, `h1/train_h1_video_cd.py`)
ran 2026-08-31 -> 09-01 and returned the verdict above; its script is the shape any wan_va
few-step recipe would take, and the seam stays a seam because the evidence says training is
not where this family's next operating point comes from.
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
    "wan_va few-step training is owned by the H1 campaign (video-stream-only consistency); "
    "wire this method to the campaign's training script per its prereg "
    "(iwm_distill/fewstep/h1_prereg.md) when it lands. The declarations (streams, grids, "
    "trainable sets) are complete and tested; only the training half is seamed. "
    "See docs/rfc/fewstep-distillation.md §5."
)


class WanVAFewStep(FamilyAdapter):
    family = "wan_va"

    def streams(self) -> tuple[StreamSpec, ...]:
        return (
            StreamSpec(
                name="video",
                scheduler="flow_match_shift",
                shift=5.0,
                teacher_nfe=25,
                guidance_scale=5.0,
                guidance_mode="cfg",
                cfg_batched=True,
                commit_forwards=1,  # the pred-commit forward that closes the loop (E8)
                certified_nfe=2,
                notes=(
                    "high-noise regime; the untrained 2->1 cliff at the shipped w=5 (E4, −16.8 pp) "
                    "is a GUIDANCE artifact: 1V/4A@w1 and @w3 sit within ~3.5 pp of the baseline "
                    "(evidence(): wan_va_robotwin_guidance_frontier.json)"
                ),
            ),
            StreamSpec(
                name="action",
                scheduler="flow_match_shift",
                shift=1.0,
                teacher_nfe=50,
                guidance_scale=1.0,
                guidance_mode="positive_only",
                cfg_batched=True,
                commit_forwards=1,
                certified_nfe=4,
                notes=(
                    "linear grid; negative branch computed then discarded today — dropping it "
                    "is a serving change with its own certificate, independent of training (H4)"
                ),
            ),
        )

    def trainable_set(self, stream: str) -> TrainableSet:
        if stream == "video":
            return TrainableSet(
                stream="video",
                unfreeze=("transformer",),
                keep_frozen=("text_encoder", "vae"),
                note=(
                    "H1 trains the trunk through the video entry point; heads-only capacity "
                    "already tied its control at 2V (PDD post-mortem E6), and the open "
                    "question — 1V — sits on the untrained cliff"
                ),
            )
        if stream == "action":
            return TrainableSet(
                stream="action",
                unfreeze=(),
                note=(
                    "H1: the action stream needs nothing at >=4 steps (certified 2V/4A; "
                    "4A ~= 50A in the sweep) and barely profits from training even at 1-2 "
                    "steps elsewhere (SB control: +0.3 pp); preserving its distribution "
                    "bit-for-bit is the property a policy certificate cares about"
                ),
            )
        raise KeyError(f"wan_va has no stream {stream!r}")

    # -- the H1 seam --------------------------------------------------------------------------

    def build_student(self, teacher: Any, schedule: Mapping[str, ScheduleGrid]) -> Any:
        raise ExperimentalSeam(_SEAM)

    def consistency_target(
        self, stream: str, batch: Any, teacher: Any, grid: ScheduleGrid
    ) -> Any:
        raise ExperimentalSeam(_SEAM)


register_family(WanVAFewStep())

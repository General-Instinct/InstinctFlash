"""InstinctFlash few-step distillation — the trainer tier of the framework.

    screen  ->  train  ->  verify

THREE PILLARS, TWO HOMES. The framework is (1) schedule screening, (2) the trainer, (3)
certification. Screening lives in `benchmarks/vla/schedule_sweep.py` and certification in
`instinctflash/verify/certify.py` + `descriptors/distillation.py`.
This package contains the training interfaces and currently available recipes. Nothing
under `instinctflash/runtime/` or `serving/` imports it; a distilled checkpoint is served through its declaration exactly like
any other checkpoint (CHECKPOINTS doctrine: recipes make checkpoints, never runtimes).

WHAT ADAPTS TO ALL MODELS. A model family joins the framework by declaring, in a
`FamilyAdapter` (`adapter.py`):

    streams()            the denoise streams and their schedule/guidance facts
    schedule_grid()      the EXACT t-grid the deployed sampler visits at a given NFE
    trainable_set()      which parameters a stream's distillation may touch
    build_student()      student construction (init = teacher, per stream)
    consistency_target() the teacher-side target for one training step

Everything else — the sweep machinery, the trainer loop (`train/trainer.py`), the three-arm
verification, the provenance stamp — is family-agnostic and reads only those declarations.

THE HARD GATE. `pipeline.verify_point` refuses to produce a report for a trained point unless
the untrained matched-NFE control arm is present at the identical OPERATING POINT — (schedule
grid, per-stream guidance scale, CFG batching) — and is the BEST untrained configuration over
the guidance grid swept at that schedule; it states B−A and C−B separately. This is enforced in
code (`ControlGateViolation`), not by convention: our own PDD post-mortem is the precedent — the
control turned "it works" into "+0.0" (methods memo E6) — the H1 screen repeated it against the
guidance axis (+14.3 pp vs the shipped w=5, +0.0 vs the best untrained knob), and the literature
omits it almost universally (§2.1).

THE VERDICT "NO DISTILLATION NEEDED" IS FIRST-CLASS. `screen_verdict` reads a frontier report
before any trainer runs; when the best untrained configuration at the target schedule already
clears the margin, `train()` raises `NoDistillationNeeded` carrying the artifact owed — the
frontier, the intervals, and a certification prereg stub for the winning untrained point.

The command surface is `python -m instinctflash.distill steps <ckpt> <out> --schedule ...
--dataset ...`. A family's unfinished training recipe raises `ExperimentalSeam`; publishing
the source does not qualify that recipe or its output checkpoint for deployment.
"""

from instinctflash.distill.adapter import (
    ExperimentalSeam,
    FamilyAdapter,
    ScheduleGrid,
    StreamSpec,
    TrainableSet,
    flow_match_times,
    get_family,
    register_family,
    registered_families,
)
from instinctflash.distill.pipeline import (
    ControlGateViolation,
    DistillPipeline,
    NoDistillationNeeded,
    ScreenVerdict,
    ThreeArmReport,
    canonical_schedule,
    screen_verdict,
    verify_point,
)

__all__ = [
    "ControlGateViolation",
    "DistillPipeline",
    "ExperimentalSeam",
    "FamilyAdapter",
    "NoDistillationNeeded",
    "ScheduleGrid",
    "ScreenVerdict",
    "StreamSpec",
    "ThreeArmReport",
    "TrainableSet",
    "canonical_schedule",
    "flow_match_times",
    "get_family",
    "register_family",
    "registered_families",
    "screen_verdict",
    "verify_point",
]

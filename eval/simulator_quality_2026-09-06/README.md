# Simulator evaluation evidence — 2026-09-06

All five currently runnable model families have completed 10 tasks × two seeds
per setting, with separate original/candidate arms. RoboTwin clean/randomized
settings are reported separately: `robotwin50_easy` is clean and `robotwin50_hard`
is randomized. These suite IDs name the 50-task catalog; this screening selects
the first 10 tasks in the frozen registry order, without outcome-based selection.
The earlier LingBot-VA 2V/4A comparison remains
an additional one-task smoke result and is not promoted to 10-task coverage.

| Model | Suite / setting | Success, original → candidate | Delta pp [central 95% interval] | Identical action pairs |
| --- | --- | --- | --- | --- |
| LingBot-VA LIBERO | wan_va_libero_long | 19/20 → 19/20 | +0.0 [-16.1, +16.1] | 20/20 |
| pi05 LIBERO | libero_10 | 19/20 → 19/20 | +0.0 [-16.1, +16.1] | 20/20 |
| GR00T N1.7 LIBERO | groot_libero_10 | 17/20 → 17/20 | +0.0 [-16.1, +16.1] | 20/20 |
| LingBot-VLA 4B RoboTwin | robotwin50_easy | 15/20 → 15/20 | +0.0 [-16.1, +16.1] | 20/20 |
| LingBot-VLA 4B RoboTwin | robotwin50_hard | 15/20 → 15/20 | +0.0 [-16.1, +16.1] | 20/20 |
| LingBot-VLA V2 RoboTwin | robotwin50_easy | 18/20 → 18/20 | +0.0 [-16.1, +16.1] | 2/20 |
| LingBot-VLA V2 RoboTwin | robotwin50_hard | 17/20 → 17/20 | +0.0 [-16.1, +16.1] | 2/20 |
| LingBot-VA RoboTwin (smoke) | robotwin50_easy | 1/1 → 1/1 | +0.0 [-79.3, +79.3] | 0/1 |
| LingBot-VA RoboTwin (smoke) | robotwin50_hard | 1/1 → 1/1 | +0.0 [-79.3, +79.3] | 0/1 |

These are **screening results, not quality certificates**. The intervals assume
matched Bernoulli episode pairs and do not model task clustering. Every task and
unsuccessful rollout is retained. Matched executed actions cover these observations
and seeds only; hidden states, untested inputs and Thor are outside this evidence.
V2's numeric changes remain numeric even when success rates match. The earlier
clean pilot observed 18/20 → 17/20; it is retained in the reset incident and pilot
diagnostics, not pooled into the new run. Differences across these runs are not
a controlled attribution experiment.
[V2 trace diagnostics](vla2-trace-diagnostics.json) record the first differing
controller step; later same-index actions have different observations and are not
isolated kernel-error measurements. Reproduce with
`python trace_diagnostics.py --run /path/to/vla2/evidence --output diagnostics.json`
using the installed InstinctFlash package. 2V/4A is an
explicit behavioral change. No reduced NFE or distillation was added to the new
pi05, GR00T, 4B or V2 comparisons.

pi05 uses the same verified 812 checkpoint tensors, NFE=10, a 50-action generated
chunk and 10 executed actions, TF32/cuDNN benchmark enabled and compilation disabled.
Its candidate installs the adapter's hoists/static-KV capture on the same LeRobot
policy; it does not evaluate the full Runtime facade/FP32 precision lease. Startup
capture checks and any eager fallback are recorded in each result. The old pi05
original-repeat smoke remains historical baseline evidence.

GR00T uses its pinned LIBERO fine-tune and native reset/action processor. The model
families have different native reset and controller contracts; these scores are
within-model comparisons and must not be used as a cross-model leaderboard.
RoboTwin uses `fresh-clutter-bounds-v1`: upstream mutable default bounds otherwise
accumulate table offsets across resets. Both preparation and rollout now copy
those bounds per call. Cold-reset and post-expert observations were checked for
identical hashes on the previously failing randomized scene. The two corrected
campaigns use fresh frozen scenes and fresh runs; earlier incomplete runs remain
under `joint/` as infrastructure diagnostics and are excluded from these scores.

The corrected RoboTwin runs execute the two arms of each pair concurrently on
independent policy endpoints, with a barrier before the next pair. The scheduling
choice is bound in the run environment.
Shared H100 inference/rendering is suitable for this paused quality evaluation;
its wall-clock timings are not isolated speedups or a Thor realtime measurement.

The [workflow guide](../../benchmarks/vla/SIMULATOR_EVALUATION.md) documents installed
planning, isolated resumable scene preparation, execution, reporting and export.
[screening-index.json](screening-index.json) links each immutable plan/report to its
verified portable evidence bundle on this experiment host. Bundles include full
results/actions, requests, logs, environment, registry and frozen scenes. The JSON
reports in this directory are summaries, not substitutes for the raw bundles.
[Validation details](validation.json) retain test coverage, optional-dependency skips
and the unresolved existing GPU performance-ratio assertion; it is not a simulator
quality failure. The [reset incident](robotwin-reset-incident.md) records the
RoboTwin repair and frozen execution identities.
Older smoke reports are retained as historical evidence; `screening-index.json`
identifies the current primary comparisons.

Cosmos Edge/Nano and DreamZero still need supported RTX rendering hardware and
native DROID simulator adapter integration. Their official Isaac-based routes are
listed explicitly as unevaluated. H100 lacks the required RT cores; see the
[Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html).
An RTX simulator may connect to H100 inference, but that deployment is not part of
these completed runs.

# Same-Thor original controls: preparation

The deployed student screen currently compares Thor students with H100 original
controls. Additional Thor original controls will separate checkpoint/sampler
changes from cross-device execution differences on the same historical requests.
They are diagnostic controls, not fresh confirmation data.

The cached Thor revision differs from the H100 historical revision in README
and transformer config bytes. All weight/tokenizer files matched. To avoid
assuming the config additions are irrelevant, an isolated control checkpoint
reuses the 32 identical files and copies the two H100-version files. All 34
frozen inventory entries passed SHA256/size checks. The original cache is unchanged.

`preparation_receipt.json` binds the full external preparation report and the
additional offline declaration. CPU package loading passed with `servable=False`.
The controls are native UniPC4/CFG4/shift1 with native padding, and original-weight
SDE[1,0]/CFG1 and CFG4 with zero action padding. `plan.json` binds their exact
historical request matrix, source hashes and 384-request/1,408-branch budget.
No quality qualification has been issued.

The initial v1 observer incorrectly expected UniPC to receive conditioning as
sampler arguments. It stopped before producing a valid action record; its frozen
source, plan and failure report remain in `failed-v1` and on Thor. Version 2 reads
UniPC conditioning from the native preparation captured by the velocity callback,
while SDE reads actual sampler arguments. It also checks the native model's SDE
sampler shift of zero separately from the explicit generation shift. Three CPU
tests passed using real native samplers: SDE CFG1/CFG4 endpoint checks and UniPC
observational endpoint parity. No model arithmetic was changed by this correction.

The v2 queue runs from `realtime-controls-v2/run_all.py` on Thor. Its live status is
`/home/guanming/ifl_eval/cosmos_distill_20260912/original-controls-live-v2.json`.
The first original UniPC4 request passed: native clocks `[999,749,499,249]`, eight
branches, exact historical command targets. The same-Thor conditioning reference
matched the student's first request, and initial noise matched outside the
intentionally different action-padding coordinates. Cross-device original action
maxabs was 0.0412843 on this one request; this is not a full-panel quality result.

`validate_control.py BANK --source FROZEN_SOURCE --output FRESH_RECEIPT` checks a
completed bank's full 128-request hashes, execution contract, finite fields and
action endpoint slices. Raw generation is instrumented and is not a latency
benchmark. All three banks subsequently completed capture validation and pairing.

`export_control.py BANK FRESH_OUTPUT --source FROZEN_SOURCE --student-source
STUDENT_SOURCE --student-validator VALIDATOR_FILE --study STUDY_ROOT` validates all full banks and compares every
request to both seeds of each applicable native student. The matched SDE controls
require identical full reference/noise/mask/target bytes and normalizers. UniPC4
allows noise/mask differences only in the declared action-padding columns8–63;
references and targets must still match completely. Six CPU tests verify that
this exception cannot hide video, active action or history-coordinate changes.

`score_controls.py COMPACT_ROOT FRESH_REPORT` retains all eight deployed variants
and four within-episode seed means. It compares each with same-Thor UniPC4 and its
matched original SDE1 control (24 comparisons), using the frozen producer math
and a separate implementation for numerical cross-checking. The independent
teacher bank remains on H100; command targets remain the primary endpoints.

The live CPU postprocessor is
`/home/guanming/ifl_eval/cosmos_distill_20260912/controls-postprocess-live-v2.json`.
It runs from `controls-postprocess-v2` without touching the frozen capture code.
The local continuation transferred all three compact banks and executed scoring
into `/home/ubuntu/ifl_cosmos_quality_preflight/same_thor_scores_v2.json`.
Its launch receipt is `same_thor_score_launch_v2.json` in that directory.
The process exited zero after all three postprocessed banks succeeded.

The first postprocessor stopped because it looked for the student validator in
the frozen capture directory; that validator lives in `quality-orchestration-v1`.
Its source and failure log are retained in `postprocess-failed-v1`. Version 2
takes an explicit validator path, checks it before reading the control bank,
and uses new output directories. Only CPU postprocessing was restarted; no GPU
capture or existing data was rerun or changed. `postprocess_all.py` records the
live child PID and waits on the existing capture scheduler.

## Completed same-Thor diagnostic results

All 384 original-control requests passed full validation. UniPC4 inputs matched
all four native students outside the declared noise/mask padding exception;
each original SDE1 control matched both same-CFG student seeds on full inputs.
The independently implemented formulas agreed on 288 control metric vectors and
2,304 bootstrap comparisons across 24 comparison groups. The complete report and
every compact input hash are bound by `same_thor_score_receipt.json`.

The following are relative changes in **energy scores**, not task success rates.
Lower is better. Values average both training runs within each of 16 historical
episodes. These same-Thor controls supersede the earlier cross-device controls
for interpreting student-versus-original deltas; the earlier reports remain frozen.

| Student vs original UniPC4 on Thor | Joint h1 | Gripper h1 | Joint h32 | Gripper h32 |
|---|---:|---:|---:|---:|
| CFG1 native | +15.97% | +2.64% | −0.61% | +4.45% |
| CFG1 cuDNN | +16.58% | +2.45% | −0.70% | +4.47% |
| CFG4 native | +7.84% | −7.06% | +1.44% | +17.65% |
| CFG4 cuDNN | +9.81% | −5.51% | +1.49% | +17.98% |

The uncorrected 95% paired intervals for CFG1's short-horizon joint delta lie
above zero for both attention paths. CFG4's long-horizon gripper interval lies
above zero for cuDNN, while its native interval crosses zero. Most other intervals
cross zero; this does not establish equivalence or noninferiority.

Against its matched untrained SDE1/CFG1 control, the CFG1 cuDNN student's gripper
scores improve by 28.56% at h1 and 13.02% at h32; joint deltas are +2.31% and
+0.39%, with intervals crossing zero. Thus the training changed quality, but it
has not recovered the original four-step policy's complete primary score profile.
All seed-specific and secondary results remain in the full report.

Measured student latency remains about 270–271 ms for CFG1 cuDNN and 436 ms for
CFG4 cuDNN (the separately frozen latency benchmark). These diagnostics do not
admit either checkpoint as a quality-preserving realtime policy. Fresh data,
controller deadline/executed chunk and closed-loop task success remain pending.

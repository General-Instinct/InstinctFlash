# Native V2 execution evidence and budget selection

Two independent original/native-capture A/B repeats completed **160 real episodes**:
10 tasks × 2 seeds × clean/randomized × 2 arms × 2 repeats. The checkpoint is
`robbyant/lingbot-vla-v2-6b-robotwin@0451855729ec904f970600e0aec8b84661423afe`.
Both arms retain native precision, NFE 10, 50 × 14 action chunks, TF32 enabled and
cuDNN benchmark disabled. Capture is NUMERIC; no FP8, distillation or reduced schedule was evaluated.

## Closed-loop screening

A/A compares originals across repeats; B/B compares accepted capture endpoints across
repeats. These reuse the same episodes and are not additional independent seeds.
Episodes use the same declared seeds and frozen scenes; actual-noise tensors were
not recorded or injected. A/A and B/B include full simulator/runtime repeatability,
not isolated kernel error.
Intervals are descriptive central 95% paired Tango score intervals in percentage
points; task clustering is not modeled. `identical` compares complete executed
action-stream digests. These 20 pairs per setting remain SCREEN, not a 5-point
noninferiority certificate or a BITEXACT claim.

| Comparison | Setting | Successes | Delta (pp) | 95% interval (pp) | Identical |
| --- | --- | --- | --- | --- | --- |
| ab_repeat1 | clean | 18/20 → 18/20 | +0.0 | [-16.1, +16.1] | 0/20 |
| ab_repeat1 | randomized | 16/20 → 17/20 | +5.0 | [-11.9, +23.6] | 0/20 |
| ab_repeat2 | clean | 16/20 → 17/20 | +5.0 | [-16.1, +26.6] | 0/20 |
| ab_repeat2 | randomized | 17/20 → 17/20 | +0.0 | [-16.1, +16.1] | 1/20 |
| aa | clean | 18/20 → 16/20 | -10.0 | [-30.1, +7.7] | 2/20 |
| aa | randomized | 16/20 → 17/20 | +5.0 | [-11.9, +23.6] | 0/20 |
| bb | clean | 18/20 → 17/20 | -5.0 | [-23.6, +11.9] | 1/20 |
| bb | randomized | 17/20 → 17/20 | +0.0 | [-16.1, +16.1] | 2/20 |

Capture admission: **2/3** valid eight-warmup
startups passed. One independent startup was rejected at max-absolute delta
0.0859375 against the unchanged 0.05083918571472168 gate.
The extra independent startup was limited to one prospectively recorded attempt.
Quality results are conditional on accepted endpoints; three trials do not estimate
a stable startup failure rate. Failed setup and partial pilot campaigns are retained
externally and excluded from the 160 analyzed episodes.

## Separately measured endpoint latency

Each arm: 8 warmups, 128 measured calls using real observations, after both simulation
campaigns. Times include serialization and transport; they are not GPU-only latency.

| Execution | p50 (ms) | p99 (ms) | max (ms) | Quality |
| --- | --- | --- | --- | --- |
| original | 822.82 | 832.67 | 833.82 | screen |
| native_capture | 131.08 | 134.73 | 135.11 | screen |

These are H100 execution records. They do not certify Thor, another checkpoint or
new source/precision/schedule. Paused simulation and 128 calls do not certify a
sustained robot control loop. Independent accepted endpoints match the recorded
hardware/software profile; physical GPU UUID and device load are not profile keys.

## Budget choices and reproducibility

The four checked-in `*.selection.json` files use illustrative explicit budgets:
20 Hz pipelined chunks (2500 ms), an added 500 ms reaction deadline, 50 Hz blocking
calls (20 ms), and a Thor target requiring Thor evidence. Baseline is retained when
it meets the observed budget. SCREEN cannot authorize selecting the accelerated
candidate when baseline misses. Precision permission, step permission and the exact
quality protocol/task set are separate inputs.

`summary.json` contains all episode pairs, intervals, timing summaries, campaign wall times and decisions.
`evidence-index.json` pins external raw bundles, measured receipts, latency samples,
records, scripts and the installed wheel. Evidence records verify those sources and
recompute gates; editing a summary and rehashing it cannot create a certificate.
The frozen GPU execution predates the added reporting/explain interface; its own
source digest is retained rather than assigning its measurements to newer code.

After restoring the external root, reproduce postprocessing with the installed wheel:

```sh
python eval/precision_evidence_2026-09-06/analyze.py --root "$EVIDENCE_ROOT" --output /tmp/v2-analysis.json
python eval/precision_evidence_2026-09-06/summarize.py --root "$EVIDENCE_ROOT" --output /tmp/v2-summary
```

Replay scripts and exact environment paths are retained in the external root;
rerunning simulation requires the pinned RoboTwin assets and model/environment
inventories in each verified bundle. See [execution workflow](../../benchmarks/vla/SIMULATOR_EVALUATION.md#bind-quality-and-latency-to-an-execution)
for rebuilding records after relocation and using the three CLI commands.

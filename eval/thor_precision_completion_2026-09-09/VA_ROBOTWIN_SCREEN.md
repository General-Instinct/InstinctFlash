# LingBot-VA RoboTwin paired screen

Final verified result: 10 tasks × 2 seeds × clean/randomized, 40 paired scenes (80 policy episodes). Original checkpoint and 25-video/50-action schedule on Thor; paused simulation.

| Condition | Native success | FP8 success | Observed difference |
|---|---:|---:|---:|
| Clean | 19/20 | 17/20 | −10 percentage points |
| Randomized | 20/20 | 19/20 | −5 percentage points |
| Combined descriptive count | 39/40 | 36/40 | −7.5 percentage points |

Three pairs succeeded only with native precision; no pair succeeded only with FP8. This small screen measures observed differences, not statistical non-inferiority or real-time qualification. Clean and randomized conditions remain separately reported.

All policy traces/controller actions, recovery provenance, paired initial inputs, and final Thor source/weight checks passed. The automatic completion chain exited successfully in all four stages.

The original FP8 run stopped before policy inference on clean dump_bin_bigbin/110101. Its original 13 results were retained. Recovery repeated only expert checks with zero policy calls, using the same pinned seed; the third new attempt entered policy inference. Model task failures were not retried. See [recovery amendment](VA_ROBOTWIN_RECOVERY.md).

Evidence: [paired results](va-runtime-robotwin-paired-comparison.json), [FP8 completion](va-runtime-robotwin-fp8-recovery-complete.json), [final source verification](va-runtime-robotwin-fp8-source-final.json).

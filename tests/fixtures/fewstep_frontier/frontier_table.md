## The untrained low-step frontier (SCREENING, paired, pinned scenes; n<=250/point)

| point | forwards/cycle | cycle p50 ms | eff. Hz | success | Δ vs 2V/4A (paired) | 95% interval (most conservative) | paired n | discordance |
|---|---|---|---|---|---|---|---|---|
| **2V/4A (baseline arm)** | (V+1)+(A+1)+2 | n/e | n/e | 0.9138 | (control) | repeat Δ-0.0087, floor 11.4% | 232 | — |
| 3V/4A | (V+1)+(A+1)+2 | n/e | n/e | 0.9389 | +0.0175 | [-0.0175, +0.0524] (bootstrap_iid) | 229 | 7.0% |
| 1V/4A | (V+1)+(A+1)+2 | n/e | n/e | 0.7478 | -0.1681 | [-0.2443, -0.0961] (bootstrap_task_cluster) | 226 | 23.9% |
| 2V/2A | (V+1)+(A+1)+2 | n/e | n/e | 0.8821 | -0.0306 | [-0.0716, +0.0105] (mcnemar_se) | 229 | 10.0% |
| 1V/2A | (V+1)+(A+1)+2 | n/e | n/e | 0.6978 | -0.2267 | [-0.3122, -0.1475] (bootstrap_task_cluster) | 225 | 28.0% |
| 1V/1A | (V+1)+(A+1)+2 | n/e | n/e | 0.5625 | -0.3571 | [-0.4484, -0.2696] (bootstrap_task_cluster) | 224 | 39.3% |

Noise floor (baseline repeat, 229 pairs): delta -0.0087, discordance 11.4%, most-conservative CI [-0.0568, +0.0426].
Historical cross-check (rerun base vs actsweep_v2a4, 229 overlapping scenes): delta -0.0131 (expected within the noise floor).

Screening sweep, explicitly not certification: intervals are the result; no ship verdicts are issued from this table (prereg §0).

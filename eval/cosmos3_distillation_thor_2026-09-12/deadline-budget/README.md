# Illustrative controller deadline budget

These are calculations from frozen Thor timing receipts, not realtime or quality admission. No controller deadline or executed action chunk has been supplied.

| Model / seed | Measured requests | p50 ms | p95 ms | Observed max ms | Margin at 500 ms | Margin at 750 ms |
|---|---:|---:|---:|---:|---:|---:|
| Edge V17 SDE2 CFG1 cuDNN / 12031 | 10 | 442.2 | 442.7 | 442.8 | +57.2 | +307.2 |
| Edge V17 SDE2 CFG1 cuDNN / 12032 | 10 | 444.3 | 444.7 | 444.8 | +55.2 | +305.2 |
| Nano V18 SDE1 CFG1 combined / 12031 | 24 | 715.7 | 717.5 | 718.4 | -218.4 | +31.6 |
| Nano V18 SDE1 CFG1 combined / 12032 | 24 | 717.0 | 718.2 | 721.3 | -221.3 | +28.7 |

Positive margin only means the observed timed call fits that illustrative budget. It must also accommodate any additional required work and deployment variability.

For illustration only, with pipelined execution at15 Hz and50 ms of additional budget, the minimum executed chunk implied by the observed maximum is:

- Edge V17 SDE2 CFG1 cuDNN, seed12031: 8 steps.
- Edge V17 SDE2 CFG1 cuDNN, seed12032: 8 steps.
- Nano V18 SDE1 CFG1 combined, seed12031: 12 steps.
- Nano V18 SDE1 CFG1 combined, seed12032: 12 steps.

- Deadlines, control rates and extra budgets are illustrative, not controller settings.
- Execution-window arithmetic assumes pipelined scheduling: N/control_hz >= observed model latency + extra budget. A blocking controller instead has one control period, not N periods; actual overlap, action age and startup remain unverified.
- Conditioning FPS is not evidence of actuator/control frequency; output horizon is not executed chunk length.
- Measured maximum is a finite-sample observation, not a worst-case bound; zero observed exceedances does not certify a deadline.
- Receipts cover timed Runtime prediction under their frozen protocols, not a certified sensor-to-actuator path; do not double-count preprocessing already timed.
- Small frozen-fixture samples exclude warmup from this table and cannot establish sustained latency tails under deployment load.
- Both trained families retain their historical UniPC4 quality limitations; timing feasibility alone cannot admit them.

The shared deployment assessor is `benchmarks/vla/realtime.py::assess`, exposed by `instinctflash eval realtime-report`. Its example50 Hz/50-action budget is not a deployment setting. These receipts do not record an actual executed chunk or controller schedule, so this analysis does not fabricate those fields to produce a passing assessment.

Reproduce with `python analyze.py` from this directory. [Full calculations and source hashes](results.json).

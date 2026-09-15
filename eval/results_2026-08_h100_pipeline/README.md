# First real-model runs through the VLA benchmark pipeline (2026-08-28, 4xH100 box)

Four families through `benchmarks/vla` end to end — serving profile (model_contract x2 repeats
+ single_gpu_latency, 30 timed chunks after 3 warm), `instinctflash_driver.py` serving
`stock_upstream` (the family's upstream in-process eager surface) vs `runtime_default`
(`Runtime.from_pretrained`, no flags), one fresh process per job, one idle H100 per pair,
counterbalanced arm order. Every report here is `complete=true`, `gates_passed=true`,
`reportable=true`, `synthetic=false`.

| report | driver revision | stock p50 | runtime_default p50 | speedup | action gate |
|---|---|---|---|---|---|
| `report_v2.json` | e8712e9 | 661.1 ms | 127.5 ms | 5.19x | numeric: max abs 4.87e-2 < 5.08e-2 envelope |
| `report_4b.json` | e8712e9 | 532.7 ms | 164.8 ms | 3.23x | bitexact 9/9 |
| `report_groot.json` | bb46142 | 92.1 ms | 51.0 ms | 1.81x | bitexact 9/9 |
| `report_pi05.json` | e8712e9 | 205.7 ms | 91.2 ms | 2.26x | bitexact 9/9 (full select_action serve path) |

Same-box agreement with the independent `reproduce_h100`/verify protocols is within ~2% for
V2/4B/GROOT; pi05's serve path carries ~31-37 ms of processor pipeline on both arms over the
module-protocol chunk pair (169 -> 61 ms on this box), so both protocols are stated separately.

GROOT is the bb46142 rerun: the first pass (driver e8712e9) measured the runtime arm at
57.4 ms and refused to match the row's 51.8 — the discrepancy was a live product bug
(`KNOWN_DECLARATIONS` had drifted from the pointer package and dropped the bit-exact
`fast_decode`/`backbone_fastpath` flags, so the bare Hub id served ~11% slower than the
published row). Fixed in bb46142 with an equality test pinning the two declaration sources.

`m3_replay_verdict_reproduction.json`: the closed-loop gate path validated on recorded
evidence — the V2 M3 RoboTwin 50x10 paired outcomes (500 episodes/arm) replayed through
plan -> run -> report via `replay_driver.py` (synthetic=true, never fresh evidence); the
report-stage certificate is byte-for-byte the frozen `instinctflash.verify.certify` output,
verdict PASS non-inferiority at -0.05 (delta -0.020, discordant 25/15, Tango lower bound
-0.0418). The M3 archive's "FRAGILE, slack +0.004" is its bootstrap_task_cluster primary
interval; the verdict is identical and the numbers are stated, not averaged.

Raw runs (plans, per-job requests/results/logs, environment manifests) live on the 4xH100 box
under `~/bench_pipeline_runs/`; the M3 replay run under
`/home/ubuntu/iwm_distill/m3_replay/`.

Still owed before this pipeline replaces the per-family scripts as the canonical protocol:
LingBot-VA, DreamZero, and both Cosmos3 policies through the same path (VA additionally needs
an arm-level numeric gate declaration — its default arm is NUMERIC-tier while registry mode
would demand cross-arm bitexact — and an upstream-server stock arm in the driver).

# Edge single-step latency budgets on Thor

Original Edge weights, native UniPC with one step, BF16 throughout. These are
operating-point cost measurements, not trained SDE students or task-quality results.

| Attention | CFG | p50 ms | p95 ms | Branches/request |
|:--|--:|--:|--:|--:|
| Native | 1 | 413.35 | 413.63 | 1 |
| cuDNN | 1 | 275.62 | 276.30 | 1 |
| cuDNN | 4 | 441.73 | 446.92 | 2 |

The matched CFG1 attention pair is **1.50× faster** with cuDNN. Across all 16
full 32×8 action outputs, maximum absolute drift is 0.015625 and mean drift
is 0.00248936. This is a numerical screen; no task-quality margin is certified.
CFG1 reduces the cuDNN arm's p50 by 37.6% versus CFG4, but changes guidance and
requires a separately trained and evaluated student before deployment selection.

Each arm ran in a fresh process under `/tmp/thor_gpu.lock`, with six warmup and
ten measured requests, fixed fixture/seeds, GPU-contention checks, actual branch
counts and complete finite actions. Source, checkpoint index/revision, fixture
and benchmark hashes were validated. Exact pointwise kernels and layer graphs
were enabled. Production attention admission and padding contracts are unchanged.

All three arms recorded 0/10 misses against a hypothetical 533 ms budget (eight
executed actions at 15 Hz). Neither an actual controller deadline nor closed-loop
realtime behavior has been established. These UniPC results do not substitute
for measuring the new SDE `[1,0]` students and their zero-padding contract.

The frozen remote scripts and outputs are under
`/home/guanming/ifl_eval/cosmos_distill_20260912/edge-budget-v1` and
`edge-budget-results-v1`. The environment uses the sibling frozen `flash` and
`compress` packages plus `flash/examples/cosmos3_policy` on `PYTHONPATH`, vendor
Cosmos venv, one visible GPU, OMP_NUM_THREADS=4, offline Hugging Face access,
PYTHONHASHSEED=0 and CUDA ptxas at `/usr/local/cuda/bin/ptxas`.

```bash
python edge-budget-v1/run.py FRESH_OUTPUT_DIRECTORY --fixture flash/eval/native_total_2026-09-10/fixtures/va_eval_obs.npz
python summarize.py receipts
```

[Validated raw summary](receipts/summary.json) includes conditional budgets and
paired numerical deltas. The JSON, NPZ and log files retain the raw evidence.

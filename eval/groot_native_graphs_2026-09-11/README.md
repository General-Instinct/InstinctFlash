# Native GR00T DiT and text graphs on Thor

This compares existing native BF16 DiT capture with the larger text-block fusion
from `eval/groot_shared_blocks_2026-09-11`. FP8, attention algorithms, GEMMs and
action step count are unchanged. The four fresh-process arms are:

- `baseline_a`: current eager Runtime before the planner update.
- `graph_only`: existing `StaticDiT`, with its six-case exact startup check.
- `fused_graph`: `StaticDiT` plus the 16 shared Norm/MLP/residual text graphs.
- `baseline_b`: eager Runtime repeated.

| Round, p50 ms | Baseline A | DiT graph | DiT + fused text | Baseline B |
|---|---:|---:|---:|---:|
| Initial | 114.40 | 111.84 | 110.50 | 119.46 |
| Repeat | 112.55 | 110.75 | 108.14 | 117.86 |

All 23 × 40 × 17 actions in each arm matched baseline bytes and were finite.
DiT alone records 92 replays; combined graphs record 460 replays. Every graph's
exact admission passed. The reused `norm_graphs` receipt field includes the DiT
receipt and optional text-block graphs. Baseline drift is visible, so report
both baseline comparisons; do not present the largest ratio as a stable gain.

The production planner update admits **only the existing DiT graph** on Thor
for GR00T's four-step action schedule, subject to static shapes and the startup
exact check. It does not enable the experimental text fusions. Other schedules
retain the prior Thor gate. The existing `IFL_GROOT_NO_CAPTURE=1` kill switch is
preserved. This does not change FP8 permission or certify arbitrary checkpoints.

## Protocol and profiling

The same recorded-input protocol is retained: fixed checkpoint revision, two
prompts, recorded camera frames, synthetic state, fixed seeds, 3 warmups and
20 measured requests, TF32/cuDNN benchmark disabled. Every request checks for
competing GPU PIDs; fresh processes are serialized with `/tmp/thor_gpu.lock`.
Raw actions and source hashes accompany each report. No simulator is involved.

Baseline A additionally profiles one request **after** its timed requests. The
profile records actual CUDA event durations by kernel name and CPU operator
self time separately; do not add those two overlapping quantities. The largest
GPU kernel groups are GEMMs, with LayerNorm and pointwise work also present.
These profiles are diagnostic and are not used as latency samples.

`run.py ROOT OUTPUT_DIR` runs the offline ablation; `compare.py OUTPUT_DIR`
checks protocol, graph admission and saved action bytes. The source root must
contain the frozen repo files and fixture; environment paths are explicit in
`run.py`. Original sources remain in
`/home/guanming/ifl_eval/groot_shared_bf16_20260911` on the Thor host.

## Public Runtime validation

`run_public.py ROOT OUTPUT_DIR` uses `public_benchmark.py` with no experimental
installer. Baseline A/B set `IFL_GROOT_NO_CAPTURE=1`; `current` sets it to zero.
`compare_public.py` requires all action bytes to agree, default capture/replay
to be active and control captures to be absent. The updated source is isolated
at `/home/guanming/ifl_eval/groot_default_graph_20260911`; it does not overwrite
the earlier ablation's source snapshot.

The public-path check passed: capture disabled A **113.93 ms**, default DiT
capture **111.14 ms**, capture disabled B **113.74 ms** (p50). This is a
**2.3–2.5%** end-to-end improvement in this matched run. All 23 action chunks
matched bytes, current recorded one capture and 92 replays, and both controls
recorded zero captures/replays. The public check applies no text-block patch.
CPU checks: 40 passed, 4 CUDA-only tests skipped locally; actual Thor model
checks and capture admission are recorded separately in these receipts.

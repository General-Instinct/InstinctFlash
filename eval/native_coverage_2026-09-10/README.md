# Native coverage — H100 and Jetson Thor

This campaign fills the missing native rows for LingBot-VLA-V2, Cosmos Edge/Nano,
DreamZero, and LingBot-VA on H100. It measures the current BITEXACT policy; it does
not promote historical NUMERIC implementations or change a model's precision,
guidance, step count, attention backend, or determinism settings.

Completed: **27 fresh GPU processes, nine model/device comparisons, 378 saved action chunks**.
Eight comparisons pass every finite action-byte check. V2 on H100 retains the
`BASELINE_VARIABLE` result; there are no remaining unmeasured cells in this campaign.
[All comparisons](completed-results.json) · [Raw receipts and actions](receipts/)
· [Retained setup failures](setup-failures.json) · [Validation](validation.json).

## H100 results

| Model | Baseline A p50 | Current p50 | Baseline B p50 | Action comparison |
| --- | ---: | ---: | ---: | --- |
| LingBot-VLA-V2 | 908.13 ms | 902.78 ms | 823.62 ms | Baseline varies |
| LingBot-VA | 2655.61 ms | 2496.01 ms | 2558.56 ms | BITEXACT, 12/12 calls in every pair |
| Cosmos Edge | 630.95 ms | 661.89 ms | 660.11 ms | BITEXACT, 15/15 calls in every pair |
| Cosmos Nano | 1171.61 ms | 1177.81 ms | 1172.97 ms | BITEXACT, 15/15 calls in every pair |
| DreamZero | 3001.88 ms | 3016.38 ms | 3018.41 ms | BITEXACT, 12/12 calls in every pair |

VA's P010 ratio is 1.025–1.064× against the two references. The other families retain
their existing native execution; their timing differences are controls, not newly
promoted optimizations. All three V2 processes run without capture. Its two
references match only 1/15 calls (max absolute action delta 0.0351493); baseline A
versus current matches 3/15 (0.0371020). These action units are not accuracy-loss
percentages. V2's existing numerical capture is not used to fill the native row.

## Thor results

| Model | Baseline A p50 | Current p50 | Baseline B p50 | Action comparison |
| --- | ---: | ---: | ---: | --- |
| LingBot-VLA-V2 | 752.25 ms | 738.76 ms | 730.01 ms | BITEXACT, 15/15 calls in every pair |
| Cosmos Edge | 3439.28 ms | 3454.49 ms | 3442.93 ms | BITEXACT, 15/15 calls in every pair |
| Cosmos Nano | 10324.49 ms | 10287.38 ms | 10336.43 ms | BITEXACT, 15/15 calls in every pair |
| DreamZero | 22894.64 ms | 23046.03 ms | 22968.35 ms | BITEXACT, 12/12 calls in every pair |

All four retain existing native execution. V2's observed exactness on these Thor
inputs does not erase its previous longer-run variability or the H100 failure.
No numerical capture is admitted by these measurements.

The README's existing VLA-4B, pi05 and GR00T improvements retain their
[separate native optimization campaign](../native_optimization_2026-09-10/README.md).
VA's Thor P010 comparison retains its [48-cycle replay](../edge_defaults_2026-09-06/README.md).

## Protocol

Each model/device runs in fresh processes in the fixed order baseline A, current
Runtime, baseline B. The baseline uses the same native adapter and checkpoint with
transformation passes excluded. VA retains the existing native optimizations and
excludes only action terminal forward elision, matching its earlier Thor comparison.
An unchanged execution path is a control, not a new acceleration result.

VLA-V2 and Cosmos use three warmups and twelve measured generation calls. Inputs
cycle through recorded camera images, three synthetic states and two prompts.
DreamZero and VA use four three-cycle episodes; the first episode is warmup. The
benchmark preserves causal history and VA's recorded executed-action feedback.
All warmup and measured actions are retained for byte comparisons. The camera
fixture is external and identified by SHA-256 in each receipt.

Timing synchronizes CUDA around the public prediction call. Reset and GPU process
checks are outside the measured interval. Competing GPU processes invalidate the
pair. These short latency samples do not qualify tail latency or simulator success.
The owned source snapshot is hashed before each arm and after the campaign.
Upstream Python sources were observed during the initial campaign and rechecked
after completion; those manifests retain their observation times.

## Interpreting the comparisons

- `BITEXACT`: finite action bytes match across all three arm comparisons.
- `BASELINE_VARIABLE`: the two native reference processes already differ. No
  tolerance is introduced and no BITEXACT speedup is claimed.
- `ACTION_MISMATCH`: native references match but the current Runtime differs.
- `INVALID_PROTOCOL` / `FAILED`: measurement prerequisites were not satisfied.

The comparison CLI saves its report and exits nonzero unless all three pairs
are BITEXACT. `unchanged_transform_plan` identifies pass-through controls. Small timing changes
on such controls are not evidence of a new optimization. Repeated baselines expose
drift; the comparison reports both baseline/current ratios rather than selecting
the faster reference.

## Reproduction

With the model's documented environment, pinned checkpoint cache and camera archive:

```bash
for arm in baseline_a current baseline_b; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
    python eval/native_coverage_2026-09-10/benchmark.py \
    edge native "edge-${arm}.json" --arm "$arm" \
    --iterations 12 --input-archive "$OBS_ARCHIVE"
done
python eval/native_coverage_2026-09-10/compare.py . edge --output edge-comparison.json
```

`run.py` serializes a device's arms and preserves failed attempts. Its paths describe
the campaign hosts and should be adapted for another installation. Frozen source,
logs and running status are retained outside the repository under
`/home/ubuntu/ifl_eval/native_coverage_20260910` and the corresponding Thor directory.

## Execution-plan correction

The frozen H100 Cosmos receipt lists `graph_capture` as applicable, while its
backend correctly reports `captured: false`. The DROID adapter has no graph
installer. The planner now honors an adapter's `capture_supported: false` note,
so it no longer proposes that unsupported transformation. This correction changes
planning metadata, not Cosmos arithmetic; timings retain their frozen source hashes.
The regression covers H100, Thor and SM120.

The initial Thor Cosmos processes failed before model loading because the installed
environment lacked the `cosmos3-iwm` entry point. Installing the shipped adapter
with `uv pip install --no-deps -e .../examples/cosmos3_policy` fixes discovery without
updating model dependencies. The original failed receipts and a separate recovery
campaign are retained; failed attempts are not silently replaced.

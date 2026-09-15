# Native Nano weight-layout ablation on Thor

Changing 72 generation-tower gate/up projection weights from contiguous `(N,K)`
to a transposed contiguous `(K,N)` backing preserves every weight byte, BF16
dtype, four UniPC steps, CFG 3 and full DROID actions. This is an experiment
around the current native Runtime on pinned upstream `f734253f`; it does not
use the newer compiled numerical attention lane.

| Arm | Warm p50 ms | Full action bytes vs baseline A |
|---|---:|---|
| Native A | 8637.85 | equal |
| Packed weights | 8667.48 | equal |
| Native B | 8664.19 | equal |

The candidate is 0.04–0.34% slower, within the reference drift. The synthetic
GEMM benefit did **not** translate into an end-to-end gain. Reject promotion;
leave Runtime defaults and the accepted nightly baseline unchanged. These
results do not isolate the reason for the missing end-to-end benefit.

Each fresh process runs six warmup and ten measured requests under the shared
Thor GPU lock. All 16 × 32 × 8 finite action values match at the byte level,
including changed prompts, images, states and resets. The candidate records all
72 original/new strides and verifies the packed strides remain live after the
full run. Every arm admits 72 layer graphs with no graph rejection. This is a
recorded-input comparison, not a proof for all inputs or a task-success score.

`run.py ROOT` runs native A / packed / native B using the explicit installed
Cosmos environment and a frozen InstinctFlash source/fixture snapshot at ROOT.
It invokes `benchmark.py`, which wraps the existing Cosmos regression workload
without adding per-forward hooks. All source, checkpoint, fixture and action
hashes are retained in `receipts/`.

Recompute the comparison with:

```bash
python compare.py /path/to/weight-layout/receipts
```

The comparator verifies matched protocols and sources, action archive hashes,
finite full chunks, actual layout installation and baseline byte repeatability.
It reports candidate differences separately instead of assuming a storage-only
change must preserve model outputs.

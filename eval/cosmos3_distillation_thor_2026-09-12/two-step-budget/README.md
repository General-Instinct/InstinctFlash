# Original-weight SDE2 latency budget on Thor

This is **not a trained two-step student**. It measures the released original
Edge weights with complete SDE `[1,0.5,0]`, CFG1, zero action padding and BF16.
It establishes a cost budget for a possible two-step distillation experiment.

| Attention | p50 | p95 | Actual branches/request |
|---|---:|---:|---:|
| Native BF16 | 719.32 ms | 719.59 ms | 2 |
| cuDNN BF16 | 442.98 ms | 444.45 ms | 2 |

The matched attention pair is 1.62× faster. Across all 16 finite 32×8 outputs,
maxabs drift is 0.0163574 and mean absolute drift is 0.00256189. These are
numerical screens; no task-quality margin has been established for this model.
The six warmup and ten measured calls use the same fixture and reset pattern
as the earlier student latency pairs. Complete `[1000,500]` callback clocks,
actual branch counts, padding restoration, artifact hashes and loaded external
Wan VAE were checked. The pair comparator passed on Thor and after local transfer.

The two-step CFG1 cost is close to the measured one-step CFG4 student's 436 ms;
both execute two denoiser branches. This supports testing a two-step CFG1 student
as a quality/latency alternative, not assuming it will improve quality. The
one-step CFG1 students remain faster at 270–271 ms.

Against a **hypothetical** 533.33 ms budget (eight executed actions at15 Hz),
the cuDNN run had 0/10 misses, with a maximum of444.92 ms. This short in-process
benchmark does not establish an actual controller contract, tail-latency
reliability, fresh-observation age, or closed-loop realtime task performance.

## Reproduction and provenance

`prepare.py ORIGINAL_SOURCE ORIGINAL_INVENTORY FRESH_PACKAGE` checks all 34
historical original checkpoint entries. It creates package-local regular files
(hardlinks where available), writes a private copy of the native config with the
two-step SDE schedule, and adds the explicit unqualified execution declaration
and padding sidecar. It never modifies source weights. The CPU reproduction
generated the exact measured36-file manifest bytes; see `preparation_receipt.json`.
The inventory is the earlier `realtime-controls/inventory.json`.

With the frozen Flash/Compress/vendor environment and `realtime-deployment`
helpers on PYTHONPATH, run `run_pair.py PACKAGE FRESH_RESULTS --fixture FIXTURE`,
then `compare.py FRESH_RESULTS`. The measured source runs from
`/home/guanming/ifl_eval/cosmos_distill_20260912/two-step-budget-v1`; successful
package/results directories end in `original-sde2-cost-v3` and
`results-original-sde2-cost-v3`. Full reports and arrays are in `receipts/`.

Two startup rejections are retained: v1 had external symlinks, rejected by the
existing artifact gate; v2 lacked the native fixed-step config, rejected by the
existing sampler-contract gate. Neither produced valid actions or timing rows.
The successful v3 package satisfies both gates. No checks were weakened, and
no previously measured checkpoint or production default was changed.

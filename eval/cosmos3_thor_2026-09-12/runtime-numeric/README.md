# Cosmos native BF16 numerical Runtime on Thor

The public Runtime now supports the screened cuDNN attention route with explicit
`precision="native", tier_ceiling="numeric"`. It preserves weights, BF16
execution, four UniPC steps and CFG 3. Conditioning reuse, checked pointwise
fusion and shared-pool layer graphs compose with that route; Nano retains the
action-only loader. Strict BITEXACT and FP8 remain separate selections.

The installer owns each model instance's attention dispatcher. It does not
patch a vendor module's global functions, and closing the Runtime restores its
owned dispatch references. Unsupported dense-attention shapes retain the vendor
implementation; the eligible BF16 route forces cuDNN. Installation checks the
screened source, Thor hardware, cuDNN 9.15.1, configuration and layer parameters.
Backend statistics expose actual calls (excluding graph replays) and explicitly
state that task quality is not certified.

## Matched Runtime measurements

Each arm runs in a fresh process, with the same fixture, checkpoint, source,
32×8 output contract, two prompts, resets, six warmup and ten measured requests.
Both arms enable guarded conditioning reuse. The comparison isolates numerical
attention's gain over our already optimized native reference, not upstream.

| Model | Strict + cache p50 ms | Numeric + cache p50 ms | Speedup | Max action delta | Mean action delta |
|:--|--:|--:|--:|--:|--:|
| Edge | 2469 | 1390 | 1.78× | 0.167817 | 0.015321 |
| Nano | 8305 | 5537 | 1.50× | 0.977512 | 0.085857 |

All 16 full action chunks per arm are finite. Both integrated strict references
and numerical candidates match their corresponding prior offline action bytes.
The public execution policy reports native precision, NUMERIC transforms and no
schedule changes. See [integration audit](receipts/integration-audit.json).

These are **SCREEN** measurements. Raw action deltas are not percentages of task
accuracy. There is no closed-loop success-rate certificate or drift-bracketed
performance claim. In particular, Nano's observed action differences should not
be described as uniformly small simply because its attention inputs stay BF16.

## Reproduce and inspect

From the qualified Thor Cosmos environment:

```bash
python -m benchmarks.regression.run_cosmos_numeric /absolute/path/to/new-results
```

Use `--family edge` or `--family nano` for one model. The runner records complete
actions and source hashes, rejects missing backend execution/cache qualification,
and keeps numerical screens separate from the strict nightly regression.
Default p50 speedup must be at least 1.1×; `--max-abs-action-delta` optionally adds
an explicit numerical regression bound, without certifying task quality.

The Runtime source was tested in the frozen remote tree
`/home/guanming/ifl_eval/cosmos_numeric_runtime_20260912`.
JSON/NPZ receipts are under [receipts](receipts/); the committed comparison tool
can recheck them with `--compare-only`. The frozen run predates a diagnostic-only
wording fix in `runtime/execution.py`: adapters can install inside
`build_in_process`, so a missing separate `install` hook no longer prints that
the model is unoptimized. The measured model execution code is unchanged.
The final comparison gate was re-run locally on the immutable GPU receipts.
The measured benchmark contents are byte-identical to `cosmos_numeric.py`; that
entrypoint was renamed after measurement to leave the strict nightly benchmark
and its accepted protocol hash unchanged. The numerical runner has its own
entrypoint deliberately so further screens do not invalidate strict history.
Seventy-one relevant CPU tests passed, including permission, isolation, cleanup,
artifact integrity and regression-gate failures.

Prior [offline cuDNN screens](../cudnn/README.md) explain the narrow backend
admission and contain operator profiling. This integration replaces the global
experimental override with instance-local dispatch. Slower NATTEN variants,
fixed-KV buffers and the host-timestep experiment are not enabled. Compiler
experiments on a different upstream source still require a matched integration
run before their gains can be combined with this route.

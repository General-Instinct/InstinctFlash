# Guarded conditioning cache through public Runtime

The native cache is now available through `IFL_COSMOS3_CONDITIONING_CACHE=1`.
It is opt-in, limited to declared Edge/Nano on Thor, native BF16, four steps and
CFG 3. It does not enable FP8, compilation or an altered sampling schedule.

```bash
IFL_COSMOS3_CONDITIONING_CACHE=1 python your_policy.py
```

Use the same `Runtime.from_pretrained(...)` and prediction API. The native
backend statistics expose `conditioning_cache_status` and `conditioning_cache`,
including admission, first-use checks, graph replays and rejection reasons.

## Guards and lifetime

The installer checks the actual upstream source-file hashes and module wrappers
before interception. Unsupported stacks retain their existing native forwards.
Each token/CFG branch first executes a complete request-local prefill, then a
reference decode that checks every cached tensor. That decode propagates the
original results through all layers. Only a successful finite-byte check permits
cached replay. A mismatch disables the cache and returns the reference result;
it does not retry the sampler or advance the RNG a second time.

Every request refills cached values. Slots may retain storage and graph admission
across requests, but no earlier request's payload is consumed without refill.
Changed cached output geometry disables the optimization. Failed requests clear
active state; the next request must complete prefill again. `close()` restores
intercepted forwards and releases slots. Cache operations are serialized on the
owning model; reentrant generation is rejected.

This port reuses the [qualified experimental mechanism](../conditioning-cache/README.md)
with source admission, reference-returning validation, safe bypass, explicit
statistics and shutdown. It remains off by default and does not replace the
accepted nightly baseline. Retained conditioning and branch graphs cost memory.

## Matched public Runtime results

| Thor model | Native A p50 ms | Cache p50 ms | Native B p50 ms | Speedup | Full actions |
|---|---:|---:|---:|---:|---|
| Edge | 2559.26 | 2469.07 | 2557.26 | 1.0357–1.0365× | identical |
| Nano | 8692.50 | 8304.78 | 8687.98 | 1.0461–1.0467× | identical |

Edge checks 756 tensors during admission and records 2436 cached graph replays;
Nano checks 972 and records 3132. Neither disables the cache or rejects a graph.
Peak allocated memory is 10.91 GiB for Edge and 35.01 GiB for Nano, about
1.0 / 2.15 GiB more than the respective native controls.

Each arm uses six warmup and ten measured requests in a fresh process, with
changing recorded images, states, prompts and resets. There is no model/service
monkeypatch in `benchmark.py`; it selects the actual Runtime option. Every
report retains full 16 × 32 × 8 finite action arrays, source/checkpoint/fixture
hashes, actual schedule, numeric settings, contention checks and backend stats.
The candidate must execute source admission, real-input checks and graph replays.

`run.py ROOT edge` runs native A / cache / native B using the explicit frozen
Thor environment; `nano` selects Nano. `compare.py RECEIPTS edge
--minimum-speedup 1.02` checks the full pair, including action identity, drift,
p50 gain and p95 regression. An invalid run clears any older passing comparison.

The official upstream/current/upstream regression also supports:

```bash
python -m benchmarks.regression.run_cosmos /absolute/new-results --candidate conditioning
```

That command compares against the upstream service. The table above instead
isolates the incremental gain over the existing optimized native Runtime.

## Validation

The real Thor `probe.py` checks request refill, branch order, eviction, retained
outputs, in-place weight updates, parameter replacement, transparent fallback
when invariance fails, failed-prefill recovery and idempotent close. All ten
checks passed; `receipts/lifecycle-probe.json` records the implementation hash.
CPU tests cover unsupported hardware/source/schedule admission, duplicate
installation, invalid capacity, close and regression rejection of silent fallback.
These are finite action-equivalence checks, not closed-loop task-success scores.

# Request-local Cosmos conditioning cache on Thor

This adapts existing InstinctFlash prefill, transactional cache and shared graph
patterns to the old native Cosmos source. It caches text-tower Q/K/V projections,
norms, output projection and MLP results within a request, separately for CFG
branches. BF16 dtype, attention math, four UniPC steps, CFG 3 and complete DROID
actions remain unchanged. See [the reuse audit](REUSE.md) for exact boundaries.

## Edge

| Arm | p50 ms | Full action bytes vs native A |
|---|---:|---|
| Native A | 2558.27 | equal |
| Conditioning cache | 2468.11 | equal |
| Native B | 2559.74 | equal |
| Diagnostic recomputation | 2591.02 | equal |

The candidate is **1.0365–1.0371×** faster (3.52–3.58% lower latency).
Reference drift is 0.057%. All four arms return finite, identical 16 × 32 × 8
action arrays. The diagnostic arm independently recomputes and checks 24,192
cached tensor outputs. The speed arm uses 84 slot/layer graph objects and records
2520 replays with no rejection. Python module-hit counts do not count replayed
GPU operations.

Measured p95 also improves from about 2561 to 2469 ms. Peak allocated memory
increases from 9.92 to 10.91 GiB; peak reserved memory increases from 11.10 to
12.29 GiB. Cached conditioning and separate branch/slot graphs trade about
1 GiB of additional allocation for this latency gain.

Each request executes two complete branch prefills and six cached velocity
forwards. No request can consume a previous request's payload before refilling
it. The LRU retains storage and graphs, not cross-request validity. The speed
includes prefill and payload-copy costs; this is not a decode-only measurement.

## Nano

| Arm | p50 ms | Full action bytes vs native A |
|---|---:|---|
| Native A | 8693.07 | equal |
| Conditioning cache | 8304.93 | equal |
| Native B | 8681.19 | equal |
| Diagnostic recomputation | 8626.80 | equal |

The candidate is **1.0453–1.0467×** faster (4.33–4.46% lower latency).
Reference drift is 0.137%. All four arms return finite, identical 16 × 32 × 8
action arrays. Diagnostic recomputation checks 31,104 cached tensor outputs.
The speed arm records 3240 replays across 108 slot/layer graph objects, with
no rejection. Both models pass the declared 1.02 minimum-speedup gate.

Nano p95 improves from 8687–8701 to 8311 ms. Peak allocated memory increases
from 32.86 to 35.01 GiB; reserved memory increases from 34.71 to 37.79 GiB.
The candidate therefore costs about 2.15 GiB of extra allocation on this model.

## Reproduction and gates

The archived reports record checkpoint identity, source hashes, fixture hashes,
numeric settings, execution schedule, per-call timings and action archive hashes.
The benchmark uses six warmup and ten measured requests, changing prompts,
recorded camera frames, states and resets. The runner holds the Thor GPU lock
over all four fresh processes; each request also checks GPU contention.

`run.py ROOT edge verify baseline_a cached baseline_b` uses the explicitly
configured installed Cosmos environment and a frozen InstinctFlash source and
fixture snapshot at ROOT. Use `nano` for Nano. `benchmark.py` installs the
experimental wrapper after native pointwise/graph setup, before first inference.

```bash
python compare.py /path/to/receipts edge --minimum-speedup 1.02
python compare.py /path/to/receipts nano --minimum-speedup 1.02
python verify_compare.py
```

The comparator requires matching protocols, finite full actions, successful
module recomputation, actual graph replay coverage and stable reference actions.
It separately reports action-byte and performance gates and exits nonzero if
either fails. The default drift limit is 5%; the default minimum speedup is 1.0.
The p95 guard rejects more than 10% regression against the slower reference p95.
The archived Edge receipt uses a 1.02 speedup threshold. Malformed evidence first
invalidates the output receipt so an older passing result cannot survive a
failed new comparison. `verify_compare.py` injects a bad action hash, changed CFG
protocol, doubled latency and an isolated tail-latency regression to exercise
rejection and stale-result prevention.

`probe.py OUTPUT.json` runs small BF16 tests with real CUDA Graphs on an idle
Thor. It exercises changed payloads under the same token key, reversed branch
order, LRU eviction, retained output ownership, weight updates/replacement, detection of violated text
invariance, partial-prefill exceptions and successful retries. The probe owns
the shared GPU lock independently and must run after the model benchmark.
All nine lifecycle checks passed on Thor; see `receipts/lifecycle-probe.json`.
The four CPU comparison fault-injection checks also passed.

## Admission

This is an offline candidate, not a default Runtime installation. It does not
modify the accepted nightly baseline. The measurements qualify the recorded
inputs and checkpoint; they do not certify task success or all possible inputs.
Production integration still needs guarded source/module admission and serving
fallback behavior. The experimental wrapper intentionally rejects unsupported
conditions instead of silently reusing questionable state.

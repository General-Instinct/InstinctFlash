# Shared native tensor-result cache

`instinctflash/runtime/tensor_cache.py` supplies a bounded, owned tensor-result
cache. It does not assume a model family or change arithmetic. The adapter must
prove its function is pure and supply a complete immutable dependency key.
Non-contiguous outputs are left uncached to preserve native output layout.

The first adapter is Cosmos Edge/Nano's original FP32 timestep embedder:

```sh
IFL_COSMOS3_TIMESTEP_CACHE=1 <your existing native Runtime command>
```

This switch is independent of NUMERIC compilation. It does not permit FP8,
change the sampling schedule, or certify the surrounding pipeline's task quality.
It defaults off and rejects the currently unqualified FP8 adapter path. Runtime
backend statistics report occupancy, hits/misses, fallbacks and invalidations;
Runtime.close() restores the wrapped function and releases cached tensors.

## Ownership and validity

- 128 MiB Edge / 256 MiB Nano tensor-storage bounds and a 16-entry LRU.
  The initial frozen Nano run uses 128 MiB and exposes capacity thrashing; the
  additive Nano v2 study qualifies its larger budget. Small host
  keys and transient caller-owned output clones are additional memory.
- Complete FP32 input bytes, shape/stride/device; parameter/buffer identity and
  mutation version; relevant autocast/matmul execution context.
- Original full GEMMs on cold misses, owned clones on hits. No action/noise cache.
- Serialized cache operations; CUDA events order cross-stream reads and
  record_stream protects stored allocations after eviction on another stream.
- Training, gradients, CUDA capture, unsupported inputs, local module hooks and
  unavailable parameter version tracking retain native execution.

The native adapter assumes its inspected function implementation and module
structure remain fixed, and external mutation is serialized with inference.
Arbitrary `.data` writes, global hooks, code monkeypatches or concurrent model
mutation are not supported cache invalidation mechanisms. Disable/close and
reinstall the optimization when changing the model implementation. This primitive
is not a general safe wrapper around arbitrary neural network modules.

## Regression

CPU tests cover ownership, bounded eviction, concurrent single computation,
training/hooks, complete input keys, checkpoint reload and autocast invalidation.
CUDA stream lifetime is a separate opt-in test on the reserved GPU:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:examples/cosmos3_policy python -m pytest -q \
  tests/test_tensor_cache.py tests/test_cosmos_timestep_cache.py
# Only on a reserved GPU:
IFL_TEST_CUDA=1 python -m pytest -q tests/test_tensor_cache.py -k cuda_cross_stream
```

`run.py` runs fresh reference/cache/reference processes for Edge and Nano under
the exclusive Thor lock. Each retains six warmup plus ten measured full 32x8
action chunks and per-request compilation counters. `compare.py` requires active
cache hits, matched protocols, unchanged compilation coverage and byte-identical
full actions, including changing inputs/prompts/resets. It reports latency
separately from task quality. The additional non-contiguous-output fallback was
added after freezing the benchmark source; tested native outputs are contiguous.

`matched/` is the shared Edge cache's own six-process comparison against pinned
Omni, with 12 warmups and 30 measured requests each. Cross-framework action deltas
are not precision-loss estimates because their initial-noise generators differ.
See ARCHITECTURE.md for transferable boundaries and remaining family-specific gaps.

## Initial A/B/A results

| Model | Reference A | Cache | Reference B | Full actions |
|---|---:|---:|---:|---|
| Edge | 1090.55 ms | 1042.72 ms | 1093.47 ms | All 16 chunks byte-identical |
| Nano, 128 MiB bound | 5104.47 ms | 5027.10 ms | 5112.56 ms | All 16 chunks byte-identical |

Edge gains 4.39–4.64%, with 248 hits / 8 cold misses and no eviction. Nano gains
1.52–1.67%, but 123 evictions reveal the 128 MiB budget does not retain its full
working set. The additive Nano v2 study uses a 256 MiB bound and separately tests
split prefill. These numbers do not establish cross-framework or task-quality
superiority. Frozen initial implementations are retained in frozen_source/.

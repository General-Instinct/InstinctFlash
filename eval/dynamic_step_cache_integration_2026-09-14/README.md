# Shared dynamic step-cache integration

The implementation adds an opt-in shared controller, frozen public schedule
selection and native DreamZero adapter hooks. Both native and explicit FP8
arithmetic can select `step_cache="dynamic"` with a behavioral ceiling. The
shipped fixed eight-of-sixteen mask remains the default. See the
[runtime guide](../../docs/dynamic-step-cache.md).

All four Thor processes and the independent audit passed: **72 finite action
arrays, 24 exact same-policy comparisons**. Dynamic-cache p50 speedups were
**1.78× native and 1.76× FP8**. The [results](results.md) include raw latencies,
scope and recheck commands; [audit.json](audit.json) contains the independent
source, action and receipt verification.

This study compares four fresh processes on an exclusively locked Jetson Thor.
[protocol.json](protocol.json) was frozen before GPU outputs. Each process runs
nine complete public predictions (one warmup and two measured three-cycle
episodes). The primary latency screen has **four continuation samples per arm**;
it does not replace the larger published benchmark or establish tail latency.

Each dynamic process additionally compares the shared implementation, the
unmodified native dynamic loop and the reinstalled shared implementation on six
matched requests apiece, using the same loaded weights and arithmetic. This
checks actions, raw endpoint hashes, all retained native KV hashes, actual DiT
and CFG branch counts, both sixteen-slot solver clocks and RNG states. It does
not compare dynamic outputs with fixed-mask outputs or establish task quality.

The queue stops on failure and has no automatic retry. Original source,
checkpoints and prior benchmark results are retained. No H100 jobs are part of
this study. All four jobs completed without an inference retry, and Thor was
idle at completion. The results are latency SCREEN evidence; task quality
relative to the fixed-mask baseline remains unqualified.

`benchmark.py`, `run.py`, `load_memory.py`, `freeze.py` and `protocol.json` are
bound by the frozen driver manifest. `source_manifest.json` inventories the
isolated runtime source and native binaries; `source.tar.gz` preserves that
source. `external_manifest.json` binds the external DreamZero source and input
fixture. The frozen bundle uses the existing checkpoint loader and separately
records its checkpoint configuration and loading receipts.

[cpu_validation.json](cpu_validation.json) records **215 passing CPU tests** for
the final working tree, including three post-freeze refinements that do not
change successful inference: public detached statistics snapshots, detached
nested execution metadata, and cleanup ownership after close refusal/conflicts.
The GPU source remains frozen; these refinements have separate CPU evidence.

The research audit and original algorithm comparison are retained in
[the preceding study](../dynamic_step_cache_2026-09-14/README.md).

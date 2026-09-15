# Cosmos3 native acceleration on Jetson Thor

The native pairs below use the pinned upstream eager path. The separately
[audited newer upstream](BASELINE_AUDIT.md) and experimental compiled BF16
attention results have different action bytes and must not replace this
BITEXACT reference.

This pass preserves BF16 parameters, four UniPC steps, CFG 3 and complete 32×8
DROID actions. It adapts the existing GR00T shared graph-pool and owned-output
patterns, and Nano's existing action-only weight loader. Pointwise fusion keeps
the native mean reduction and intermediate rounding. Neither FP8 nor the
experimental SwiGLU lookup path is enabled.

## Results

| Model | Upstream A p50 (ms) | InstinctFlash p50 (ms) | Upstream B p50 (ms) | Speedup range | Full-action bytes |
| --- | ---: | ---: | ---: | ---: | --- |
| Edge | 3398.3 | 2554.4 | 3388.0 | 1.326–1.330× | PASS |
| Nano | 10236.8 | 8672.6 | 10210.9 | 1.177–1.180× | PASS |

Both A/B/A gates pass: zero action-byte differences, reference drift below 0.4%,
and no observed competing GPU process. [Edge receipt](receipts/edge-comparison.json)
and [Nano receipt](receipts/nano-comparison.json).

## Protocol

Each model runs upstream A, InstinctFlash, upstream B in separate processes on
one idle Thor. Every arm executes six warmup and twenty measured requests, with
two prompts, changing recorded camera frames and states, continuous requests,
and resets. All 26 complete action chunks are compared, including warmup.
Edge retains JSON prompt formatting; Nano retains plain prompt formatting.

The reference directly constructs NVIDIA's RoboLab DROID service from the installed
`cosmos-framework` checkout at `f734253f0f6af3e268372402f44435c38f55ef3e`;
the receipts record its actual imported-file hashes. It does not
install the adapter's optimizations. Inputs use the same public Runtime wrapper
and the same pinned checkpoint. TF32 and cuDNN benchmark are disabled in all
arms. The GPU lock spans the complete suite, and competing processes are checked
before loading and each request. The receipts retain per-call timings, complete
action arrays, source hashes, checkpoint identity, effective schedule, runtime
versions, and graph admission statistics.

The measured inference sources match commit
`8fce33a7ac5f57cfa2705126301043957f5b93e2`, with `--candidate graphs`. The later
native Thor default selects those same installers. H100 and FP8 defaults are
unchanged. Reproduce the frozen candidate on Thor from that commit:

```bash
python -m benchmarks.regression.run_cosmos /absolute/path/to/new-results \
  --candidate graphs --iterations 20
```

The existing Cosmos environment and cached checkpoints are required; the runner
installs no dependencies. Native defaults can be disabled individually with
`IFL_COSMOS3_EXACT_POINTWISE=0`, `IFL_COSMOS3_LAYER_GRAPHS=0`, and
`IFL_COSMOS3_NANO_ACTION_ONLY=0`.

## Regression and limits

Further Thor experiments are archived separately:

| Experiment | Outcome | Evidence |
| --- | --- | --- |
| BF16 attention reuse, newer compiled upstream | Edge 2126 → 1309 ms; Nano 7404 → 5283 ms; action bytes differ | [Attention](attention-reuse/README.md) |
| Static compilation | No speed gain; additional action differences | [Compile shapes](static_compile/README.md) |
| Selective CUDA Graph capture | No speed gain; additional action differences | [Graphs](graph_compile/README.md) |
| Persistent attention tile scheduler | Slower on actual generation shapes | [Tiles](attention-tiles/README.md) |
| Existing BF16 GEMM and weight packing | Synthetic gain did not improve native Nano end-to-end latency; action bytes match | [GEMM](gemm-reuse/README.md), [full model](weight-layout/README.md) |
| Request-local text conditioning cache | Edge 2558–2560 → 2468 ms; Nano 8681–8693 → 8305 ms; all tested action bytes match | [Cache qualification](conditioning-cache/README.md) |
| Guarded conditioning cache in public Runtime | Edge 2557–2559 → 2469 ms; Nano 8688–8693 → 8305 ms; all tested action bytes match | [Runtime integration](runtime-conditioning/README.md) |
| Fixed interleaved KV buffers on guarded Runtime | Edge 2468–2473 → 2466 ms; bytes match, performance gate fails, +1.03 GiB | [Fixed KV](fixed-kv/README.md) |

All of these experiments preserve BF16 storage, but that alone does not qualify
them as BITEXACT. None changes the accepted native regression baseline.

[The regression runner](../../benchmarks/regression/README.md) separates action
identity from performance. Unstable references, competing GPU processes,
non-finite or incomplete action chunks, mismatched protocols, and changed source
hashes cannot produce a passing paired result. Nightly runs also compare the
candidate/reference latency ratio against an explicitly accepted fixed baseline.

These are recorded-input action-equivalence checks, not closed-loop task-success
measurements or a claim of equivalence for every possible input. Graph admission
checks actual inputs and falls back to eager execution on failure. Cold requests
and capture costs remain in the raw receipts; the table reports warm p50.

[Reuse audit and deferred work](REUSE.md) records which existing optimizations
apply, why guidance=1 prompt caching cannot simply be enabled at CFG 3, and why
a shared graph pool needs owned output copies. The independent-pool Nano probe
was stopped when memory consumption stalled progress; no speed is claimed for
that failed probe. Attention replacement and whole-denoiser changes remain
separate model-specific work after this reuse pass.

The current NVIDIA model card also reports a faster Edge PyTorch Thor result
under a similarly described four-step request. That result has not been
reproduced on this pinned stack and is not substituted for our measured
reference. Resolving the software/protocol gap is a priority before claiming
that Edge needs distillation. See the [few-step research note](../../docs/rfc/cosmos3-fewstep-distillation.md).

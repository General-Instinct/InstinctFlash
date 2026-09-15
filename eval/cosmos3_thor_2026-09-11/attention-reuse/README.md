# Thor BF16 Blackwell FMHA reuse — experimental numerical screen

This adapts the existing engine's Blackwell strided FMHA layout/GQA mapping,
Thor architecture compatibility shim and preallocated-workspace pattern. It
instantiates **BF16 Q/K/V and output**, with FP32 accumulators. No FP8 conversion,
distillation, CFG interval or step-count change is used.

## Full-model results

All arms use the pinned upstream `2b6c9a7061ae78dc83e29a4910ec5f8c9fe4b6ce`,
torch 2.10/cu130 on Thor, native BF16, compilation on, four UniPC steps, full
CFG 3, complete 32×8 actions, six warmup and ten measured requests. TF32 and
cuDNN benchmark are off. GPU jobs are serialized under `/tmp/thor_gpu.lock`.

| Model | Compiled reference p50 | BF16 FMHA candidate p50 | Ratio | Max / mean absolute action delta |
| --- | ---: | ---: | ---: | ---: |
| Edge | 2125.79 ms | 1309.41 ms | 1.623× | 0.36227 / 0.01746 |
| Nano | 7404.12 ms | 5283.22 ms | 1.401× | 1.00111 / 0.08759 |

**Neither pair is BITEXACT.** These are raw action differences, not task-success
measurements or declared acceptable error margins. The comparison uses archived
compiled references and single candidate arms, not a fresh A/current/B
qualification. No production Runtime default, accepted nightly baseline or
published Results cell is changed.

Both candidates return finite full chunks on all 16 requests, with no observed
competing GPU process. Actual CUDA operator counts are **3584 Edge / 4608 Nano**
(16 requests × 4 steps × 2 CFG branches × 28 / 36 layers).

Actual GQA input geometries:

- Edge: Q `[1,3094,16,128]`; K/V `[1,3245,8,128]` or `[1,3112,8,128]`.
- Nano: Q `[1,3094,32,128]`; K/V `[1,3182,8,128]` or `[1,3103,8,128]`.

`summary.json`, per-arm JSON/NPZ and instrumentation sidecars retain the timings,
complete actions, checkpoint identity, imported-source hashes and actual call
coverage. The initial equal-head-only adapter executed zero replacement calls;
that diagnostic run is not an accelerated result. The reusable audit now fails
when attention replacement is requested but executes zero calls.

## Correctness boundaries

Only single-batch, non-causal dense attention, head_dim=128 and compatible
strides/head ratios are eligible. Causal attention, varlen document boundaries,
LSE requests and other explicit options fall back to upstream. This prototype
patches two module dispatch references inside an isolated experiment process;
it is not a multi-model/concurrent serving integration.

**Tail masking is required even for non-causal dense attention.** The engine's
original `NoMask` template lets zero-padded KV tail positions enter softmax when
KV length is not a multiple of 128. The first synthetic probe exposed errors
up to 0.826 versus FP32 on a short sequence. This candidate uses CUTLASS
`ResidualMask`, which excludes those positions; it does not introduce a causal
or task mask. The rejected unmasked version is not used in model measurements.

The corrected synthetic screen includes odd tails and GQA. On the large
expanded-head layouts seen in NATTEN profiling, attention falls from about
9.90–10.12 ms to 3.36–3.37 ms. Max absolute output delta versus NATTEN is 0.00098.
Small operator deltas do **not** establish small final-action deltas: the full
model numbers above are the relevant observed discrepancy.

`verification.json` records passing checks for:

- Constant-value attention with KV lengths 1, 31 and 257, catching tail mass leakage.
- Strided/interleaved inputs versus contiguous inputs.
- Retaining an output across later calls using the same workspace.
- CUDA Graph capture and replay after changing the input.

These check kernel mechanics; they do not certify task quality or byte
identity with another attention algorithm.

## Build and reproduce

CUTLASS checkout: `da5e086dab31d63815acafdac9a9c5893b1c69e2` (clean worktree).
The build uses its stock example-77 mainloop, not the engine's FP8-specific
probability rescaling modification. The graph-safe workspace is caller-owned
and allocated before execution.

```bash
CUTLASS_ROOT=/absolute/pinned/cutlass bash eval/cosmos3_thor_2026-09-11/attention-reuse/build.sh
flock -n /tmp/thor_gpu.lock python eval/cosmos3_thor_2026-09-11/attention-reuse/probe.py /absolute/new-screen.json
flock -n /tmp/thor_gpu.lock python eval/cosmos3_thor_2026-09-11/attention-reuse/verify.py /absolute/new-verification.json
```

For full-model runs, set `PYTHONPATH` to the pinned Cosmos upstream checkout,
this repo and `examples/cosmos3_policy`; use the cached policy checkpoints and
Thor Cosmos Python environment. Set `AUDIT_COMPILE=1` and
`AUDIT_ENGINE_ATTENTION=/absolute/path/bf16_fmha.so`, then run
`audit_upstream.py FAMILY baseline_a OUTPUT --iterations 10` under the GPU lock.
Do not also enable the SwiGLU or CFG-interval experiments.

Next: investigate layer/action divergence and run checkpoint-appropriate
closed-loop quality evaluation before considering an opt-in native numerical
route. Further BITEXACT work must keep its own reference and acceptance gate.

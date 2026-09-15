# GR00T BF16 text-block graph ablation on Thor

This extends the norm-only experiment to a complete post-attention region:
`x + MLP(RMSNorm(x))`. Attention and its input norm retain upstream execution.
All three BF16 GEMMs stay separate; no projection concatenation or quantization.

`norm_candidate.py` verifies exact upstream decoder-layer, norm and MLP source
hashes. It replaces only each decoder layer's forward, with unchanged attention
argument forwarding and residual order. Original child modules remain intact.
The graph-only arm captures the original region. The fused arm uses the shared
BF16 norm and lookup-table SwiGLU. Both use original-region byte admission,
owned output copies and normal eager fallback on a failed capture. The benchmark
rejects failed admission rather than counting fallback as optimized execution.

| Arm | p50 ms | Speedup vs baseline B |
|---|---:|---:|
| Baseline A | 115.52 | — |
| Original region graph | 112.28 | 1.025x |
| Shared fusion region graph | 113.65 | 1.013x |
| Baseline B | 115.07 | — |

All 23 × 40 × 17 actions are finite and byte-identical. Each candidate records
16 active graphs and 368 replays. Protocol matches the preceding norm-only
experiment: same checkpoint, recorded cameras, synthetic state, two prompts,
3 warmups + 20 measured requests, serialized fresh processes and TF32 disabled.
The reused report field `norm_graphs` contains complete text-block graphs here.

The larger graph shows modest gains in this sweep. Shared norm/SwiGLU fusion
does not outperform the original region graph, so it is not enabled by default.
There is no simulator-quality claim or qualification for other checkpoints.

`run.py ROOT OUTPUT_DIR` executes all four arms against a frozen source root;
`compare.py OUTPUT_DIR` validates protocol and every saved action byte.
`profile_runtime.py` is a separate optional profiling harness: setting
`IFL_BENCH_PROFILE=1` profiles an extra request after the timed requests and
records CUDA-kernel durations separately from CPU operator self time.

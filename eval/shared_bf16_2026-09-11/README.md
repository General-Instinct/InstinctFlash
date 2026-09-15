# Shared BF16 primitive qualification on Thor

`verification.json`: all four groups passed on NVIDIA Thor, Torch 2.10.0+cu130.
The receipt records shared backend, Cosmos adapter and graph implementation hashes.
CPU contract/fallback tests: 8 passed; 2 CUDA tests skipped locally and covered by
this standalone Thor run. No H100 execution or model baseline replacement.

Reproduce from the repository root in the existing Cosmos environment:

```bash
PYTHONPATH="$PWD:$PWD/examples/cosmos3_policy" \
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
flock -n /tmp/thor_gpu.lock python benchmarks/regression/verify_cosmos_kernels.py \
  --output /tmp/shared-bf16-verification.json
```

This validates the extraction and declared primitive semantics on tested inputs.
It is not a new latency measurement, full-model action certificate, or evidence
that other adapters can enable these operations without their own qualification.
SwiGLU graph replay includes changed gate/up values and a separate invocation
that must not overwrite the captured output. The exhaustive finite BF16 SiLU
screen uses an up multiplier of one; it does not enumerate all gate/up pairs.

## Qwen3-VL module screen and residual fusion

`qwen3-residual-modules.json` compares the actual Transformers Qwen3-VL module
classes used by the GR00T text architecture, with random weights and synthetic
inputs (hidden 2048, MLP intermediate 6144). It is **not a GR00T checkpoint run**.
The source of each reference class is embedded in the receipt. No adapter defaults
were changed. Measurements use CUDA events, 5 warmups and 30 samples per arm,
reference/candidate/reference order; ratios below use the repeated reference.
Both sides in the graph rows use CUDA Graph. Compilation was not tested.

| Operation, CUDA Graph | 1 token | 128 tokens | 512 tokens |
|---|---:|---:|---:|
| RMSNorm | 1.43x | 2.04x | 3.39x |
| Residual + RMSNorm | 1.43x | 2.08x | 3.16x |
| Full MLP with shared SwiGLU | 1.00x | 1.00x | 1.08x |

All tested module outputs were finite and byte-identical, including changed-input
replay. Small eager norm cases are slower (0.80–0.87x), so these results support
prioritizing graph integration rather than blanket eager replacement. Full MLP
results include all three unchanged BF16 GEMMs. These are module speedups, not
whole-model speedups or task-quality evidence.

The new `residual_norm` fuses residual addition and square preparation while
preserving the BF16 addition boundary and Torch FP32 reduction. Its two outputs
are owned tensors. `residual-verification.json` records all five check groups
passing, including both norm rounding contracts, cancellation, graph replay and
output ownership. Local tests: 10 passed, 2 CUDA-only tests skipped.

Run `benchmark_qwen3.py --output /tmp/qwen3-screen.json` with the same environment
and lock as above. It refuses existing output and checks for competing GPU PIDs
at the start and end; it does not continuously monitor transient contention.

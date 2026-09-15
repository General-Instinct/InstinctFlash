# Existing C++ GemmRunner on Nano BF16 MLP geometry

This experiment compiles the existing `serving/csrc/gemm/gemm_runner.cu` with a
small diagnostic C ABI. It uses the BF16 NN path and its existing 32-candidate
cuBLASLt autotuner. Weight transposes are prepared before timing; weights and
activations stay BF16. No FP8/FP4 quantization or schedule change is involved.

The benchmark uses original released generation MLP weights at layers 0 and 35
and synthetic inputs with the profiled `[3093,4096]` shape. It compares native
PyTorch, shared SwiGLU, default C++ BF16 NN and tuned C++ BF16 NN in interleaved
trials. Output deltas, source/weight hashes and timing samples are recorded. The
C++ variants own reusable output buffers; this is an operator cost screen, not
an end-to-end speed qualification or task-quality certificate.

The existing small-M `bf16_matmul_qwen36` kernel uses per-row SIMT dot products;
it is not the appropriate reuse target for this 3093-token tensor-core workload.
The generic cuBLASLt runner is tested before introducing a new GEMM kernel.

On Thor, stage this directory with dereferenced source links, then build:

```sh
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC \
  shim.cu gemm_runner.cu -lcublasLt -lcudart -o libdiagnostic_gemm.so
```

Run `benchmark.py ORIGINAL_NANO_PACKAGE FRESH_REPORT --library
libdiagnostic_gemm.so` with the study Flash source on PYTHONPATH. The worker holds
the shared Thor GPU lock, rejects GPU contention, and disables TF32. CUDA13 build
completed; results must be inspected before claiming any benefit or admission.

## Completed operator result

| Layer | PyTorch | Shared SwiGLU | C++ default BF16 | C++ tuned BF16 |
|---|---:|---:|---:|---:|
| 0 | 9.111 ms | 8.645 ms | 9.313 ms | 9.107 ms |
| 35 | 10.352 ms | 9.941 ms | 10.215 ms | 10.026 ms |

The tuner requested up to32 candidates; cuBLASLt returned7 for these shapes.
Raw tuning output is in `run.log`. Tuned C++ was essentially tied on layer0 and
about3% faster on layer35. Shared SwiGLU remained faster in both sampled cases.
All sampled synthetic-input outputs matched the PyTorch baseline byte for byte;
this does not establish model action equivalence. Source/weight hashes and all
interleaved timing samples are in `receipt.json`. The result does not justify
replacing the model's BF16 projections or claiming a large C++ kernel speedup.

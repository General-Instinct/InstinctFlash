# Nano BF16 merged-projection cost screen

Original released generation MLP weights from layers 0 and 35; synthetic BF16
inputs shaped `[3093,4096]`, matching the measured generation-token geometry.
Five interleaved trials of twenty iterations per variant after warmup. CUDA-event
spans include host launch gaps. This is not an end-to-end or policy-quality test.

| Layer | Separate GEMMs | Separate + shared SwiGLU | Merged gate/up | Merged + copies + shared SwiGLU |
|---|---:|---:|---:|---:|
| 0 | 9.035 ms | 8.548 ms | 10.461 ms | 11.011 ms |
| 35 | 10.257 ms | 9.824 ms | 11.887 ms | 12.500 ms |

The plain merged projection was approximately 16% slower for both sampled layers.
Adding contiguous copies for the shared activation made it slower still. All
sampled outputs matched on these synthetic inputs; this is not an action-equivalence
qualification. Source and selected original weight-tensor hashes are in the receipt.
No merged projection is installed in Runtime. The next candidate is the existing
C++ GemmRunner BF16 NN layout and algorithm tuning, tested separately.

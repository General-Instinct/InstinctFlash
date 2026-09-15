# Optional SM120 kernels

The core InstinctFlash package remains CUDA-free. Build this small raw-pointer library only on an
RTX 5090 / SM120 host:

```bash
cmake -S instinctflash/native -B /tmp/instinctflash-sm120-build \
  -DCMAKE_CUDA_COMPILER=/path/to/nvcc -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/instinctflash-sm120-build -j
export IFL_SM120_KERNEL_LIBRARY=/tmp/instinctflash-sm120-build/libinstinctflash_sm120.so
export IFL_SM120_STAGE2_LIBRARY=/tmp/instinctflash-sm120-build/libinstinctflash_sm120_wan_stage2.so
export IFL_SM120_STAGE3_LIBRARY=/tmp/instinctflash-sm120-build/libinstinctflash_sm120_wan_stage3.so
export IFL_SM120_QK_ROPE_LIBRARY=/tmp/instinctflash-sm120-build/libinstinctflash_sm120_wan_qk_rope.so
export IFL_SM120_GEMM_LIBRARY=/tmp/instinctflash-sm120-build/libinstinctflash_sm120_wan_gemm.so
export IFL_SM120_RING_CONCAT_LIBRARY=/tmp/instinctflash-sm120-build/libinstinctflash_sm120_wan_ring_concat.so
```

`DeviceProfile` reports `sm120_kernels` only when the library loads and exports the expected ABI.
P009-A1 therefore stays out of the plan on every other architecture and on a 5090 without a valid
optional build. The runtime also checks SM120, BF16/FP32 dtypes, device identity, and the certified
LingBot production shapes before every launch.

P009-A2 has a separate library, environment variable, feature (`sm120_stage2_kernels`), and ABI.
It is selected only together with P009-A1 and additionally validates Welford-specific LayerNorm,
stride, alignment, alias, upstream-source, 30-block, and single-stream preconditions.

P009-A3 is another independent ABI (`sm120_stage3_kernels`). It requires A1+A2 and fuses the
remaining norm1 FP32 LayerNorm + Ada modulation chain only for the certified D3072, rows 64/480,
Torch 2.9/CUDA 12.8 operating point; all other shapes and builds fail closed.

P009-A4 (`sm120_qk_rope_kernels`) requires A1+A2+A3 and replaces only the 30 self-attention
Q/K RMSNorm+RoPE regions. It pins P003's ring-aware forward and validates BF16 weights,
complex64 frequencies, H24/D128 geometry, alignment, aliasing, and one-stream execution.

P009-A5 (`sm120_gemm_kernels`) requires A1+A2+A3+A4 and pins two no-split-K cuBLASLt
configurations for the certified `(M,N,K)=(480,3072,3072)` and `(64,14336,3072)` BF16
Linear shapes. Every other shape keeps upstream `torch.nn.Linear`; weight mutation, pointer,
dtype, contiguity, module-graph, Torch/CUDA/cuBLASLt version, device, and stream mismatches fail
closed. The tactic IDs are certified specifically against cuBLASLt 12.8.4 (`120804`).

P009-A6 (`sm120_ring_concat_kernels`) requires A1-A5. For a wrapped P003 ring interval it
copies K and V together into one shared persistent scratch arena, preserving ascending physical
slot order exactly. It is certified only for the production `[2,9792,24,128]` BF16 pools;
shape, wrap state, dtype, contiguity, alignment, device, ABI, and stream mismatches fail closed.

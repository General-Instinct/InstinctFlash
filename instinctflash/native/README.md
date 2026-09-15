# Optional SM120 kernels

The core InstinctFlash package remains CUDA-free. Build this small raw-pointer library only on an
RTX 5090 / SM120 host:

```bash
cmake -S instinctflash/native -B /tmp/instinctflash-sm120-build \
  -DCMAKE_CUDA_COMPILER=/path/to/nvcc -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/instinctflash-sm120-build -j
export IFL_SM120_KERNEL_LIBRARY=/tmp/instinctflash-sm120-build/libinstinctflash_sm120.so
export IFL_SM120_STAGE2_LIBRARY=/tmp/instinctflash-sm120-build/libinstinctflash_sm120_wan_stage2.so
```

`DeviceProfile` reports `sm120_kernels` only when the library loads and exports the expected ABI.
P009-A1 therefore stays out of the plan on every other architecture and on a 5090 without a valid
optional build. The runtime also checks SM120, BF16/FP32 dtypes, device identity, and the certified
LingBot production shapes before every launch.

P009-A2 has a separate library, environment variable, feature (`sm120_stage2_kernels`), and ABI.
It is selected only together with P009-A1 and additionally validates Welford-specific LayerNorm,
stride, alignment, alias, upstream-source, 30-block, and single-stream preconditions.

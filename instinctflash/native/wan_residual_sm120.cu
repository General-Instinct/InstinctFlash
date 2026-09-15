// Wan gated residual for SM120.
//
// Exact eager semantics:
//   bf16(fp32(hidden) + round_fp32(fp32(update) * gate_fp32))
//
// The two RN intrinsics are load-bearing. A plain `hidden + update * gate` contracts to FMA under
// nvcc -O3, retaining a guard bit that PyTorch's separate multiply and add kernels discard. That
// changes a small number of BF16 words and eventually actions. The emitted SASS/PTX gate checks
// for separate FMUL and FADD instructions.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
extern "C" int instinctflash_sm120_abi_version() { return 1; }


__global__ void wan_gate_residual_bf16_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ update,
    const float* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    int rows,
    int dim,
    int gate_row_stride) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= dim) return;
  const int index = row * dim + col;
  const float product = __fmul_rn(__bfloat162float(update[index]),
                                  gate[row * gate_row_stride + col]);
  const float value = __fadd_rn(__bfloat162float(hidden[index]), product);
  output[index] = __float2bfloat16_rn(value);
}

extern "C" int wan_gate_residual_bf16(
    uintptr_t hidden,
    uintptr_t update,
    uintptr_t gate,
    uintptr_t output,
    int rows,
    int dim,
    int gate_row_stride,
    uintptr_t stream) {
  const dim3 block(256);
  const dim3 grid((dim + block.x - 1) / block.x, rows);
  wan_gate_residual_bf16_kernel<<<grid, block, 0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden),
      reinterpret_cast<const __nv_bfloat16*>(update),
      reinterpret_cast<const float*>(gate),
      reinterpret_cast<__nv_bfloat16*>(output),
      rows, dim, gate_row_stride);
  return static_cast<int>(cudaGetLastError());
}

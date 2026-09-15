// P009-A2: Wan block stage-2 fusion for SM120.
//
// The Welford update/combine order and launch geometry mirror PyTorch v2.9.0's vectorized
// LayerNorm path for aligned FP32 D=3072: float4 groups, threads=(32,4), six groups per thread,
// then the same x-warp and y-warp reduction tree. Residuals materialize to BF16 before the
// reduction, exactly where the eager Wan block materializes them.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" int instinctflash_sm120_wan_stage2_abi_version() { return 1; }

namespace {

constexpr int kDim = 3072;
constexpr int kVec = 4;

struct WelfordData {
  float mean;
  float sigma2;
  float count;
  __device__ WelfordData() : mean(0.0f), sigma2(0.0f), count(0.0f) {}
  __device__ WelfordData(float m, float s, float c) : mean(m), sigma2(s), count(c) {}
};

__device__ __forceinline__ WelfordData welford_online(
    float value, const WelfordData& current) {
  const float delta = value - current.mean;
  const float new_count = current.count + 1.0f;
  const float new_mean = current.mean + delta * (1.0f / new_count);
  return WelfordData(
      new_mean, current.sigma2 + delta * (value - new_mean), new_count);
}

// Argument order intentionally matches PyTorch's cuWelfordCombine(dataB, dataA).
__device__ __forceinline__ WelfordData welford_combine(
    const WelfordData data_b, const WelfordData data_a) {
  const float delta = data_b.mean - data_a.mean;
  const float count = data_a.count + data_b.count;
  if (count > 0.0f) {
    const float coefficient = 1.0f / count;
    const float n_a = data_a.count * coefficient;
    const float n_b = data_b.count * coefficient;
    const float mean = n_a * data_a.mean + n_b * data_b.mean;
    const float sigma2 = data_a.sigma2 + data_b.sigma2 +
                         delta * delta * data_a.count * n_b;
    return WelfordData(mean, sigma2, count);
  }
  return WelfordData(0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ WelfordData warp_reduce(WelfordData value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const WelfordData other(
        __shfl_down_sync(0xffffffffu, value.mean, offset),
        __shfl_down_sync(0xffffffffu, value.sigma2, offset),
        __shfl_down_sync(0xffffffffu, value.count, offset));
    value = welford_combine(value, other);
  }
  return value;
}

__device__ __forceinline__ WelfordData finish_block_stats(
    WelfordData value, int dim, float* buffer) {
  value = warp_reduce(value);
  float* mean_sigma = buffer;
  float* counts = buffer + blockDim.y;
  for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
    if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
      const int write_y = threadIdx.y - offset;
      mean_sigma[2 * write_y] = value.mean;
      mean_sigma[2 * write_y + 1] = value.sigma2;
      counts[write_y] = value.count;
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y < offset) {
      const WelfordData other(
          mean_sigma[2 * threadIdx.y], mean_sigma[2 * threadIdx.y + 1],
          counts[threadIdx.y]);
      value = welford_combine(value, other);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    mean_sigma[0] = value.mean;
    mean_sigma[1] = value.sigma2 / static_cast<float>(dim);
  }
  __syncthreads();
  return WelfordData(mean_sigma[0], mean_sigma[1], 0.0f);
}

__device__ __forceinline__ int linear_thread() {
  return threadIdx.x + threadIdx.y * blockDim.x;
}

__global__ void gate_residual_affine_layer_norm_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ update,
    const float* __restrict__ gate,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ normed,
    float* __restrict__ means,
    float* __restrict__ rstds,
    int dim,
    int gate_row_stride,
    float eps) {
  extern __shared__ float buffer[];
  const int row = blockIdx.x;
  const int base = row * dim;
  const int vectors = dim / kVec;
  const int threads = blockDim.x * blockDim.y;
  const int tid = linear_thread();
  const float* gate_row = gate + row * gate_row_stride;
  WelfordData stats;
  for (int i = tid; i < vectors; i += threads) {
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const int col = i * kVec + j;
      const int index = base + col;
      const float product = __fmul_rn(__bfloat162float(update[index]), gate_row[col]);
      const float sum = __fadd_rn(__bfloat162float(hidden[index]), product);
      const __nv_bfloat16 rounded = __float2bfloat16_rn(sum);
      residual[index] = rounded;
      stats = welford_online(__bfloat162float(rounded), stats);
    }
  }
  stats = finish_block_stats(stats, dim, buffer);
  const float rstd = rsqrtf(stats.sigma2 + eps);
  for (int i = tid; i < vectors; i += threads) {
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const int col = i * kVec + j;
      const float value = weight[col] *
                              (rstd * (__bfloat162float(residual[base + col]) - stats.mean)) +
                          bias[col];
      normed[base + col] = __float2bfloat16_rn(value);
    }
  }
  if (tid == 0) {
    means[row] = stats.mean;
    rstds[row] = rstd;
  }
}

__global__ void cross_residual_ada_layer_norm_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ update,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ normed,
    float* __restrict__ means,
    float* __restrict__ rstds,
    int dim,
    int scale_row_stride,
    int shift_row_stride,
    float eps) {
  extern __shared__ float buffer[];
  const int row = blockIdx.x;
  const int base = row * dim;
  const int vectors = dim / kVec;
  const int threads = blockDim.x * blockDim.y;
  const int tid = linear_thread();
  WelfordData stats;
  for (int i = tid; i < vectors; i += threads) {
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const int col = i * kVec + j;
      const int index = base + col;
      const float sum = __fadd_rn(
          __bfloat162float(hidden[index]), __bfloat162float(update[index]));
      const __nv_bfloat16 rounded = __float2bfloat16_rn(sum);
      residual[index] = rounded;
      stats = welford_online(__bfloat162float(rounded), stats);
    }
  }
  stats = finish_block_stats(stats, dim, buffer);
  const float rstd = rsqrtf(stats.sigma2 + eps);
  const float* scale_row = scale + row * scale_row_stride;
  const float* shift_row = shift + row * shift_row_stride;
  for (int i = tid; i < vectors; i += threads) {
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const int col = i * kVec + j;
      const float centered =
          __fsub_rn(__bfloat162float(residual[base + col]), stats.mean);
      const float layer_norm = __fmul_rn(rstd, centered);
      const float one_plus_scale = __fadd_rn(1.0f, scale_row[col]);
      const float scaled = __fmul_rn(layer_norm, one_plus_scale);
      const float shifted = __fadd_rn(scaled, shift_row[col]);
      normed[base + col] = __float2bfloat16_rn(shifted);
    }
  }
  if (tid == 0) {
    means[row] = stats.mean;
    rstds[row] = rstd;
  }
}

inline dim3 layer_norm_threads() { return dim3(32, 4, 1); }
inline int layer_norm_smem() { return 6 * sizeof(float); }

}  // namespace

extern "C" int wan_gate_residual_affine_layer_norm_bf16(
    uintptr_t hidden,
    uintptr_t update,
    uintptr_t gate,
    uintptr_t weight,
    uintptr_t bias,
    uintptr_t residual,
    uintptr_t normed,
    uintptr_t means,
    uintptr_t rstds,
    int rows,
    int dim,
    int gate_row_stride,
    float eps,
    uintptr_t stream) {
  if (rows <= 0 || dim != kDim || gate_row_stride < dim) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  gate_residual_affine_layer_norm_kernel<<<
      rows, layer_norm_threads(), layer_norm_smem(),
      reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden),
      reinterpret_cast<const __nv_bfloat16*>(update),
      reinterpret_cast<const float*>(gate), reinterpret_cast<const float*>(weight),
      reinterpret_cast<const float*>(bias), reinterpret_cast<__nv_bfloat16*>(residual),
      reinterpret_cast<__nv_bfloat16*>(normed), reinterpret_cast<float*>(means),
      reinterpret_cast<float*>(rstds), dim, gate_row_stride, eps);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int wan_cross_residual_ada_layer_norm_bf16(
    uintptr_t hidden,
    uintptr_t update,
    uintptr_t scale,
    uintptr_t shift,
    uintptr_t residual,
    uintptr_t normed,
    uintptr_t means,
    uintptr_t rstds,
    int rows,
    int dim,
    int scale_row_stride,
    int shift_row_stride,
    float eps,
    uintptr_t stream) {
  if (rows <= 0 || dim != kDim || scale_row_stride < dim || shift_row_stride < dim) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cross_residual_ada_layer_norm_kernel<<<
      rows, layer_norm_threads(), layer_norm_smem(),
      reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden),
      reinterpret_cast<const __nv_bfloat16*>(update),
      reinterpret_cast<const float*>(scale), reinterpret_cast<const float*>(shift),
      reinterpret_cast<__nv_bfloat16*>(residual),
      reinterpret_cast<__nv_bfloat16*>(normed), reinterpret_cast<float*>(means),
      reinterpret_cast<float*>(rstds), dim, scale_row_stride, shift_row_stride, eps);
  return static_cast<int>(cudaGetLastError());
}

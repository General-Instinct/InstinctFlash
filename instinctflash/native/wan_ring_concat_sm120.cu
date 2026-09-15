#include <cuda_runtime.h>
#include <cstdint>

extern "C" int instinctflash_sm120_wan_ring_concat_abi_version() { return 1; }

__global__ void ring_concat_bf16_vec8(
    const uint4* __restrict__ key,
    const uint4* __restrict__ value,
    uint4* __restrict__ out_key,
    uint4* __restrict__ out_value,
    int batch,
    int total_tokens,
    int inner_vec,
    int start,
    int count,
    int end) {
  const int64_t work = int64_t(batch) * count * inner_vec;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < work;
       index += int64_t(blockDim.x) * gridDim.x) {
    const int lane = int(index % inner_vec);
    const int64_t logical = index / inner_vec;
    const int token = int(logical % count);
    const int b = int(logical / count);
    const int source_token = token < end ? token : start + (token - end);
    const int64_t source = (int64_t(b) * total_tokens + source_token) * inner_vec + lane;
    out_key[index] = key[source];
    out_value[index] = value[source];
  }
}

extern "C" int wan_ring_concat_bf16(
    uintptr_t key,
    uintptr_t value,
    uintptr_t out_key,
    uintptr_t out_value,
    int batch,
    int total_tokens,
    int inner,
    int start,
    int count,
    uintptr_t stream) {
  if (!key || !value || !out_key || !out_value || batch <= 0 ||
      total_tokens <= 0 || inner <= 0 || inner % 8 != 0 ||
      start < 0 || start >= total_tokens || count <= 0 ||
      count >= total_tokens || start + count <= total_tokens ||
      start + count > 2 * total_tokens) {
    return -1;
  }
  const int end = start + count - total_tokens;
  const int inner_vec = inner / 8;
  const int64_t work = int64_t(batch) * count * inner_vec;
  const int threads = 256;
  const int blocks = int((work + threads - 1) / threads);
  ring_concat_bf16_vec8<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const uint4*>(key),
      reinterpret_cast<const uint4*>(value),
      reinterpret_cast<uint4*>(out_key),
      reinterpret_cast<uint4*>(out_value),
      batch,
      total_tokens,
      inner_vec,
      start,
      count,
      end);
  return int(cudaGetLastError());
}

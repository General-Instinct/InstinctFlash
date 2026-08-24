// ============================================================================
//  flash_wm — TRUE SiLU-mul + FP4 + SFA fused kernel for BAGEL (Class 1c).
//
//  Upstream `gate_silu_mul_fp4_sfa_v2_fp16` is mis-named: it actually runs
//  GELU-tanh-approx (see csrc/fused_fp4/silu_mul_fp4_sfa_v2.cu line 54).
//  BAGEL's FFN uses true SiLU (swish = x·sigmoid(x)) — same activation as
//  `fwk.silu_mul_fp16`. The upstream silu/gelu naming inversion is a
//  long-standing footgun documented in docs/kernel_fusion.md §1.
//
//  This kernel mirrors the upstream v2 layout (1 thread = 1 NVFP4 block)
//  with only the activation formula swapped:
//      gate_out = g / (1 + exp(-g)) * u        # true SiLU
//  vs upstream:
//      gate_out = g / (1 + exp(-1.5957*g*(1+0.044715*g²))) * u   # GELU tanh
//
//  Input:  merged [S, 2H] fp16   ([:, 0:H) = gate, [:, H:2H) = up)
//  Output: packed [S, H/2] uint8 (NVFP4) + SFA (CUTLASS tile-interleaved)
// ============================================================================
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace flash_wm {
namespace bagel_silu {

using CfgV2 = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t fp32_to_e2m1(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t mant;
    if      (ax <= 0.25f) mant = 0u;
    else if (ax <= 0.75f) mant = 1u;
    else if (ax <= 1.25f) mant = 2u;
    else if (ax <= 1.75f) mant = 3u;
    else if (ax <= 2.5f)  mant = 4u;
    else if (ax <= 3.5f)  mant = 5u;
    else if (ax <= 5.0f)  mant = 6u;
    else                  mant = 7u;
    return sign | mant;
}

__device__ __forceinline__ float true_silu_mul(float g, float u) {
    // True SiLU (swish): g · sigmoid(g) = g / (1 + exp(-g)).
    return g / (1.0f + __expf(-g)) * u;
}

template <class LayoutSF>
__global__ void bagel_silu_mul_fp4_sfa_kernel(
    const __half* __restrict__ merged,   // [S, 2H]
    uint8_t* __restrict__ packed,        // [S, H/2]
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int H) {
    const int block_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int row       = blockIdx.x;
    const int n_blocks  = H / 16;
    if (block_idx >= n_blocks) return;

    const int col_base = block_idx * 16;
    const __half* merged_row = merged + row * 2 * H;
    const __half* gate_ptr = merged_row + col_base;
    const __half* up_ptr   = merged_row + H + col_base;

    const __half2* gate2 = reinterpret_cast<const __half2*>(gate_ptr);
    const __half2* up2   = reinterpret_cast<const __half2*>(up_ptr);

    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        __half2 g2 = gate2[i];
        __half2 u2 = up2[i];
        float v0 = true_silu_mul(__half2float(g2.x), __half2float(u2.x));
        float v1 = true_silu_mul(__half2float(g2.y), __half2float(u2.y));
        vals[2*i]   = v0;
        vals[2*i+1] = v1;
        float a0 = fabsf(v0), a1 = fabsf(v1);
        if (a0 > amax) amax = a0;
        if (a1 > amax) amax = a1;
    }

    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    float bs_dq = static_cast<float>(bs_q);

    int sfa_off = layout(row, col_base, 0);
    dst_sfa[sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

    uint8_t* packed_row = packed + row * (H / 2);
    const float inv_bs = 1.f / bs_dq;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e2m1(vals[2 * p    ] * inv_bs);
        uint8_t hi = fp32_to_e2m1(vals[2 * p + 1] * inv_bs);
        packed_row[block_idx * 8 + p] = lo | (hi << 4);
    }
}

void run(const __half* merged, uint8_t* packed, uint8_t* sfa,
         int seq_len, int half_dim, cudaStream_t stream) {
    auto shape = cute::make_shape(seq_len, 1, half_dim, 1);
    auto layout = CfgV2::tile_atom_to_shape_SFA(shape);

    const int n_blocks = half_dim / 16;
    const int threads = 256;
    const int y_groups = (n_blocks + threads - 1) / threads;
    dim3 grid(seq_len, y_groups);
    dim3 block(threads);
    bagel_silu_mul_fp4_sfa_kernel<<<grid, block, 0, stream>>>(
        merged, packed, sfa, layout, half_dim);
}

}  // namespace bagel_silu
}  // namespace flash_wm

// Public C entry: true-SiLU fused gate_silu_mul + fp4 + SFA.
//   merged: fp16 [seq_len, 2*half_dim]  (gate in [:, :half_dim), up in [:, half_dim:])
//   packed: uint8 [seq_len, half_dim/2]  NVFP4 packed
//   sfa:    uint8 SFA block scales (CUTLASS tile-interleaved)
extern "C" void bagel_silu_mul_fp4_sfa_v2_fp16(
    const __half* merged, uint8_t* packed, uint8_t* sfa,
    int seq_len, int half_dim, cudaStream_t stream) {
    flash_wm::bagel_silu::run(merged, packed, sfa, seq_len, half_dim, stream);
}

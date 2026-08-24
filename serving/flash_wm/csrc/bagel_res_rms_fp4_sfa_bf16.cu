// ============================================================================
//  flash_wm — Fused residual_add + rms_norm(×weight) + FP4 + SFA  (BF16 input).
//
//  ROI A: eliminates the fp16 bridging buffer (b_normed_fp16 [Sq, D] fp16)
//  between the existing 2-kernel pattern:
//     (1) residual_add_rms_norm_bf16_to_fp16(x, r, w, b_normed_fp16, ...)
//     (2) quantize_fp4_dynamic_sfa_fp16(b_normed_fp16, sc_gu.packed, sc_gu.sfa, ...)
//
//  Replaces with a single kernel that does (x ← x+r); (normed = x/rms · w);
//  (per-16 FP4 quant + SFA) in one pass. Keeps the residual stream in BF16
//  (important: FP16 would overflow on L5/L9 at low t — documented in
//   debug_l9_t04.py).
//
//  Upstream `residual_add_rms_norm_mul_fp4_sfa_fp16` exists but takes FP16
//  residual (kills the overflow safety). This is the BF16 variant.
//
//  Weight `w` is the AWQ-baked `ln_baked = ln2_w × inv_s_gu` [D] bf16. The
//  inv_s_gu AWQ factor is absorbed into the weight, identical to what
//  `residual_add_rms_norm_bf16_to_fp16` already consumes.
//
//  Shape: D must be divisible by 16 (FP4 block size). D=3584 for BAGEL. One
//  CUDA block per row, blockDim.x = D/16 threads (each thread owns exactly
//  one FP4 block of 16 consecutive columns).
// ============================================================================
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace flash_wm {
namespace bagel_res_rms {

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

template <class LayoutSF>
__global__ void bagel_res_rms_fp4_sfa_kernel(
    __nv_bfloat16* __restrict__ x,           // [rows, D]  in-place: x ← x + r
    const __nv_bfloat16* __restrict__ r,     // [rows, D]
    const __nv_bfloat16* __restrict__ w,     // [D]  ln_baked (ln2_w × inv_s_gu)
    uint8_t* __restrict__ packed,            // [rows, D/2]  NVFP4 packed
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int D,
    float eps)
{
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int nblk = D / 16;   // threads per block = blocks-of-16 per row

    __nv_bfloat16* x_row   = x + row * D;
    const __nv_bfloat16* r_row = r + row * D;

    // Pass 1: load x + r, cache in registers, in-place bf16 write, accumulate ssq.
    const int col_base = tid * 16;
    float cache[16];
    float ssq = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int c = col_base + i;
        float v = __bfloat162float(x_row[c]) + __bfloat162float(r_row[c]);
        cache[i] = v;
        x_row[c] = __float2bfloat16(v);
        ssq += v * v;
    }

    // Pass 2: block-wide sum-of-squares reduce.
    __shared__ float sh[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, off);
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = (nblk + 31) >> 5;
    if (lane == 0) sh[warp] = ssq;
    __syncthreads();
    if (warp == 0) {
        ssq = (tid < num_warps) ? sh[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, off);
        if (lane == 0) sh[0] = ssq;
    }
    __syncthreads();
    const float rstd = rsqrtf(sh[0] / (float)D + eps);

    // Pass 3: normalize × ln_baked, amax over this thread's 16 elements.
    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        float v = cache[i] * rstd * __bfloat162float(w[col_base + i]);
        vals[i] = v;
        float av = fabsf(v);
        if (av > amax) amax = av;
    }

    // Pass 4: per-16-block FP4 quantize + SFA write.
    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    float bs_dq = static_cast<float>(bs_q);
    int sfa_off = layout(row, col_base, 0);
    dst_sfa[sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

    uint8_t* packed_row = packed + row * (D / 2);
    const float inv_bs = 1.f / bs_dq;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e2m1(vals[2 * p    ] * inv_bs);
        uint8_t hi = fp32_to_e2m1(vals[2 * p + 1] * inv_bs);
        packed_row[tid * 8 + p] = lo | (hi << 4);
    }
}

void run(__nv_bfloat16* x, const __nv_bfloat16* r, const __nv_bfloat16* w,
         uint8_t* packed, uint8_t* sfa,
         int rows, int D, float eps, cudaStream_t stream) {
    auto shape = cute::make_shape(rows, 1, D, 1);
    auto layout = CfgV2::tile_atom_to_shape_SFA(shape);
    const int threads = D / 16;   // one thread per FP4 block
    dim3 grid(rows);
    dim3 block(threads);
    bagel_res_rms_fp4_sfa_kernel<<<grid, block, 0, stream>>>(
        x, r, w, packed, sfa, layout, D, eps);
}

}  // namespace bagel_res_rms
}  // namespace flash_wm

// Public C entry.
//   x:  bf16 [rows, D]   in-place residual update: x ← x + r
//   r:  bf16 [rows, D]   delta (attention output b_o)
//   w:  bf16 [D]         ln_baked = ln2_w × inv_s_gu (AWQ-pre-applied)
//   packed: fp4 [rows, D/2]
//   sfa:    SFA block scales (CUTLASS tile-interleaved)
extern "C" void bagel_res_rms_mul_fp4_sfa_bf16(
    __nv_bfloat16* x, const __nv_bfloat16* r, const __nv_bfloat16* w,
    uint8_t* packed, uint8_t* sfa,
    int rows, int D, float eps, cudaStream_t stream) {
    flash_wm::bagel_res_rms::run(x, r, w, packed, sfa, rows, D, eps, stream);
}

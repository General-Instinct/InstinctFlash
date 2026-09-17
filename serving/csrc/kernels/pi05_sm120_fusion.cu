// SM120 pi0.5/SigLIP fixed-shape fusions. Keep every legacy BF16 rounding barrier.
#include "common.cuh"
#include <stdexcept>

namespace {
using BF = __nv_bfloat16;
constexpr int S = 512, D = 1152, H = 4304;

__device__ __forceinline__ BF bias_value(BF x, BF bias) {
    // The reference uses bias_residual(x, zero, bias), including its signed-zero behavior.
    return __float2bfloat16(__fadd_rn(__fadd_rn(__bfloat162float(x), 0.f), __bfloat162float(bias)));
}

__device__ __forceinline__ __nv_fp8_e4m3 quant(BF x, float inv_scale) {
    float value = __bfloat162float(x) * inv_scale;
    return __nv_fp8_e4m3(fminf(fmaxf(value, -448.f), 448.f));
}

__global__ void bias_split_qkv(const BF* __restrict__ input, const BF* __restrict__ bias,
                              BF* __restrict__ q, BF* __restrict__ k, BF* __restrict__ v) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (index >= S * 3 * D) return;
    int row = index / (3 * D), column = index % (3 * D);
    uint4 raw = *reinterpret_cast<const uint4*>(input + index);
    uint4 offset = *reinterpret_cast<const uint4*>(bias + column);
    uint4 result;
    auto* x = reinterpret_cast<BF*>(&raw);
    auto* b = reinterpret_cast<BF*>(&offset);
    auto* y = reinterpret_cast<BF*>(&result);
    #pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = bias_value(x[j], b[j]);
    BF* destination = column < D ? q + row * D + column
                    : column < 2 * D ? k + row * D + column - D
                                     : v + row * D + column - 2 * D;
    *reinterpret_cast<uint4*>(destination) = result;
}

template<int Vector>
__global__ void bias_gelu_static_fp8(const BF* __restrict__ input, const BF* __restrict__ bias,
                                   __nv_fp8_e4m3* __restrict__ output, const float* scale) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * Vector;
    if (index >= S * H) return;
    float inv_scale = 1.f / *scale;
    uint4 raw{}, offset{};
    if constexpr (Vector == 8) {
        raw = *reinterpret_cast<const uint4*>(input + index);
        offset = *reinterpret_cast<const uint4*>(bias + index % H);
    } else {
        *reinterpret_cast<uint2*>(&raw) = *reinterpret_cast<const uint2*>(input + index);
        *reinterpret_cast<uint2*>(&offset) = *reinterpret_cast<const uint2*>(bias + index % H);
    }
    const auto* x = reinterpret_cast<const BF*>(&raw);
    const auto* b = reinterpret_cast<const BF*>(&offset);
    alignas(8) __nv_fp8_e4m3 result[Vector];
    #pragma unroll
    for (int j = 0; j < Vector; ++j) {
        float value = to_f32(bias_value(x[j], b[j]));
        float t = tanhf(0.7978845608f * (value + 0.044715f * value * value * value));
        BF activated = from_f32<BF>(value * .5f * (1.f + t));
        result[j] = quant(activated, inv_scale);
    }
    if constexpr (Vector == 8)
        *reinterpret_cast<uint2*>(output + index) = *reinterpret_cast<const uint2*>(result);
    else
        *reinterpret_cast<uint32_t*>(output + index) = *reinterpret_cast<const uint32_t*>(result);
}

void checked_launch() {
    auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace

void pi05_bias_qkv(const void* input, const void* bias, void* q, void* k, void* v, cudaStream_t stream) {
    bias_split_qkv<<<(S * 3 * D / 8 + 255) / 256, 256, 0, stream>>>(static_cast<const BF*>(input),
        static_cast<const BF*>(bias), static_cast<BF*>(q), static_cast<BF*>(k), static_cast<BF*>(v));
    checked_launch();
}

void pi05_bias_gelu_fp8(const void* x, const void* b, void* out, const float* scale, int vector, cudaStream_t stream) {
    if (vector == 8)
        bias_gelu_static_fp8<8><<<(S * H / 8 + 255) / 256, 256, 0, stream>>>(static_cast<const BF*>(x),
            static_cast<const BF*>(b), static_cast<__nv_fp8_e4m3*>(out), scale);
    else if (vector == 4)
        bias_gelu_static_fp8<4><<<(S * H / 4 + 255) / 256, 256, 0, stream>>>(static_cast<const BF*>(x),
            static_cast<const BF*>(b), static_cast<__nv_fp8_e4m3*>(out), scale);
    else throw std::invalid_argument("supported vector widths are 4 and 8");
    checked_launch();
}

// ================================================================
// FlashRT — LingBot-VA (wan_va) fused kernels. See wan_va_fused.cuh.
// ================================================================
#include "wan_va_fused.cuh"

// One thread per __half2 pair: the residual and the gemm output are read
// once, the gate row is read through L1/L2 (6 KB, hot). fp32 math.
__global__ void gate_row_mul_residual_fp16_kernel(__half2* __restrict__ res,
                                                  const __half2* __restrict__ x,
                                                  const __half2* __restrict__ gate,
                                                  int n2, int d2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    int c = i % d2;
    float2 r = __half22float2(res[i]);
    float2 v = __half22float2(x[i]);
    float2 g = __half22float2(gate[c]);
    r.x += v.x * g.x;
    r.y += v.y * g.y;
    res[i] = __float22half2_rn(r);
}

void gate_row_mul_residual_fp16(__half* res, const __half* x,
                                const __half* gate_row, int S, int D,
                                cudaStream_t stream) {
    int n2 = (S * D) >> 1;
    int d2 = D >> 1;
    gate_row_mul_residual_fp16_kernel<<<(n2 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<__half2*>(res),
        reinterpret_cast<const __half2*>(x),
        reinterpret_cast<const __half2*>(gate_row), n2, d2);
}

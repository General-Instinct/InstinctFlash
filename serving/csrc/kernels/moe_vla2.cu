// ================================================================
// FlashRT — T2-V2 token-MoE kernels (implementation)
// See moe_vla2.cuh for contracts; moe_ref.py for the torch baseline.
// ================================================================

#include "moe_vla2.cuh"

#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>

// ────────────────────────────────────────────────────────────────
// Router device core: sigmoid + biased top-K + norm + scale.
// One WARP owns one token; lane e holds expert e (E <= 32).
// Iterative arg-max with lowest-index tie-break (deterministic).
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__ void moe_router_warp_reduce(
    float logit, int lane, const float* __restrict__ e_bias,
    int* __restrict__ ids_row, float* __restrict__ w_row,
    int E, int K, float routed_scale) {
    const bool active = lane < E;
    float score = active ? (1.0f / (1.0f + expf(-logit))) : -INFINITY;
    float sel = active ? (score + e_bias[lane]) : -INFINITY;

    float raw_k[8];
    int id_k[8];
    for (int k = 0; k < K; ++k) {
        float v = sel;
        int idx = lane;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, v, o);
            int oi = __shfl_down_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        idx = __shfl_sync(0xffffffffu, idx, 0);
        raw_k[k] = __shfl_sync(0xffffffffu, score, idx);
        id_k[k] = idx;
        if (lane == idx) sel = -INFINITY;   // knock out the winner
    }
    if (lane == 0) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += raw_k[k];
        const float denom = s + 1e-20f;     // norm_topk_prob eps (expert:297)
        for (int k = 0; k < K; ++k) {
            ids_row[k] = id_k[k];
            // op order mirrors moe_ref.route: divide, then * 4.0
            w_row[k] = (raw_k[k] / denom) * routed_scale;
        }
    }
}

// ── moe_router_topk: fp32 logits already computed ──
__global__ void moe_router_topk_kernel(
    const float* __restrict__ logits, const float* __restrict__ e_bias,
    int* __restrict__ ids, float* __restrict__ weights,
    int T, int E, int K, float routed_scale) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const int lane = threadIdx.x;
    const float logit = (lane < E) ? logits[t * E + lane] : 0.0f;
    moe_router_warp_reduce(logit, lane, e_bias, ids + t * K, weights + t * K,
                           E, K, routed_scale);
}

void moe_router_topk(const float* logits, const float* e_bias,
                     int* ids, float* weights,
                     int T, int E, int K, float routed_scale,
                     cudaStream_t stream) {
    moe_router_topk_kernel<<<T, 32, 0, stream>>>(
        logits, e_bias, ids, weights, T, E, K, routed_scale);
}

// ── Fused fp32 gate GEMM + router ──
// Block = 128 threads (4 warps) per token. x row staged in smem as
// fp32; warp w computes experts [w*(E/4), ...) by lane-strided dot +
// warp reduce; warp 0 then runs the top-K core on the smem logits.
// D <= MOE_ROUTER_MAX_D floats of dynamic smem (V2: D=768 → 3 KB).
template <typename LOADER>
__global__ void moe_router_gemm_topk_kernel(
    LOADER loader, const float* __restrict__ gate_w,
    const float* __restrict__ e_bias, int* __restrict__ ids,
    float* __restrict__ weights, float* __restrict__ logits_out,
    int T, int D, int E, int K, float routed_scale) {
    extern __shared__ float smem[];        // [D] x row | [32] logits
    float* xs = smem;
    float* ls = smem + D;

    const int t = blockIdx.x;
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;

    for (int i = threadIdx.x; i < D; i += blockDim.x) xs[i] = loader(t, i);
    __syncthreads();

    for (int e = warp; e < E; e += nwarp) {
        const float* w_row = gate_w + (size_t)e * D;
        float acc = 0.0f;
        for (int i = lane; i < D; i += 32) acc += xs[i] * w_row[i];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (lane == 0) {
            ls[e] = acc;
            if (logits_out != nullptr) logits_out[t * E + e] = acc;
        }
    }
    __syncthreads();

    if (warp == 0) {
        const float logit = (lane < E) ? ls[lane] : 0.0f;
        moe_router_warp_reduce(logit, lane, e_bias, ids + t * K,
                               weights + t * K, E, K, routed_scale);
    }
}

struct LoadFp16 {
    const __half* x;
    int D;
    __device__ float operator()(int t, int i) const {
        return __half2float(x[(size_t)t * D + i]);
    }
};

struct LoadFp8Descale {
    const __nv_fp8_e4m3* x;
    const float* descale;
    int D;
    __device__ float operator()(int t, int i) const {
        return float(x[(size_t)t * D + i]) * (*descale);
    }
};

void moe_router_gemm_topk_fp16x(const __half* x, const float* gate_w,
                                const float* e_bias, int* ids,
                                float* weights, float* logits_out,
                                int T, int D, int E, int K,
                                float routed_scale, cudaStream_t stream) {
    const size_t smem = (D + 32) * sizeof(float);
    moe_router_gemm_topk_kernel<<<T, 128, smem, stream>>>(
        LoadFp16{x, D}, gate_w, e_bias, ids, weights, logits_out,
        T, D, E, K, routed_scale);
}

void moe_router_gemm_topk_fp8x(const void* x_fp8, const float* gate_w,
                               const float* e_bias, int* ids,
                               float* weights, float* logits_out,
                               int T, int D, int E, int K,
                               float routed_scale, const float* act_descale,
                               cudaStream_t stream) {
    const size_t smem = (D + 32) * sizeof(float);
    moe_router_gemm_topk_kernel<<<T, 128, smem, stream>>>(
        LoadFp8Descale{reinterpret_cast<const __nv_fp8_e4m3*>(x_fp8),
                       act_descale, D},
        gate_w, e_bias, ids, weights, logits_out, T, D, E, K, routed_scale);
}

// ────────────────────────────────────────────────────────────────
// Combine: gather-weighted sum over the dense all-expert slab
// (+ optional ungated shared-expert add). fp32 accumulate, k-major
// add order identical to moe_ref.combine_gather.
// ────────────────────────────────────────────────────────────────
template <typename T>
__global__ void moe_combine_kernel(
    const T* __restrict__ slab, const int* __restrict__ ids,
    const float* __restrict__ weights, const T* __restrict__ shared,
    T* __restrict__ out, int T_, int D, int K) {
    const int t = blockIdx.x;
    if (t >= T_) return;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            const int e = ids[t * K + k];
            float v;
            if constexpr (sizeof(T) == 2)
                v = __half2float(reinterpret_cast<const __half*>(
                        slab)[((size_t)e * T_ + t) * D + d]);
            else
                v = reinterpret_cast<const float*>(
                        slab)[((size_t)e * T_ + t) * D + d];
            acc += weights[t * K + k] * v;
        }
        if (shared != nullptr) {
            if constexpr (sizeof(T) == 2)
                acc += __half2float(reinterpret_cast<const __half*>(
                           shared)[(size_t)t * D + d]);
            else
                acc += reinterpret_cast<const float*>(
                           shared)[(size_t)t * D + d];
        }
        if constexpr (sizeof(T) == 2)
            reinterpret_cast<__half*>(out)[(size_t)t * D + d] =
                __float2half(acc);
        else
            reinterpret_cast<float*>(out)[(size_t)t * D + d] = acc;
    }
}

void moe_combine_fp16(const __half* slab, const int* ids,
                      const float* weights, const __half* shared,
                      __half* out, int T, int D, int K,
                      cudaStream_t stream) {
    moe_combine_kernel<__half><<<T, 256, 0, stream>>>(
        slab, ids, weights, shared, out, T, D, K);
}

void moe_combine_fp32(const float* slab, const int* ids,
                      const float* weights, const float* shared,
                      float* out, int T, int D, int K,
                      cudaStream_t stream) {
    moe_combine_kernel<float><<<T, 256, 0, stream>>>(
        slab, ids, weights, shared, out, T, D, K);
}

// ────────────────────────────────────────────────────────────────
// True-SiLU merged gate|up (exact x*sigmoid(x), matching F.silu —
// NOT the GELU-tanh the pi05-lineage merged kernels apply).
// ────────────────────────────────────────────────────────────────
__global__ void silu_mul_merged_fp8_fp16_kernel(
    const __half* __restrict__ merged, __nv_fp8_e4m3* __restrict__ out,
    int S, int H, const float* __restrict__ descale_ptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S * H) return;
    const int s = i / H, h = i % H;
    const float g = __half2float(merged[(size_t)s * 2 * H + h]);
    const float u = __half2float(merged[(size_t)s * 2 * H + H + h]);
    const float silu_g = g / (1.0f + expf(-g));
    const float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);
    out[i] = __nv_fp8_e4m3(
        fminf(fmaxf(silu_g * u * inv_scale, -448.0f), 448.0f));
}

void silu_mul_merged_fp8_fp16(const __half* merged, __nv_fp8_e4m3* out,
                              int S, int H, const float* descale_ptr,
                              cudaStream_t stream) {
    silu_mul_merged_fp8_fp16_kernel<<<(S * H + 255) / 256, 256, 0, stream>>>(
        merged, out, S, H, descale_ptr);
}

__global__ void silu_mul_merged_fp16_kernel(
    const __half* __restrict__ merged, __half* __restrict__ out,
    int S, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S * H) return;
    const int s = i / H, h = i % H;
    const float g = __half2float(merged[(size_t)s * 2 * H + h]);
    const float u = __half2float(merged[(size_t)s * 2 * H + H + h]);
    out[i] = __float2half((g / (1.0f + expf(-g))) * u);
}

void silu_mul_merged_fp16(const __half* merged, __half* out,
                          int S, int H, cudaStream_t stream) {
    silu_mul_merged_fp16_kernel<<<(S * H + 255) / 256, 256, 0, stream>>>(
        merged, out, S, H);
}

// ────────────────────────────────────────────────────────────────
// cuBLASLt strided-batched FP8 GEMM with device descale → FP16.
// Layout family: fp8_gemm_descale_fp16 (decoder_fused.cu) with
// CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT / STRIDED_BATCH_OFFSET added.
// Self-contained cache — nothing shared with the existing entries.
// ────────────────────────────────────────────────────────────────
namespace {

cublasLtHandle_t g_moe_lt = nullptr;
void* g_moe_ws = nullptr;
size_t g_moe_ws_sz = 32 * 1024 * 1024;   // matches production (32 MB)

struct BatchedKey {
    int M, N, K, batch, layout;   // layout: 0 = KN/NN, 1 = NK/TN
    long long sA, sB, sC;
    bool operator==(const BatchedKey& o) const {
        return M == o.M && N == o.N && K == o.K && batch == o.batch &&
               layout == o.layout &&
               sA == o.sA && sB == o.sB && sC == o.sC;
    }
};
struct BatchedKeyHash {
    size_t operator()(const BatchedKey& k) const {
        size_t h = std::hash<int>()(k.M);
        auto mix = [&h](size_t v) {
            h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        mix(std::hash<int>()(k.N));
        mix(std::hash<int>()(k.K));
        mix(std::hash<int>()(k.batch));
        mix(std::hash<int>()(k.layout));
        mix(std::hash<long long>()(k.sA));
        mix(std::hash<long long>()(k.sB));
        mix(std::hash<long long>()(k.sC));
        return h;
    }
};
struct CachedBatched {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatmulAlgo_t algo;
};
std::unordered_map<BatchedKey, CachedBatched, BatchedKeyHash> g_moe_cache;

std::string shape_str(int M, int N, int K, int batch) {
    return "fp8_gemm_batched_descale_fp16 [" + std::to_string(M) + "," +
           std::to_string(N) + "," + std::to_string(K) + "]x" +
           std::to_string(batch);
}

void moe_check(cublasStatus_t status, int M, int N, int K, int batch,
               const char* op) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            shape_str(M, N, K, batch) + ": " + op +
            " failed with cuBLAS status " +
            std::to_string(static_cast<int>(status)));
    }
}

}  // namespace

// Shared implementation for both weight layouts:
//   layout=0 (KN/NN): B stored [K, N] row-major (the pipeline's
//     fp8_gemm_descale_fp16 convention). NOTE (measured): cuBLASLt
//     12.8 on sm_90 returns NOT_SUPPORTED for NN fp8 — the NN combo
//     is proven on Thor/CUDA-13 (pi05/vla4b production) only.
//   layout=1 (NK/TN): B stored [N, K] row-major (native HF [out, in]
//     orientation; the gmm_fp8 / fp8_nt_dev / torch._scaled_mm
//     combo — supported on BOTH sm_90/cu12.8 and sm_110/cu13).
static void fp8_gemm_batched_impl(const void* A_fp8, const void* B_fp8,
                                  void* C_fp16, int layout,
                                  int M, int N, int K, int batch,
                                  long long strideA, long long strideB,
                                  long long strideC,
                                  const float* act_descale,
                                  const float* w_descale,
                                  cudaStream_t stream) {
    if (!g_moe_lt) {
        moe_check(cublasLtCreate(&g_moe_lt), M, N, K, batch,
                  "cublasLtCreate");
        if (cudaMalloc(&g_moe_ws, g_moe_ws_sz) != cudaSuccess)
            throw std::runtime_error(
                shape_str(M, N, K, batch) + ": workspace cudaMalloc failed");
    }

    BatchedKey key{M, N, K, batch, layout, strideA, strideB, strideC};
    auto it = g_moe_cache.find(key);
    if (it == g_moe_cache.end()) {
        CachedBatched cg{};
        moe_check(cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F,
                                           CUDA_R_32F),
                  M, N, K, batch, "cublasLtMatmulDescCreate");
        cublasOperation_t opN = CUBLAS_OP_N, opT = CUBLAS_OP_T;
        moe_check(cublasLtMatmulDescSetAttribute(
                      cg.desc, CUBLASLT_MATMUL_DESC_TRANSA,
                      layout == 1 ? &opT : &opN, sizeof(opN)),
                  M, N, K, batch, "set TRANSA");
        moe_check(cublasLtMatmulDescSetAttribute(
                      cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)),
                  M, N, K, batch, "set TRANSB");
        if (layout == 1) {
            // first operand = B (weight [N,K] row = [K,N] col, ld=K; OP_T)
            moe_check(cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3,
                                                 K, N, K),
                      M, N, K, batch, "create A layout");
        } else {
            // first operand = B (weight [K,N] row = [N,K] col, ld=N; OP_N)
            moe_check(cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3,
                                                 N, K, N),
                      M, N, K, batch, "create A layout");
        }
        // second operand = A (act [M,K] row = [K,M] col, ld=K)
        moe_check(cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3,
                                             K, M, K),
                  M, N, K, batch, "create B layout");
        // out C ([M,N] row = [N,M] col, ld=N)
        moe_check(cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F,
                                             N, M, N),
                  M, N, K, batch, "create C layout");
        const int32_t bc = batch;
        const int64_t soB = strideB, soA = strideA, soC = strideC;
        auto set_batch = [&](cublasLtMatrixLayout_t lay, const int64_t* so,
                             const char* what) {
            moe_check(cublasLtMatrixLayoutSetAttribute(
                          lay, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc,
                          sizeof(bc)),
                      M, N, K, batch, what);
            moe_check(cublasLtMatrixLayoutSetAttribute(
                          lay, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                          so, sizeof(*so)),
                      M, N, K, batch, what);
        };
        set_batch(cg.Adesc, &soB, "set A batch");   // weight strides
        set_batch(cg.Bdesc, &soA, "set B batch");   // activation strides
        set_batch(cg.Cdesc, &soC, "set C batch");
        cublasLtMatmulPreference_t pref;
        moe_check(cublasLtMatmulPreferenceCreate(&pref), M, N, K, batch,
                  "preference create");
        moe_check(cublasLtMatmulPreferenceSetAttribute(
                      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                      &g_moe_ws_sz, sizeof(g_moe_ws_sz)),
                  M, N, K, batch, "set workspace preference");
        cublasLtMatmulHeuristicResult_t result;
        int ret = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
            g_moe_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc,
            pref, 1, &result, &ret);
        cublasLtMatmulPreferenceDestroy(pref);
        moe_check(hs, M, N, K, batch, "cublasLtMatmulAlgoGetHeuristic");
        if (ret == 0)
            throw std::runtime_error(
                shape_str(M, N, K, batch) +
                ": cuBLASLt returned no batched FP8 GEMM algorithm");
        cg.algo = result.algo;
        g_moe_cache[key] = cg;
        it = g_moe_cache.find(key);
    }
    auto& cg = it->second;
    moe_check(cublasLtMatmulDescSetAttribute(
                  cg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                  &w_descale, sizeof(w_descale)),
              M, N, K, batch, "set A scale pointer");
    moe_check(cublasLtMatmulDescSetAttribute(
                  cg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                  &act_descale, sizeof(act_descale)),
              M, N, K, batch, "set B scale pointer");
    float alpha = 1.0f, beta = 0.0f;
    moe_check(cublasLtMatmul(g_moe_lt, cg.desc, &alpha,
                             B_fp8, cg.Adesc, A_fp8, cg.Bdesc,
                             &beta, C_fp16, cg.Cdesc, C_fp16, cg.Cdesc,
                             &cg.algo, g_moe_ws, g_moe_ws_sz, stream),
              M, N, K, batch, "cublasLtMatmul");
}

void fp8_gemm_batched_descale_fp16(const void* A_fp8, const void* B_fp8,
                                   void* C_fp16,
                                   int M, int N, int K, int batch,
                                   long long strideA, long long strideB,
                                   long long strideC,
                                   const float* act_descale,
                                   const float* w_descale,
                                   cudaStream_t stream) {
    fp8_gemm_batched_impl(A_fp8, B_fp8, C_fp16, /*layout=*/0,
                          M, N, K, batch, strideA, strideB, strideC,
                          act_descale, w_descale, stream);
}

void fp8_gemm_batched_descale_nt_fp16(const void* A_fp8, const void* B_fp8,
                                      void* C_fp16,
                                      int M, int N, int K, int batch,
                                      long long strideA, long long strideB,
                                      long long strideC,
                                      const float* act_descale,
                                      const float* w_descale,
                                      cudaStream_t stream) {
    fp8_gemm_batched_impl(A_fp8, B_fp8, C_fp16, /*layout=*/1,
                          M, N, K, batch, strideA, strideB, strideC,
                          act_descale, w_descale, stream);
}

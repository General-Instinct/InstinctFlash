#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <unordered_map>

typedef __half bf16;  // FP16 throughout — no more BF16
typedef __nv_fp8_e4m3 fp8e4;

// Cached cublasLtMatmul FP8 GEMM
// C[M,N] = A[M,K] fp8 × B_NK[N,K] fp8 → bf16
// B must be stored as [N,K] row-major (transposed weight layout)
// This matches torch._scaled_mm(A, B.t()) convention

static cublasLtHandle_t g_lt = nullptr;
static void* g_lt_ws = nullptr;
static size_t g_lt_ws_sz = 32 * 1024 * 1024;

struct LtGemmKey { int M, N, K;
    bool operator==(const LtGemmKey& o) const { return M==o.M && N==o.N && K==o.K; }
};
struct LtGemmKeyHash { size_t operator()(const LtGemmKey& k) const { return k.M*1000003 + k.N*1009 + k.K; } };
struct CachedLtGemm {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatmulAlgo_t algo;
};
static std::unordered_map<LtGemmKey, CachedLtGemm, LtGemmKeyHash> g_lt_cache;

static void gmm_fp8(const fp8e4* A, const fp8e4* B_NK, bf16* C,
                     int M, int N, int K, float alpha, float beta, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }

    LtGemmKey key{M, N, K};
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        // A_desc: B_NK stored [N,K] row = [K,N] col, ld=K
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, K, N, K);
        // B_desc: A stored [M,K] row = [K,M] col, ld=K
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        // C_desc: [N,M] col = [M,N] row, ld=N
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N);

        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }

    auto& cg = it->second;
    cublasLtMatmul(g_lt, cg.desc, &alpha,
                    B_NK, cg.Adesc,  // first operand = B_NK (weight, transposed)
                    A, cg.Bdesc,     // second operand = A (activation)
                    &beta, C, cg.Cdesc, C, cg.Cdesc,
                    &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

// Variant: B stored as [K,N] row-major (standard weight layout, NOT transposed)
// C[M,N] = A[M,K] fp8 × B_KN[K,N] fp8 → bf16
static void gmm_fp8_kn(const fp8e4* A, const fp8e4* B_KN, bf16* C,
                        int M, int N, int K, float alpha, float beta, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }

    // Key includes a flag to distinguish KN from NK layout
    LtGemmKey key{M, N + 1000000, K};  // offset N to avoid collision with NK variant
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        // B_KN stored [K,N] row = [N,K] col. With OP_N → reads as [N,K]. Need [K,N] → use OP_T
        // A stored [M,K] row = [K,M] col. With OP_N → reads as [K,M].
        // D = OP_T([N,K]) × OP_N([K,M]) = [K,N] × [K,M] → dims wrong!
        // Need: D[N,M] = first_op × second_op
        // first = B_KN with OP_N: reads [N,K] col-major → shape (N,K)
        // second = A with OP_N: reads [K,M] col-major → shape (K,M) 
        // matmul: (N,K) × (K,M) = (N,M) ✓
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        // A_desc: B_KN stored [K,N] row = [N,K] col, ld=N
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        // B_desc: A stored [M,K] row = [K,M] col, ld=K
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        // C_desc: [N,M] col = [M,N] row, ld=N
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N);

        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        if (s != CUBLAS_STATUS_SUCCESS || ret == 0) {
            printf("[gmm_fp8_kn] Heuristic FAILED for [%d,%d,%d] status=%d\n", M, N, K, s);
        }
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }

    auto& cg = it->second;
    cublasLtMatmul(g_lt, cg.desc, &alpha,
                    B_KN, cg.Adesc,  // first operand = B_KN
                    A, cg.Bdesc,     // second operand = A
                    &beta, C, cg.Cdesc, C, cg.Cdesc,
                    &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

// cublasLt FP8 GEMM with bias epilogue: D = alpha * A@B + bias
static void gmm_fp8_kn_bias(const fp8e4* A, const fp8e4* B_KN, bf16* C, const bf16* bias,
                              int M, int N, int K, float alpha_in, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 2000000, K};  // unique key for bias variant
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    float beta = 0.0f;
    cublasLtMatmul(g_lt, cg.desc, &alpha_in, B_KN, cg.Adesc, A, cg.Bdesc, &beta, C, cg.Cdesc, C, cg.Cdesc,
                    &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

// cublasLt FP8 GEMM with bias epilogue → FP16 output: D = fp16(alpha * A@B + bias)
// bias must be FP16!
static void gmm_fp8_kn_bias_fp16out(const fp8e4* A, const fp8e4* B_KN, half* C, const half* bias,
                                      int M, int N, int K, float alpha_in, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 6000000, K};
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
        cudaDataType_t btype = CUDA_R_16F;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &btype, sizeof(btype));
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        if (ret == 0) printf("[gmm_fp8_kn_bias_fp16out] Heuristic FAILED for [%d,%d,%d]\n", M, N, K);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    float beta = 0.0f;
    cublasLtMatmul(g_lt, cg.desc, &alpha_in, B_KN, cg.Adesc, A, cg.Bdesc, &beta, C, cg.Cdesc, C, cg.Cdesc,
                    &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

// cublasLt FP8 GEMM with bias epilogue → FP8 output: D = fp8(alpha * A@B + bias)
// bias must be FP16!  Output is FP8 e4m3.
static void gmm_fp8_kn_bias_fp8out(const fp8e4* A, const fp8e4* B_KN, fp8e4* C, const half* bias,
                                     int M, int N, int K, float alpha_in, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 5000000, K};  // unique key
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
        cudaDataType_t btype = CUDA_R_16F;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &btype, sizeof(btype));
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_8F_E4M3, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        if (ret == 0) printf("[gmm_fp8_kn_bias_fp8out] Heuristic FAILED for [%d,%d,%d]\n", M, N, K);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    float beta = 0.0f;
    cublasLtMatmul(g_lt, cg.desc, &alpha_in, B_KN, cg.Adesc, A, cg.Bdesc, &beta, C, cg.Cdesc, C, cg.Cdesc,
                    &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

// cublasLt FP8 GEMM with GELU+bias → FP8 output: D = fp8(GELU(alpha * A@B + bias))
static void gmm_fp8_kn_gelu_bias_fp8out(const fp8e4* A, const fp8e4* B_KN, fp8e4* C, const bf16* bias,
                                          int M, int N, int K, float alpha_in, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 7000000, K};
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU_BIAS;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
        cudaDataType_t btype = CUDA_R_16F;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &btype, sizeof(btype));
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_8F_E4M3, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        if (ret == 0) printf("[gmm_fp8_kn_gelu_bias_fp8out] Heuristic FAILED for [%d,%d,%d]\n", M, N, K);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    float beta = 0.0f;
    cublasLtMatmul(g_lt, cg.desc, &alpha_in, B_KN, cg.Adesc, A, cg.Bdesc, &beta, C, cg.Cdesc, C, cg.Cdesc,
                    &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

// cublasLt FP8 GEMM with bias+GELU epilogue: D = GELU(alpha * A@B + bias)
static void gmm_fp8_kn_gelu_bias(const fp8e4* A, const fp8e4* B_KN, bf16* C, const bf16* bias,
                                   int M, int N, int K, float alpha_in, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 3000000, K};
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU_BIAS;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        if (ret == 0) printf("[gmm_fp8_kn_gelu_bias] Heuristic FAILED for [%d,%d,%d]\n", M, N, K);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    float beta = 0.0f;
    cublasLtMatmul(g_lt, cg.desc, &alpha_in, B_KN, cg.Adesc, A, cg.Bdesc, &beta, C, cg.Cdesc, C, cg.Cdesc,
                    &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

// cublasLt FP8 GEMM + bias + residual: D = A@B + bias + D_initial
// Combines GEMM + bias epilogue + residual add in one cublasLt call
static void gmm_fp8_kn_bias_res(const fp8e4* A, const fp8e4* B_KN, bf16* D, const bf16* bias,
                                  int M, int N, int K, float alpha_in, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 4000000, K};  // unique key
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    float beta = 1.0f;  // beta=1 for residual!
    cublasLtMatmul(g_lt, cg.desc, &alpha_in, B_KN, cg.Adesc, A, cg.Bdesc, &beta, D, cg.Cdesc, D, cg.Cdesc,
                    &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

// cublasLt FP8 GEMM + residual only (no bias): D = A@B + D_initial
static void gmm_fp8_kn_res(const fp8e4* A, const fp8e4* B_KN, bf16* D,
                             int M, int N, int K, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 5000000, K};
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        // No epilogue — just plain GEMM with beta=1
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    float alpha = 1.0f, beta = 1.0f;
    cublasLtMatmul(g_lt, cg.desc, &alpha, B_KN, cg.Adesc, A, cg.Bdesc, &beta, D, cg.Cdesc, D, cg.Cdesc,
                    &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

// ============================================================
// cuBLASLt FP8 GEMM with per-tensor descale factors
// D = alpha * (descale_A * A_fp8) @ (descale_B * B_fp8) + beta * C
// act_descale, w_descale: DEVICE float pointers (amax/448 or just scale)
// B is [K,N] row-major (same as gmm_fp8_kn)
// ============================================================
static void gmm_fp8_kn_descale(const fp8e4* A, const fp8e4* B_KN, bf16* C,
                                 int M, int N, int K,
                                 const float* act_descale, const float* w_descale,
                                 cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 6000000, K};  // unique key for descale variant
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        // Layout: same as gmm_fp8_kn
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        if (ret == 0) printf("[gmm_fp8_kn_descale] Heuristic FAILED for [%d,%d,%d]\n", M, N, K);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    // Set descale pointers (device float*) before each call
    // In cuBLASLt: A_desc corresponds to first operand = B_KN (weight), B_desc = A (activation)
    // So: A_SCALE → weight descale, B_SCALE → activation descale
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_descale, sizeof(w_descale));
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_descale, sizeof(act_descale));
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(g_lt, cg.desc, &alpha, B_KN, cg.Adesc, A, cg.Bdesc, &beta, C, cg.Cdesc, C, cg.Cdesc,
                    &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

// Same but with N,K weight layout (for CUTLASS-style weights [N,K])
static void gmm_fp8_nk_descale(const fp8e4* A, const fp8e4* B_NK, bf16* C,
                                 int M, int N, int K,
                                 const float* act_descale, const float* w_descale,
                                 cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 7000000, K};
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, K, N, K);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        if (ret == 0) printf("[gmm_fp8_nk_descale] Heuristic FAILED for [%d,%d,%d]\n", M, N, K);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_descale, sizeof(w_descale));
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_descale, sizeof(act_descale));
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(g_lt, cg.desc, &alpha, B_NK, cg.Adesc, A, cg.Bdesc, &beta, C, cg.Cdesc, C, cg.Cdesc,
                    &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

// ============================================================
// Autotune nk_descale: test all algos, cache the best
// Call once per shape at init time. Subsequent gmm_fp8_nk_descale
// calls for that shape will use the autotuned algo.
// ============================================================
static void autotune_nk_descale(int M, int N, int K, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 7000000, K};

    CachedLtGemm cg;
    cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));

    cublasLtMatmulHeuristicResult_t results[64];
    int num = 0;
    cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 64, results, &num);
    if (num == 0) { printf("[autotune_nk] FAILED [%d,%d,%d]\n", M, N, K); return; }

    fp8e4* d_a; fp8e4* d_b; bf16* d_c;
    cudaMalloc(&d_a, (size_t)M*K); cudaMalloc(&d_b, (size_t)K*N); cudaMalloc(&d_c, (size_t)M*N*2);
    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    float best_ms = 1e9f; int best_idx = 0;
    for (int a = 0; a < num; a++) {
        for (int w = 0; w < 3; w++)
            cublasLtMatmul(g_lt, cg.desc, &alpha, d_b, cg.Adesc, d_a, cg.Bdesc, &beta, d_c, cg.Cdesc, d_c, cg.Cdesc,
                           &results[a].algo, g_lt_ws, g_lt_ws_sz, st);
        cudaStreamSynchronize(st);
        cudaEventRecord(t0, st);
        for (int i = 0; i < 20; i++)
            cublasLtMatmul(g_lt, cg.desc, &alpha, d_b, cg.Adesc, d_a, cg.Bdesc, &beta, d_c, cg.Cdesc, d_c, cg.Cdesc,
                           &results[a].algo, g_lt_ws, g_lt_ws_sz, st);
        cudaEventRecord(t1, st); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 20;
        if (ms < best_ms) { best_ms = ms; best_idx = a; }
    }
    cg.algo = results[best_idx].algo;
    g_lt_cache[key] = cg;
    printf("[autotune_nk] [%d,%d,%d]: best=%d/%d (%.1fus)\n", M, N, K, best_idx, num, best_ms*1000);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cublasLtMatmulPreferenceDestroy(pref);
}

// Autotune kn_descale (AE layout)
static void autotune_kn_descale(int M, int N, int K, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 6000000, K};

    CachedLtGemm cg;
    cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opN_op = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN_op, sizeof(opN_op));
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN_op, sizeof(opN_op));
    cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
    cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));

    cublasLtMatmulHeuristicResult_t results[64];
    int num = 0;
    cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 64, results, &num);
    if (num == 0) { printf("[autotune_kn] FAILED [%d,%d,%d]\n", M, N, K); return; }

    fp8e4* d_a; fp8e4* d_b; bf16* d_c;
    cudaMalloc(&d_a, (size_t)M*K); cudaMalloc(&d_b, (size_t)K*N); cudaMalloc(&d_c, (size_t)M*N*2);
    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    float best_ms = 1e9f; int best_idx = 0;
    for (int a = 0; a < num; a++) {
        for (int w = 0; w < 3; w++)
            cublasLtMatmul(g_lt, cg.desc, &alpha, d_b, cg.Adesc, d_a, cg.Bdesc, &beta, d_c, cg.Cdesc, d_c, cg.Cdesc,
                           &results[a].algo, g_lt_ws, g_lt_ws_sz, st);
        cudaStreamSynchronize(st);
        cudaEventRecord(t0, st);
        for (int i = 0; i < 20; i++)
            cublasLtMatmul(g_lt, cg.desc, &alpha, d_b, cg.Adesc, d_a, cg.Bdesc, &beta, d_c, cg.Cdesc, d_c, cg.Cdesc,
                           &results[a].algo, g_lt_ws, g_lt_ws_sz, st);
        cudaEventRecord(t1, st); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 20;
        if (ms < best_ms) { best_ms = ms; best_idx = a; }
    }
    cg.algo = results[best_idx].algo;
    g_lt_cache[key] = cg;
    printf("[autotune_kn] [%d,%d,%d]: best=%d/%d (%.1fus)\n", M, N, K, best_idx, num, best_ms*1000);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cublasLtMatmulPreferenceDestroy(pref);
}

// Autotune all shapes used in the pipeline
extern "C" void autotune_pipeline(cudaStream_t st) {
    printf("[autotune_pipeline] Starting...\n");
    // Encoder NK shapes (QKV, O, Gate+Up — Down uses CUTLASS)
    autotune_nk_descale(968, 2560, 2048, st);    // QKV
    autotune_nk_descale(968, 2048, 2048, st);    // O
    autotune_nk_descale(968, 32768, 2048, st);   // Gate+Up
    // AE KN shapes
    autotune_kn_descale(10, 2560, 1024, st);     // QKV
    autotune_kn_descale(10, 1024, 2048, st);     // O
    autotune_kn_descale(10, 8192, 1024, st);     // Gate+Up
    autotune_kn_descale(10, 1024, 4096, st);     // Down
    printf("[autotune_pipeline] Done!\n");
}

// ============================================================
// GEMM Autotune: benchmark top algos, cache the fastest
// ============================================================
static void autotune_gemm_fp8_kn(int M, int N, int K, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    
    // Create desc for this shape
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayout_t Ad, Bd, Cd;
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, N, K, N);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16F, N, M, N);
    
    // Get top 16 algos
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
    
    cublasLtMatmulHeuristicResult_t results[16];
    int num_results = 0;
    cublasLtMatmulAlgoGetHeuristic(g_lt, desc, Ad, Bd, Cd, Cd, pref, 16, results, &num_results);
    
    if (num_results == 0) {
        printf("[autotune] No algos for [%d,%d,%d]\n", M, N, K);
        cublasLtMatmulDescDestroy(desc);
        cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
        cublasLtMatmulPreferenceDestroy(pref);
        return;
    }
    
    // Allocate temp buffers
    fp8e4* d_a; fp8e4* d_b; bf16* d_c;
    cudaMalloc(&d_a, (size_t)M*K); cudaMalloc(&d_b, (size_t)K*N); cudaMalloc(&d_c, (size_t)M*N*2);
    float alpha = 1.0f, beta = 0.0f;
    
    // Benchmark each algo
    float best_ms = 1e9f;
    int best_idx = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    for (int a = 0; a < num_results; a++) {
        // Warmup
        for (int w = 0; w < 3; w++)
            cublasLtMatmul(g_lt, desc, &alpha, d_b, Ad, d_a, Bd, &beta, d_c, Cd, d_c, Cd,
                           &results[a].algo, g_lt_ws, g_lt_ws_sz, st);
        cudaStreamSynchronize(st);
        
        // Benchmark
        cudaEventRecord(start, st);
        for (int i = 0; i < 20; i++)
            cublasLtMatmul(g_lt, desc, &alpha, d_b, Ad, d_a, Bd, &beta, d_c, Cd, d_c, Cd,
                           &results[a].algo, g_lt_ws, g_lt_ws_sz, st);
        cudaEventRecord(stop, st);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        ms /= 20.0f;
        
        if (ms < best_ms) { best_ms = ms; best_idx = a; }
    }
    
    // Cache the best algo
    LtGemmKey key{M, N + 1000000, K};
    CachedLtGemm cg;
    cg.desc = desc; cg.Adesc = Ad; cg.Bdesc = Bd; cg.Cdesc = Cd;
    cg.algo = results[best_idx].algo;
    g_lt_cache[key] = cg;
    
    printf("[autotune] [%d,%d,%d]: best=%d/%d (%.1fus, %.1fx vs heuristic)\n",
           M, N, K, best_idx, num_results, best_ms*1000, 
           results[0].workspaceSize == results[best_idx].workspaceSize ? 1.0f : best_ms*1000/1.0f);
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cublasLtMatmulPreferenceDestroy(pref);
}

// Autotune all unique shapes used in the model
extern "C" void autotune_all_gemms(cudaStream_t st) {
    // SigLIP shapes
    autotune_gemm_fp8_kn(768, 3456, 1152, st);   // QKV
    autotune_gemm_fp8_kn(768, 1152, 1152, st);   // O proj
    autotune_gemm_fp8_kn(768, 4304, 1152, st);   // Up
    autotune_gemm_fp8_kn(768, 1152, 4304, st);   // Down
    // AE shapes
    autotune_gemm_fp8_kn(10, 2560, 2048, st);    // QKV
    autotune_gemm_fp8_kn(10, 2048, 2048, st);    // O proj
    autotune_gemm_fp8_kn(10, 8192, 2048, st);    // merged gate+up
    autotune_gemm_fp8_kn(10, 2048, 4096, st);    // Down
    printf("[autotune] Done: 8 shapes tuned\n");
}

// Autotune bias variant
static void autotune_bias(int M, int N, int K, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
    cublasLtMatrixLayout_t Ad, Bd, Cd;
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, N, K, N);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16F, N, M, N);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
    cublasLtMatmulHeuristicResult_t results[16]; int num = 0;
    cublasLtMatmulAlgoGetHeuristic(g_lt, desc, Ad, Bd, Cd, Cd, pref, 16, results, &num);
    if (num == 0) { cublasLtMatmulDescDestroy(desc); cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd); cublasLtMatmulPreferenceDestroy(pref); return; }
    fp8e4* d_a; fp8e4* d_b; bf16* d_c; bf16* d_bias;
    cudaMalloc(&d_a, (size_t)M*K); cudaMalloc(&d_b, (size_t)K*N); cudaMalloc(&d_c, (size_t)M*N*2); cudaMalloc(&d_bias, (size_t)N*2);
    float alpha=1,beta=0; float best_ms=1e9; int best_idx=0;
    cudaEvent_t start,stop; cudaEventCreate(&start);cudaEventCreate(&stop);
    for (int a=0;a<num;a++){
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias));
        for(int w=0;w<3;w++) cublasLtMatmul(g_lt,desc,&alpha,d_b,Ad,d_a,Bd,&beta,d_c,Cd,d_c,Cd,&results[a].algo,g_lt_ws,g_lt_ws_sz,st);
        cudaStreamSynchronize(st);
        cudaEventRecord(start,st);
        for(int i=0;i<20;i++) cublasLtMatmul(g_lt,desc,&alpha,d_b,Ad,d_a,Bd,&beta,d_c,Cd,d_c,Cd,&results[a].algo,g_lt_ws,g_lt_ws_sz,st);
        cudaEventRecord(stop,st);cudaEventSynchronize(stop);
        float ms;cudaEventElapsedTime(&ms,start,stop);ms/=20;
        if(ms<best_ms){best_ms=ms;best_idx=a;}
    }
    LtGemmKey key{M, N+2000000, K};
    CachedLtGemm cg; cg.desc=desc; cg.Adesc=Ad; cg.Bdesc=Bd; cg.Cdesc=Cd; cg.algo=results[best_idx].algo;
    g_lt_cache[key] = cg;
    printf("[autotune-bias] [%d,%d,%d]: best=%d/%d (%.1fus)\n",M,N,K,best_idx,num,best_ms*1000);
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);cudaFree(d_bias);
    cudaEventDestroy(start);cudaEventDestroy(stop);cublasLtMatmulPreferenceDestroy(pref);
}

extern "C" void autotune_all_v2(cudaStream_t st) {
    autotune_all_gemms(st);
    // SigLIP bias variants
    autotune_bias(768, 1152, 1152, st);  // O proj bias
    autotune_bias(768, 1152, 4304, st);  // Down bias
    printf("[autotune-v2] All shapes tuned\n");
}

// ═══════════════════════════════════════════════════════════════
// FP32 Output Variants — for models with activations > FP16 range
// Same FP8 inputs, FP32 accumulation, but OUTPUT to FP32 (not FP16)
// Avoids FP16 output overflow when alpha * accumulated > 65504
// ═══════════════════════════════════════════════════════════════

// B stored as [N,K] row-major → FP32 output
static void gmm_fp8_f32out(const fp8e4* A, const fp8e4* B_NK, float* C,
                            int M, int N, int K, float alpha, float beta, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 9000000, K};  // unique key for f32out variant
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, K, N, K);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_32F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        if (ret == 0) printf("[gmm_fp8_f32out] Heuristic FAILED for [%d,%d,%d]\n", M, N, K);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    cublasLtMatmul(g_lt, cg.desc, &alpha, B_NK, cg.Adesc, A, cg.Bdesc,
                    &beta, C, cg.Cdesc, C, cg.Cdesc, &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

// B stored as [K,N] row-major → FP32 output
static void gmm_fp8_kn_f32out(const fp8e4* A, const fp8e4* B_KN, float* C,
                               int M, int N, int K, float alpha, float beta, cudaStream_t st) {
    if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_lt_ws, g_lt_ws_sz); }
    LtGemmKey key{M, N + 9100000, K};  // unique key
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_32F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_sz, sizeof(g_lt_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc, pref, 1, &result, &ret);
        if (ret == 0) printf("[gmm_fp8_kn_f32out] Heuristic FAILED for [%d,%d,%d]\n", M, N, K);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    cublasLtMatmul(g_lt, cg.desc, &alpha, B_KN, cg.Adesc, A, cg.Bdesc,
                    &beta, C, cg.Cdesc, C, cg.Cdesc, &cg.algo, g_lt_ws, g_lt_ws_sz, st);
}

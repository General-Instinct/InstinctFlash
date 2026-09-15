#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

// P009-A5: pinned, no-split-K BF16 cuBLASLt tactics for two LingBot production shapes.
extern "C" int instinctflash_sm120_wan_gemm_abi_version() { return 1; }
extern "C" uint64_t wan_gemm_cublaslt_version() { return uint64_t(cublasLtGetVersion()); }
namespace {
struct Plan {
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{}, b{}, c{}, d{};
  cublasLtMatmulAlgo_t algo{};
  uintptr_t weight{}, bias{};
  ~Plan() {
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (d) cublasLtMatrixLayoutDestroy(d);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};
struct Context {
  cublasLtHandle_t handle{};
  std::vector<std::unique_ptr<Plan>> plans;
  int last_status{};
  ~Context() {
    plans.clear();
    if (handle) cublasLtDestroy(handle);
  }
};
Context* get(uintptr_t value) { return reinterpret_cast<Context*>(value); }
int fail(Context* ctx, cublasStatus_t status) {
  if (ctx) ctx->last_status = int(status);
  return -1;
}
}

extern "C" uintptr_t wan_gemm_context_create() {
  auto ctx = std::make_unique<Context>();
  cublasStatus_t status = cublasLtCreate(&ctx->handle);
  if (status != CUBLAS_STATUS_SUCCESS) return 0;
  return reinterpret_cast<uintptr_t>(ctx.release());
}

extern "C" int wan_gemm_plan_register(
    uintptr_t context, int m, int n, int k, uintptr_t weight, uintptr_t bias,
    int algo_id, uint32_t tile, uint32_t stages, uint32_t swizzle, uint32_t custom) {
  Context* ctx = get(context);
  if (!ctx || m <= 0 || n <= 0 || k <= 0 || !weight || !bias) return -1;
  auto plan = std::make_unique<Plan>();
  plan->weight = weight;
  plan->bias = bias;
  cublasStatus_t status;
  if ((status = cublasLtMatmulDescCreate(
           &plan->op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) != CUBLAS_STATUS_SUCCESS)
    return fail(ctx, status);
  cublasOperation_t trans_a = CUBLAS_OP_T, trans_b = CUBLAS_OP_N;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  cudaDataType_t bias_type = CUDA_R_16BF;
  void* bias_pointer = reinterpret_cast<void*>(bias);
  if ((status = cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &trans_a, sizeof(trans_a))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &trans_b, sizeof(trans_b))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &bias_pointer, sizeof(bias_pointer))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                &bias_type, sizeof(bias_type))) != CUBLAS_STATUS_SUCCESS)
    return fail(ctx, status);

  // Torch row-major buffers viewed as column-major matrices:
  // D[N,M] = transpose(W[K,N]) * X[K,M], whose storage is Torch's Y[M,N].
  if ((status = cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_16BF, k, n, k)) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_16BF, k, m, k)) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatrixLayoutCreate(&plan->c, CUDA_R_16BF, n, m, n)) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatrixLayoutCreate(&plan->d, CUDA_R_16BF, n, m, n)) != CUBLAS_STATUS_SUCCESS)
    return fail(ctx, status);
  if ((status = cublasLtMatmulAlgoInit(
           ctx->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
           CUDA_R_16BF, CUDA_R_16BF, algo_id, &plan->algo)) != CUBLAS_STATUS_SUCCESS)
    return fail(ctx, status);
  int split_k = 1;
  uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
  if ((status = cublasLtMatmulAlgoConfigSetAttribute(&plan->algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                                      &tile, sizeof(tile))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulAlgoConfigSetAttribute(&plan->algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                                      &stages, sizeof(stages))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulAlgoConfigSetAttribute(&plan->algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                                                      &swizzle, sizeof(swizzle))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulAlgoConfigSetAttribute(&plan->algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                                      &custom, sizeof(custom))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulAlgoConfigSetAttribute(&plan->algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                      &split_k, sizeof(split_k))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulAlgoConfigSetAttribute(&plan->algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                      &reduction, sizeof(reduction))) != CUBLAS_STATUS_SUCCESS)
    return fail(ctx, status);
  cublasLtMatmulHeuristicResult_t checked{};
  status = cublasLtMatmulAlgoCheck(
      ctx->handle, plan->op, plan->a, plan->b, plan->c, plan->d, &plan->algo, &checked);
  if (status != CUBLAS_STATUS_SUCCESS || checked.state != CUBLAS_STATUS_SUCCESS ||
      checked.workspaceSize != 0)
    return fail(ctx, status != CUBLAS_STATUS_SUCCESS ? status : checked.state);
  int index = int(ctx->plans.size());
  ctx->plans.push_back(std::move(plan));
  ctx->last_status = 0;
  return index;
}

extern "C" int wan_gemm_bf16(
    uintptr_t context, int index, uintptr_t x, uintptr_t weight, uintptr_t bias,
    uintptr_t output, uintptr_t stream) {
  Context* ctx = get(context);
  if (!ctx || index < 0 || index >= int(ctx->plans.size()) || !x || !output) return -1;
  Plan& plan = *ctx->plans[index];
  if (weight != plan.weight || bias != plan.bias) return -2;
  float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t status = cublasLtMatmul(
      ctx->handle, plan.op, &alpha,
      reinterpret_cast<void*>(weight), plan.a,
      reinterpret_cast<void*>(x), plan.b,
      &beta, reinterpret_cast<void*>(output), plan.c,
      reinterpret_cast<void*>(output), plan.d,
      &plan.algo, nullptr, 0, reinterpret_cast<cudaStream_t>(stream));
  ctx->last_status = int(status);
  return int(status);
}

extern "C" int wan_gemm_context_last_status(uintptr_t context) {
  Context* ctx = get(context);
  return ctx ? ctx->last_status : -1;
}
extern "C" int wan_gemm_context_plan_count(uintptr_t context) {
  Context* ctx = get(context);
  return ctx ? int(ctx->plans.size()) : -1;
}
extern "C" void wan_gemm_context_destroy(uintptr_t context) { delete get(context); }

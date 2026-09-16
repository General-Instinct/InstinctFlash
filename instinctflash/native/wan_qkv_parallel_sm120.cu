#include <cublasLt.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

// P009-A7: three pinned Q/K/V GEMMs on private streams, joined to the caller stream.
extern "C" int instinctflash_sm120_wan_qkv_parallel_abi_version() { return 1; }
extern "C" uint64_t wan_qkv_parallel_cublaslt_version() {
  return uint64_t(cublasLtGetVersion());
}

namespace {
struct MatmulPlan {
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{}, b{}, c{}, d{};
  cublasLtMatmulAlgo_t algo{};
  uintptr_t weight{}, bias{};
  ~MatmulPlan() {
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (d) cublasLtMatrixLayoutDestroy(d);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};
struct TripletPlan { std::array<std::unique_ptr<MatmulPlan>, 3> lanes; };
struct Context {
  std::array<cublasLtHandle_t, 3> handles{};
  std::array<cudaStream_t, 3> streams{};
  cudaEvent_t ready{};
  std::array<cudaEvent_t, 3> done{};
  std::vector<std::unique_ptr<TripletPlan>> triplets;
  int last_status{};
  ~Context() {
    for (auto stream : streams) if (stream) cudaStreamSynchronize(stream);
    triplets.clear();
    if (ready) cudaEventDestroy(ready);
    for (auto event : done) if (event) cudaEventDestroy(event);
    for (auto stream : streams) if (stream) cudaStreamDestroy(stream);
    for (auto handle : handles) if (handle) cublasLtDestroy(handle);
  }
};
Context* get(uintptr_t value) { return reinterpret_cast<Context*>(value); }
int fail(Context* context, int status) {
  if (context) context->last_status = status;
  return -1;
}

std::unique_ptr<MatmulPlan> make_plan(
    Context* context, int lane, int m, int n, int k, uintptr_t weight,
    uintptr_t bias, int algo_id, uint32_t tile, uint32_t stages,
    uint32_t swizzle, uint32_t custom) {
  auto plan = std::make_unique<MatmulPlan>();
  plan->weight = weight;
  plan->bias = bias;
  cublasStatus_t status;
  if ((status = cublasLtMatmulDescCreate(
           &plan->op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) != CUBLAS_STATUS_SUCCESS)
    return nullptr;
  cublasOperation_t trans_a = CUBLAS_OP_T, trans_b = CUBLAS_OP_N;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  cudaDataType_t bias_type = CUDA_R_16BF;
  void* bias_pointer = reinterpret_cast<void*>(bias);
  if ((status = cublasLtMatmulDescSetAttribute(
           plan->op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulDescSetAttribute(
           plan->op, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulDescSetAttribute(
           plan->op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulDescSetAttribute(
           plan->op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulDescSetAttribute(
           plan->op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type))) != CUBLAS_STATUS_SUCCESS)
    return nullptr;
  if ((status = cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_16BF, k, n, k)) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_16BF, k, m, k)) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatrixLayoutCreate(&plan->c, CUDA_R_16BF, n, m, n)) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatrixLayoutCreate(&plan->d, CUDA_R_16BF, n, m, n)) != CUBLAS_STATUS_SUCCESS)
    return nullptr;
  if ((status = cublasLtMatmulAlgoInit(
           context->handles[lane], CUBLAS_COMPUTE_32F, CUDA_R_32F,
           CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
           algo_id, &plan->algo)) != CUBLAS_STATUS_SUCCESS)
    return nullptr;
  int split_k = 1;
  uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
  if ((status = cublasLtMatmulAlgoConfigSetAttribute(&plan->algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulAlgoConfigSetAttribute(&plan->algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulAlgoConfigSetAttribute(&plan->algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulAlgoConfigSetAttribute(&plan->algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof(custom))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulAlgoConfigSetAttribute(&plan->algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k, sizeof(split_k))) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasLtMatmulAlgoConfigSetAttribute(&plan->algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction))) != CUBLAS_STATUS_SUCCESS)
    return nullptr;
  cublasLtMatmulHeuristicResult_t checked{};
  status = cublasLtMatmulAlgoCheck(
      context->handles[lane], plan->op, plan->a, plan->b, plan->c, plan->d,
      &plan->algo, &checked);
  if (status != CUBLAS_STATUS_SUCCESS || checked.state != CUBLAS_STATUS_SUCCESS ||
      checked.workspaceSize != 0)
    return nullptr;
  return plan;
}
}  // namespace

extern "C" uintptr_t wan_qkv_parallel_context_create() {
  auto context = std::make_unique<Context>();
  for (int lane = 0; lane < 3; ++lane) {
    if (cublasLtCreate(&context->handles[lane]) != CUBLAS_STATUS_SUCCESS ||
        cudaStreamCreateWithFlags(&context->streams[lane], cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&context->done[lane], cudaEventDisableTiming) != cudaSuccess)
      return 0;
  }
  if (cudaEventCreateWithFlags(&context->ready, cudaEventDisableTiming) != cudaSuccess)
    return 0;
  return reinterpret_cast<uintptr_t>(context.release());
}

extern "C" int wan_qkv_parallel_plan_register(
    uintptr_t value, int m, int n, int k,
    uintptr_t wq, uintptr_t bq, uintptr_t wk, uintptr_t bk,
    uintptr_t wv, uintptr_t bv, int algo_id, uint32_t tile,
    uint32_t stages, uint32_t swizzle, uint32_t custom) {
  Context* context = get(value);
  std::array<uintptr_t, 3> weights{wq, wk, wv}, biases{bq, bk, bv};
  if (!context || m <= 0 || n <= 0 || k <= 0) return -1;
  auto triplet = std::make_unique<TripletPlan>();
  for (int lane = 0; lane < 3; ++lane) {
    if (!weights[lane] || !biases[lane]) return -1;
    triplet->lanes[lane] = make_plan(
        context, lane, m, n, k, weights[lane], biases[lane], algo_id,
        tile, stages, swizzle, custom);
    if (!triplet->lanes[lane]) return fail(context, -2);
  }
  int index = int(context->triplets.size());
  context->triplets.push_back(std::move(triplet));
  context->last_status = 0;
  return index;
}

extern "C" int wan_qkv_parallel_bf16(
    uintptr_t value, int index, uintptr_t x,
    uintptr_t wq, uintptr_t bq, uintptr_t wk, uintptr_t bk,
    uintptr_t wv, uintptr_t bv, uintptr_t oq, uintptr_t ok,
    uintptr_t ov, uintptr_t caller_value) {
  Context* context = get(value);
  if (!context || index < 0 || index >= int(context->triplets.size()) ||
      !x || !oq || !ok || !ov)
    return -1;
  TripletPlan& triplet = *context->triplets[index];
  std::array<uintptr_t, 3> weights{wq, wk, wv}, biases{bq, bk, bv};
  std::array<uintptr_t, 3> outputs{oq, ok, ov};
  cudaStream_t caller = reinterpret_cast<cudaStream_t>(caller_value);
  cudaError_t cuda_status = cudaEventRecord(context->ready, caller);
  if (cuda_status != cudaSuccess) return fail(context, int(cuda_status));
  float alpha = 1.0f, beta = 0.0f;
  for (int lane = 0; lane < 3; ++lane) {
    MatmulPlan& plan = *triplet.lanes[lane];
    if (weights[lane] != plan.weight || biases[lane] != plan.bias) return -2;
    cuda_status = cudaStreamWaitEvent(context->streams[lane], context->ready, 0);
    if (cuda_status != cudaSuccess) return fail(context, int(cuda_status));
    cublasStatus_t status = cublasLtMatmul(
        context->handles[lane], plan.op, &alpha,
        reinterpret_cast<void*>(weights[lane]), plan.a,
        reinterpret_cast<void*>(x), plan.b, &beta,
        reinterpret_cast<void*>(outputs[lane]), plan.c,
        reinterpret_cast<void*>(outputs[lane]), plan.d, &plan.algo,
        nullptr, 0, context->streams[lane]);
    if (status != CUBLAS_STATUS_SUCCESS) return fail(context, int(status));
    cuda_status = cudaEventRecord(context->done[lane], context->streams[lane]);
    if (cuda_status != cudaSuccess) return fail(context, int(cuda_status));
  }
  for (auto event : context->done) {
    cuda_status = cudaStreamWaitEvent(caller, event, 0);
    if (cuda_status != cudaSuccess) return fail(context, int(cuda_status));
  }
  context->last_status = 0;
  return 0;
}

extern "C" int wan_qkv_parallel_context_last_status(uintptr_t value) {
  Context* context = get(value);
  return context ? context->last_status : -1;
}
extern "C" void wan_qkv_parallel_context_destroy(uintptr_t value) { delete get(value); }

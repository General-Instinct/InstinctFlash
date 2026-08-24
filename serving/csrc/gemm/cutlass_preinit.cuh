#pragma once
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cute/tensor.hpp>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

using namespace cute;

using cutlass_fp8 = cutlass::float_e4m3_t;
using cutlass_fp16 = cutlass::half_t;  // FP16 throughout

// Small tile GEMM for preinit (128x128x32, Cluster 1x1x1)
using SmTile = Shape<_128, _128, _32>;
using SmCluster = Shape<_1, _1, _1>;

using SmFusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp16, float>;
using SmEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    SmTile, SmCluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, SmFusion>::CollectiveOp;

using SmMain = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp8, cutlass::layout::RowMajor, 16,
    cutlass_fp8, cutlass::layout::ColumnMajor, 16,
    float, SmTile, SmCluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename SmEpi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using SmGemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, SmMain, SmEpi>>;

using SmParams = typename SmGemm::GemmKernel::Params;

// Pre-initialized GEMM runner: build Params once, run many times
// Shared workspace for all PreinitGemm (allocated once, 32MB)
static uint8_t* g_preinit_ws = nullptr;
static size_t g_preinit_ws_sz = 32 * 1024 * 1024;  // 32MB

struct PreinitGemm {
    SmParams params;
    uint8_t* workspace = nullptr;
    size_t ws_size = 0;
    bool initialized = false;
    
    void init(void* A, void* B, void* D, int M, int N, int K, cudaStream_t st, float alpha = 1.0f) {
        if (!g_preinit_ws) cudaMalloc(&g_preinit_ws, g_preinit_ws_sz);
        using SA = typename SmGemm::GemmKernel::StrideA;
        using SB = typename SmGemm::GemmKernel::StrideB;
        using SC = typename SmGemm::GemmKernel::StrideC;
        using SD = typename SmGemm::GemmKernel::StrideD;

        typename SmGemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {(cutlass_fp8*)A, cutlass::make_cute_packed_stride(SA{}, {M, K, 1}),
             (cutlass_fp8*)B, cutlass::make_cute_packed_stride(SB{}, {N, K, 1})},
            {{alpha, 0.0f},
             (cutlass_fp16*)D, cutlass::make_cute_packed_stride(SC{}, {M, N, 1}),
             (cutlass_fp16*)D, cutlass::make_cute_packed_stride(SD{}, {M, N, 1})}
        };
        
        if (!workspace) {
            ws_size = SmGemm::get_workspace_size(args);
            if (ws_size > 0) cudaMalloc(&workspace, ws_size);
        }

        // Cache args for initialize() at run time
        cached_args = args;
        params = SmGemm::GemmKernel::to_underlying_arguments(args, workspace);
        initialized = true;
    }
    
    typename SmGemm::Arguments cached_args;

    void run(cudaStream_t st) {
        // SM110: must initialize() each call (cached params alone doesn't work)
        SmGemm gemm;
        gemm.initialize(cached_args, workspace, st);
        gemm.run(st);
    }

    // Graph-safe run: bypasses cudaLaunchKernelExC, uses cudaLaunchKernel directly
    // For Cluster<1,1,1> these are identical, but cudaLaunchKernel IS graph-capturable
    void run_graph_safe(cudaStream_t st) {
        dim3 block = SmGemm::GemmKernel::get_block_shape();
        dim3 grid = SmGemm::get_grid_shape(params);
        int smem = SmGemm::GemmKernel::SharedStorageSize;
        void* kernel_fn = (void*)cutlass::device_kernel<typename SmGemm::GemmKernel>;
        void* args[] = {&params};
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaLaunchKernel(kernel_fn, grid, block, args, smem, st);
    }
};

// AE pre-initialized GEMM pool: 180 layers × 4 GEMMs = 720 pre-built params
#define AE_MAX_LAYERS 180
#define AE_GEMMS_PER_LAYER 4


static PreinitGemm ae_gemm_pool[AE_MAX_LAYERS * AE_GEMMS_PER_LAYER];
static bool ae_pool_initialized = false;

// Initialize all 720 GEMM params at startup
static void ae_preinit_gemms(
    void* xn_fp8, void* qkv, void* ctx_fp8, void* fg, void* hid_fp8,
    void* qw, void* ow, void* gw, void* dw,
    int S, int D, int H, int NH, int HD, int layers,
    cudaStream_t st) {
    
    for (int l = 0; l < layers; l++) {
        int base = l * AE_GEMMS_PER_LAYER;
        int D3 = 2560;  // QKV output dim
        
        // QKV: [S, 2560, D]
        ae_gemm_pool[base + 0].init(xn_fp8, (char*)qw + l*D*D3, qkv, S, D3, D, st);
        // O proj: [S, D, NH*HD]
        ae_gemm_pool[base + 1].init(ctx_fp8, (char*)ow + l*NH*HD*D, fg, S, D, NH*HD, st);
        // Gate+Up merged: [S, 2*H, D]
        ae_gemm_pool[base + 2].init(xn_fp8, (char*)gw + l*D*H*2, fg, S, H*2, D, st);
        // Down: [S, D, H] — alpha=5.0 compensates SILU_SCALE=0.2 in silu_mul_fp8_k
        ae_gemm_pool[base + 3].init(hid_fp8, (char*)dw + l*H*D, fg, S, D, H, st, 5.0f);
    }
    ae_pool_initialized = true;
}


// Encoder pre-initialized GEMM pool
// Uses the large-tile PlainGemm from main engine (256x128x64)
// Can't define here because PlainGemm is in .cu file
// Instead, export as extern "C" functions from .cu



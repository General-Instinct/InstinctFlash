#include "gemm_types.h"
#include "cutlass/util/device_memory.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// ============================================================
//  Helper: run a CUTLASS GEMM and return elapsed ms
// ============================================================
template <typename GemmOp, typename Arguments>
float run_gemm(Arguments& args, int warmup, int iters) {
    GemmOp gemm;
    size_t workspace_size = GemmOp::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS cannot implement this GEMM: "
                  << cutlassGetStatusString(status) << std::endl;
        return -1.0f;
    }

    status = gemm.initialize(args, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS init failed: "
                  << cutlassGetStatusString(status)
                  << " (workspace=" << workspace_size << ")" << std::endl;
        auto cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            std::cerr << "  CUDA error: " << cudaGetErrorString(cerr) << std::endl;
        }
        return -1.0f;
    }

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        gemm.run();
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        gemm.run();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

// ============================================================
//  BF16 GEMM: C = alpha * A @ B + beta * C
// ============================================================
extern "C" float launch_bf16_gemm(
    void* A_ptr, void* B_ptr, void* C_ptr, void* D_ptr,
    int M, int N, int K,
    float alpha, float beta,
    int warmup, int iters)
{
    using namespace bf16_gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<ElementA*>(A_ptr), stride_A,
            reinterpret_cast<ElementB*>(B_ptr), stride_B,
        },
        {
            {alpha, beta},
            reinterpret_cast<ElementC*>(C_ptr), stride_C,
            reinterpret_cast<ElementD*>(D_ptr), stride_D,
        }
    };

    return run_gemm<Gemm>(arguments, warmup, iters);
}

// ============================================================
//  FP8 GEMM: D = alpha * A_fp8 @ B_fp8 + beta * C_bf16
// ============================================================
extern "C" float launch_fp8_gemm(
    void* A_ptr, void* B_ptr, void* C_ptr, void* D_ptr,
    int M, int N, int K,
    float alpha, float beta,
    int warmup, int iters)
{
    using namespace fp8_gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<ElementA*>(A_ptr), stride_A,
            reinterpret_cast<ElementB*>(B_ptr), stride_B,
        },
        {
            {alpha, beta},
            reinterpret_cast<ElementC*>(C_ptr), stride_C,
            reinterpret_cast<ElementD*>(D_ptr), stride_D,
        }
    };

    return run_gemm<Gemm>(arguments, warmup, iters);
}

// ============================================================
//  NVFP4 Block-Scaled GEMM: D = alpha * A_fp4 @ B_fp4 + beta * C
// ============================================================
extern "C" float launch_fp4_gemm(
    void* A_ptr, void* SFA_ptr,
    void* B_ptr, void* SFB_ptr,
    void* C_ptr, void* D_ptr,
    int M, int N, int K,
    float alpha, float beta,
    int warmup, int iters)
{
    using namespace fp4_gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<typename ElementA::DataType*>(A_ptr), stride_A,
            reinterpret_cast<typename ElementB::DataType*>(B_ptr), stride_B,
            reinterpret_cast<typename ElementA::ScaleFactorType*>(SFA_ptr), layout_SFA,
            reinterpret_cast<typename ElementB::ScaleFactorType*>(SFB_ptr), layout_SFB,
        },
        {
            {alpha, beta},
            reinterpret_cast<ElementC*>(C_ptr), stride_C,
            reinterpret_cast<ElementD*>(D_ptr), stride_D,
        }
    };

    return run_gemm<Gemm>(arguments, warmup, iters);
}

// ============================================================
//  W4A8 Mixed Block-Scaled GEMM (benchmark version)
//  A: (M, K) MX-FP8 + per-32-block UE4M3 scale factors
//  B: (K, N) MX-FP4 + per-16-block UE4M3 scale factors
//  D: (M, N) BF16
// ============================================================
extern "C" float launch_w4a8_gemm(
    void* A_ptr, void* SFA_ptr,
    void* B_ptr, void* SFB_ptr,
    void* C_ptr, void* D_ptr,
    int M, int N, int K,
    float alpha, float beta,
    int warmup, int iters)
{
    using namespace w4a8_gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<typename ElementA::DataType*>(A_ptr), stride_A,
            reinterpret_cast<typename ElementB::DataType*>(B_ptr), stride_B,
            reinterpret_cast<typename ElementA::ScaleFactorType*>(SFA_ptr), layout_SFA,
            reinterpret_cast<typename ElementB::ScaleFactorType*>(SFB_ptr), layout_SFB,
        },
        {
            {alpha, beta},
            reinterpret_cast<ElementC*>(C_ptr), stride_C,
            reinterpret_cast<ElementD*>(D_ptr), stride_D,
        }
    };

    return run_gemm<Gemm>(arguments, warmup, iters);
}

// ============================================================
//  W4A8 GEMM (stream version for inference — no benchmark loop)
// ============================================================
extern "C" int run_w4a8_gemm(
    void* A_ptr, void* SFA_ptr,
    void* B_ptr, void* SFB_ptr,
    void* D_ptr,
    int M, int N, int K,
    cudaStream_t stream)
{
    using namespace w4a8_gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<typename ElementA::DataType*>(A_ptr), stride_A,
            reinterpret_cast<typename ElementB::DataType*>(B_ptr), stride_B,
            reinterpret_cast<typename ElementA::ScaleFactorType*>(SFA_ptr), layout_SFA,
            reinterpret_cast<typename ElementB::ScaleFactorType*>(SFB_ptr), layout_SFB,
        },
        {
            {1.0f, 0.0f},
            nullptr, stride_D,
            reinterpret_cast<ElementD*>(D_ptr), stride_D,
        }
    };

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (workspace.size() < workspace_size) {
        workspace.reset(workspace_size);
    }

    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "W4A8 can_implement failed: " << cutlassGetStatusString(status)
                  << " M=" << M << " N=" << N << " K=" << K << std::endl;
        return -1;
    }
    status = gemm.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        auto cerr = cudaGetLastError();
        std::cerr << "W4A8 initialize failed: " << cutlassGetStatusString(status)
                  << " CUDA: " << cudaGetErrorString(cerr) << std::endl;
        return -1;
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "W4A8 run failed: " << cutlassGetStatusString(status) << std::endl;
    }
    return (status == cutlass::Status::kSuccess) ? 0 : -1;
}

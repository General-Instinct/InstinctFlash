// Shared SM110 BF16 GEMM + rounded ReLU squared. Raw-pointer ABI v1.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

namespace flash_wm {
namespace bf16_relu2 {

using namespace cute;

using ElementA   = cutlass::bfloat16_t;
using LayoutA    = cutlass::layout::RowMajor;
using ElementB   = cutlass::bfloat16_t;
using LayoutB    = cutlass::layout::ColumnMajor;
using ElementC   = cutlass::bfloat16_t;
using ElementD   = cutlass::bfloat16_t;
using LayoutC    = cutlass::layout::RowMajor;
using LayoutD    = cutlass::layout::RowMajor;
using ElementAcc     = float;
using ElementCompute = float;

static constexpr int AlignA    = 8;                                  // 128b / 16b = 8
static constexpr int AlignB    = 8;
static constexpr int AlignC    = 128 / cutlass::sizeof_bits<ElementC>::value;     // 8
static constexpr int AlignD    = AlignC;

using ArchTag   = cutlass::arch::Sm100;
using OpClass   = cutlass::arch::OpClassTensorOp;

// Thor tile admitted by the paired Edge regression.
using TileShape = Shape<_256, _128, _128>;
using ClusterShape = Shape<_2, _1, _1>;

// Keep the explicit BF16 GEMM rounding boundary before ReLU squared.
template<class T> struct RoundedReluSquared {
  static constexpr bool kIsHeavy = false;
  CUTLASS_HOST_DEVICE T operator()(T value) const {
    float rounded = float(cutlass::bfloat16_t(float(value)));
    float positive = cutlass::maximum<float, true>{}(rounded, 0.0f);
    return T(positive * positive);
  }
};
template<class T, int N> struct RoundedReluSquared<cutlass::Array<T,N>> {
  static constexpr bool kIsHeavy = false;
  CUTLASS_HOST_DEVICE cutlass::Array<T,N> operator()(cutlass::Array<T,N> const& x) const {
    cutlass::Array<T,N> y;
    CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<N;++i) y[i]=RoundedReluSquared<T>{}(x[i]);
    return y;
  }
};
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<RoundedReluSquared,
    ElementD, ElementCompute, ElementC, ElementCompute>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    Fusion
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

static int run(void const* A, void const* B, void* workspace, size_t capacity,
               void* D, int M, int N, int K, float alpha,
               cudaStream_t stream) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        { reinterpret_cast<ElementA const*>(A), stride_A,
          reinterpret_cast<ElementB const*>(B), stride_B },
        { {},  // epilogue.thread (fusion args filled below)
          reinterpret_cast<ElementC const*>(D), stride_C,  // C unused (beta=0)
          reinterpret_cast<ElementD*>(D), stride_D }
    };

    auto& fusion_args = args.epilogue.thread;
    fusion_args.alpha = alpha;
    fusion_args.beta  = 0.0f;

    Gemm gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        fprintf(stderr, "[bf16_relu2] can_implement failed: M=%d N=%d K=%d code=%d\n",
                M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }

    size_t ws_sz = Gemm::get_workspace_size(args);
    if (ws_sz > capacity) return -2;
    st = gemm.initialize(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) {
        fprintf(stderr, "[bf16_relu2] init failed: M=%d N=%d K=%d code=%d\n",
                M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x20000;
    }

    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace bf16_relu2
}  // namespace flash_wm

extern "C" int instinctflash_bf16_abi_version() { return 1; }

extern "C" int instinctflash_bf16_linear_relu2(void const* A, void const* B, void* D,
 int M, int N, int K, void* workspace, size_t capacity, cudaStream_t stream) {
 if (!A || !B || !D || !workspace || M != 3093 || N != 9216 || K != 2048) return -3;
 if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) |
      reinterpret_cast<uintptr_t>(D) | reinterpret_cast<uintptr_t>(workspace)) % 16) return -4;
 try {
   return flash_wm::bf16_relu2::run(A,B,workspace,capacity,D,M,N,K,1.0f,stream);
 } catch (...) { return -5; }
}

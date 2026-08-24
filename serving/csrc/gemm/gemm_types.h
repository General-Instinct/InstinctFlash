#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ============================================================
//  Kernel Type 1: BF16 GEMM (baseline, non-quantized)
// ============================================================
namespace bf16_gemm {

using ElementA       = cutlass::bfloat16_t;
using LayoutA        = cutlass::layout::RowMajor;
constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 8

using ElementB       = cutlass::bfloat16_t;
using LayoutB        = cutlass::layout::ColumnMajor;
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC       = cutlass::bfloat16_t;
using LayoutC        = cutlass::layout::RowMajor;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD       = cutlass::bfloat16_t;
using LayoutD        = cutlass::layout::RowMajor;
constexpr int AlignD = AlignC;

using ElementAcc     = float;
using Arch           = cutlass::arch::Sm100;
using OpClass        = cutlass::arch::OpClassTensorOp;
using TileShape      = Shape<_256, _128, _64>;
using ClusterShape   = Shape<_2, _2, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, OpClass,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // namespace bf16_gemm

// ============================================================
//  Kernel Type 2: FP8 GEMM (E4M3 × E4M3 → BF16)
// ============================================================
namespace fp8_gemm {

using ElementA       = cutlass::float_e4m3_t;
using LayoutA        = cutlass::layout::RowMajor;
constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16

using ElementB       = cutlass::float_e4m3_t;
using LayoutB        = cutlass::layout::ColumnMajor;
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC       = cutlass::bfloat16_t;
using LayoutC        = cutlass::layout::RowMajor;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD       = cutlass::bfloat16_t;
using LayoutD        = cutlass::layout::RowMajor;
constexpr int AlignD = AlignC;

using ElementAcc     = float;
using Arch           = cutlass::arch::Sm100;
using OpClass        = cutlass::arch::OpClassTensorOp;
using TileShape      = Shape<_256, _128, _64>;
using ClusterShape   = Shape<_2, _2, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, OpClass,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // namespace fp8_gemm

// ============================================================
//  Kernel Type 3: NVFP4 Block-Scaled GEMM (FP4 × FP4 → BF16)
//  — For Encoder FFN large GEMMs (max speedup)
// ============================================================
namespace fp4_gemm {

using ElementA       = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutA        = cutlass::layout::RowMajor;
constexpr int AlignA = 32;

using ElementB       = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutB        = cutlass::layout::ColumnMajor;
constexpr int AlignB = 32;

using ElementC       = cutlass::bfloat16_t;
using LayoutC        = cutlass::layout::RowMajor;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD       = cutlass::bfloat16_t;
using LayoutD        = cutlass::layout::RowMajor;
constexpr int AlignD = AlignC;

using ElementAcc     = float;
using Arch           = cutlass::arch::Sm120;
using OpClass        = cutlass::arch::OpClassBlockScaledTensorOp;
using TileShape      = Shape<_128, _128, _128>;
using ClusterShape   = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, OpClass,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // namespace fp4_gemm

// ============================================================
//  Kernel Type 4: W4A8 Mixed Block-Scaled GEMM (MX-FP8 × MX-FP4 → BF16)
//  — Weight FP4, Activation FP8, on-the-fly dequant in registers
//  — Same MMA throughput as FP8, but halves weight memory traffic
//  — Based on CUTLASS example 72c_blackwell_mixed_mxfp8_bf16_gemm
// ============================================================
namespace w4a8_gemm {

// A = activation: MX-FP8 block-scaled (per-32 UE4M3 scale factors)
using ElementA       = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutA        = cutlass::layout::RowMajor;
constexpr int AlignA = 16;

// B = weight: MX-FP4 block-scaled (per-16 UE4M3 scale factors)
using ElementB       = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using LayoutB        = cutlass::layout::ColumnMajor;
constexpr int AlignB = 128;

// Output: BF16
using ElementC       = cutlass::bfloat16_t;
using LayoutC        = cutlass::layout::RowMajor;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD       = cutlass::bfloat16_t;
using LayoutD        = cutlass::layout::RowMajor;
constexpr int AlignD = AlignC;

using ElementAcc     = float;
using Arch           = cutlass::arch::Sm120;
using OpClass        = cutlass::arch::OpClassBlockScaledTensorOp;
using TileShape      = Shape<_128, _128, _128>;
using ClusterShape   = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, OpClass,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // namespace w4a8_gemm

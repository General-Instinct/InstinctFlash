/**
 * fmha_fp16_strided.cu — FP16 FMHA with custom sequence stride
 * Allows reading Q/K/V from interleaved QKV buffer WITHOUT deinterleave.
 * Key: sequence stride can be != NH*HD (e.g., 3*NH*HD for [S, 3D] layout)
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "device/fmha.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_load_tma_warpspecialized.hpp"
#include "collective/fmha_fusion.hpp"

using namespace cute;
using Element = cutlass::half_t;
using ElementAccQK = float;
using ElementAccPV = float;
using ElementOut = cutlass::half_t;
using TileShape = Shape<_256, _128, _128>;

using StrideQ   = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK   = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideV   = StrideK;
using StrideO   = StrideQ;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;
using ProblemShape = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;

using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
    Element, ElementAccQK, ElementAccPV, TileShape,
    StrideQ, StrideK, StrideV, cutlass::fmha::collective::NoMask>;
using Epilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
    ElementOut, ElementAccPV, typename Mainloop::TileShapePV, StrideO, StrideLSE>;
using Kernel = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
    ProblemShape, Mainloop, Epilogue,
    cutlass::fmha::kernel::IndividualTileScheduler>;
using FmhaOp = cutlass::fmha::device::FMHA<Kernel>;

static void* g_ws = nullptr; static size_t g_ws_sz = 0;
static float* g_lse = nullptr; static size_t g_lse_sz = 0;

// Standard API: Q/K/V contiguous [S, NH/NKV, HD]
extern "C" int fmha_fp16_attn(
    const void* Q, const void* K, const void* V, void* O,
    int B, int SQ, int SK, int NQ, int NKV, int HD,
    cudaStream_t stream)
{
    int H_Q = NQ/NKV, H_K = NKV, H = H_Q*H_K;
    int D = cutlass::round_up(HD, 8);
    auto ps = cute::make_tuple(SQ, SK, D, cute::make_tuple(cute::make_tuple(H_Q, H_K), B));

    StrideQ sQ = make_stride(H*D, _1{}, make_stride(make_stride(D, H_Q*D), H*D*SQ));
    StrideO sO = sQ;
    StrideK sK = make_stride(H_K*D, _1{}, make_stride(make_stride(_0{}, D), H_K*D*SK));
    int SQ_r = ((SQ+127)/128)*128;
    StrideLSE sL = make_stride(_1{}, make_stride(make_stride(SQ_r, SQ_r*H_Q), SQ_r*H));

    size_t lsz = (size_t)B*H*SQ_r*sizeof(float);
    if (lsz > g_lse_sz) { if(g_lse) cudaFree(g_lse); cudaMalloc(&g_lse,lsz); g_lse_sz=lsz; }
    int sm = 0; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);

    typename FmhaOp::Arguments args{ps,
        {{(Element const*)Q, sQ, (Element const*)K, sK, (Element const*)V, sK},
         0.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {(ElementOut*)O, sO, g_lse, sL}, {0, sm}};

    FmhaOp op;
    if (op.can_implement(args) != cutlass::Status::kSuccess) return -1;
    size_t wsz = FmhaOp::get_workspace_size(args);
    if (wsz > g_ws_sz) { if(g_ws) cudaFree(g_ws); cudaMalloc(&g_ws,wsz); g_ws_sz=wsz; }
    if (op.initialize(args, g_ws, stream) != cutlass::Status::kSuccess) return -2;
    return (op.run(stream) == cutlass::Status::kSuccess) ? 0 : -3;
}

// ═══════════════════════════════════════════════════════════════════
// Strided API: Q/K/V read from interleaved buffer with custom stride
// Q at qkv_ptr + q_offset, stride between tokens = seq_stride (not NH*HD)
// Output O is always contiguous [S, NH, HD]
// ═══════════════════════════════════════════════════════════════════
extern "C" int fmha_fp16_strided(
    const void* Q, const void* K, const void* V, void* O,
    int B, int SQ, int SK, int NQ, int NKV, int HD,
    int q_seq_stride,  // stride between tokens for Q (e.g., 3*NH*HD for interleaved QKV)
    int k_seq_stride,  // stride between tokens for K
    cudaStream_t stream)
{
    int H_Q = NQ/NKV, H_K = NKV, H = H_Q*H_K;
    int D = cutlass::round_up(HD, 8);
    auto ps = cute::make_tuple(SQ, SK, D, cute::make_tuple(cute::make_tuple(H_Q, H_K), B));

    // Q stride: (q_seq_stride, 1, ((D, H_Q*D), q_seq_stride*SQ))
    StrideQ sQ = make_stride(q_seq_stride, _1{}, make_stride(make_stride(D, H_Q*D), q_seq_stride*SQ));
    // O stride: contiguous [S, NH, HD]
    StrideO sO = make_stride(H*D, _1{}, make_stride(make_stride(D, H_Q*D), H*D*SQ));
    // K/V stride: (k_seq_stride, 1, ((_0, D), k_seq_stride*SK))
    StrideK sK = make_stride(k_seq_stride, _1{}, make_stride(make_stride(_0{}, D), k_seq_stride*SK));

    int SQ_r = ((SQ+127)/128)*128;
    StrideLSE sL = make_stride(_1{}, make_stride(make_stride(SQ_r, SQ_r*H_Q), SQ_r*H));

    size_t lsz = (size_t)B*H*SQ_r*sizeof(float);
    if (lsz > g_lse_sz) { if(g_lse) cudaFree(g_lse); cudaMalloc(&g_lse,lsz); g_lse_sz=lsz; }
    int sm = 0; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);

    typename FmhaOp::Arguments args{ps,
        {{(Element const*)Q, sQ, (Element const*)K, sK, (Element const*)V, sK},
         0.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {(ElementOut*)O, sO, g_lse, sL}, {0, sm}};

    FmhaOp op;
    auto st = op.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        printf("[FMHA strided] can_implement FAILED (%d) qstride=%d kstride=%d\n", (int)st, q_seq_stride, k_seq_stride);
        return -1;
    }
    size_t wsz = FmhaOp::get_workspace_size(args);
    if (wsz > g_ws_sz) { if(g_ws) cudaFree(g_ws); cudaMalloc(&g_ws,wsz); g_ws_sz=wsz; }
    if (op.initialize(args, g_ws, stream) != cutlass::Status::kSuccess) return -2;
    return (op.run(stream) == cutlass::Status::kSuccess) ? 0 : -3;
}

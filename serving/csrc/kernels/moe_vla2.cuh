// ================================================================
// FlashRT — T2-V2 token-MoE kernels (LingBot-VLA-V2-6B action expert)
//
// ADDITIVE module (new file, nothing existing modified). Ground truth
// for all semantics: Qwen2TokenMoeBlock.forward (lingbotvla
// qwen2_action_expert.py :274-362) under the shipped ckpt config —
// torch parity baseline: flash_rt/models/vla2/moe_ref.py, plan:
// /home/ubuntu/iwm_distill/thor_t2v2/mapping_memo.md §C.
//
// Router math (fp32 end-to-end, matching the stock autocast-disabled
// router):
//   scores          = sigmoid(logits)                    (fp32)
//   selection       = top-K over (scores + e_score_correction_bias)
//                     — bias biases SELECTION only
//   weights[k]      = scores[sel[k]] / (sum_k scores[sel[k]] + 1e-20)
//                     * routed_scaling_factor (4.0)
// Tie-break: lowest expert index wins (deterministic; stock fp32
// router ties are measure-zero).
//
// All kernels are shape-static and capture-safe: fixed grids from the
// shape args, no host sync, no allocation. The cuBLASLt batched entry
// allocates its descriptor cache + workspace on FIRST call per shape
// (same discipline as fp8_gemm_descale_fp16) — run each shape once
// before CUDA-graph capture.
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ── Router: consume an fp32 logits matrix [T, E] ──
// ids: [T, K] int32; weights: [T, K] fp32. E <= 32, K <= 8.
void moe_router_topk(const float* logits, const float* e_bias,
                     int* ids, float* weights,
                     int T, int E, int K, float routed_scale,
                     cudaStream_t stream = 0);

// ── Fused router: fp32 gate GEMM in-kernel + sigmoid/top-K/norm ──
// x fp16 [T, D] (Stage-A / calibrate path: post-AdaRMS fp16 acts);
// gate_w fp32 [E, D] row-major (HF layout, the fp32 router weight);
// logits_out optional ([T, E] fp32, pass nullptr to skip) for
// layerwise diffing vs moe_ref. D arbitrary; E <= 32, K <= 8.
void moe_router_gemm_topk_fp16x(const __half* x, const float* gate_w,
                                const float* e_bias, int* ids,
                                float* weights, float* logits_out,
                                int T, int D, int E, int K,
                                float routed_scale,
                                cudaStream_t stream = 0);

// Same, consuming the fp8 post-AdaRMS activation the routed/shared
// experts read (one activation tensor feeds router + all experts —
// repack_calib_plan §2). act_descale: DEVICE float* (amax/448).
// PRECISION NOTE: the stock router sees the unquantized bf16 hidden;
// this variant sees its fp8 quantization — gated at M2d, fallback =
// adarms_fp16 + the fp16x variant above.
void moe_router_gemm_topk_fp8x(const void* x_fp8, const float* gate_w,
                               const float* e_bias, int* ids,
                               float* weights, float* logits_out,
                               int T, int D, int E, int K,
                               float routed_scale,
                               const float* act_descale,
                               cudaStream_t stream = 0);

// ── Combine: out[t] = sum_k w[t,k] * slab[ids[t,k], t] (+ shared[t]) ──
// slab: [E_slab, T, D] all-expert outputs (dense v0/v0.5 layout);
// shared: UNGATED shared-expert output [T, D], pass nullptr/0 to get
// the routed-only contribution (the pipeline_thor contract: moe_out
// excludes the shared expert, which is summed by residual_add_fp16).
// fp32 accumulation; k-major add order == moe_ref.combine_gather.
void moe_combine_fp16(const __half* slab, const int* ids,
                      const float* weights, const __half* shared,
                      __half* out, int T, int D, int K,
                      cudaStream_t stream = 0);

void moe_combine_fp32(const float* slab, const int* ids,
                      const float* weights, const float* shared,
                      float* out, int T, int D, int K,
                      cudaStream_t stream = 0);

// ── True-SiLU merged gate|up activation (the existing merged kernels
// are GELU-tanh — pi05 lineage; the V2 MoE experts use exact SiLU) ──
// merged: [S, 2H] rows = [gate(H) | up(H)]; S may be batch*tokens for
// the (E, T, 2H) batched slab. fp8 out uses amax/448 device descale.
void silu_mul_merged_fp8_fp16(const __half* merged, __nv_fp8_e4m3* out,
                              int S, int H, const float* descale_ptr,
                              cudaStream_t stream = 0);

// fp16-out variant (calibrate path: quantize_fp8_device_fp16 reads it
// to write the amax slot in place).
void silu_mul_merged_fp16(const __half* merged, __half* out,
                          int S, int H, cudaStream_t stream = 0);

// ── cuBLASLt strided-batched FP8 GEMM with device descale → FP16 ──
// Per batch e: C[e] = (act_descale * w_descale) * A[e] @ B[e]
//   A: fp8e4m3 [M, K] row-major, batch stride strideA ELEMENTS
//      (strideA = 0 broadcasts one activation to all experts);
//   B: fp8e4m3 [K, N] row-major per expert (KN layout, matching
//      fp8_gemm_descale_fp16), batch stride strideB elements;
//   C: fp16 [M, N] row-major, batch stride strideC elements.
// act_descale / w_descale: DEVICE float* — cuBLASLt per-tensor scales,
// i.e. ONE scale per (layer, matrix) 3D tensor: the batched path is
// the R3 shared-scale mode (per-expert scales require the per-expert
// loop of fp8_gemm_descale_fp16 — mapping_memo §C precision plan).
// Descriptor+algo cached per (M, N, K, batch, strides); scale
// pointers are set per call (pointer-stable slots, recalibration
// never recaptures).
void fp8_gemm_batched_descale_fp16(const void* A_fp8, const void* B_fp8,
                                   void* C_fp16,
                                   int M, int N, int K, int batch,
                                   long long strideA, long long strideB,
                                   long long strideC,
                                   const float* act_descale,
                                   const float* w_descale,
                                   cudaStream_t stream = 0);

// TN-layout variant: B stored [N, K] row-major per expert (native HF
// [out, in] orientation — no repack transpose). Same batching/scale
// semantics. MEASURED: cuBLASLt 12.8/sm_90 rejects the NN fp8 combo
// (status 15) — NN above is Thor/CUDA-13-only (pi05/vla4b-proven
// there); this TN combo is supported on both sm_90 and sm_110, so it
// is the default the vla2 MoE engine and unit tests use.
void fp8_gemm_batched_descale_nt_fp16(const void* A_fp8, const void* B_fp8,
                                      void* C_fp16,
                                      int M, int N, int K, int batch,
                                      long long strideA, long long strideB,
                                      long long strideC,
                                      const float* act_descale,
                                      const float* w_descale,
                                      cudaStream_t stream = 0);

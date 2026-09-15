// ================================================================
// FlashRT — LingBot-VA (wan_va) fused kernels (Stage 2, 2026-09-02)
//
// ADDITIVE module (new file, nothing existing modified). The wan_va DiT
// maps onto existing fvk kernels everywhere except ONE site (mapping_memo
// §A.2): the AdaLN gated residual
//
//     hidden = (hidden.float() + y.float() * gate).type_as(hidden)
//
// where `gate` is a per-CHANNEL row [D] from the step tables (per stream,
// per t, per layer) — model.py:543-544 (attn) and :564-565 (ffn). The
// existing `gate_res_fp16` takes a full [S, D] gate tensor (the AdaRMS
// style of pi05/vla4b/vla2); broadcasting the row into an [S, D] buffer
// per call costs an extra S*D write+read for 60 sites per forward. This
// kernel reads the [D] row directly. fp32 math, fp16 out — the same
// rounding structure as the stock fp32 multiply-add -> bf16 cast.
//
// Capture-safe: fixed grid from the shape args, no host sync, no
// allocation.
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// res[s, c] += x[s, c] * gate_row[c]   for s in [0, S), c in [0, D)
// res, x: [S, D] fp16 row-major (may be sub-slices of a larger buffer,
// rows are contiguous); gate_row: [D] fp16. D must be even.
void gate_row_mul_residual_fp16(__half* res, const __half* x,
                                const __half* gate_row, int S, int D,
                                cudaStream_t stream = 0);

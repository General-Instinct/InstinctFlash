"""FlashRT — LingBot-VA (wan_va) model package (Stage 2, 2026-09-02).

Target: LingBot-VA posttrain-robotwin — a single-stack Wan2.2 DiT
(30 blocks x hidden 3072, 24h x hd128, ffn 14336, 5.089B ckpt-verified)
serving interleaved video/action chunks. The engine is built FOR ONE declared
operating point (``operating_point.WanVaOperatingPoint``): the CERTIFIED
2V/4A@w5 point (10 DiT forwards / 32-control-step cycle, CFG batch 2) or the
guidance-off 2V/2A@w1 point (8 forwards, batch 1) — step tables, slab stream
count and forward count all derive from the point, and the frontend declines
any other request (h2_thor_realtime_design.md §5, engine_backend 9bf2337).

Ground truth: /home/ubuntu/lingbot-va/wan_va (modules/model.py,
wan_va_server.py, utils/scheduler.py); mapping + gate ladder:
/home/ubuntu/iwm_distill/thor_va_engine/{mapping_memo,repack_calib_plan}.md.

Modules:
    operating_point — the declared build point: grids, guidance, cfg batch,
                   forwards/cycle, the mismatch rule (mirror of
                   engine_operating_point_problem), worker-flag parsing.
    rope_table   — f64 wan 3D rope (44/42/42) + the interleaved->split-half
                   channel permutation that makes ``rope_rotate_half_fp16``
                   exact; video/action grid builders (fractional action
                   frames, h=w=-1).
    step_tables  — AdaLN modulation tables through the DEPLOYED dtype chain
                   (fp32 sinusoid/time_embedder -> bf16 cast -> bf16 silu ->
                   bf16 time_proj GEMM -> fp32 add), sized from the point's grids.
    ring_ref     — stock KV allocator reference (mask/id/is_pred) + the
                   engine linear-episode-slab model + set-parity self-check.
    wan_ref      — engine-structured torch reference block forward (tables +
                   permuted rope + explicit KV window) vs a literal
                   transcription of the stock block math (the M1 gate arm).
    pipeline_thor— the Thor pointer-op DiT program (existing fvk kernels +
                   the ONE new device kernel gate_row_mul_residual_fp16 in
                   csrc/kernels/wan_va_fused.cu, with a no-rebuild fallback).

Everything here is import-clean without the flash_rt_kernels extension and
CPU-runnable for the self-checks.
"""

DIT_L = 30
DIT_D = 3072
DIT_NH = 24
DIT_HD = 128
DIT_FFN = 14336
TEXT_LEN = 512          # fixed; zero-pad rows are LIVE in cross attention
TEXT_DIM = 4096
LATENT_C = 48
ACTION_DIM = 30
ACTION_PER_FRAME = 16
FRAME_CHUNK = 2
LATENT_H, LATENT_W = 24, 20          # robotwin_tshape post-VAE
PATCH = (1, 2, 2)
VIDEO_TOKENS = FRAME_CHUNK * (LATENT_H // 2) * (LATENT_W // 2)   # 240
ACTION_TOKENS = FRAME_CHUNK * ACTION_PER_FRAME                   # 32
POOL_SLOTS = 9792                    # attn_window 72 -> 36*(240+32)
SLAB_ROWS = 15360                    # linear episode slab (memo §A.3)
CFG_BATCH = 2
VIDEO_NFE, ACTION_NFE = 2, 4         # the CERTIFIED point (operating_point.POINT_2V4A_W5)
GUIDANCE_VIDEO, GUIDANCE_ACTION = 5.0, 1.0
MAX_COMMIT_FRAMES = 3                # cycle-0 kv commit: init keyframe + 2 keyframes
MAX_VIDEO_TOKENS = MAX_COMMIT_FRAMES * (LATENT_H // 2) * (LATENT_W // 2)   # 360

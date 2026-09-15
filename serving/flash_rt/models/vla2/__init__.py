"""LingBot-VLA-V2-6B model package (T2-V2, Stage 1 scaffold).

Qwen3-VL-4B backbone (ViT 24L/1024/16h x hd64 full-attention-per-image
+ deepstack taps [5,11,17]; LM 36L/2560 GQA 32q/8kv x hd128 with
per-head q/k RMSNorm and interleaved 3D M-RoPE theta 5e6) with a
36-layer Qwen2-skeleton action expert (hidden 768, joint attention in
LM head geometry 32q/8kv x hd128, AdaRMS time conditioning, and a
sparse token-MoE MLP on every layer: 32 experts x ffn 512, top-4
sigmoid routing with e_score_correction_bias, routed_scaling_factor
4.0, plus an ungated shared expert ffn 704).

Chunk 50, 10 Euler steps, state/action dim 55. Prefix layout per
infer: 3 x [vision_start | 64 merged patches | vision_end] + language
(Qwen3 chat template, dense) + 16 trained align-query tokens
(current-task 8 + future-task 8 — constants at inference). Prefill is
CAUSAL (vlm_causal=true in the shipped checkpoint).

Ground truth: lingbotvla modeling_lingbot_vla_v2.py + qwen2_action_expert.py
+ qwen3vl_in_vla.py at /home/ubuntu/lingbot-vla-v2-repo, checkpoint
robbyant/lingbot-vla-v2-6b-robotwin (global_step_50000). Full mapping:
/home/ubuntu/iwm_distill/thor_t2v2/mapping_memo.md.

Files:
    rope_table.py     — host-side ViT 2D rope + LM/expert 3D M-RoPE tables
                        (reuses the bit-exact groot_n17 Qwen3-VL builders)
    step_tables.py    — AdaRMS step tables (re-exports vla4b: identical math)
    moe_ref.py        — correct-by-construction torch reference for the
                        token-MoE block (CPU-runnable parity baseline)
    pipeline_thor.py  — Thor SM110 forward skeletons (fvk pointer ops;
                        MoE router/combine marked STAGE2)
    moe_engine.py     — M2: routed_moe_fn on the flash_rt_kernels MoE
                        entries (moe_router_*/moe_combine_*/batched
                        fp8 GEMMs; v0 per-expert loop + v0.5 batched)
    repack_v2.py      — M2: HF ckpt (+ calib scales) → engine-layout
                        artifact dir, incl. the standalone R1
                        pos-embed step (stock HF interpolation)
"""

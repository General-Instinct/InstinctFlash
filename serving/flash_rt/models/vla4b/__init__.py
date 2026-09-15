"""LingBot-VLA-4B model package (T2-V1).

Qwen2.5-VL-3B backbone (ViT 32L/1280/16h×hd80 windowed + LM 36L/2048
GQA 16q/2kv×hd128) with a 36-layer Qwen2-skeleton action expert
(hidden 768, joint attention in LM head geometry, AdaRMS time
conditioning). Chunk 50, 10 Euler steps, state/action dim 75.

Files:
    rope_table.py     — host-side ViT 2D + LM 1D rope tables
    pipeline_thor.py  — Thor SM110 forward functions (fvk pointer ops)
"""

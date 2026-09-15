"""Host-side RoPE table construction for LingBot-VLA-V2-6B.

Three table families, all consumed unchanged by the existing
``rope_rotate_half_fp16`` kernel (split-half rotation, x in
``[S, NH*HD]``, cos/sin ``[S, HD]``):

1. **ViT 2D rope** (Qwen3-VL vision tower, hd64, theta 1e4).
   Qwen3-VL has NO window attention: every block runs full attention
   over each image's 256 patch tokens (``cu_seqlens`` per image →
   batched FMHA with batch = num_images). Token order is the raw HF
   merge-grouped flatten order — there is NO window permutation
   (unlike Qwen2.5-VL / vla4b). Ground truth:
   ``Qwen3VLVisionModel.rot_pos_emb`` via
   ``qwen3vl_in_vla.preprcess_grid_thw`` (lingbot repo, :260-276).

2. **LM prefix 3D M-RoPE** (theta 5e6, mrope_section [24,20,20],
   INTERLEAVED — Qwen3-VL text). Built by reusing the groot_n17
   builders, which were validated bit-exact against HF Qwen3-VL
   (``models/groot_n17/mrope_table.py`` docstring). V2 prefix
   segments: per image [text 1 | image grid | text 1] ×3, then
   language (text), then 16 align-query tokens (text).

3. **Suffix M-RoPE** (state + 50 action tokens). The reference
   (``FlowMatchingV2._build_full_position_ids``, modeling_..._v2.py
   :741-747) assigns suffix tokens sequential positions starting at
   ``max(valid prefix position) + 1`` with all three mrope axes equal
   — i.e. exactly what ``build_position_ids_for_segments`` produces
   for a trailing text segment. With t==h==w the interleaved M-RoPE
   degenerates to plain 1D rope at theta 5e6 per channel; we still
   build it through the same mrope path so there is a single code
   path to verify.

NOTE (correction vs T2-V1): VLA-4B's joint attention used pi0-style
1D rope theta 1e4. V2 is genuinely Qwen3-VL 3D M-RoPE theta 5e6 for
BOTH prefix and suffix (``QwenvlWithExpertV2Model.apply_mrope``,
modeling_..._v2.py :275-277 → ``Qwen3VLTextRotaryEmbedding``).

DEPLOYED-NUMERICS NOTE (M1 measurement, 2026-08-25): the stock server
casts the whole policy to bf16 (deploy:214 ``self.vla.to(torch.
bfloat16)``), which also casts the HF rotary modules' non-persistent
``inv_freq`` buffers to bf16.  Every rope phase the deployed model
computes (prefix KV at fill time AND suffix q/k every step) therefore
uses bf16-ROUNDED inverse frequencies (~2^-9 relative).  At V2
position ranges this moves cos/sin by up to ~1.3e-1 vs exact-fp32
tables (measured 9.95e-2 at Se=234; reproduced exactly by rounding
inv_freq through bf16).  Stage-A hands the STOCK-roped prefix KV to
the engine loop, so the engine suffix tables must be phase-consistent
with it: ``build_joint_rope_tables`` rounds inv_freq through bf16 by
default (``inv_freq_like="stock_bf16"``).  The same applies to the
vision tables for the M2 ViT port (visual rotary is cast too).
Pass ``inv_freq_like="fp32"`` for exact tables.

Pure-PyTorch (no flash_rt_kernels import). Stage-2 gate: bitwise
comparison of these tables against the stock model's
``rotary_emb``/``rot_pos_emb`` outputs on the deployed prefix.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from flash_rt.models.groot_n17.mrope_table import (
    RopeConfig,
    apply_interleaved_mrope,
    build_cos_sin_tables,
    build_position_ids_for_segments,
    compute_inv_freq,
)

# ────────────────────────────────────────────────────────────────────
# Constants (locked by the lingbot-vla-v2-6b ckpt + Qwen3-VL-4B config)
# ────────────────────────────────────────────────────────────────────

VIS_HD = 64                  # ViT head dim (1024 / 16 heads)
VIS_THETA = 10_000.0         # Qwen3VLVisionRotaryEmbedding default
VIS_SPATIAL_MERGE = 2

LM_HD = 128
LM_ROPE = RopeConfig(
    head_dim=128,
    rope_theta=5_000_000.0,
    mrope_section=(24, 20, 20),
    attention_scaling=1.0,
    spatial_merge_size=2,
)

IMG_GRID = (1, 16, 16)       # 256px / patch16 → (t, h, w) raw patch grid
MERGED_PER_IMAGE = 64        # 16*16 / merge 2*2
BOUNDARY_TOKENS = 2          # vision_start + vision_end per image
ALIGN_TOKENS = 16            # current-task 8 + future-task 8
SUF = 51                     # state + 50 action tokens


# ────────────────────────────────────────────────────────────────────
# ViT 2D rope (per-patch, raw merge-grouped order — NO window permute)
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VisionLayout:
    """Fixed multi-image ViT bookkeeping.

    Attributes:
        cos / sin: ``(S_patches, VIS_HD)`` fp32 tables, raw HF token
            order (ready for ``rope_rotate_half_fp16`` after fp16 cast).
        image_seqlens: per-image patch-token counts (256 each at 256px)
            — feeds the batched FMHA (batch axis = num_images).
    """

    cos: torch.Tensor
    sin: torch.Tensor
    image_seqlens: list


def _rot_pos_ids_one_image(grid_h: int, grid_w: int) -> torch.Tensor:
    """(h, w) ids per patch token in merge-grouped flatten order.

    Replicates ``Qwen3VLVisionModel.rot_pos_emb`` for one (1, h, w)
    image — same merge-grouping as Qwen2.5-VL (the merge_size reshape
    is unchanged between the two vision towers).
    """
    m = VIS_SPATIAL_MERGE
    hpos = torch.arange(grid_h).unsqueeze(1).expand(-1, grid_w)
    hpos = hpos.reshape(grid_h // m, m, grid_w // m, m).permute(0, 2, 1, 3).flatten()
    wpos = torch.arange(grid_w).unsqueeze(0).expand(grid_h, -1)
    wpos = wpos.reshape(grid_h // m, m, grid_w // m, m).permute(0, 2, 1, 3).flatten()
    return torch.stack([hpos, wpos], dim=-1)  # (h*w, 2)


def _round_inv_freq(inv_freq: torch.Tensor, inv_freq_like: str) -> torch.Tensor:
    """Match the deployed model's inv_freq numerics (see module note).

    ``"stock_bf16"``: round through bf16 — what the bf16-served stock
    model actually multiplies positions by (non-persistent HF buffer is
    cast by ``.to(torch.bfloat16)``).  ``"fp32"``: exact.
    """
    if inv_freq_like == "stock_bf16":
        return inv_freq.to(torch.bfloat16).to(torch.float32)
    if inv_freq_like == "fp32":
        return inv_freq
    raise ValueError(f"unknown inv_freq_like: {inv_freq_like!r}")


def build_vision_layout(
    *,
    num_images: int = 3,
    grid_h: int = 16,
    grid_w: int = 16,
    inv_freq_like: str = "stock_bf16",
) -> VisionLayout:
    """Fixed ViT rope tables for N images (deployment: 3 × 256px)."""
    half = VIS_HD // 2                    # 32
    freq_dim = half // 2                  # 16 per axis
    inv_freq = 1.0 / (
        VIS_THETA ** (torch.arange(0, half, 2, dtype=torch.float32) / half)
    )                                     # (16,)
    inv_freq = _round_inv_freq(inv_freq, inv_freq_like)
    pos_ids = torch.cat(
        [_rot_pos_ids_one_image(grid_h, grid_w) for _ in range(num_images)],
        dim=0,
    )                                     # (S, 2)
    max_grid = max(grid_h, grid_w)
    table = torch.outer(torch.arange(max_grid, dtype=torch.float32), inv_freq)
    freqs = table[pos_ids].flatten(1)     # (S, 32) = [h(16) | w(16)]
    emb = torch.cat([freqs, freqs], dim=-1)  # (S, 64) split-half duplicate
    assert freqs.shape[1] == 2 * freq_dim
    return VisionLayout(
        cos=emb.cos(),
        sin=emb.sin(),
        image_seqlens=[grid_h * grid_w] * num_images,
    )


# ────────────────────────────────────────────────────────────────────
# LM / expert 3D M-RoPE (prefix + suffix)
# ────────────────────────────────────────────────────────────────────

def build_prefix_segments(
    n_lang: int,
    *,
    num_images: int = 3,
    align_tokens: int = ALIGN_TOKENS,
) -> tuple[list, list, list]:
    """(lengths, kinds, grids) for the V2 prefix, dense (pads dropped).

    Reference layout (FlowMatchingV2.embed_prefix, modeling_..._v2.py
    :543-708 with qwen3vl_use_vision_boundaries=True and the shipped
    align config → segments [language, current_depth, future_depth]):

        [start(1) img(64) end(1)] × num_images | lang(n) | align(16)

    Language pads (right-padded to tokenizer_max_length=72 in the
    reference) are fully masked there → dropped here; Qwen3-VL
    ``get_rope_index`` consumes the attention mask, so dense positions
    are identical for every surviving token.
    """
    lengths: list[int] = []
    kinds: list[str] = []
    grids: list = []
    for _ in range(num_images):
        lengths += [1, MERGED_PER_IMAGE, 1]
        kinds += ["text", "image", "text"]
        grids += [None, IMG_GRID, None]
    lengths.append(n_lang)
    kinds.append("text")
    grids.append(None)
    if align_tokens:
        lengths.append(align_tokens)
        kinds.append("text")
        grids.append(None)
    return lengths, kinds, grids


def build_joint_rope_tables(
    n_lang: int,
    *,
    num_images: int = 3,
    align_tokens: int = ALIGN_TOKENS,
    suffix_len: int = SUF,
    dtype: torch.dtype = torch.float16,
    inv_freq_like: str = "stock_bf16",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """cos/sin for the full [prefix | suffix] joint sequence.

    Returns (pre_cos, pre_sin, suf_cos, suf_sin); prefix tables
    ``(Se, 128)``, suffix tables ``(51, 128)``. The suffix is appended
    as one more text segment, which reproduces
    ``_build_full_position_ids`` exactly (offset = max valid prefix
    position + 1, all three axes equal).

    ``inv_freq_like="stock_bf16"`` (default) makes the tables
    phase-consistent with the bf16-served stock model — REQUIRED for
    Stage-A, where the prefix KV rows were roped by the stock model
    (see the deployed-numerics module note; exact-fp32 tables sit up to
    ~1.3e-1 away in cos and fail the M1 envelope).
    """
    lengths, kinds, grids = build_prefix_segments(
        n_lang, num_images=num_images, align_tokens=align_tokens)
    lengths.append(suffix_len)
    kinds.append("text")
    grids.append(None)
    pos = build_position_ids_for_segments(
        segment_lengths=lengths,
        segment_kinds=kinds,
        segment_grids=grids,
        cfg=LM_ROPE,
    )                                              # (3, 1, S)
    if inv_freq_like == "fp32":
        cos, sin = build_cos_sin_tables(pos, LM_ROPE, dtype=dtype)
    else:
        # groot's assembly with the deployed (bf16-rounded) inv_freq
        inv_freq = _round_inv_freq(compute_inv_freq(LM_ROPE),
                                   inv_freq_like)
        freqs = inv_freq[None, None, None, :] \
            * pos[..., None].to(torch.float32)     # (3, 1, S, 64)
        freqs_t = apply_interleaved_mrope(freqs, LM_ROPE.mrope_section)
        emb = torch.cat([freqs_t, freqs_t], dim=-1)
        cos = (emb.cos() * LM_ROPE.attention_scaling).to(dtype).squeeze(0)
        sin = (emb.sin() * LM_ROPE.attention_scaling).to(dtype).squeeze(0)
    se = int(sum(lengths[:-1]))
    return (cos[:se], sin[:se],
            cos[se:].contiguous(), sin[se:].contiguous())


__all__ = [
    "VisionLayout",
    "build_vision_layout",
    "build_prefix_segments",
    "build_joint_rope_tables",
    "LM_ROPE",
    "VIS_HD",
    "LM_HD",
    "MERGED_PER_IMAGE",
    "ALIGN_TOKENS",
    "SUF",
]

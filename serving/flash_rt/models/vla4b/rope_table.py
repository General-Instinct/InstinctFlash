"""Host-side RoPE cos/sin table construction for LingBot-VLA-4B.

Two table families, both consumed unchanged by the existing
``rope_rotate_half_fp16`` kernel (``csrc/kernels/rope_qwen3.cu``:
split-half rotation, x in ``[S, NH*HD]``, cos/sin ``[S, HD]`` with the
first ``HD/2`` entries per row used and the second half a duplicate):

1. **ViT 2D rope** (Qwen2.5-VL vision tower, hd80, theta 1e4).
   Per-token frequencies are ``[h_freqs(20) | w_freqs(20)]`` in the
   *window-permuted* token order the vision blocks run in — the
   window_index permutation is baked into the table at build time so
   the pipeline never permutes at runtime.

2. **LM/expert 1D rope** (joint attention, hd128, theta 1e4).
   Ground truth is ``lingbotvla...pi0.utils.apply_rope`` — plain 1D
   positions (``max_wavelength=10_000``), split-half rotation, fp32
   trig. NOT the HF Qwen2.5-VL 3D M-RoPE: the VLA's custom joint
   forward never calls the HF text rotary path.

This module is pure-PyTorch (no flash_rt_kernels import). Tables are
built once per ``set_prompt`` (LM) / once per construction (ViT — the
224px grid is fixed) and uploaded as fp16 device tensors.

Validated against the torch reference in
``/home/ubuntu/iwm_distill/thor_t2/p1_verify_rope.py`` (fp32 bitwise
for the tables; see the gate JSON for the applied-rotation deltas).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# ────────────────────────────────────────────────────────────────────
# Constants (locked by the lingbot-vla-4b checkpoint + base config)
# ────────────────────────────────────────────────────────────────────

VIS_HD = 80              # ViT head dim (1280 / 16 heads)
VIS_THETA = 10_000.0
VIS_SPATIAL_MERGE = 2    # spatial_merge_size
VIS_MERGE_UNIT = 4       # spatial_merge_size ** 2
VIS_WINDOW = 112         # window_size (pixels)
VIS_PATCH = 14           # patch_size

LM_HD = 128              # LM + expert head dim (joint attention geometry)
LM_THETA = 10_000.0      # apply_rope max_wavelength default — NOT 1e6


# ────────────────────────────────────────────────────────────────────
# ViT 2D rope + window bookkeeping
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VisionLayout:
    """Window/permutation bookkeeping for a fixed multi-image grid.

    All index tensors are CPU int64; the caller uploads what it needs.

    Attributes:
        cos / sin: ``(S_patches, VIS_HD)`` fp32 tables in the
            window-permuted order (ready for ``rope_rotate_half_fp16``
            after ``.to(torch.float16)``).
        window_index: ``(S_patches/4,)`` merge-unit permutation applied
            to patch tokens before block 0 (HF semantics).
        patch_perm: ``(S_patches,)`` patch-level gather indices:
            ``x_permuted = x[patch_perm]``.
        reverse_index: ``(S_patches/4,)`` merged-token gather that
            restores raw (row-major merged grid) order after the
            merger: ``out = merged[reverse_index]``.
        window_seqlens: list of per-window token counts (uniform 64 for
            224px inputs) — feeds the batched FMHA (batch axis).
        image_seqlens: per-image token counts (256 each) for the
            fullatt layers.
    """

    cos: torch.Tensor
    sin: torch.Tensor
    window_index: torch.Tensor
    patch_perm: torch.Tensor
    reverse_index: torch.Tensor
    window_seqlens: list
    image_seqlens: list


def _rot_pos_ids_one_image(grid_h: int, grid_w: int) -> torch.Tensor:
    """(h, w) position ids per patch token in merge-grouped flatten order.

    Replicates ``Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb``
    for one (1, h, w) image.
    """
    m = VIS_SPATIAL_MERGE
    hpos = torch.arange(grid_h).unsqueeze(1).expand(-1, grid_w)
    hpos = hpos.reshape(grid_h // m, m, grid_w // m, m).permute(0, 2, 1, 3).flatten()
    wpos = torch.arange(grid_w).unsqueeze(0).expand(grid_h, -1)
    wpos = wpos.reshape(grid_h // m, m, grid_w // m, m).permute(0, 2, 1, 3).flatten()
    return torch.stack([hpos, wpos], dim=-1)  # (h*w, 2)


def _window_index_one_image(grid_h: int, grid_w: int) -> tuple[torch.Tensor, list]:
    """Merge-unit window permutation + per-window merge-unit counts.

    Replicates ``get_window_index`` for one (1, h, w) image (grid_t=1).
    Returns (index (llm_h*llm_w,), seqlens_in_merge_units list — zeros
    already dropped).
    """
    m = VIS_SPATIAL_MERGE
    llm_h, llm_w = grid_h // m, grid_w // m
    vw = VIS_WINDOW // m // VIS_PATCH  # merger window size in llm cells (4)
    index = torch.arange(llm_h * llm_w).reshape(1, llm_h, llm_w)
    pad_h = vw - llm_h % vw
    pad_w = vw - llm_w % vw
    nwh = (llm_h + pad_h) // vw
    nww = (llm_w + pad_w) // vw
    index_p = torch.nn.functional.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
    index_p = index_p.reshape(1, nwh, vw, nww, vw)
    index_p = index_p.permute(0, 1, 3, 2, 4).reshape(1, nwh * nww, vw, vw)
    seqlens = (index_p != -100).sum([2, 3]).reshape(-1)
    index_p = index_p.reshape(-1)
    index_new = index_p[index_p != -100]
    seqlens = [int(s) for s in seqlens.tolist() if s > 0]
    return index_new, seqlens


def build_vision_layout(
    *,
    num_images: int = 3,
    grid_h: int = 16,
    grid_w: int = 16,
) -> VisionLayout:
    """Build the fixed ViT rope tables + permutations for N images.

    Default is the deployment shape: 3 cameras × 224 px → (1, 16, 16)
    grids, 256 patch tokens per image.
    """
    tokens_per_image = grid_h * grid_w
    units_per_image = tokens_per_image // VIS_MERGE_UNIT

    # ── rot_pos_emb: (S, 40) freqs in raw merge-grouped order ──
    half = VIS_HD // 2                    # 40
    freq_dim = half // 2                  # 20 per axis
    inv_freq = 1.0 / (VIS_THETA ** (torch.arange(0, half, 2, dtype=torch.float32) / half))
    pos_ids = torch.cat(
        [_rot_pos_ids_one_image(grid_h, grid_w) for _ in range(num_images)], dim=0
    )  # (S, 2)
    max_grid = max(grid_h, grid_w)
    seq = torch.arange(max_grid, dtype=torch.float32)
    table_full = torch.outer(seq, inv_freq)          # (max_grid, 20)
    freqs = table_full[pos_ids]                      # (S, 2, 20)
    freqs = freqs.flatten(1)                         # (S, 40) = [h | w]

    # ── window_index over all images (merge-unit granularity) ──
    win_parts, win_seqlens = [], []
    for i in range(num_images):
        idx, seqlens = _window_index_one_image(grid_h, grid_w)
        win_parts.append(idx + i * units_per_image)
        win_seqlens.extend(s * VIS_MERGE_UNIT for s in seqlens)
    window_index = torch.cat(win_parts, dim=0)       # (S/4,)

    # ── permute freqs by window_index on merge-unit groups ──
    S = num_images * tokens_per_image
    freqs = freqs.reshape(S // VIS_MERGE_UNIT, VIS_MERGE_UNIT, -1)
    freqs = freqs[window_index].reshape(S, -1)       # (S, 40)

    emb = torch.cat([freqs, freqs], dim=-1)          # (S, 80)
    cos = emb.cos()
    sin = emb.sin()

    patch_perm = (
        window_index[:, None] * VIS_MERGE_UNIT
        + torch.arange(VIS_MERGE_UNIT)[None, :]
    ).reshape(-1)                                    # (S,)
    reverse_index = torch.argsort(window_index)

    return VisionLayout(
        cos=cos,
        sin=sin,
        window_index=window_index,
        patch_perm=patch_perm,
        reverse_index=reverse_index,
        window_seqlens=win_seqlens,
        image_seqlens=[tokens_per_image] * num_images,
    )


# ────────────────────────────────────────────────────────────────────
# LM / expert 1D rope (apply_rope ground truth)
# ────────────────────────────────────────────────────────────────────

def build_lm_rope_tables(
    positions: torch.Tensor,
    *,
    head_dim: int = LM_HD,
    theta: float = LM_THETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables ``(S, head_dim)`` fp32 for 1D positions.

    Math source: ``apply_rope`` in lingbotvla pi0/utils.py —
    ``timescale = theta ** (2i/d)``, ``radians = pos / timescale``,
    split-half rotation. The second half of each row duplicates the
    first (kernel reads only the first ``head_dim/2`` entries but
    expects row width ``head_dim``).
    """
    positions = positions.to(torch.float32)
    d_half = head_dim // 2
    freq_exp = (2.0 / head_dim) * torch.arange(
        d_half, dtype=torch.float32, device=positions.device)
    inv_timescale = 1.0 / (theta ** freq_exp)
    radians = torch.outer(positions, inv_timescale)  # (S, d_half)
    emb = torch.cat([radians, radians], dim=-1)      # (S, head_dim)
    return emb.cos(), emb.sin()


__all__ = [
    "VisionLayout",
    "build_vision_layout",
    "build_lm_rope_tables",
    "VIS_HD",
    "LM_HD",
]

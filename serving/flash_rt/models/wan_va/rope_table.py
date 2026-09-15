"""Host-side RoPE tables for LingBot-VA (wan_va) — f64, permuted split-half.

Ground truth (`wan_va/modules/model.py`):
  * ``WanRotaryPosEmbed`` (:243-286): head_dim 128 split f 44 / h 42 / w 42
    (:258-260); frequency bases are computed in float64 and stored as PLAIN
    attributes (:262-276) — ``.to(bfloat16)`` does NOT touch them (verified on
    CPU 2026-08-28), so the deployed phases are float64-exact. This is the
    OPPOSITE of the T2-V2 bf16-inv_freq trap: do NOT round these tables.
  * ``forward`` (:278-286): freqs = grid * base (f64), cat, ``.float()``,
    ``torch.polar`` -> complex64 ``freqs_cis``.
  * ``apply_rotary_emb`` (:434-440): x -> float64 -> view_as_complex over
    ADJACENT CHANNEL PAIRS -> multiply -> back to x.dtype. i.e. the rotation
    convention is complex-pair INTERLEAVED, not rotate-half.
  * grids (`wan_va/utils/utils.py::get_mesh_id` :33-53): video tokens get
    integer (f, h, w) with f offset by ``frame_st_id`` (int64 grid); ACTION
    tokens get FRACTIONAL frames f + k/17 (k = 1..16) and h = w = -1
    (float32 grid, cat-promotion). A 4th grid row carries t but the rope
    consumes rows 0-2 only — dead at inference.

Engine mapping — the permutation repack (mapping_memo §0.1/§A.2):
the fixed per-head channel permutation P (2j -> j, 2j+1 -> j+64) turns the
interleaved pair convention into split-half EXACTLY: for permuted channels,
``y[j]' = y[j]*cos_j - y[j+64]*sin_j`` and ``y[j+64]' = y[j+64]*cos_j +
y[j]*sin_j`` are literally the complex multiply's two components. Because the
SAME permutation is applied to q and k (weight rows + biases + the q/k RMS
norm weight vectors — RMS statistics over the full 3072 are permutation-
invariant), every attention dot product is unchanged, v/o are untouched, and
the existing ``rope_rotate_half_fp16`` kernel consumes cos/sin tables built
as ``cat([freqs, freqs])``. ``self_check()`` proves the equivalence at f64
(<= 1e-12) and the dot-product invariance.

Pure PyTorch, no flash_rt_kernels import. M0 gate: bitwise comparison of
``freqs_cis`` (and cos/sin extraction) against the stock
``WanRotaryPosEmbed`` on the deployed grids at several ``frame_st_id``.
"""
from __future__ import annotations

import torch

HD = 128
THETA = 10_000.0
F_DIM = HD - 2 * (HD // 3)          # 44
H_DIM = HD // 3                     # 42
W_DIM = HD // 3                     # 42
HALF = HD // 2                      # 64 = 22 + 21 + 21 frequency channels


def freqs_base() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """float64 bases, byte-for-byte the stock `_precompute_freqs_base`."""
    f = 1.0 / (THETA ** (torch.arange(0, F_DIM, 2)[: F_DIM // 2].double() / F_DIM))
    h = 1.0 / (THETA ** (torch.arange(0, H_DIM, 2)[: H_DIM // 2].double() / H_DIM))
    w = 1.0 / (THETA ** (torch.arange(0, W_DIM, 2)[: W_DIM // 2].double() / W_DIM))
    return f, h, w


# ────────────────────────────────────────────────────────────────────
# Grids (get_mesh_id semantics, minus the dead t row)
# ────────────────────────────────────────────────────────────────────

def video_grid(num_frames: int, grid_h: int = 12, grid_w: int = 10,
               frame_st_id: int = 0) -> torch.Tensor:
    """(3, F*h*w) int64 — integer (f, h, w) per post-patch video token.

    ``grid_h/grid_w`` are POST-patch (latent 24x20 / patch 2x2 -> 12x10).
    Token order is frame-major (f, h, w) — the property that makes the
    cycle-0 mixed-t modulation two contiguous row spans.
    """
    f_idx = torch.arange(frame_st_id, num_frames + frame_st_id)
    ff, hh, ww = torch.meshgrid(f_idx, torch.arange(grid_h),
                                torch.arange(grid_w), indexing="ij")
    return torch.stack([ff, hh, ww], dim=0).flatten(1)


def action_grid(num_frames: int, action_per_frame: int = 16,
                frame_st_id: int = 0) -> torch.Tensor:
    """(3, F*apf) float32 — fractional frames f + k/(apf+1), h = w = -1.

    Byte-for-byte the ``action=True`` branch of ``get_mesh_id``:
    ``ff_offset = cumsum(ones(apf)) / (apf + 1)`` (k = 1..apf, so the last
    action of frame f sits at f + 16/17 and frame f+1 starts at f + 1 + 1/17).
    """
    f_idx = torch.arange(frame_st_id, num_frames + frame_st_id)
    ff, hh, ww = torch.meshgrid(f_idx, torch.arange(action_per_frame),
                                torch.arange(1), indexing="ij")
    off = (torch.ones([action_per_frame]).cumsum(0)
           / (action_per_frame + 1)).view(1, -1, 1)
    ff = ff + off
    hh = torch.ones_like(hh) * -1
    ww = torch.ones_like(ww) * -1
    g = torch.cat([ff.unsqueeze(0), hh.unsqueeze(0).to(ff.dtype),
                   ww.unsqueeze(0).to(ff.dtype)], dim=0).flatten(1)
    return g.to(torch.float32)


# ────────────────────────────────────────────────────────────────────
# Stock-path freqs (reference) and engine tables
# ────────────────────────────────────────────────────────────────────

def freqs_f64(grid: torch.Tensor) -> torch.Tensor:
    """(S, 64) float64 phase angles, stock op order (grid * f64 base, cat)."""
    fb, hb, wb = freqs_base()
    f = grid[0].unsqueeze(-1) * fb
    h = grid[1].unsqueeze(-1) * hb
    w = grid[2].unsqueeze(-1) * wb
    return torch.cat([f, h, w], dim=-1)


def freqs_cis_stock(grid: torch.Tensor) -> torch.Tensor:
    """complex64 (S, 64) — byte-for-byte ``WanRotaryPosEmbed.forward``."""
    fr = freqs_f64(grid).float()
    return torch.polar(torch.ones_like(fr), fr)


def build_cos_sin(grid: torch.Tensor, dtype: torch.dtype = torch.float16
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """(S, 128) cos/sin split-half tables for ``rope_rotate_half_fp16``.

    DEPLOYED-NUMERICS (found by the wan_ref self-check, 2026-08-28): the
    stock forward computes the phase ANGLE in f64 but rounds it to fp32
    BEFORE ``torch.polar`` (model.py:283 ``.float()``), so the deployed
    cos/sin values are fp32(cos(fp32(angle))). The tables reproduce that
    chain exactly (f64-phase tables sit ~1e-7 away — wrong direction).
    Channel j and j+64 share frequency j — valid in the P-permuted layout.
    """
    fr = freqs_f64(grid).float()          # model:283 rounds the angle here
    # use torch.polar — the IDENTICAL op the stock forward runs; its
    # cos/sin differ from Tensor.cos()/.sin() by isolated fp32 ulps
    # (measured 6e-8), so .cos() tables would not be byte-faithful
    c = torch.polar(torch.ones_like(fr), fr)
    cos = torch.cat([c.real, c.real], dim=-1)
    sin = torch.cat([c.imag, c.imag], dim=-1)
    return cos.to(dtype), sin.to(dtype)


# ────────────────────────────────────────────────────────────────────
# The interleaved -> split-half permutation (repack helpers)
# ────────────────────────────────────────────────────────────────────

def pair_to_half_perm(hd: int = HD) -> torch.Tensor:
    """Index p with ``x_perm = x[..., p]``: p[j] = 2j, p[j+hd/2] = 2j+1."""
    half = hd // 2
    p = torch.empty(hd, dtype=torch.long)
    p[:half] = torch.arange(half) * 2
    p[half:] = torch.arange(half) * 2 + 1
    return p


def permute_qk_rows(w: torch.Tensor, nh: int = 24, hd: int = HD) -> torch.Tensor:
    """Permute OUTPUT rows of a q/k projection weight [nh*hd, in] (or a bias
    / norm vector [nh*hd]) into the split-half layout, per head."""
    p = pair_to_half_perm(hd)
    if w.dim() == 1:
        return w.view(nh, hd)[:, p].reshape(-1).contiguous()
    return w.view(nh, hd, -1)[:, p, :].reshape(nh * hd, -1).contiguous()


def rotate_half_ref(x_perm: torch.Tensor, cos: torch.Tensor,
                    sin: torch.Tensor) -> torch.Tensor:
    """Reference of the split-half rotation the kernel performs.
    x_perm: (S, NH, HD) in the permuted layout; cos/sin: (S, HD)."""
    half = x_perm.shape[-1] // 2
    x1, x2 = x_perm[..., :half], x_perm[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x_perm * cos[:, None, :].to(x_perm.dtype) \
        + rot * sin[:, None, :].to(x_perm.dtype)


def apply_rotary_stock(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Literal transcription of the stock ``apply_rotary_emb``
    (model.py:434-440). x: (B, S, NH, HD); freqs_cis: (S, 64) -> broadcast."""
    xc = torch.view_as_complex(
        x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs_cis[None, :, None, :]).flatten(3)
    return out.to(x.dtype)


# ────────────────────────────────────────────────────────────────────
# Self-check (CPU, no GPU, no flash_rt_kernels)
# ────────────────────────────────────────────────────────────────────

def self_check() -> dict:
    torch.manual_seed(0)
    report = {}

    gv = video_grid(2, frame_st_id=4)
    ga = action_grid(2, frame_st_id=4)
    assert gv.dtype == torch.int64 and gv.shape == (3, 240)
    assert ga.dtype == torch.float32 and ga.shape == (3, 32)
    # fractional action frames: first token of frame 4 at 4 + 1/17
    assert abs(float(ga[0, 0]) - (4 + 1 / 17)) < 1e-6
    assert float(ga[1, 0]) == -1.0 and float(ga[2, 0]) == -1.0

    for name, grid in (("video", gv), ("action", ga)):
        fc = freqs_cis_stock(grid)
        assert fc.dtype == torch.complex64 and fc.shape == (grid.shape[1], HALF)

        # equivalence: stock complex-pair rotation == permuted rotate-half,
        # both in float64 (isolates convention from precision)
        S = grid.shape[1]
        x = torch.randn(1, S, 24, HD, dtype=torch.float64)
        y_stock = apply_rotary_stock(x, torch.polar(
            torch.ones(S, HALF, dtype=torch.float64), freqs_f64(grid)))
        p = pair_to_half_perm()
        fr = freqs_f64(grid)
        emb = torch.cat([fr, fr], dim=-1)
        y_half = rotate_half_ref(x[0, :, :, :][..., :][:, :, p].contiguous(),
                                 emb.cos(), emb.sin())
        diff = (y_stock[0][:, :, p] - y_half).abs().max().item()
        assert diff <= 1e-12, (name, diff)
        report[f"{name}_rotation_equiv_f64_maxdiff"] = diff

        # dot-product invariance under P (attention is unchanged; the
        # nonzero residue is f64 reduction order, not the permutation)
        q = torch.randn(S, HD, dtype=torch.float64)
        k = torch.randn(S, HD, dtype=torch.float64)
        d0 = q @ k.t()
        d1 = q[:, p] @ k[:, p].t()
        report[f"{name}_dot_invariance_maxdiff"] = (d0 - d1).abs().max().item()
        assert report[f"{name}_dot_invariance_maxdiff"] <= 1e-12

        cos, sin = build_cos_sin(grid)
        assert cos.shape == (S, HD) and cos.dtype == torch.float16
        # fp32 tables must be BITWISE the deployed complex64 phases
        # (polar(1, theta32) == (theta32.cos(), theta32.sin()))
        c64 = freqs_cis_stock(grid)
        cos32, sin32 = build_cos_sin(grid, dtype=torch.float32)
        assert torch.equal(cos32[:, :HALF], c64.real)
        assert torch.equal(sin32[:, :HALF], c64.imag)
        report[f"{name}_table_vs_stock_phases"] = "bitwise"

    # weight-repack round trip: permuting rows then feeding permuted tables
    # equals stock projection + stock rotation (single random head)
    w = torch.randn(HD, 16, dtype=torch.float64)
    xin = torch.randn(5, 16, dtype=torch.float64)
    grid = video_grid(1, grid_h=1, grid_w=5)
    q_stock = apply_rotary_stock((xin @ w.t()).view(1, 5, 1, HD),
                                 torch.polar(torch.ones(5, HALF,
                                                        dtype=torch.float64),
                                             freqs_f64(grid)))[0, :, 0]
    wp = permute_qk_rows(w, nh=1)
    fr = freqs_f64(grid)
    emb = torch.cat([fr, fr], dim=-1)
    q_eng = rotate_half_ref((xin @ wp.t()).view(5, 1, HD),
                            emb.cos(), emb.sin())[:, 0]
    p = pair_to_half_perm()
    diff = (q_stock[:, p] - q_eng).abs().max().item()
    assert diff <= 1e-12, diff
    report["weight_repack_equiv_f64_maxdiff"] = diff
    return report


__all__ = [
    "HD", "THETA", "F_DIM", "H_DIM", "W_DIM", "HALF",
    "freqs_base", "video_grid", "action_grid",
    "freqs_f64", "freqs_cis_stock", "build_cos_sin",
    "pair_to_half_perm", "permute_qk_rows", "rotate_half_ref",
    "apply_rotary_stock", "self_check",
]

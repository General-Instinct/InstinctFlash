"""Engine-structured torch reference for the wan_va DiT block (M1 gate arm).

Two implementations of ONE block forward, compared in ``self_check()``:

* ``stock_block_forward`` — a literal transcription of
  ``WanTransformerBlock.forward`` + the ``WanAttention`` inference path
  (`wan_va/modules/model.py` :414-465, :515-566): FP32LayerNorm modulation,
  full-3072 q/k RMSNorm before the head split, f64 complex-pair rope,
  sdpa over [KV window | current], fp32 gated residuals, cross attention
  recomputing text K/V, gelu-tanh ffn.

* ``engine_block_forward`` — the SAME math restructured the way the engine
  runs it: modulation from a precomputed step-table row (mod = per-block
  scale_shift_table + timestep_proj, fp32), q/k projections with
  P-PERMUTED weight rows + split-half rotate (rope_table.pair_to_half_perm),
  cross K/V HOISTED (computed once, reused), explicit contiguous KV window
  (the linear slab view).

Equivalence claims proven by the self-check:
  f64 arms agree to <= 1e-10 (convention equivalence — permutation, table
  precompute, hoisting are exact restructurings); bf16 arms are reported
  (RMS reduction-order under permutation can flip isolated bf16 ulps —
  precision-tier, covered by the M1/M2 gates, NOT a convention error).

CPU-runnable, no flash_rt_kernels, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from flash_rt.models.wan_va.rope_table import (
    apply_rotary_stock,
    build_cos_sin,
    freqs_cis_stock,
    pair_to_half_perm,
    permute_qk_rows,
    rotate_half_ref,
    video_grid,
)

D_REF = 3072
NH_REF = 24
HD = 128
EPS = 1e-6


@dataclass
class WanBlockRefWeights:
    """Raw stock-layout tensors for one block (any float dtype)."""
    q_w: torch.Tensor; q_b: torch.Tensor
    k_w: torch.Tensor; k_b: torch.Tensor
    v_w: torch.Tensor; v_b: torch.Tensor
    nq_w: torch.Tensor; nk_w: torch.Tensor
    o_w: torch.Tensor; o_b: torch.Tensor
    cq_w: torch.Tensor; cq_b: torch.Tensor
    ck_w: torch.Tensor; ck_b: torch.Tensor
    cv_w: torch.Tensor; cv_b: torch.Tensor
    cnq_w: torch.Tensor; cnk_w: torch.Tensor
    co_w: torch.Tensor; co_b: torch.Tensor
    n2_w: torch.Tensor; n2_b: torch.Tensor
    ff1_w: torch.Tensor; ff1_b: torch.Tensor
    ff2_w: torch.Tensor; ff2_b: torch.Tensor
    sst: torch.Tensor            # scale_shift_table [1, 6, D]

    @staticmethod
    def random(d: int = D_REF, ffn: int = 512,
               dtype: torch.dtype = torch.bfloat16) -> "WanBlockRefWeights":
        def lin(o, i):
            return (torch.randn(o, i) / (i ** 0.5)).to(dtype), \
                   (torch.randn(o) / 30).to(dtype)
        q = lin(d, d); k = lin(d, d); v = lin(d, d); o = lin(d, d)
        cq = lin(d, d); ck = lin(d, d); cv = lin(d, d); co = lin(d, d)
        f1 = lin(ffn, d); f2 = lin(d, ffn)
        return WanBlockRefWeights(
            q_w=q[0], q_b=q[1], k_w=k[0], k_b=k[1], v_w=v[0], v_b=v[1],
            nq_w=(1 + torch.randn(d) / 20).to(dtype),
            nk_w=(1 + torch.randn(d) / 20).to(dtype),
            o_w=o[0], o_b=o[1],
            cq_w=cq[0], cq_b=cq[1], ck_w=ck[0], ck_b=ck[1],
            cv_w=cv[0], cv_b=cv[1],
            cnq_w=(1 + torch.randn(d) / 20).to(dtype),
            cnk_w=(1 + torch.randn(d) / 20).to(dtype),
            co_w=co[0], co_b=co[1],
            n2_w=(1 + torch.randn(d) / 20).to(dtype),
            n2_b=(torch.randn(d) / 20).to(dtype),
            ff1_w=f1[0], ff1_b=f1[1], ff2_w=f2[0], ff2_b=f2[1],
            sst=(torch.randn(1, 6, d) / d ** 0.5).to(dtype))

    def to(self, dtype):
        return WanBlockRefWeights(**{
            k: v.to(dtype) for k, v in self.__dict__.items()})


def _up(x: torch.Tensor) -> torch.Tensor:
    """The stock code's ``.float()`` upcast — kept at f64 in the f64
    self-check arm so convention equivalence is measured at 1e-12, not at
    fp32 reduction-order noise."""
    return x if x.dtype == torch.float64 else x.float()


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """torch.nn.RMSNorm semantics: fp32 internal, cast back (verified)."""
    xf = _up(x)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * _up(w)
    return y.to(x.dtype)


def _ln_no_affine(x32: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x32, (x32.shape[-1],), eps=EPS)


def _sdpa(q, k, v):
    """custom_sdpa (model:37-40): (B,S,NH,HD) in/out, default scale."""
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


# ────────────────────────────────────────────────────────────────────
# Arm 1 — literal stock transcription
# ────────────────────────────────────────────────────────────────────

def stock_block_forward(w: WanBlockRefWeights, x: torch.Tensor,
                        text_hidden: torch.Tensor,
                        timestep_proj: torch.Tensor,
                        freqs_cis: torch.Tensor,
                        win_k: torch.Tensor, win_v: torch.Tensor,
                        nh: int = NH_REF):
    """x (1,S,D); text_hidden (1,T,D); timestep_proj (6,D) fp32 (uniform-t
    forward); freqs_cis (S,64) complex; win_k/win_v (1,W,nh,hd) = live pool
    rows in presentation order. Returns (x_out, k_cur, v_cur)."""
    table = (_up(w.sst)[0] + timestep_proj[None].squeeze(0).to(_up(w.sst).dtype))  # (6, D)
    shift, scale, gate, c_shift, c_scale, c_gate = table.unbind(0)

    xn = (_ln_no_affine(_up(x)) * (1. + scale) + shift).type_as(x)
    q = _rms(xn @ w.q_w.t() + w.q_b, w.nq_w).unflatten(2, (nh, -1))
    k = _rms(xn @ w.k_w.t() + w.k_b, w.nk_w).unflatten(2, (nh, -1))
    v = (xn @ w.v_w.t() + w.v_b).unflatten(2, (nh, -1))
    q = apply_rotary_stock(q, freqs_cis)
    k = apply_rotary_stock(k, freqs_cis)
    k_all = torch.cat([win_k, k], dim=1)
    v_all = torch.cat([win_v, v], dim=1)
    attn = _sdpa(q, k_all, v_all).flatten(2, 3).type_as(q)
    attn = attn @ w.o_w.t() + w.o_b
    x = (_up(x) + _up(attn) * gate).type_as(x)

    # norm2: FP32LayerNorm WITH affine (fp32 weights at deploy, bf16 values)
    xn = (F.layer_norm(_up(x), (x.shape[-1],), _up(w.n2_w),
                       _up(w.n2_b), EPS)).type_as(x)
    cq = _rms(xn @ w.cq_w.t() + w.cq_b, w.cnq_w).unflatten(2, (nh, -1))
    ck = _rms(text_hidden @ w.ck_w.t() + w.ck_b, w.cnk_w).unflatten(2, (nh, -1))
    cv = (text_hidden @ w.cv_w.t() + w.cv_b).unflatten(2, (nh, -1))
    cattn = _sdpa(cq, ck, cv).flatten(2, 3).type_as(cq)
    x = x + (cattn @ w.co_w.t() + w.co_b)

    xn = (_ln_no_affine(_up(x)) * (1. + c_scale) + c_shift).type_as(x)
    h = F.gelu(xn @ w.ff1_w.t() + w.ff1_b, approximate="tanh")
    ff = h @ w.ff2_w.t() + w.ff2_b
    x = (_up(x) + _up(ff) * c_gate).type_as(x)
    return x, k, v


# ────────────────────────────────────────────────────────────────────
# Arm 2 — engine-structured (tables + permuted rope + hoisted cross KV)
# ────────────────────────────────────────────────────────────────────

def repack(w: WanBlockRefWeights, nh: int = NH_REF) -> dict:
    """The frontend's rope repack: P-permute q/k projection rows + biases +
    q/k norm vectors; precompute the hoisted cross K/V weights unchanged."""
    return {
        "q_w": permute_qk_rows(w.q_w, nh), "q_b": permute_qk_rows(w.q_b, nh),
        "k_w": permute_qk_rows(w.k_w, nh), "k_b": permute_qk_rows(w.k_b, nh),
        "nq_w": permute_qk_rows(w.nq_w, nh),
        "nk_w": permute_qk_rows(w.nk_w, nh),
    }


def hoist_cross_kv(w: WanBlockRefWeights, text_hidden: torch.Tensor,
                   nh: int = NH_REF):
    """Episode-scope cross K/V (post-norm_k K, raw V) — byte-for-byte the
    shipped conditioning-prefill projection (lingbot_install.py:124-129)."""
    ck = _rms(text_hidden @ w.ck_w.t() + w.ck_b, w.cnk_w).unflatten(2, (nh, -1))
    cv = (text_hidden @ w.cv_w.t() + w.cv_b).unflatten(2, (nh, -1))
    return ck, cv


def engine_block_forward(w: WanBlockRefWeights, rp: dict, x: torch.Tensor,
                         cross_k: torch.Tensor, cross_v: torch.Tensor,
                         mod_row: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor,
                         win_k_perm: torch.Tensor, win_v: torch.Tensor,
                         nh: int = NH_REF):
    """mod_row (6,D) fp32 = step-table row (sst + proj, precomputed);
    cos/sin (S,HD) split-half tables; win_k_perm holds P-permuted roped keys
    (the engine slab layout). Rope applied in f64 here to isolate the
    convention (the device kernel is fp16 — gate-covered)."""
    shift, scale, gate, c_shift, c_scale, c_gate = mod_row.unbind(0)

    xn = (_ln_no_affine(_up(x)) * (1. + scale) + shift).type_as(x)
    q = _rms(xn @ rp["q_w"].t() + rp["q_b"], rp["nq_w"]).unflatten(2, (nh, -1))
    k = _rms(xn @ rp["k_w"].t() + rp["k_b"], rp["nk_w"]).unflatten(2, (nh, -1))
    v = (xn @ w.v_w.t() + w.v_b).unflatten(2, (nh, -1))
    q = rotate_half_ref(q[0].to(torch.float64), cos.double(),
                        sin.double()).to(q.dtype)[None]
    k = rotate_half_ref(k[0].to(torch.float64), cos.double(),
                        sin.double()).to(k.dtype)[None]
    k_all = torch.cat([win_k_perm, k], dim=1)
    v_all = torch.cat([win_v, v], dim=1)
    attn = _sdpa(q, k_all, v_all).flatten(2, 3).type_as(q)
    attn = attn @ w.o_w.t() + w.o_b
    x = (_up(x) + _up(attn) * gate).type_as(x)   # gate_row_mul_residual

    xn = (F.layer_norm(_up(x), (x.shape[-1],), _up(w.n2_w),
                       _up(w.n2_b), EPS)).type_as(x)
    cq = _rms(xn @ w.cq_w.t() + w.cq_b, w.cnq_w).unflatten(2, (nh, -1))
    cattn = _sdpa(cq, cross_k, cross_v).flatten(2, 3).type_as(cq)
    x = x + (cattn @ w.co_w.t() + w.co_b)

    xn = (_ln_no_affine(_up(x)) * (1. + c_scale) + c_shift).type_as(x)
    h = F.gelu(xn @ w.ff1_w.t() + w.ff1_b, approximate="tanh")
    ff = h @ w.ff2_w.t() + w.ff2_b
    x = (_up(x) + _up(ff) * c_gate).type_as(x)
    return x, k, v


# ────────────────────────────────────────────────────────────────────
# Self-check
# ────────────────────────────────────────────────────────────────────

def self_check() -> dict:
    torch.manual_seed(2)
    report = {}
    S, W_ROWS, T = 8, 16, 5
    grid = video_grid(1, grid_h=2, grid_w=4)          # 8 tokens
    fc = freqs_cis_stock(grid)
    cos, sin = build_cos_sin(grid, dtype=torch.float64)
    p = pair_to_half_perm()

    for dtype, tol, tag in ((torch.float64, 1e-10, "f64"),
                            (torch.bfloat16, None, "bf16")):
        w = WanBlockRefWeights.random(dtype=torch.bfloat16).to(dtype)
        x = (torch.randn(1, S, D_REF) / 4).to(dtype)
        text = (torch.randn(1, T, D_REF) / 4).to(dtype)
        proj = (torch.randn(6, D_REF) / 8).float()
        # a live window: random roped keys (stock layout) + values
        wk = (torch.randn(1, W_ROWS, NH_REF, HD) / 4).to(dtype)
        wv = (torch.randn(1, W_ROWS, NH_REF, HD) / 4).to(dtype)

        y_stock, k_s, v_s = stock_block_forward(
            w, x, text, proj, fc, wk, wv)

        rp = repack(w)
        ck, cv = hoist_cross_kv(w, text)
        mod_row = _up(w.sst)[0] + proj.to(_up(w.sst).dtype)  # step-table row
        y_eng, k_e, v_e = engine_block_forward(
            w, rp, x, ck, cv, mod_row, cos, sin,
            wk[:, :, :, p], wv)

        d_out = (y_stock.float() - y_eng.float()).abs().max().item()
        d_k = (k_s[:, :, :, p].float() - k_e.float()).abs().max().item()
        d_v = (v_s.float() - v_e.float()).abs().max().item()
        report[f"{tag}_out_maxdiff"] = d_out
        report[f"{tag}_k_perm_maxdiff"] = d_k
        report[f"{tag}_v_maxdiff"] = d_v
        if tol is not None:
            assert d_out <= tol and d_k <= tol and d_v <= tol, (tag, d_out,
                                                                d_k, d_v)
        # bf16 arm: reported only — isolated ulp flips from reduction-order
        # changes (RMS over permuted channels) are precision-tier.
    return report


__all__ = ["WanBlockRefWeights", "stock_block_forward", "repack",
           "hoist_cross_kv", "engine_block_forward", "self_check"]

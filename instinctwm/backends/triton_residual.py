"""Triton kernels for the residual paths, with the rounding point preserved explicitly.

Break-even, computed before writing anything (see `test_triton_residual.py`):

    eager measured   29.16 us   (50.14 MB of traffic, 1.9x off its own roofline)
    fused ideal       4.40 us   (14.75 MB -- one read of each input, one write)
    budget for launch overhead: ~20 us

`torch.compile` on this region measured 70 us and LOST. The traffic argument was always fine; what
killed it was per-call guard and dispatch overhead. Triton is worth trying precisely because the
arithmetic was never the problem.

THE NUMERICS CONSTRAINT
-----------------------
The eager chain is

    (hidden.float() + attn_out * gate).type_as(hidden)

with `hidden`/`attn_out` bf16 and `gate` **fp32** (it comes from the fp32 modulation table). So:

  * `attn_out * gate` promotes to fp32 and is NOT rounded -- there is no intermediate bf16 store
  * the multiply and the add are two SEPARATE fp32 operations, each rounded to fp32
  * exactly ONE bf16 rounding happens, at the final `type_as`

That middle point is the trap, and it is not hypothetical -- it cost us a wrong diagnosis. Triton
3.5.0 contracts `h + a*g` into a single `fma.rn.f32`, which does NOT round the product to fp32
before adding. Measured against eager:

    stage                       max|delta|    differing elements
    product a*g   in fp32       0.000e+00     0        / 1,474,560
    full h + a*g  in fp32       9.537e-07     285,833  / 1,474,560   <-- created here
    conversion fp32 -> bf16     0.000e+00     0        / 1,474,560

So bf16 conversion matches PyTorch's RNE exactly and FTZ is not involved; the add is the whole
story. 9.537e-07 is 2**-20, one fp32 ULP at magnitude ~8: a single retained guard bit. After the
bf16 round only 33 elements still differ, by 2**-7 -- one bf16 ULP. **A loose tolerance would have
called this kernel correct.** The audit is what caught it, and only because it demands zero.

The fix is a compiler flag, not a source trick:

    enable_fp_fusion=False   ->   PTX emits mul.rn.f32 + add.rn.f32   ->   bit-exact

A first attempt used `tl.where(mask, p, p)` as an optimization barrier. It was removed by the
compiler, so BOTH paths contracted and both showed the same delta -- which read as "FMA ruled out"
when in fact neither path had ever disabled it. Verify at the PTX level, not by differential
testing against your own flag.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                    # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _gated_residual_kernel(H, A, G, O, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        h = tl.load(H + offs, mask=mask, other=0.0).to(tl.float32)   # exact widening
        a = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)   # exact widening
        g = tl.load(G + offs, mask=mask, other=0.0)                  # already fp32

        # Two separately-rounded fp32 ops, matching eager. Contraction is prevented by
        # `enable_fp_fusion=False` at the launch below -- NOT by anything written here.
        r = h + a * g

        tl.store(O + offs, r.to(tl.bfloat16), mask=mask)             # THE rounding point

    def gated_residual(hidden: torch.Tensor, attn_out: torch.Tensor, gate: torch.Tensor,
                       *, allow_fma: bool = False, block: int = 1024) -> torch.Tensor:
        """out = bf16( fp32(hidden) + fp32(attn_out) * gate )

        `gate` is broadcast-expanded by the caller if needed; the kernel is flat over elements,
        which keeps it shape-agnostic and therefore reusable for the FFN residual path.

        `allow_fma=True` re-enables contraction. It exists so the audit can measure the
        difference; it is NUMERIC tier and must never be the default.
        """
        assert hidden.is_contiguous() and attn_out.is_contiguous()
        g = gate if gate.shape == hidden.shape else gate.expand_as(hidden)
        g = g.contiguous()
        out = torch.empty_like(hidden)
        n = hidden.numel()
        grid = (triton.cdiv(n, block),)
        _gated_residual_kernel[grid](hidden, attn_out, g, out, n, BLOCK=block, num_warps=4,
                                     enable_fp_fusion=allow_fma)
        return out

    # -- registration ---------------------------------------------------------------------------
    # `matches_reference_contraction=True` is asserted at the PTX level, not declared:
    # `tests/test_triton_residual.py:test_ptx` fails the build if `fma.rn.f32` reappears.
    from instinctwm.backends.registry import HardwareReq, register_kernel

    @register_kernel(
        region="post_attention_gated_residual",
        hardware=HardwareReq(requires=frozenset({"triton"})),
        preserves_intermediate_rounding=True,
        preserves_reduction_order=True,
        matches_reference_contraction=True,
        compute_dtype="fp32",
        note="raw Triton, enable_fp_fusion=False. Bit-exact on 17.7M adversarial elements; "
             "1.21-1.26x over eager. Contraction disabled because the eager reference is three "
             "separate kernels, each rounding at its boundary")
    def post_attn_gated_residual_triton(hidden, attn_out, gate):
        return gated_residual(hidden, attn_out, gate)


else:                                                   # pragma: no cover
    def gated_residual(*a, **k):
        raise RuntimeError("triton not available")


def gated_residual_eager(hidden, attn_out, gate):
    """The reference. Every audit compares against this, not against intuition."""
    return (hidden.float() + attn_out * gate).type_as(hidden)

"""Fused rotary position embedding — a Layer 5 kernel family. WRITTEN, GATED, AND **NOT SHIPPED**.

VERDICT FIRST. This kernel is correct and it is faster, and it is rejected:

    correctness   bit-exact (0 differing bf16 words) at both shapes the warm profile reports, and on
                  the SPLIT_HALF layout. One documented limit, below.
    region scale  1.10x over eager, 58.5 -> 53.2 us across both shapes
    cycle scale   ~1.6 ms of a 487 ms cycle = 0.3%, against a cycle-to-cycle spread of 1-3%.
                  A cycle-level gate cannot resolve it, so it would return NOT EVALUATED -- and a
                  change that cannot be shown to help does not ship. `passes/contract.py` already
                  says this: a pass that does not improve its declared cost term is rejected
                  regardless of its tier.

WHY IT WAS PICKED, AND WHY THAT WAS WRONG. `aten::copy_` is 66.4 ms of the cycle across 34,710 calls,
and dispatcher-level attribution named this site as 47.4% of watched calls. But TorchDispatchMode saw
only 7,589 calls where the profiler counted 64,391 copy_/fill_ -- about 12%. So "47.4%" was 47.4% of an
unrepresentative eighth of the population, and the real share of this site is roughly 1,200 of 34,710
copies: ~3%, which is what the 0.3% cycle extrapolation reflects. The sample was partial and I ranked
against it anyway.

WHAT IS WORTH KEEPING. The kernel family, the region declaration, and one finding that applies to every
future Layer 5 kernel: torch narrows fp64 -> bf16 **via fp32**, i.e. it double-rounds. A kernel that
rounds fp64 -> bf16 directly is more accurate and NOT bit-exact -- 2 words in 393,216, one ULP apart.
Reproducing the reference's rounding structure means reproducing its double rounding, not improving on
it. That cost two debugging cycles here and would cost them again in the next kernel.


WHY THIS ONE, from the warm 2V/4A profile (PROFILE.md) rather than from the roadmap.

`aten::copy_` is 66.4 ms of a 487 ms cycle across 34,710 calls at 1.9 us each — the largest single
Layer 5 cost, and launch-bound rather than bandwidth-bound. Dispatcher-level attribution put 1,200 of
those calls per cycle in one place, `passes/lingbot/ring_kv.py`, applying RoPE like this:

    def apply_rotary_emb(x, freqs):
        x_out = torch.view_as_complex(
            x.to(torch.float64).reshape(B, S, H, -1, 2))     # bf16 -> fp64: 4x the bytes
        return torch.view_as_real(x_out * freqs).flatten(3).to(x.dtype)

Six ops and three materializations to rotate a vector. The eager path reads the bf16 input, writes an
fp64 copy 4x its size, reads that plus the frequencies, writes an fp64 complex product, then reads it
back and writes a bf16 result. Fused, it is one read and one write.

WHY IT IS A FAMILY AND NOT A PATCH. Rotary embedding is not a LingBot detail — it is in essentially
every current transformer, and in every WAM we have surveyed. The only real variation is the pairing
convention:

    INTERLEAVED   pairs are (x[2i], x[2i+1]).   `view_as_complex` requires this. Wan/LingBot, GPT-NeoX
                  in its original form, and anything built on `torch.view_as_complex`.
    SPLIT_HALF    pairs are (x[i], x[i + D/2]).  Llama/HF `rotate_half`, and most LLM code today.

Both are supported, selected by a declared enum rather than by a flag, so a second backbone needs a
declaration and not a kernel.

BIT-EXACTNESS IS THE INTERESTING CONSTRAINT, and it is the reason this file has two variants rather
than one. The reference computes the complex product in **float64** and rounds to bf16 exactly once,
at the end. A kernel that computes in fp32 is faster still and is *not* bit-exact — the fp64 products
`a*c - b*d` differ from fp32 ones well above bf16's resolution once they cancel. So:

    rope_fused_exact    computes in fp64, one rounding, FP contraction DISABLED  -> Tier.BITEXACT
    rope_fused_fast     computes in fp32                                        -> Tier.NUMERIC

The tier is DERIVED from those declarations by `backends/registry.derive_tier`, not claimed here. The
exact variant is the default precisely because the cheap win is the launch count and the memory
traffic, not the arithmetic width — dropping to fp32 buys a little more and costs the bit-exact claim
that the whole gating regime for Layers 2-3 depends on.

CONTRACTION. `enable_fp_fusion=False` is not decoration. `a*c - b*d` contracted into an FMA keeps the
product of `a*c` at full width instead of rounding it, which changes the result in exactly the
cancelling case that matters here. The eager reference does not contract, so neither may we.
"""

from __future__ import annotations

import enum

import torch


class RopeLayout(enum.Enum):
    """How a rotated pair is laid out in the feature dimension.

    Declared by the adapter, never sniffed. Getting it wrong is silent: both conventions produce
    plausible rotated vectors and only one matches the weights the checkpoint was trained with.
    """

    #: (x[2i], x[2i+1]) — what `torch.view_as_complex` consumes. Wan / LingBot-VA.
    INTERLEAVED = "interleaved"
    #: (x[i], x[i + D/2]) — HF `rotate_half`. Llama-family.
    SPLIT_HALF = "split_half"


def rope_reference(x: torch.Tensor, freqs: torch.Tensor,
                   layout: RopeLayout = RopeLayout.INTERLEAVED) -> torch.Tensor:
    """The eager path, verbatim. THE definition of correct for the kernels below.

    Kept in this file rather than imported from the pass, so the kernel and its reference cannot drift
    apart, and so the bit-exactness test compares against the arithmetic that actually ships.
    """
    if layout is RopeLayout.INTERLEAVED:
        x_out = torch.view_as_complex(
            x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
        return torch.view_as_real(x_out * freqs).flatten(3).to(x.dtype)
    # SPLIT_HALF: (a, b) = (x[:D/2], x[D/2:]), rotated by (cos, sin) = (freqs.real, freqs.imag)
    d = x.shape[-1] // 2
    xd = x.to(torch.float64)
    a, b = xd[..., :d], xd[..., d:]
    c, s = freqs.real.to(torch.float64), freqs.imag.to(torch.float64)
    return torch.cat([a * c - b * s, a * s + b * c], dim=-1).to(x.dtype)


try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                             # pragma: no cover
    HAVE_TRITON = False


#: Split (real, imag) views per frequency tensor. The split is pure view arithmetic; the
#: cache exists only to avoid re-deriving strides per call, not to avoid a copy.
_FREQ_CACHE: dict = {}


if HAVE_TRITON:

    @triton.jit
    def _rope_kernel(X, F_RE, F_IM, OUT,
                     n_pairs, S, H, D2,
                     x_stride_b, x_stride_s, x_stride_h,
                     f_stride_s, f_stride_d,
                     BLOCK: tl.constexpr, INTERLEAVED: tl.constexpr, FP64: tl.constexpr):
        """One program handles BLOCK rotated pairs.

        Indexing rather than reshaping: the eager path's `reshape` + `view_as_complex` +
        `view_as_real` + `flatten` are pure index arithmetic (OpKind.RESHAPE, no numerics), so they
        cost nothing here and disappear entirely. That is where most of the launches went.
        """
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        mask = off < n_pairs

        # Unflatten the pair index into (b, s, h, d) without materialising anything.
        d = off % D2
        rest = off // D2
        h = rest % H
        rest = rest // H
        s = rest % S
        b = rest // S

        base = b * x_stride_b + s * x_stride_s + h * x_stride_h
        if INTERLEAVED:
            i0 = base + 2 * d
            i1 = i0 + 1
        else:
            i0 = base + d
            i1 = base + d + D2

        a = tl.load(X + i0, mask=mask, other=0.0)
        bb = tl.load(X + i1, mask=mask, other=0.0)
        fo = s * f_stride_s + d * f_stride_d
        c = tl.load(F_RE + fo, mask=mask, other=0.0)
        sn = tl.load(F_IM + fo, mask=mask, other=0.0)

        # WIDEN, THEN COMPUTE. bf16 -> fp64 is exact, so this reproduces the reference's `.to(float64)`
        # without materialising a tensor four times the input's size.
        if FP64:
            a64 = a.to(tl.float64)
            b64 = bb.to(tl.float64)
            c64 = c.to(tl.float64)
            s64 = sn.to(tl.float64)
        else:
            a64 = a.to(tl.float32)
            b64 = bb.to(tl.float32)
            c64 = c.to(tl.float32)
            s64 = sn.to(tl.float32)

        # (a + bi)(c + si) = (ac - bs) + (as + bc)i. Written as separate products and one add so that
        # with fp contraction disabled it matches the reference term for term.
        o0 = a64 * c64 - b64 * s64
        o1 = a64 * s64 + b64 * c64

        # ROUNDING PATH, and this is where "preserve the reference's rounding structure" stops being
        # a slogan. Rounding fp64 -> bf16 directly is ONE rounding. Torch's `.to(bfloat16)` from a
        # float64 complex tensor goes through float32 first, which is TWO roundings, and the two
        # disagree on the tie cases -- 2 words in 393,216 at the profile's shapes, exactly one bf16
        # ULP apart. Matching bit-for-bit means reproducing the double rounding, not improving on it.
        if FP64:
            tl.store(OUT + i0, o0.to(tl.float32).to(a.dtype), mask=mask)
            tl.store(OUT + i1, o1.to(tl.float32).to(a.dtype), mask=mask)
        else:
            tl.store(OUT + i0, o0.to(a.dtype), mask=mask)
            tl.store(OUT + i1, o1.to(a.dtype), mask=mask)

    def _split_freqs(freqs: torch.Tensor, s_len: int, d2: int):
        """Real and imaginary parts as 2-D (S, D2) real tensors, plus their strides.

        `view_as_real` on a complex tensor is a VIEW, so this materialises nothing. Broadcast
        dimensions arrive as stride 0 and are indexed as such in the kernel.
        """
        f = freqs
        while f.dim() > 2:
            f = f.squeeze(0) if f.shape[0] == 1 else f.squeeze(-2) if f.shape[-2] == 1 else f
        if f.dim() != 2:
            raise ValueError(f"rope: cannot reduce freqs of shape {tuple(freqs.shape)} to (S, D/2)")
        r = torch.view_as_real(f) if f.is_complex() else f.unsqueeze(-1).expand(*f.shape, 2)
        re, im = r[..., 0], r[..., 1]
        if re.shape != (s_len, d2):
            raise ValueError(f"rope: freqs {tuple(re.shape)} != expected ({s_len}, {d2})")
        # NO `.contiguous()` HERE. `view_as_real(...)[..., 0]` is a stride-2 view, so contiguous()
        # allocated and copied both halves on EVERY call -- two extra launches and two allocations per
        # invocation, which is what made the first benchmark report a constant 77 us regardless of
        # shape. The kernel takes explicit strides, so it reads the view directly.
        return re, im

    def rope_fused(x: torch.Tensor, freqs: torch.Tensor, *,
                   layout: RopeLayout = RopeLayout.INTERLEAVED,
                   fp64: bool = True) -> torch.Tensor:
        """Fused RoPE. `fp64=True` is bit-exact against `rope_reference`; `False` is NUMERIC."""
        if x.dim() != 4:
            raise ValueError(f"rope_fused expects (B, S, H, D), got {tuple(x.shape)}")
        B, S, H, D = x.shape
        if D % 2:
            raise ValueError(f"rope_fused needs an even feature dim, got {D}")
        D2 = D // 2
        if not x.is_contiguous():
            x = x.contiguous()
        key = (id(freqs), S, D2)
        cached = _FREQ_CACHE.get(key)
        if cached is None:
            cached = _split_freqs(freqs, S, D2)
            if len(_FREQ_CACHE) > 64:
                _FREQ_CACHE.clear()
            _FREQ_CACHE[key] = cached
        re, im = cached
        out = torch.empty_like(x)
        n_pairs = B * S * H * D2
        BLOCK = 256
        grid = (triton.cdiv(n_pairs, BLOCK),)
        _rope_kernel[grid](
            x, re, im, out,
            n_pairs, S, H, D2,
            x.stride(0), x.stride(1), x.stride(2),
            re.stride(0), re.stride(1),
            BLOCK=BLOCK,
            INTERLEAVED=(layout is RopeLayout.INTERLEAVED),
            FP64=fp64,
            # NOT decoration: contracting `a*c - b*s` into an FMA keeps the product at full width
            # instead of rounding it, which differs from the reference exactly when the terms cancel.
            enable_fp_fusion=False,
        )
        return out

    def _register() -> None:
        from instinctwm.backends.registry import HardwareReq, register_kernel

        @register_kernel(
            region="rotary_position_embedding",
            hardware=HardwareReq(requires=frozenset({"triton"})),
            preserves_intermediate_rounding=True,
            preserves_reduction_order=True,
            matches_reference_contraction=True,
            compute_dtype="fp64",
            note="fused RoPE, fp64 math, one rounding, enable_fp_fusion=False. Replaces 6 eager ops "
                 "and 3 materialisations (including an fp64 copy 4x the input) with one launch. "
                 "Bit-exact against rope_reference by construction and by test.")
        def rope_exact(x, freqs, layout=RopeLayout.INTERLEAVED):
            return rope_fused(x, freqs, layout=layout, fp64=True)

        @register_kernel(
            region="rotary_position_embedding",
            hardware=HardwareReq(requires=frozenset({"triton"})),
            preserves_intermediate_rounding=False,
            preserves_reduction_order=True,
            matches_reference_contraction=True,
            compute_dtype="fp32",
            note="fused RoPE in fp32. Faster than the exact variant and NOT bit-exact: the reference "
                 "contracts in fp64, and fp32 products differ above bf16 resolution when they cancel. "
                 "Offered so the choice is explicit and tiered, not so it is taken by default.")
        def rope_fast(x, freqs, layout=RopeLayout.INTERLEAVED):
            return rope_fused(x, freqs, layout=layout, fp64=False)

    _register()

else:                                                           # pragma: no cover
    def rope_fused(x, freqs, *, layout=RopeLayout.INTERLEAVED, fp64=True):
        raise RuntimeError("rope_fused needs Triton; the eager reference is rope_reference()")


def rope_region():
    """The fusible region this family serves, for the planner.

    Declared as what the EAGER path does, op by op, so `rounding_points()` reports the one rounding a
    bit-exact kernel has to reproduce. The reshape ops materialise nothing and are marked accordingly
    — they are exactly what fusing deletes.
    """
    from instinctwm.backends.regions import FusibleRegion, OpKind, OpSpec

    return FusibleRegion(
        name="rotary_position_embedding",
        ops=(
            OpSpec("widen_to_fp64", OpKind.ELEMENTWISE, materializes_as="fp64", computes_in="fp64"),
            OpSpec("view_as_complex", OpKind.RESHAPE, materializes_as=None, computes_in="fp64"),
            OpSpec("complex_mul", OpKind.ELEMENTWISE, materializes_as="fp64", computes_in="fp64"),
            OpSpec("view_as_real", OpKind.RESHAPE, materializes_as=None, computes_in="fp64"),
            OpSpec("flatten", OpKind.RESHAPE, materializes_as=None, computes_in="fp64"),
            OpSpec("narrow_to_bf16", OpKind.ELEMENTWISE, materializes_as="bf16", computes_in="fp64"),
        ),
        boundary_in=("x", "freqs"),
        boundary_out=("x_rotated",),
        phases=("video", "action", "kv_refresh"),
        #: q and k, per block, per forward. 30 blocks x 2 = 60 per forward; measured at 1,200
        #: dispatcher-level casts per cycle across 10 forwards.
        occurrences_per_forward=60,
        note="RoPE as the eager path computes it: widen to fp64, complex multiply, narrow to bf16. "
             "Three of the six ops are pure index arithmetic and vanish under fusion.",
    )

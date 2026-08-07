# Fused QKV: theorem, accident, or something else?

Measured 2026-08-07, H100 80GB, torch 2.9.0+cu126, cuDNN 9.10, warm 2V/4A with P007 applied.
Probe: [`probe_qkv_exactness.py`](eval/lingbot_va_robotwin/probe_qkv_exactness.py).

**Answer: neither.** The bit-exactness follows from the algebra *conditional on one invariant*, and that
invariant holds for a structural reason but is not contractual. So the pass should be **BITEXACT gated on
a load-time certificate that fails closed** — the same shape as P006's pointer-stability certificate —
rather than either claimed as a theorem or downgraded to NUMERIC.

---

## The algebra, which tells us what to measure

For `C[m,n] = Σ_k A[m,k]·B[k,n]`, concatenating `B` along **N** adds columns. Every output element's
reduction is over **K**, and K is untouched. N is embarrassingly parallel: no output element's value
depends on how many other columns exist.

So in exact arithmetic fused ≡ split, and in floating point they are bit-identical **iff the reduction
over K is performed identically**. That leaves exactly three ways to fail:

| | mechanism | risk |
|:--|:--|:--|
| `tile_k` | a different K-block size reorders the partial sums | the real one |
| **split-K** | splitting K across CTAs adds a second reduction stage, and with atomics, non-determinism | the feared one |
| accumulator | fp32 accumulation for bf16 inputs, or not | unlikely to change |

Split-K was the specific worry: cuBLAS enables it when `M·N` is too small to fill the GPU, and the
action-stream GEMM is **M = 64** — squarely in that regime. Fusing *triples* the available parallelism
(N: 3072 → 9216), so the heuristic could plausibly switch split-K **off** precisely because we fused,
which would change the reduction and break exactness.

## The production envelope

Enumerated empirically, not assumed. Self-attention only — P002 already removed the cross-attention K/V
projections, so those calls have no second or third GEMM to fuse.

| phase | input | M | K | N split | N fused | calls/cycle |
|:--|:--|--:|--:|--:|--:|--:|
| action | (2, 32, 3072) | 64 | 3072 | 3072 | 9216 | 180 |
| video | (2, 240, 3072) | 480 | 3072 | 3072 | 9216 | 60 |
| kv_refresh | (2, 240, 3072) | 480 | 3072 | 3072 | 9216 | 60 |

Three distinct shapes, 300 calls per cycle. The CFG batch is already folded in — batch 2 is the `2` in
the input shape, so there is no separate CFG variant to enumerate.

## Split vs fused, every shape

| phase | max ULP | differing words | deterministic? | split kernel | fused kernel |
|:--|--:|--:|:--|:--|:--|
| action | **0** | **0** / 589,824 | both, 3 runs | `40x64_64x16` | `72x64_64x12` |
| video | **0** | **0** / 4,423,680 | both, 3 runs | `128x96_64x7` | `384x96_64x3` |
| kv_refresh | **0** | **0** / 4,423,680 | both, 3 runs | `128x96_64x7` | `384x96_64x3_coopA` |

## Why it is bit-exact — the mechanism, not the observation

**A different kernel is selected in every case.** `nvjet_tst_128x96_64x7` becomes
`nvjet_tst_384x96_64x3_coopA`. That looks alarming and is not, because of what changes and what does not:

| tile dimension | split → fused | can it change the result? |
|:--|:--|:--|
| `tile_m` | 40 → 72, 128 → 384 | **no** — partitions rows; each row's reduction is independent |
| `tile_n` | 64 → 64, 96 → 96 | **no** — partitions columns; ditto (and unchanged anyway) |
| **`tile_k`** | **64 → 64** at every shape | **yes, and it is IDENTICAL** |
| `stages` | 16 → 12, 7 → 3 | **no** — pipeline prefetch depth, not arithmetic order |

**`tile_k = 64` in all six kernels.** So the K-loop is the same sequence of 3072/64 = 48 sequential block
accumulations in both forms, and every output element sums its 3072 products in the same order. Both
forms are also run-to-run deterministic across 3 repeats, which rules out atomic split-K.

That is the mechanism. It is not luck: **N does not influence `tile_k`.** cuBLAS picks `tile_k` from K and
the dtype — 64 is its standard bf16 K-block on Hopper — while N drives `tile_n` and the grid. We changed
N and left K alone, so the quantity that governs the arithmetic was never a candidate to change.

> An earlier version of this analysis compared the whole tile tuple and reported *"DIFFERENT — this is
> the risk"* at all three shapes, which would have forced a NUMERIC classification on evidence that
> actually supports the opposite conclusion. The kernels do differ; the K-loop does not.

## Guaranteed by the envelope, or observed on today's stack?

**Both, in different senses, and the distinction is the deliverable.**

**The algebra is guaranteed.** Fused ≡ split for any (M, N, K) whenever `tile_k` matches and split-K is
absent. That is a property of matrix multiplication, not of a library. It will not change.

**The invariant is not guaranteed.** Nothing in the cuBLAS API promises `tile_k` stability under a change
of N. The heuristic is free to differ by:

- library version (cuBLAS 12.x → 13.x)
- driver
- GPU architecture (Hopper → Blackwell, where `nvjet` tiling differs)
- available workspace, which is what gates split-K
- and, at N = 9216 with small M, a future heuristic could plausibly prefer split-K

So `tile_k = 64` is **structurally plausible and empirically confirmed, but contractual nowhere.**

## Therefore: BITEXACT with a fail-closed certificate

The two obvious classifications are both wrong:

- **Claiming BITEXACT unconditionally** would rest a `max |Δ action| = 0` promise on a vendor heuristic.
  One library upgrade could silently invalidate every downstream bit-exactness claim in the chain.
- **Classifying NUMERIC** would accept a numerical difference that *demonstrably does not exist*, and buy
  a 555-episode certification to license an error of zero. That is paying for a risk we can instead
  verify the absence of.

The right answer is the pattern the project already uses. P006 ships BITEXACT "gated by a runtime pointer
certificate that fails closed": an unguaranteed property, checked at runtime, refusing to proceed when it
fails. Apply it here:

> **At install time, for every shape in the declared envelope, compute split and fused and compare bit
> patterns. Fuse only where they match exactly. Any shape that differs falls back to the split path, and
> the fallback is reported rather than silent.**

This is cheap — three GEMM pairs, milliseconds, once per server start — and it converts an empirical
property into a checked precondition. A cuBLAS upgrade that changes `tile_k` then costs performance, not
correctness, and says so in the log.

Two design requirements follow, and both are the kind of thing that is easy to get wrong:

1. **The certificate must cover the envelope, not a sample.** Shapes come from the declared `phases`, so a
   new operating point introduces new shapes and must re-certify. This is exactly the descriptor-driven
   admission `CHECKPOINTS.md` already requires — the envelope is a declared fact, not a constant.
2. **Per-shape granularity.** Fusing is legal at the shapes that certify and illegal at the others, so the
   decision is per-site, not global. A single global boolean would either forfeit a legal win or ship an
   uncertified one.

## Verdict

**Fused QKV belongs beside P007 as a structural optimization, not as a NUMERIC one** — but its BITEXACT
claim is conditional on a runtime-verified invariant, and that condition must be in the pass, not in a
document. Recommend implementing with the certificate as part of the pass rather than as a separate gate,
so it cannot be omitted.

Expected value is unchanged: ~6.13 ms/cycle (1.9%), 600 fewer launches. The certificate makes it a
BITEXACT ship gated in hours instead of a NUMERIC one gated in 555 episodes — which, at 1.9%, is likely
the difference between worth doing and not.

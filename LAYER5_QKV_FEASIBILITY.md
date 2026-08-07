# Fused QKV: feasibility study

Measured 2026-08-07, 2V/4A warm, P007 applied. Probe:
[`probe_qkv_feasibility.py`](eval/lingbot_va_robotwin/probe_qkv_feasibility.py). **No implementation.**

**Verdict: feasible, and better than expected on the axis that mattered most — it measured BIT-EXACT at
both served shapes.** Estimated 6.13 ms/cycle (1.9%), 600 fewer launches. Recommend proceeding, with
one condition attached.

---

## 1. Which sites are legally fusible

600 attention calls in one warm cycle, classified by input identity:

| calls | share | category | fusible? |
|--:|--:|:--|:--|
| 300 | 50% | **self-attention** — `q is k is v` | **yes** |
| 300 | 50% | cross-attention **with cached K/V** — `k`/`v` arrive as `None` | **nothing to fuse** |
| 0 | — | cross-attention with distinct tensors | would have to stay split |

The second row is the interesting one, and it is a consequence of our own work: **P002
(`conditioning_prefill`) already eliminated the cross-attention K/V projections entirely.** Those calls
pass `None` because the K/V were computed once per episode and cached, so there is no second or third
GEMM left to merge. Cross-attention is not a case to be careful about here — it is a case that no longer
exists.

So the fusible population is exactly the 300 self-attention calls, and the 900 `addmm` calls at
`ring_kv.py:146` are `300 × 3`. Fusing them gives **300 GEMMs, 600 fewer launches.**

## 2. Weight compatibility

| | shape | dtype |
|:--|:--|:--|
| `to_q` | (3072, 3072) | bfloat16 |
| `to_k` | (3072, 3072) | bfloat16 |
| `to_v` | (3072, 3072) | bfloat16 |

`in_features` match at K = 3072, dtypes match, and all three carry a bias. Concatenating along
`out_features` gives a single **N = 9216, K = 3072** GEMM. No padding, no reshaping, no special case.

## 3 + 4. Launches and numerics, on the real weights

| tokens | split (3 GEMMs) | fused (1 GEMM) | | max\|Δ\| | words differing |
|--:|--:|--:|--:|--:|--:|
| 32 (action) | 51.2 µs | **24.6 µs** | **2.08×** | 0.000e+00 | **0 / 589,824** |
| 240 (video) | 52.3 µs | **44.5 µs** | **1.18×** | 0.000e+00 | **0 / 4,423,680** |

**Bit-exact at both served shapes.** That was the open risk — N going 3072 → 9216 can change cuBLAS
tile and split-K selection, and therefore the accumulation order — and at these shapes it does not.

This is a measurement, not a guarantee. cuBLAS may select differently under other shapes, other
alignments, or another library version, so the pass cannot *claim* BITEXACT on this evidence alone. But
it means **the NUMERIC tier may be avoidable**, which changes the economics substantially: the
difference between a `max |Δ action| = 0` gate (hours) and a 555-episode paired certification.

Note the 2.08× at 32 tokens versus 1.18× at 240. The small-token case is launch-and-overhead dominated,
so merging three thin GEMMs helps far more there — and 7 of the 10 forwards per cycle are 32-token
(action and KV-refresh) rather than 240-token.

## 5. Cycle-level estimate

| shape | split | fused | × 30 blocks × forwards | saving |
|:--|--:|--:|:--|--:|
| video, 240 tokens | 52.3 µs | 44.5 µs | × 3 forwards | +0.70 ms |
| action/KV, 32 tokens | 51.2 µs | 24.6 µs | × 7 forwards | +5.43 ms |
| | | | **total** | **+6.13 ms** |

**6.13 ms of a 330.2 ms cycle = 1.9%**, plus 600 fewer launches in a cycle that is 42% GPU-idle.

An estimate, not a result. It excludes the `split()` view on the fused output, assumes every fusible
call converts, and ignores any interaction with how the KV cache is written. The cycle gate decides, and
at 1.9% it must be ABBA-ordered — the 1.5–2.0% window spread is the same order as the effect.

---

## Recommendation: proceed, with one condition

It dominates Candidate 4 on every axis that survived measurement:

| | Candidate 4 (cast hoist) | **Candidate 3 (fused QKV)** |
|:--|:--|:--|
| estimated saving | 4.5 ms (1.4%) | **6.13 ms (1.9%)** |
| measured at cycle level | **0.66%, not resolvable** | not yet measured |
| launches removed | 1,740 casts | **600 GEMMs** |
| tier | BITEXACT by construction | bit-exact *measured*, not claimable |
| implementation | 20 lines, done | load-time weight concat + forward rewrite |

**The condition: verify bit-exactness across every served shape before choosing the gate.** If it holds
at all of them, this ships under `max |Δ action| = 0` and costs hours. If it fails at any, it is NUMERIC
and costs a 555-episode certification — at which point 1.9% may not justify the price, and that decision
should be made *before* the implementation, not after.

The shapes to check are enumerable: 2 streams × {32, 240} tokens × the CFG batch dimension, plus
whatever the KV-refresh forward passes. That is a short, cheap sweep and it is the next step.

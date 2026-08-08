# The critical path of one control cycle

Measured 2026-08-08, one saturated cycle (warm 70, past ring saturation), 2V/4A, P007 applied, H100.
Probe: [`probe_critical_path.py`](eval/lingbot_va_robotwin/probe_critical_path.py).

> **PARTIALLY RETRACTED (2026-08-08).** The inventory in sections 1–3 stands. **Section 4's model is
> refuted and its P007 row is a misattribution** — see [LAYER6.md](LAYER6.md) sections H and I, and
> [LAYER6_GAPS.md](LAYER6_GAPS.md) for what the ~155 ms actually is. The corrected marginal cost is
> **1.017 µs** per *Python-originated* dispatch, measured as a slope; C++-internal redispatch is free.

The cycle issues **105,130 dispatcher operations** and **66% of them launch no GPU kernel at all.** Device
work is 196 ms of a 338 ms cycle. This document concluded from that arithmetic that the host was the
critical path; the conclusion did not survive a direct test, and the ~155 ms difference turned out to be
neither host dispatch throughput nor kernel duration.

---

## 1. The timeline is a chain, not a DAG

| | |
|:--|:--|
| streams observed | **1** (stream 7, 18,605 device events) |
| host threads issuing | 1 |

With one host thread and one stream, host program order serialises the CPU side and stream order
serialises the device side. There is no parallel branch to be the "longest path" among — **the DAG's
longest path is the wall clock itself.** So the useful question is not *which chain* but *which side of
the coupling owns it*, and that is what the rest of this measures. Had multiple streams appeared, a real
longest-path computation would have been required; the stream inventory is reported so the assumption is
checked rather than assumed.

## 2. Composition of the path

Measured on the traced cycle and corrected to the untraced one (the profiler adds ~2.3 µs per host op
across 105k ops, which inflates the traced wall from 338 ms to 577 ms — all of it on the host side):

| | ms | share of 338 ms cycle |
|:--|--:|--:|
| device BUSY (interval union) | **196** | **58%** |
| device IDLE | **142** | **42%** |
| host busy | ~338 | ~100% |
| of the idle, host *blocking* on the device | ~8% | |
| of the idle, host simply *issuing* | ~92% | |

**If all device work became free the cycle would be 142 ms (2.38×). If the host could keep the queue
full it would be 196 ms (1.72×).** The host is the longer chain, so the host bounds the cycle.

Note what the idle is *not*: it is not sync stalls. There are 1,306 `aten::item` and 1,306
`_local_scalar_dense` calls per cycle — a genuine serialisation structure, one queue drain every ~14
device events — but together they cost only ~2.8 ms. The device is idle because the host cannot issue
fast enough, not because it is blocked at barriers.

## 3. The host operation inventory — the actual finding

105,130 host ops produce 18,605 device events: **5.7 host operations per kernel.**

| aten op | calls/cycle | share | launches a kernel? |
|:--|--:|--:|:--|
| `as_strided` | 12,082 | 11.5% | **no** |
| `view` | 10,796 | 10.3% | **no** |
| `to` | 8,097 | 7.7% | **no** |
| `copy_` | 6,385 | 6.1% | yes |
| `empty_strided` | 6,279 | 6.0% | **no** |
| `empty` | 5,986 | 5.7% | **no** |
| `_to_copy` | 5,543 | 5.3% | yes |
| `transpose` | 4,858 | 4.6% | **no** |
| `slice` | 4,244 | 4.0% | **no** |
| `reshape` | 3,185 | 3.0% | **no** |
| `linear` + `t` + `addmm` | 7,344 | 7.0% | yes (one kernel per three ops) |
| `item` + `_local_scalar_dense` | 2,612 | 2.5% | **no** |
| **ops launching NO kernel** | **69,601** | **66.2%** | |
| ops launching work | 35,529 | 33.8% | |

**Two thirds of the critical path is metadata.** `as_strided`, `view`, `transpose`, `slice`, `reshape`,
`empty` — 69,601 operations per cycle that move no bytes and compute nothing, each costing ~3.2 µs of
dispatcher time on the chain that determines when the cycle finishes.

## 4. The model — RETRACTED, see LAYER6.md sections H and I

> **RETRACTION (2026-08-08).** The model below was refuted by direct measurement, and P007's attribution
> here is wrong. Read [LAYER6.md](LAYER6.md) sections H and I before using anything in this section. The
> gap inventory in sections 1–3 stands; the causal claim in this section does not.

The model as originally stated:

> When host-bound, cycle time ≈ (host op count) × ~3.2 µs. The lever is the *number* of dispatcher
> operations, not the GPU time they represent.

| change | host ops removed | model predicted | measured |
|:--|--:|--:|:--|
| **P007** conv layout | 56,645 | 35% (1.54×) | **1.405×** |
| cast hoist (Candidate 4) | 1,740 | 1.6% | 0.66% |
| fused QKV (Candidate 3) | ~0 net | 0% | 0.2% *slower* |
| RoPE kernel | 0 | 0% | 0.3% |

**Two things are wrong with this table.**

**The `3.2 µs` was never measured.** It was obtained by dividing 338 ms by 105,130 operations, which
presumes the cycle is the sum of per-operation host costs — the claim the table was offered as evidence
for. Injecting no-kernel dispatches and fitting the derivative gives **1.017 µs/op**, 3.1× lower
(LAYER6.md section I). Worse, the population is not homogeneous: a Python-originated dispatch costs
~1.02 µs and a C++-internal redispatch costs approximately nothing, and the profiler counts them
identically.

**P007's row is a misattribution, but not in the direction I first corrected it to.** Crediting its
56,645 removed dispatches at 3.2 µs each gives 181 ms for a 150–160 ms effect, which only looked right
because the rate was wrong. A positive control has since measured both terms directly
([LAYER6_REGIMES.md](LAYER6_REGIMES.md)): device busy fell **56.2 ms** and device events fell **28,387**,
which against the published 160.2 ms in-process delta is **~37% device / ~63% launches removed at
~3.5 µs each**. Both terms are real and the launch term is the larger.

The deeper point is that P007 **changed the regime**: with `vol2col` the VAE encode issued 46,992 tiny
kernels and was host-bound; with cuDNN it is device-bound at slope ~1.09. A single-term attribution
cannot describe an intervention that moves work across the boundary, which is why this row resisted two
attempts to label it.

So the model retrodicted five results because op count and device time happened to move together in all of
them, and it has since made three prospective predictions and missed all three. The decisive one had no
confound: `prebound_projection` removed 12,190 operations bit-exactly, with no artefact built, no kernel
changed and no shape changed, and the cycle got 3.4 ms **slower** — because it traded 12,190 free
C++-internal redispatches for 4,888 Python-level ones.

**What survives from this document:** the gap inventory. Device work is 196 ms of a 338 ms cycle and the
~155 ms difference is real. It is neither host dispatch throughput (capped at ~56 ms) nor kernel duration.
[LAYER6_GAPS.md](LAYER6_GAPS.md) measures what it actually is.

## 5. Why each expensive off-path operator cannot help

| operator | device ms/cycle | why speeding it up cannot reduce cycle latency |
|:--|--:|:--|
| attention (`cudnn_sdpa`) | 44.2 | inside the 196 ms device chain, which has 142 ms of slack. Halving it lengthens the idle gap by 22 ms and shortens nothing. |
| `CatArrayBatchedCopy` (ring KV) | 21.0 | same slack. Eliminating the materialization removes ~60 host ops and ~21 ms of device time — the device time is free, the 60 ops are 0.06% |
| `nvjet` GEMMs (all) | 51.7 | the GEMMs are 2,444 `addmm` calls but **7,344** host ops (`linear` → `t` → `addmm`). The kernel time is off-path; the dispatch triple is on it. |
| elementwise/layout kernels | 74.0 | off-path device time. Fusing them would help only by reducing the *number of dispatches*, not by reducing their GPU cost. |
| RoPE `mul`/`_to_copy` | 8.3 | off-path, and the kernel is already written and rejected |

The general rule this produces: **an optimization's value is its host-op delta, and its device-time delta
is worth nothing until the device becomes the longer chain** — which happens only below ~196 ms.

## 6. Reclassification of every Layer 5 proposal

| proposal | class | value |
|:--|:--|:--|
| **P007 conv layout** (shipped) | **removes work on critical path** | 1.405×, measured |
| Cast hoist, Candidate 4 (implemented, unshipped) | **removes work on critical path** | 1,740 ops = 1.6% ceiling. Real but at the noise floor. |
| Fused QKV, Candidate 3 (implemented, rejected) | **accelerates work off critical path** | ~0. Removes 600 GEMM launches, adds 300 `split` + slicing ops. |
| RoPE Triton kernel (written, rejected) | **accelerates work off critical path** | 0. Removes no dispatches. |
| `cat` / ring-KV materialization removal | **removes work off critical path** | ~60 ops. The 21 ms of device time is slack. |
| FFN GEMM+GELU epilogue fusion | **accelerates work off critical path** | ~0 unless it removes the `linear`/`t`/`addmm` triple |
| Attention backend swap (Layer 4) | **accelerates work off critical path** | 0 at this operating point |
| Graph capture, P005 (shipped at Quality, off at Fast) | **removes work on critical path** | **the only mechanism that removes host dispatch wholesale** — see below |

## 7. What follows, without proposing an optimization

Three consequences, stated as findings rather than plans:

**a. The host-op count is the metric Layer 5 should be tracked against.** Not GPU time, not operator
totals, not region speedups. Every candidate should carry a predicted host-op delta before anything is
built, and that number is cheap to obtain from a trace.

**b. Two thirds of the path is metadata, and no kernel can touch it.** `as_strided`, `view`,
`transpose`, `reshape`, `empty` do not exist on the device. The only mechanisms that remove them are
ones that eliminate dispatch itself — CUDA graph replay, a compiled/traced graph, or restructuring the
model code so fewer views are created.

**c. P005's rejection at Fast deserves re-examination, and this is the sharpest implication.** Graph
capture removes host dispatch wholesale on replay, which is precisely the binding constraint identified
here. It was measured unprofitable at Fast on the grounds that its ~700 ms/cycle capture cost breaks even
near 41 forwards — but that analysis assumed the cycle was device-bound and priced only the device
saving. If capture converged instead of re-capturing every cycle (its key is the ring signature, which
changes as the ring advances), the host saving would be most of 105,130 dispatches. That is a
measurement, not a proposal, and it should be made before any further kernel work.

## Method notes

Two corrections were needed and both matter for anyone repeating this:

**The profiler distorts exactly the thing under study.** CPU+CUDA tracing costs ~2.3 µs per host op and
the cycle has 105k of them, so the traced wall is 577 ms against a real 338 ms — and *all* the inflation
lands on the host side, i.e. on the quantity being measured. Uncorrected, the idle attribution ranked
`aten::empty` and `aten::empty_strided` near the top purely because they are called often. The table
above subtracts count × per-op overhead for non-blocking ops and leaves blocking ops uncorrected.

**Device-busy must be an interval union, not a sum of kernel durations.** Summing durations
double-counts any overlap and can exceed the wall clock. The 196 ms figure is the union.

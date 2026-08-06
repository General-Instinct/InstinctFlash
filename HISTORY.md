# Implementation history

P001–P006 are **implementation milestones, not architecture**. They are the order in which things
got built and measured on one model, LingBot-VA, on one box. They are kept because each carries a
gate and a retraction that would otherwise be re-learned, and because a released pass with a version
number is a promise.

For how the project is organized, read [ARCHITECTURE.md](ARCHITECTURE.md) instead. If you are trying
to understand InstinctWM, nothing here is the right starting point.

The authoritative registry is [`instinctwm/verify/released.py`](instinctwm/verify/released.py); this
page is its narrative form.

---

## The chain

Measured with `probe_latency.py --cycles 10 --repeats 3`, first run discarded, all spreads ≤ 0.7%.

| Stage | ms/cycle | Pass | Step | Tier |
|:--|--:|:--|--:|:--|
| stock | 8431.5 | — | — | — |
| +P001 | 3994.0 | `substrate_elision` | 2.11× | BITEXACT |
| +P002 | 3567.5 | `conditioning_prefill` | 1.12× | BITEXACT |
| +P003 | 2553.9 | `ring_kv_addressing` | 1.40× | BITEXACT |
| +P004 | 2539.9 | `hoist_invariant_casts` | 1.02× | BITEXACT |
| +P005 | 1842.0 | `graph_block_stack` | 1.38× | BITEXACT |
| +P006 | 1211.3 | `stable_state_pools` | 1.52× | BITEXACT |

**Cumulative 6.96×, every stage at `max |Δ action| = 0`.**

> **These numbers are pre-order-control and Quality-profile.** They were taken with sequential A/B
> ordering — baseline first, treatment second — and this box drifts upward within a session
> (3214 → 3730 → 3964 ms across three rounds of one configuration). Whichever arm ran second is
> systematically penalised, so treatments that ran second are *understated*. ABBA is the default
> protocol now; these have not been restated under it.
>
> They are also 25 video / 50 action steps, ~79 forwards per cycle. The Fast operating point runs 6.
> Do not quote any of this for Fast — see [CHECKPOINTS.md](CHECKPOINTS.md) on why the ranking inverts.

---

## The milestones

**P001 · `substrate_elision` · 2.11×**
Removes FSDP at world-size 1, per-chunk `empty_cache`, and blocking debug dumps. The largest single
win in the project was deleting work that was never needed, not making work faster.

**P002 · `conditioning_prefill` · 1.12×**
Caches episode-constant cross-attention K/V for all 30 layers (+360 MiB), removing 89 of 226
TFLOP/cycle.

**P003 · `ring_kv_addressing` · 1.40×**
Slice-addressed ring KV. Gated over 40 cycles past the wrap at ~36, 800/800 allocator parity checks
across 5.6 full wraps, 3/3 bitwise-identical action streams over ~53 cycles/episode. This is also
what makes P005 possible at all — a stock block raises `cudaErrorStreamCaptureInvalidated`.

**P004 · `hoist_invariant_casts` · 1.02×**
Casts `FP32LayerNorm` weight/bias and the block `scale_shift_table` once per episode instead of once
per forward: 7,110 casts of a constant removed per control cycle. Cost model predicted 47.4 ms,
measured 49.7 ms — 6% error, the first time a prediction was made before the measurement.

**P005 · `graph_block_stack` · 1.38× · v1.0.1**
Runs the 30-block stack from a captured CUDA graph. Per-op dispatch of 6.2 µs — 83.6% of it
`cudaLaunchKernel` — becomes ~1.17 µs of replay. Gated *after* an episode reset, the ordering that
exposed a NaN.

**P006 · `stable_state_pools` · 1.52×**
Reset clears logical KV state in place instead of reallocating, so P005's graphs survive resets.
Gated by a runtime pointer certificate that fails closed. `probe_reset_isolation = 0` — episode 2 is
bitwise identical to a fresh episode.

---

## Corrections worth keeping

Each of these was believed, published internally, and then withdrawn. They are recorded because the
belief was reasonable and the correction was not.

**P005 v1.0.1 — the eager fallback froze the ring.** `install` sets `_iwm_defer_commit`
permanently, so only `_commit_all` advances the ring. Both fallback returns skipped it. From the
first capture failure onwards the ring froze: every later forward rewrote the same slots and
attention read a stale window, with nothing raised. The actions stayed plausible and were wrong.

A correctness bug is the only reason a frozen pass may change. No reported number was affected — no
log contains `CAPTURE FAILED` and no eval server ran `--graph-blocks`. The fix is gated by a probe
that *forces* the fallback, because the standard P005 gate cannot fail on this bug: capture succeeds
on all 6 of its cycles, so the fallback path is never entered and the gate would pass identically
before and after. **A gate that cannot fail on the bug it is gating is worse than no gate.**

**Whole-cycle graph capture — rejected by measurement.** Kept in the stack table struck through so
it is not re-proposed.

**CFG parallelization — rejected by measurement.** Same.

**A +331 ms "regression" that was contention.** An eval fleet sharing the GPUs produced it. This is
why speed gates now report NOT EVALUATED on an occupied device, and why the guard lives in one shared
module (`tests/perf_gate.py`) rather than being copied per test — it was fixed in one gate and the
copy in another kept the bug.

**`probe_latency` hid a position-dependent cost.** It resets between repeats, rewinding the ring to
`(0,0)`, so every repeat replays the keys the discarded first run captured. Graph capture's key
contains `(start, count)`, so the cost that depends on ring position was invisible. In episode mode
(45 cycles, one reset) capture is a 1.21× net win, not the 1.5–2× `probe_latency` implied. Captures
never stop: 6.0/cycle at a 92.5% hit rate.

**The runtime had a training method's name in it.** `runtime/pdd_serve.py`, entry point
`install_pdd_video_heads`. It was written before the one-runtime principle was stated, and it is
exactly what that principle forbids: a serving path that names where the weights came from. Renamed
to `runtime/block_heads.py` / `install_block_velocity_heads`, which is what it actually does — serve
a checkpoint whose output projection is a set of per-interval velocity heads, derived from three
declared facts (`n_intervals`, `block`, `guidance`) and nothing else. Behaviour is unchanged; the
`delta.json` it reads is the prototype of the `instinctwm.json` contract in
[CHECKPOINTS.md](CHECKPOINTS.md). Found by grepping for the rule instead of trusting it.

**Attention was ranked first by intuition and near-last by profile.** 7% of GPU-busy time at Quality,
and inside the 7% that is not fixed overhead at Fast.

---

## Layer 1

**The step-allocation response surface.** 7 operating points, ~3,500 paired episodes, 50 tasks. Both
streams tolerate reduction to 2 steps with minimal degradation and both cliff at 1 step: action 50→2
costs 0.02–0.03, video 25→2 costs ~0.03, either at 1 step costs 0.11–0.18. Video is the steeper side
past the cliff (2:1 = 0.790 beats 1:2 = 0.712). From 79 forwards per cycle down to 4–6 is nearly free
**with no training at all**, which is the result that reframed Layer 1.

**PDD bought nothing at 2V/50A.** A heads-only student measured 0.920 against an untrained
step-reduced control at 0.920 — 3/3 discordant, p = 1.0. The control is what made this readable;
comparing the student to the teacher alone would have conflated "PDD works" with "2 video steps are
enough anyway."

Diagnostics from that run: 15.3% endpoint RMSE, a plateau, Spearman −0.05 between per-head error and
update count, 3.5× worse error at high sigma. Two bugs found along the way — a double sign flip in
serving (0/100 on RoboTwin until fixed; both arms then 0.920) and AdamW's weight decay moving all 256
heads every step whether or not they were touched.

**Fast is certified.** 2 video / 4 action, 566 matched pairs on identical seeds: teacher 0.929,
student 0.910, delta −0.019, exact McNemar p = 0.185, non-inferiority at −0.05 with p = 0.0085.
2V/2A is dominated — same latency, worse accuracy point — and 2V/8A costs +29%, outside the band.

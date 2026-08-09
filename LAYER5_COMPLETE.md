# Layer 5 is effectively complete at 2V/4A

**Verdict: COMPLETE.** No backend-, library-, layout- or algorithm-selection candidate can remove ~100 ms
of device work, and none can return 10 ms of cycle. The screen found **zero** operators on a library
fallback path — the question that produced P007, asked again, returns nothing — and the regime model caps
*all* remaining device-side work at **~43 ms of a 331 ms cycle** even if every kernel in the model became
free.

Probes: [`probe_backend_screen.py`](eval/lingbot_va_robotwin/probe_backend_screen.py) (kernel census) and
[`probe_slope_clean.py`](eval/lingbot_va_robotwin/probe_slope_clean.py) (the exchange rate).

## 1. The arithmetic that decides it, before any candidate is named

Total device work in a warm cycle is **195.8 ms**. The bar — "remove on the order of 100 ms of device
work" — therefore asks for **half of all compute in the model**. The largest single kernel is attention at
44.3 ms. Only two things could halve device work: eliminating CFG (both branches), or halving the denoise
step count. CFG is ruled out by measurement (§4) and the step count is the operating point, not a backend
choice.

And a millisecond of device time is not a millisecond of cycle. Re-measured cleanly with a
`torch.cuda._sleep` dummy, no profiler anywhere in the loop, and a hard gate that the k=0 baseline match
the known unprofiled cycle (**330.1 ms measured against 330.7 ms reference, +0.2%**):

| injected ms | Δcycle ms | **marginal slope** | absorbed |
|--:|--:|--:|--:|
| 59.7 | +8.6 | **0.145** | 51.1 ms |
| 149.7 | +48.3 | 0.440 | 101.4 ms |
| 299.7 | +198.6 | **1.002** | 101.1 ms |
| 449.7 | +352.0 | **1.023** | 97.7 ms |

Ten `k=0` arms, counterbalanced forward and reverse, spread 2.3%. **The transformer absorbs ~100 ms of
added device work and then pays exactly 1:1** — 1.002 and 1.023 against the 1.0 the max-plus model
predicts above the knee, which is the sharpest confirmation the model has had.

> **The exchange rate at the operating point: transformer ×0.145, VAE ×1.0.**
> Ceiling if every device kernel became free: `179.2 × 0.145 + 16.6 = ` **42.6 ms of 331 ms.**

### Every candidate, priced

| eliminated **entirely** | device ms | cycle ms | clears 10 ms? |
|:--|--:|--:|:--|
| attention (cuDNN flash SDPA) | 44.3 | 6.4 | no |
| all cuBLASLt GEMMs | 51.5 | 7.5 | no |
| all 14,495 eager elementwise launches | 67.0 | 9.7 | no |
| ring-KV `CatArrayBatchedCopy` | 21.1 | 3.1 | no |
| both layer-norm families | 10.1 | 1.5 | no |
| **all remaining VAE device work** | 16.6 | **16.6** | yes — but that is deleting the encoder |

Nothing that is an *optimization* clears the bar. The only row that does is "remove the observation
encoder", which is not one.

## 2. The kernel census: no fallbacks remain

One warm cycle, 195.8 ms of device time, grouped by kernel.

| ms | calls | µs/call | kernel |
|--:|--:|--:|:--|
| 44.3 | 600 | 73.8 | `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16` |
| 21.1 | 60 | 351.8 | `CatArrayBatchedCopy` — the ring-KV window materialisation |
| 19.7 | 852 | 23.1 | `nvjet_tst_128x96_64x7_4x1_v_bz_bias_TNT` |
| 17.6 | 1272 | 13.9 | `nvjet_tst_40x64_64x16_4x1_v_bz_bias_TNN` |
| 14.7 | 2720 | 5.4 | `unrolled_elementwise_kernel<direct_copy_kernel_cuda>` |
| 8.1 | 120 | 67.8 | `nvjet_tst_192x160_64x4_2x1_v_bz_coopB_bias_TNT` |
| 6.1 | 180 | 34.1 | `nvjet_tst_112x64_64x9_2x1_v_bz_bias_TNN` |
| 5.4 / 4.7 | 910 / 900 | 5.9 / 5.2 | `vectorized_layer_norm_kernel<float>` / `<bfloat16>` |
| 2.5 / 1.8 / 1.5 / 1.3 | 8 / 4 / 17 / 4 | 310–446 | `sm90_xmma_fprop_implicit_gemm_..._nhwckrsc_...` — cuDNN NHWC conv, the VAE |

**`slow_*`, `vol2col`, `im2col`, `col2im`, `_naive`, `generic_`: zero occurrences.** The only `unrolled_`
kernels are `direct_copy_kernel_cuda`, which is a copy, not a library declining an input.

**Library coverage of everything compute-heavy:**

| library | ms | kernels |
|:--|--:|--:|
| cuBLASLt (`nvjet_*`) | 52.3 | 17 |
| cuDNN (incl. `sm90_xmma` NHWC convs) | 52.2 | 14 |
| cuDNN flash SDPA | 44.3 | 1 |

The VAE convolutions now run `sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_**nhwckrsc**`, which is
exactly the NHWC implicit-GEMM path P007 was built to reach. P007 is confirmed working in the census, not
just in its certificate.

## 3. Why each category is exhausted

**Backend dispatch.** The question that found P007 — *why is this operator on a fallback at all?* — now
has no positive answer anywhere in the cycle. Zero fallback kernels; 148.8 of 195.8 ms on vendor
libraries; the rest is elementwise work that has no library to dispatch to.

**Library selection.** GEMMs are on cuBLASLt, convolutions on cuDNN implicit GEMM, attention on cuDNN
flash with wgmma. There is no faster library in the installed stack (torch 2.9 / cuDNN 9.10) for these
operators, and even a hypothetical 2× on *all* of them returns `(44.3+51.5)/2 × 0.145 = 6.9 ms`.

**Memory layout.** P007 was the layout win and it is taken. The remaining convolutions are already NHWC;
the GEMMs are TNN/TNT, the standard orientations; SDPA receives `(B,H,S,D)` which is its preferred layout.

**Algorithm choice.** The one structural algorithm change available — attending over the live ring
interval rather than the whole 9,792-slot pool — is *already* what P003 does. `CatArrayBatchedCopy` at
21.1 ms is the cost of materialising that window, and eliminating it entirely returns 3.1 ms.

**Fallback kernels.** None, as above.

**Hidden framework fallbacks.** Layer norms are on the fused `vectorized_layer_norm_kernel`, not
decomposed. The RoPE `float64` round trip was already investigated and its whole region measured 0.3% of
the cycle. Nothing is silently decomposed into elementwise chains that a library would take.

**Duplicated execution.** Two candidates, both refuted by measurement: CFG (§4) and the terminal action
forward (§4b), the latter dead for 38 cycles and live thereafter. Beyond them: the 2,720
`direct_copy_kernel_cuda` launches (14.7 ms → 2.1 ms of cycle) are the largest remaining duplication-shaped
item and are below the bar by 5×.

## 4. CFG: the only candidate that could have cleared the bar, ruled out by measurement

`guidance_scale=5`, `action_guidance_scale=1`, and `use_cfg` is a single global flag OR-ed from both
streams, so **every** forward runs at batch 2 including all action forwards whose CFG output is then
discarded by `action_noise_pred[:1]`. That is ~98 ms of device work and half the host dispatches — the
only thing in this system that could plausibly remove 100 ms.

It is not dead compute. A two-axis liveness test found branch 1 **live on both axes**: corrupting its
returned value moved final actions by **5.64**, and suppressing only its writes to the shared KV pool
moved them by **5.39**, against a chunk-to-chunk movement of **1.03**. Both CFG branches write the shared
ring KV pool and the video stream at scale 5 reads branch 1. `dead_outputs` is a true statement about
output usage; `elidable_computations` is not. Recorded in `passes/lingbot/cfg_elision.py`, kept as a
ruled-out optimization rather than a pass.

## 4b. The one candidate that cleared the bar, and why it died

An independent search (5 parallel lenses over the source and the census) reached the same COMPLETE
verdict, and surfaced one item I had missed that **did** clear the 10 ms bar: the **terminal action
forward**.

Both denoise loops pad a terminal timestep `t=0` (`wan_va_server.py:473`, `:478`) and run one more
transformer forward at it with `update_cache=1`. On that iteration the output is *provably* discarded —
`if not last_step` guards every use (`:548` action, `:508` video) — so the forward's only possible effect
is its write to the shared ring KV pool. And `_compute_kv_cache` calls `clear_pred_cache` as its **first**
statement (`:574`), with no transformer running in between. It reads as 1 of 10 forwards, ~1,860 launches
and ~10,500 aten events, worth **~30 ms of cycle** — and because it deletes a whole host-bound segment
rather than device time, it is worth ~30 ms rather than the ~2 ms a device-only change of that size buys.

**It is not dead.** [`probe_terminal_forward.py`](eval/lingbot_va_robotwin/probe_terminal_forward.py),
45 seeded cycles spanning ring saturation:

| cycles | `max |Δ action|` |
|:--|:--|
| 0 – ~38, pre-saturation | **0** — genuinely dead |
| last 6, post-saturation | 0.0297, 0.0234, 0.266, **0.406**, 0.266, 0.102 |

Against a chunk-to-chunk movement of 1.03, the worst is ~40% of a real movement.

**The mechanism is the ring wrap.** `clear_pred_cache` rolls the count back (`ring_kv.py:132`,
`r["count"] -= r["pred"]`), which is why the write is invisible before saturation. But once
`count > total` the write has already **evicted** the oldest slot and advanced the interval
(`ring_kv.py:258–259`, `r["start"] = (r["start"] + (r["count"] - total)) % total`), and *that* is not
rolled back. The forward's effect on ring *state* outlives the rollback of its ring *contents*.

**A 12-cycle gate would have reported `max|Δ| = 0` and shipped it.** This is the third time in this
project that spanning the saturation transition has been the difference between a correct result and a
confident wrong one.

The video terminal forward was tested in the same run as a control and is live at **1.3086**, as
predicted — its KV writes *are* read, by the action loop that follows it. The predicted asymmetry between
the two loops is real; it just does not reach as far as "dead".

## 5. The structural reason, not just the empirical one

The empirical answer is "we looked and there is nothing left". The structural answer is stronger, and it
is why looking harder will not help:

```
W(α) = Σ_segments max( H_s , α · D_s )
```

The cycle is a sum of per-segment maxima, so a device-side change is worth the **device-bound segments'
share** and nothing more. Measured: the transformer holds 179.2 ms of device time against ~100 ms of
absorbable host slack, so it sits far below its knee and returns **0.145 ms per ms**. The VAE holds
16.6 ms and sits above its knee, returning ~1.0 — but there is only 16.6 ms there.

**Layer 5 operates on device time. The binding constraint is no longer device time.** That is not a
statement about how well-optimised the kernels are; it would remain true if they were twice as fast. The
ceiling on the entire layer is 42.6 ms and the bar is 100 ms, so the conclusion holds with a 2.3× margin
even before any candidate is examined.

This also retro-explains the layer's record without appealing to bad luck. The RoPE kernel, fused QKV and
every proposed transformer fusion targeted device time at ×0.145. P007 succeeded because it acted on the
VAE **and** because it moved that segment across the regime boundary — from host-bound with 46,992 tiny
`vol2col` launches to device-bound with large cuDNN convolutions. It changed which term binds. Nothing
left in the stack has that property.

## 6. What would reopen Layer 5

| condition | why it changes the answer |
|:--|:--|
| **A segment crosses the regime boundary.** | The only mechanism that has ever paid here. If a change made the transformer device-bound — or moved a host-bound region onto a library path the way P007 did — the exchange rate for that segment jumps from 0.145 to ~1.0. |
| **Host dispatch is cut hard enough to expose the device.** | The transformer's knee is ~100 ms of slack. Remove ~100 ms of host issue time from it and the slope goes to 1.0, at which point 179 ms of device work becomes fully reachable. The available host reduction is capped at ~35 ms (34,635 Python-originated dispatches × 1.017 µs), so this does not happen incrementally — it needs a mechanism that removes dispatch wholesale, and both known ones (graph capture, `torch.compile`) are measured and rejected. |
| **A different operating point with a different H/D ratio.** | Quality (25V/50A) does **not** qualify: the block runs 79 times instead of 10, but per-block host (~1.10 ms) against per-block device (~0.64 ms) is unchanged, because the denoise-step count scales both and tokens per block do not change. This follows from the ratio; it has not been measured at Quality. |
| **A different checkpoint.** | Cosmos3-Edge has a different H/D balance and its own screen would be required. Nothing here transfers to it. |
| **A larger batch or resolution.** | Both raise device work per dispatch, which pushes segments toward the device-bound side and raises the exchange rate. |

## 7. What this does not claim

- **Not "the kernels are optimal".** Several are not. The claim is that making them faster does not
  shorten this cycle at this operating point.
- **The 0.145 is a marginal slope at the current point,** measured by *adding* device work. `W` is convex
  and non-decreasing in α, so `0 ≤ W'(1⁻) ≤ W'(1⁺)`, which licenses using it as an upper bound for
  removal — but it is an upper bound, not a point estimate for large removals.
- **The 179.2 / 16.6 ms region split** comes from attributing each kernel's launch to the innermost named
  scope. The screen's raw output put 97 ms in an "other" bucket — SDPA and the `nvjet` GEMMs, which are
  unambiguously transformer work; that reassignment is by kernel identity, not by the scope lookup.
- **The largest transformer arm (450 ms injected) ran the cycle at 679 ms**, far outside the served
  regime. It is used only to establish that the above-knee slope is 1.0, not to price anything.
- **No candidate was implemented.** This is a screen. The one measurement it rests on that was *not*
  taken today is P007's own cycle delta, which comes from its published certificate.

## Further reading

- [LAYER6_REGIMES.md](LAYER6_REGIMES.md) — the two regimes and the exchange rates
- [LAYER6_GAPS.md](LAYER6_GAPS.md) — the 139 ms of device idle, diffuse, the eager floor
- [LAYER5.md](LAYER5.md) — the backend/layout flow, and P007 as its reference implementation
- [`passes/lingbot/cfg_elision.py`](instinctwm/passes/lingbot/cfg_elision.py) — the ruled-out CFG record

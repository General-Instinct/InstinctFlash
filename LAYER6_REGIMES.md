# The cycle has two regimes, and they need opposite levers

**Measured 2026-08-08. This supersedes both "the host is the clock" and "the device has 155 ms of
slack." Neither was right, because both asked a global question about a system that is not globally
one thing.**

Probes: [`probe_device_slope.py`](eval/lingbot_va_robotwin/probe_device_slope.py) and
[`probe_p007_passthrough.py`](eval/lingbot_va_robotwin/probe_p007_passthrough.py). 2V/4A, warm 70 cycles
past ring saturation, shipped stack, keyframes pre-generated, **no profiler in any timed loop**.

## The measurement

Inject one dummy GPU kernel per site and vary only its duration, holding launch count constant by
fitting against a `side=128` reference arm that also launches. Then read
`d(cycle) / d(injected device time)`.

| region | injection sites/cycle | injected device ms | cycle delta ms | **slope** |
|:--|--:|--:|--:|--:|
| **transformer blocks** | 300 | 4.1 / 19.8 / 51.6 | +2.2 / −4.2 / −2.5 | **−0.067** |
| **observation VAEs** | 81 | 13.3 / 50.9 / 122.6 | +16.3 / +57.5 / +133.0 | **+1.093** |

Per-point slopes in the VAE: 1.223, 1.130, 1.085 — linear and converging. In the transformer every
point sits inside the noise floor.

> **A millisecond of device time in the VAE costs a millisecond of cycle. A millisecond of device time
> in the transformer costs nothing.**

### Controls

| control | transformer | VAE |
|:--|--:|--:|
| dummy's effect on the **real** kernels (device busy, dummy subtracted) | −2.4 ms (−1.2%) | +2.3 ms (+1.2%) |
| `k=0` baseline spread across the sweep | 1.6% | 2.0% |
| SM clock across arms | 1980 MHz throughout | 1980 → 1950 MHz at the two largest arms |

The dummy adds device time and nothing else: it does not slow the real kernels in either region. The
VAE's mild clock droop at the largest arms is the likely reason its slope reads 1.09 rather than 1.00 —
the reading is "≈1", not "1.09 exactly".

## Why a single global slope was never going to work

The wall is not `max(host, device)`. It is a **sum over segments**:

```
W(α) = Σ_s max( H_s , α · D_s )
```

so `dW/dα` is the **device-bound segments' share of the cycle**, not a property of the runtime. An
experiment that injects only into host-bound segments is *predetermined* to return zero, and one that
injects only into device-bound segments is predetermined to return one. My first sweep did the former
and I nearly published "device time is free" from it.

This also disposes of the objection that adding device work and removing it are different. `W` is a max
of affine functions of `α`, hence **convex**, and it is non-decreasing since `D_s ≥ 0`. So
`0 ≤ W'(1⁻) ≤ W'(1⁺)`: measuring a slope of ~0 by *adding* device work does license the conclusion that
*removing* it buys nothing — within that segment.

## What each regime is

| | transformer stack | observation VAEs |
|:--|:--|:--|
| kernels | ~18,500, ~10 µs each | ~81 conv sites, 0.58–2.7 ms each |
| host issue per block/site | ~1.10 ms per block for 62 kernels | microseconds |
| device per block/site | ~0.64 ms | milliseconds |
| **bound by** | **the host** | **the device** |
| share of the cycle's gap time | 95.4% | 2.1% |
| slope | **−0.07** | **+1.09** |
| lever that works | fewer Python-originated dispatches (~35 ms total, capped) | faster/fewer device kernels, pays ~1:1 |
| lever that does nothing | any kernel or layout change | dispatch tidying |

The transformer is host-bound because it issues 62 tiny kernels per block against 0.64 ms of device
work — 1.10 ms of host issue against 0.64 ms of device, so the queue drains and stays drained. The VAE
is device-bound because a single 3×3×3 convolution runs for milliseconds while its dispatch costs
microseconds.

## Consequences for every result in this project

**P007 is NOT purely device-side, and I had that wrong twice.** A positive control toggled the conv
layout in-process and measured both quantities on the same footing
([`probe_p007_passthrough.py`](eval/lingbot_va_robotwin/probe_p007_passthrough.py)):

| | layout off | layout on | delta |
|:--|--:|--:|--:|
| device busy | 246.3 / 246.4 ms | 190.8 / 189.5 ms | **−56.2 ms** |
| device events | 46,992 / 46,986 | 18,603 / 18,602 | **−28,387** |
| cycle wall | 546.2 / 654.6 ms | 423.8 / 419.0 ms | **NOT EVALUATED** |

The off-arms drifted **18.1%** between repeats, so this run's own cycle number is refused by its gate —
toggling `memory_format` in-process leaves hysteresis that twelve settle cycles do not clear. But the
device columns agree across repeats to **0.04%**, and those are the ones that were unknown.

**The device saving is 56.2 ms, not the ~130 ms I extrapolated from per-conv microbenchmarks.** That
extrapolation took 2.659 → 0.581 ms — the *largest* full-resolution signature — and multiplied it by 62,
ignoring that most convolutions are smaller and that the half-resolution VAE's are smaller again.

Combining the measured device delta with P007's published cycle delta (490.4 → 330.2 ms in-process, its
own ABBA certificate) and the VAE slope of ~1:

| term | ms | share |
|:--|--:|--:|
| device time removed | 56–61 | **~37%** |
| 28,387 kernel launches removed, ≈3.5 µs each | 99–104 | **~63%** |
| **total** | **160.2** | |

**So the host term is the larger one, and neither of my previous labels was right.** "Host-op
elimination" (the original) and "device-side, full stop" (my correction this morning) are both
incomplete.

**What P007 actually did was change the regime.** Before it, the VAE issued 46,992 tiny `vol2col`
kernels — a host-bound region by the same arithmetic as the transformer. After it, the VAE issues 18,603
events dominated by large cuDNN convolutions and is device-bound at slope 1.09. It won on both terms
*because* it moved the work across the boundary. That is why it is the only intervention in this project
that has ever paid, and it is a better lesson than either single-term attribution.

One caveat this creates for the decomposition above: the slope of 1.09 was measured in the *post*-P007
VAE. Pre-P007 that region was host-bound, so its slope was nearer 0 and the split is approximate at the
edges. The regime moved during the intervention, which is exactly what makes a clean linear attribution
impossible.

**Every failed Layer 5 kernel attempt was in the wrong regime.** The RoPE kernel, fused QKV, and every
proposed transformer fusion targeted device time in the region where the slope is zero. They were not
too small; they were in the wrong place. `attention` at 44.4 ms and `cat` at 19.7 ms of device time are
*unreachable* — not merely low-value.

**The Layer 6 dispatch work was in the right regime but is capped.** 34,635 Python-originated
dispatches × 1.017 µs ≈ 35 ms, nearly all inside the transformer, which is exactly where host work
counts. That is the honest ceiling: ~35 ms of a 331 ms cycle, ~10%.

**The remaining ~139 ms of gaps is the host issuing the transformer's 18,500 kernels.** Diffuse at
7.5 µs each ([LAYER6_GAPS.md](LAYER6_GAPS.md)), no synchronization, no allocator. Reducing the *number*
of transformer kernels attacks it; making them faster does not.

## Where the cycle actually is

| | ms | regime | reducible by |
|:--|--:|:--|:--|
| VAE device work | ~36 *(inferred: 190.1 total − ~156 transformer)* | device-bound, slope 1.09 | kernels, layout, backends — pays ~1:1 |
| transformer device work | ~156 | host-bound, slope ~0 | **nothing on the device side** |
| transformer host issue | ~139 | the eager floor | fewer kernels, or fewer Python dispatches (≤35 ms) |

P007 already took the large win in the only region where device work pays, and it took it by a factor
of 4.35–7.24× on those convolutions. What remains there is ~36 ms.

## What this does not settle

- **The VAE slope is measured by adding compute-bound matmuls, not by shortening convolutions.**
  Convexity licenses the direction, but the segment boundaries could shift if the VAE's device time
  fell far enough to make *it* host-bound too. The ~36 ms that remains is small enough that this is a
  live concern, not a theoretical one.
- **The 1.09 exceeds 1.0** and the clock dropped 1980 → 1950 MHz on the two largest arms. Read it as
  "≈1", and do not price anything at 1.09.
- **The in-process P007 toggle failed its own drift gate** (18.1% between repeated off-arms). Its device
  columns are used because they agree to 0.04% across repeats; its cycle column is not used at all, and
  the 160.2 ms comes from P007's published certificate instead. A clean in-process cycle A/B would need
  either far longer settling after each `memory_format` toggle or one arm per process.
- **Only two regions were probed.** `_infer`, `prepare_latent_input`, the schedulers and
  `postprocess_action` together hold ~2.5% of gap time and were not measured; they are too small to
  matter but their slopes are unknown.
- **`34.1 µs/launch` from the transformer sweep is a cuBLAS `mm` dispatch cost** — heuristic lookup,
  workspace, launch — not a generic kernel launch cost. Do not generalise it. The generic figure from
  the gap inventory is ~7.5 µs.
- **The k=0 baselines here read ~390–400 ms**, above the 330.7 ms measured in
  [LAYER6_GAPS.md](LAYER6_GAPS.md), because every block or conv forward is wrapped in a Python closure
  in all arms. It inflates the baseline and differences out of the slope, but percentages against these
  baselines are not percentages of the served cycle.

## Further reading

- [LAYER6_GAPS.md](LAYER6_GAPS.md) — the 139 ms of gaps, diffuse, and the eager floor
- [LAYER6.md](LAYER6.md) — the dispatch inventory and the 1.017 µs slope
- [LAYER5.md](LAYER5.md) — P007, and why backend/layout selection is the Layer 5 flow

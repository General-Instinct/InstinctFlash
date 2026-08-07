# Fast profile decomposition — and a retraction

Measured 2026-08-07 on an idle 8×H100 box, LingBot-VA at the Fast operating point (2 video / 4 action),
in-process on one GPU with the shipped pass chain: substrate elision, conditioning prefill, ring KV,
graph capture. Probe: [`eval/lingbot_va_robotwin/profile_fixed_term.py`](eval/lingbot_va_robotwin/profile_fixed_term.py).

---

## The retraction first

The cost model quoted across this repository was

```
cycle = FIXED 1164 ms + 15.5 ms/forward        (R² = 0.994, 5 operating points)
```

and from it the conclusion that **93% of Fast latency is fixed overhead** no per-forward optimization
can touch. That conclusion appeared in README, ARCHITECTURE.md, CHECKPOINTS.md and ATTENTION.md, and it
was used to deprioritize Layer 4.

**It is wrong.** A direct phase decomposition attributes 99.0% of the Fast cycle to two components:

| phase | ms/cycle | share | calls/cycle |
|:--|--:|--:|--:|
| transformer forwards | 831.7 | **80.8%** | 10 |
| keyframe VAE encode | 181.8 | **17.7%** | 1 |
| `prepare_latent_input` | 3.2 | 0.3% | 9 |
| `_compute_kv_cache` (exclusive of the above) | 0.9 | 0.1% | 1 |
| action scheduler steps | 0.5 | 0.0% | 4 |
| `postprocess_action` | 0.3 | 0.0% | 1 |
| video scheduler steps | 0.2 | 0.0% | 2 |
| `set_timesteps` | 0.1 | 0.0% | 2 |
| `preprocess_action` | 0.1 | 0.0% | 1 |
| debug dump (elided) | 0.0 | 0.0% | 3 |
| **attributed** | **1018.9** | **99.0%** | |
| unattributed remainder | 10.6 | 1.0% | |

There is no large unexplained fixed term. The 1164 ms intercept does not correspond to any component
of the cycle.

### Why the regression lied

Three reasons, and the first is the one that generalizes:

1. **Per-forward cost is not constant across the configurations I fitted.** The fit used
   (2,2), (4,4), (8,8), (12,12), (25,50). Graph-capture behaviour, KV extent and allocator pressure all
   differ between those points, so a straight line through them has a slope that is an average of
   different regimes and an intercept that absorbs the residual. R² = 0.994 measures how straight the
   line is, not whether the model is real — five points will look linear under a wide range of wrong
   models.
2. **The forward count was wrong.** Fast runs **10** forwards per cycle, not 6. Each denoise loop pads
   a terminal timestep and runs one extra cache-only forward whose output is discarded
   (`wan_va_server.py:502-508`), and the KV refresh adds 2: `3 + 5 + 2 = 10`. Quality is
   `26 + 51 + 2 = 79`, which is the long-standing figure — so the inconsistency was visible and I did
   not check it.
3. **An intercept is not a component.** "FIXED" was a name I gave to a residual. Naming a residual
   invites treating it as a thing, and then as a target.

The specific damage: attention was ranked near-last for Fast on the strength of a 7% denominator that
is actually 81%. That ranking is withdrawn.

---

## What the components are

**Transformer forwards — 81%, 10 per cycle.** This is the cycle. Any Layer 4/5 work multiplies against
it, and step reduction multiplies the 8 denoise forwards of the 10.

**Keyframe VAE encode — 18%, one call of 182 ms.** Each cycle encodes the keyframe observations handed
to `_compute_kv_cache` (8 frames in episode mode). This is untouched by any step reduction, any
attention backend, and any kernel fusion — an entirely separate lever that nothing in the layer
roadmap addresses.

**Everything else — under 1% combined.** Schedulers, input preparation, action pre/postprocessing, and
the elided debug dump total ~5 ms. There is nothing here worth optimizing, which is itself useful: it
closes off a class of speculation.

---

## What this profile does NOT establish

Stated explicitly, because the numbers above are load-bearing and two things about them are not settled.

**The absolute per-forward cost.** 831.7 ms over 10 forwards is 83 ms/forward, and that cannot be the
steady-state figure: 79 forwards at Quality would then be 6.6 s against a measured 2315 ms. The
instrumented pass began from dropped graph pools, so it re-captures, and the per-forward number carries
capture cost. **The 81/18 split is a ratio measured within one pass and is sound; the 83 ms/forward
absolute is not.** Pinning it down needs the served multi-GPU configuration in warm steady state.

**Why the two passes differ by 47%.** Uninstrumented 1930 ms, instrumented 1030 ms — the *instrumented*
pass is faster, which is backwards. The likely cause is allocator pressure rather than
instrumentation: pass 1 ends with 64 held graphs and ~1.2 GiB free of 79 GiB, and pass 2 starts after
those pools are released with 46 GiB free. If that is right, holding graphs costs more in allocator
thrash than it saves in replay at this operating point — which would be a second, independent argument
that graph capture is unprofitable at Fast. It is a hypothesis, not a result, and the ABBA confirmation
that would settle it has not produced a number yet.

---

## Chosen next optimization

From the breakdown, not the roadmap:

**1. The keyframe VAE encode — 182 ms/cycle, 18%.** The clearest target. It is a single call, it is
independent of every other lever, and no pass in the stack addresses it. Worth understanding before
optimizing: whether all 8 keyframes need encoding every cycle, or whether the encode can be
incremental across a sliding window as the ring already is for KV. If two thirds of it is redundant
that is ~120 ms/cycle for a Layer 3-style cache, in a place nobody has looked.

**2. Per-forward cost, measured properly first.** Forwards are 81% of the cycle, so this is where the
mass is — but the absolute number above is contaminated, and optimizing against a contaminated
baseline is how the last three rankings went wrong. The prerequisite is a warm steady-state per-forward
measurement on the served configuration. Only then is it worth asking whether attention, fusion, or
step count is the right cut.

**3. Not graph capture, yet.** The pass-ordering artifact points at it being a net loss at Fast, and the
earlier single-shot measurement pointed the same way. Two weak signals agreeing is a reason to measure,
not to act.

The order matters: (1) is independent and actionable now; (2) is larger but blocked on a measurement;
(3) is blocked on the same measurement.

---

## Probe defects found and fixed

Recorded because each produced a plausible wrong answer, and the first three all reported *something*:

| | Symptom | Cause |
|:--|:--|:--|
| 1 | `transformer forwards: 0.0 ms over 0.0 forwards` | patched `instance.__call__`; Python resolves special methods on the **type**, so `module(...)` never saw it. `forward` is the right hook |
| 2 | 99.8% unattributed | timed `_infer` only. A cycle is `_infer` **plus** `_compute_kv_cache` over 8 keyframes — the entire VAE encode was outside the measured region |
| 3 | attributed 1508 ms of a 1961 ms cycle, `kv_refresh` 454 ms **and** `vae_encode_obs` 183 ms | phases nest; inclusive timing double-counts. Now exclusive: each timer subtracts its children |
| 4 | OOM at 76 GiB **allocated** in the VAE conv3d | bypassed `server.infer()`, skipping the bookkeeping that advances `frame_st_id`, so the ring never advanced and KV grew without bound. `empty_cache` cannot help when memory is allocated rather than cached |
| 5 | shares understated ~45% | shares taken against pass 1's total while components came from pass 2 |

Defect 4 is the one worth generalizing: **measure through the entry point the system actually uses.**
Calling the internals directly skipped invariants maintained between them, and the failure surfaced two
layers away as an allocation error.

---

# Warm steady state, and the Layer 4/5 re-ranking

Measured 2026-08-07, idle fleet, 2V/4A, in-process, one allocator state, **graph capture OFF** (a
per-forward number for ranking should describe the model's compute, not the launch machinery).
Probe: [`profile_forward_warm.py`](eval/lingbot_va_robotwin/profile_forward_warm.py).

Convergence was demonstrated rather than assumed — the run reports NOT EVALUATED if the last two
windows disagree by more than 5%:

| cycles | median | free |
|:--|--:|--:|
| 1–15 | 1383.8 ms | 46.1 GiB |
| 16–30 | 1387.6 ms | 46.1 GiB |
| 31–45 | 497.0 ms | 46.1 GiB |
| 46–60 | 485.5 ms | 46.1 GiB |
| 61–75 | 492.7 ms | 46.1 GiB |
| 76–90 | 487.3 ms | 46.1 GiB |

**Converged: 1.8% apart, zero free-memory drift.** Reproduced twice across separate processes at
487.3 / 489.2 / 496.0 ms.

## The 30-cycle transient is not warmup, it is deployment

Cycles 1–30 run at **1385 ms**; from cycle 31 onward, **490 ms**. A 2.8× step, with free memory flat
throughout — so not allocator pressure. A real RoboTwin episode is ~53 cycles, so **more than half of
every episode runs inside the transient.** This is why "whole episode" and "late episode" numbers
differ by ~1.5× throughout our history (`fast_full`: 1835.7 whole vs 1226.1 late) and it is the largest
single thing the regression intercept was absorbing. Both numbers are correct for different questions
and neither should be quoted without saying which.

## Per-forward, warm

| phase | forwards/cycle | ms/forward | ms/cycle |
|:--|--:|--:|--:|
| kv_refresh | 1 | 29.79 | 29.8 |
| video | 3 | 29.50 | 88.5 |
| action | 6 | 29.38 | 176.3 |
| **all forwards** | **10** | | **294.6** |
| cycle total | | | 487.3 |
| **forwards as share** | | | **60.5%** |

**~29.5 ms/forward** — not the 15.5 ms the regression claimed, and not the 83 ms the
capture-contaminated probe reported. Note the cost is *flat* across phases despite very different
token counts, which is the signature of a per-call floor rather than compute.

## GPU time by category — the re-ranking

| category | ms/cycle | share |
|:--|--:|--:|
| **elementwise / layout (Layer 5)** | **112.0** | **45.9%** |
| matmul / projections (Layer 5–6) | 60.4 | 24.7% |
| attention (Layer 4) | 44.5 | 18.3% |
| conv / VAE (Layer 5) | 14.0 | 5.7% |
| normalisation (Layer 5) | 10.1 | 4.1% |
| other | 3.1 | 1.3% |
| **total GPU busy** | **244.0** | of a 487.3 ms cycle |

Two results:

**Layer 5 outranks Layer 4 by 2.5×.** Elementwise and layout kernels are 46% of GPU time — the single
largest category, and larger than attention and matmul combined. `CatArrayBatchedCopy` alone is 16.6
ms/cycle. Attention is 18.3%, which is real but is not where the mass is. The earlier 7% figure was
too low and the roadmap's instinct to reach for attention first is still wrong, but for a different
reason than the retracted model gave.

**The GPU is idle for half the cycle.** 244 ms busy of 487 ms. Combined with the flat 29.5 ms/forward,
this says the warm cycle is launch/dispatch-bound, not compute-bound — which is Layer 2 territory, and
sits awkwardly beside the evidence that graph capture is unprofitable here. That tension is the next
thing worth resolving, and it is a measurement, not a build.

### Auditing this table took three attempts

Recorded because each wrong version produced a confident number:

1. `"mul" in name` matched **matmul**, filing every GEMM as elementwise. Matmul read 0.3% of a
   transformer's GPU time — implausible, which is the only reason it was caught.
2. Fixing that changed nothing, because the keys are raw CUDA kernel names and I was still guessing:
   the GEMMs are `nvjet_tst_128x96_64x7_4x1_v_bz_bias_TNT`, which contains no `gemm` or `matmul`
   substring at all. `"add_"` also matched `badd_` inside one, pulling a GEMM into elementwise.
3. Only after printing the raw keys did the buckets become correct. The probe now always prints the
   top 22 keys with their assigned bucket, because **a bucketing whose inputs are invisible cannot be
   audited**, and an unauditable 46% is not a finding.

---

# The observation encode: the proposed optimization already exists

Probe: [`probe_obs_encode.py`](eval/lingbot_va_robotwin/probe_obs_encode.py). Three cameras, 8
keyframes, 256×320.

**The incremental-reuse proposal is already implemented, upstream, and bit-exact.** `StreamingVAE`
threads `feat_cache` through 26 `WanCausalConv3d` layers; after one encode, 10 of 26 slots hold
temporal context. Prior frames are carried, never recomputed. And the real client
(`eval_polict_client_openpi.py:655-662`) builds `key_frame_list` fresh each cycle from *new* simulator
observations — it is not a sliding window, so there is nothing repeated to cache in the first place.

**A content-keyed latent cache would be a correctness bug, not a missed optimization.** Since the
encode is a function of (pixels, history), caching on pixel content returns a latent computed against
the wrong temporal context.

**The chunk size is not a free parameter.** Every attempt to vary it fails, and the failures are the
finding: 2 frames gives *"padded input (2 × 16 × 20), kernel (3 × 1 × 1)"* — the causal conv needs a
temporal extent of ≥ 3, so "encode one new frame per cycle" cannot exist. 4/8/16 give *"size of tensor
a (n) must match tensor b (n/2)"*, because the full-res and half-res streaming VAEs carry independent
caches that desynchronise when the chunk size changes. So "send fewer keyframes" is not a knob that
turns in isolation.

**And the host-side preprocessing is not the problem.** `_encode_obs` stacks, promotes to fp32 and
bilinear-resizes on the CPU before uploading. Measured standalone: **4.7 ms**, against **1.6 ms** for
uploading uint8 and resizing on the device. Real, but ~3 ms/cycle — 0.6% of a cycle. Not worth a
NUMERIC-tier change (CPU and GPU bilinear are different reductions, so it would need the paired
protocol).

**So the 182 ms figure needs re-reading.** In the warm profile, conv/VAE GPU time is only 14.0
ms/cycle. The encode's wall time is therefore mostly *not* GPU compute and *not* host preprocessing —
it is launch overhead across 26 tiny causal convolutions × 3 cameras. That makes it the same problem as
the 50%-idle finding above, not a separate one.

## Revised recommendation

The keyframe VAE encode was the previous recommendation; **it is withdrawn.** Nothing in it is
cacheable, its chunk size is not adjustable, and its host half is worth 3 ms.

What the two clean measurements point at instead, in order:

1. **The 30-cycle transient (~900 ms/cycle for the first 30 cycles).** Largest single effect measured,
   affects more than half of every real episode, and nothing in the stack targets it. Find out what
   changes at cycle 31.
2. **Elementwise/layout fusion (Layer 5), 46% of GPU time.** Concentrated enough to attack —
   `CatArrayBatchedCopy` at 16.6 ms/cycle is one kernel.
3. **The launch-bound half of the cycle (Layer 2).** 244 ms busy of 487 ms, with a flat per-forward
   cost. Resolve against the graph-capture evidence before building anything.

Attention (Layer 4) is 18.3% — better than the retracted 7%, still third.

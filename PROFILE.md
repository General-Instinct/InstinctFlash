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

---

# Layer 5, target 2: the VAE convolutions were on a fallback path

Measured 2026-08-07, idle fleet. Probes:
[`probe_vae_conv_backend.py`](eval/lingbot_va_robotwin/probe_vae_conv_backend.py),
[`probe_vae_channels_last.py`](eval/lingbot_va_robotwin/probe_vae_channels_last.py).

## The cause is memory layout, not a missing kernel

`slow_conv_dilated3d` was 21.7 ms/cycle at 42 calls. The name is misleading: `WanCausalConv3d` sets
`padding=(0,0,0)`, pads explicitly with `F.pad`, and never sets `dilation`, so nothing is dilated. It is
simply where PyTorch lands when its 3D backends decline.

Per-signature, in NCDHW vs NDHWC:

| input | weight | as-is | channels_last_3d | |
|:--|:--|--:|--:|--:|
| (1,160,8,128,160) | (160,160,3,3,3) | 2.659 ms `slow_conv_dilated3d` | 0.581 ms `cudnn_convolution` | **4.58×** |
| (1,320,8,64,80) | (320,320,3,3,3) | 2.702 ms | 0.540 ms | **5.00×** |
| (1,12,8,128,160) | (160,12,3,3,3) | 1.336 ms | 0.307 ms | **4.35×** |
| (1,160,8,64,80) | (320,160,3,3,3) | 2.593 ms | 0.358 ms | **7.24×** |

Every 3×3×3 convolution in NCDHW falls back; every one of them reaches cuDNN in NDHWC. The 1×1×1
convolution already reached cuDNN, which is why 16 of 62 convs were never on the slow path.

**`cudnn.benchmark=True` changes nothing (1.00× on all four).** So this is not heuristic search failing
— cuDNN has no NCDHW bf16 3D kernel for these shapes on H100 / torch 2.9 / cuDNN 9.10, and PyTorch
falls back rather than transposing. That distinction matters: `benchmark` would have been a
search-strategy flag with no arithmetic change and therefore BITEXACT-eligible. Layout is not.

## At whole-encode scale it is much larger than the conv time

Converting only the 62 Conv3d weights (`module.to(memory_format=...)` raises *"required rank 5 tensor"*
on the rank-1 RMSNorm weights) and the input, once:

| | fallback convs | cuDNN convs | conv time | **whole encode** |
|:--|--:|--:|--:|--:|
| NCDHW (as shipped) | 46 | 16 | 16.69 ms | **175.72 ms** |
| NDHWC | **0** | 62 | 6.16 ms | **17.00 ms** |

**10.34× on the whole encode**, transforms included, at 0.8–1.3% spread.

The important part is the discrepancy: convolution time fell by 10.5 ms while the encode fell by
**158.7 ms**. The fallback was not merely a slow kernel — it was generating an order of magnitude more
surrounding work than it spent computing. `slow_conv_dilated3d` lowers via `vol2col`, materialising
column buffers, and that is almost certainly a large share of the **34,710 `copy_` and 29,681 `fill_`
calls per cycle** that four attribution attempts failed to place. The copy storm and the conv fallback
look like one problem, not two.

## What is NOT yet established

**The cycle-level number.** This standalone encode measures 175.7 ms for ONE camera, while the in-cycle
profile attributes 181.8 ms to `_encode_obs` for all THREE. Those cannot both describe the same work,
so the standalone figure is not transferable: `clear_cache()` on every iteration forces a full
recompute, and the chunk sequence differs from the real incremental one. **Do not quote 32.6%.** The
defensible bound from the full-cycle profile is 21.7 ms (the conv line) up to ~88 ms (conv plus the
copy/fill population it plausibly generates) — 4.5% to 18% of a 487 ms cycle. Only the cycle gate
settles it.

**The numerics.** `max|delta|` on the encoder output is 1.25e-01, relative 6.67e-03 — about 1.7× bf16
resolution at that magnitude, so it is a real difference, not a rounding artefact. NDHWC changes the
convolution's accumulation order. This is **NUMERIC tier**: it cannot ship under `max|Δ| = 0` and needs
paired non-inferiority on pinned seeds. The encoded latents feed the KV cache, so it propagates to
actions.

## Also corrected

PROFILE.md previously said the VAE chunk size was bound by cache state and that `_reset` "does not
resynchronise" the two VAEs. The real rule is sharper and is a property of the architecture: the Wan
causal VAE downsamples time by 4, so the **first** chunk after `clear_cache` must have `T = 4k+1` and
every later chunk `T = 4k`. Feeding 8 frames as the first chunk raises *"size of tensor a (8) must match
tensor b (4)"* inside the residual shortcut. That is exactly why the real flow works: `_infer` encodes
one frame at `frame_st_id 0`, then `_compute_kv_cache` sends 4 and then 8.

## Next

1. **Cycle-level gate for the layout change**, at 2V/4A warm. It is the only number that decides.
2. If it holds, it needs paired non-inferiority, not a bit-exactness gate — budget the episodes.
3. The `cat` path (21.6 ms, 172 calls, 125.6 µs/call) and the remaining `copy_` population are both
   still open, but the copy population should be re-profiled AFTER the layout fix: if `vol2col` was
   generating most of it, a large part of that 66 ms disappears and the target list changes.

---

# The conv dispatch layer, and the full-cycle result

Backends: [`instinctwm/backends/conv/`](instinctwm/backends/conv/). Gate:
[`tests/test_conv_backend.py`](tests/test_conv_backend.py). Cycle measurement:
`profile_forward_warm.py --conv-layout ndhwc`.

## Full 2V/4A cycle, warm, both arms converged

| | cycle | GPU busy | forwards | transient (cycles 1-30) |
|:--|--:|--:|--:|--:|
| as shipped (NCDHW) | **490.4 ms** | 244.0 ms | 296.4 ms (60.4%) | 1391 ms |
| conv dispatch (NDHWC) | **330.2 ms** | 191.7 ms | 288.6 ms (87.4%) | 1228 ms |
| | **1.49×** | −52.3 ms | −7.8 ms | 1.13× |

Convergence 1.2% and 2.1% between final windows, zero free-memory drift in both. **No kernel was
written**; this is dispatch.

**The cycle fell 160.2 ms while GPU-busy fell only 52.3 ms.** So two thirds of the win is host-side —
kernel launches that no longer happen. That is the quantitative confirmation of the earlier suspicion
that the fallback and the copy storm were one problem.

## The copy population, finally explained — by removing it

| operator | before | after | |
|:--|--:|--:|:--|
| `copy_` | 66.42 ms / **34,710** calls | 28.34 ms / **6,385** calls | −82% of calls |
| `fill_` | 1.69 ms / **29,681** calls | 1.70 ms / **1,361** calls | −95% of calls |
| `slow_conv_dilated3d` | 21.74 ms / 42 | **gone** | |
| `vol2col_kernel` | 13.98 ms | **gone** | |
| `cudnn_convolution` | 0.72 ms / 22 | 7.94 ms / 64 | now serves all 62 |
| `addmm` | 52.21 ms | 52.48 ms | unchanged |
| attention | 44.92 ms | 44.95 ms | unchanged |
| `cat` | 21.60 ms / 172 | 19.68 ms / 172 | unchanged |

**28,325 of the 34,710 `copy_` calls were the conv fallback's own `vol2col` lowering**, along with 95%
of the `fill_` calls. Four attribution methods failed to place them; the answer was that they were not a
target at all but a symptom. The per-call cost also rose from 1.9 µs to 4.4 µs, which is the signature
of what remains being real data movement rather than a launch storm.

This is the whole argument for optimizing the execution graph after dispatch rather than the largest
operator in a trace. `copy_` was the biggest line in the profile and writing a copy kernel would have
been wasted work: the fix was one layout decision two layers away.

## Stabilized distribution, and what it says about the next target

| category | ms/cycle | share of GPU busy |
|:--|--:|--:|
| elementwise / layout (L5) | 74.0 | 38.6% |
| matmul / projections (L5-6) | 60.2 | 31.4% |
| attention (L4) | 44.4 | 23.2% |
| normalisation (L5) | 10.1 | 5.3% |
| other | 2.9 | 1.5% |
| **total GPU busy** | **191.7** | of a 330.2 ms cycle |

Forwards are now **87.4%** of the cycle, up from 60.4% — the non-forward work collapsed. GPU busy is
191.7 of 330.2 ms, so **42% of the cycle is still idle** and the remaining problem is launch-bound.

By best-attributed contribution the next candidates are:

1. **`cat`, 19.68 ms, 172 calls at 114.4 µs, one shape signature.** Unchanged by the layout fix and
   still the most concentrated line in the profile. At ~114 µs per call it is moving real bytes —
   consistent with assembling a 9792-token KV window per layer. The right move remains eliminating the
   materialization or teaching the consumer to read segmented KV, not writing a faster `cat`.
2. **`addmm`, 52.48 ms, 2,444 calls.** Already cuBLAS via `nvjet`; unlikely to beat, but it is now the
   largest single operator and worth confirming it is on the best path — the same question that just
   paid 1.49× for convolutions.
3. **The 42% idle.** Launch-bound, which is Layer 2, and it sits against the graph-capture evidence.

Attention is 23.2% of GPU busy — its share rose because everything around it shrank, not because it
changed.

## What still gates this

**NUMERIC tier, not BITEXACT.** NDHWC changes the convolution's accumulation order:
`max|delta|` 1.25e-01 on the encoder output, relative 6.67e-03, ~1.7× bf16 resolution. The latents feed
the KV cache, so it propagates to actions. It **cannot** ship under `max|Δ action| = 0` and needs paired
non-inferiority on pinned seeds. `select()` enforces this structurally: with the default ceiling it
returns the incumbent even when handed measurements showing a 10× win, and only an explicit
`prefer_bitexact=False` lets the NUMERIC pair through.

**Also required:** the half-resolution VAE that serves the wrist cameras has its own 62 convolutions and
must be converted too, or two thirds of the encode stays on the fallback path.

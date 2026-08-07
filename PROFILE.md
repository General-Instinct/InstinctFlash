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

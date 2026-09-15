# Dynamic denoising-step caching

Status: source audit and CPU decision-rule verification, 2026-09-14. This RFC
specifies the shared implementation; the proposed public API below is **not
implemented yet**. Existing native DreamZero dynamic scheduling remains available
through its explicit behavioral option. No new Thor latency or task-quality
result is claimed here.

## Decision

Add a shared, opt-in controller for approximate reuse of denoiser predictions.
First integrate DreamZero with the exact upstream decision rule; qualify native
precision and FP8 separately. Share lifecycle, configuration, owned storage and
telemetry across adapters, while keeping each model's prediction semantics and
sampler inside its adapter. Do not apply DreamZero thresholds to every family.

This belongs beside the exact [tensor result cache](../../instinctflash/runtime/tensor_cache.py),
not inside it. That cache reuses a pure computation with identical dependencies.
Dynamic step caching reuses an older prediction after the latent and timestep
have changed, so it changes the computation even when every tensor remains BF16.

## What the audited implementations actually do

Sources are DreamZero `ab790c198fbce33503358efbbd4187ce9a89adf3` and the benchmark's
vLLM-Omni `f7d9deb45ab56e6a2ccc1690279bd9e6bdefbfe3`. Exact files, hashes and line
references are in the [source audit](../../eval/dynamic_step_cache_2026-09-14/source_audit.md).

1. Execute the first two denoiser steps to establish prediction history.
2. Flatten the two most recent **computed video velocities**, convert the
   similarity inputs to FP32, and calculate mean batch cosine similarity.
3. Above `0.95`, reuse the last prediction for four consecutive slots; above
   `0.93`, reuse it for two. The trigger slot is included in that count. After
   the countdown, force one new denoiser evaluation before another decision.
4. Reuse both CFG-combined video velocity and conditional-branch action velocity.
   The action stream does not receive the video's CFG combination and is **not
   consulted by the similarity test**.
5. Advance both original schedulers at every slot, including skipped DiT slots.
   Retain the original timestep grids, noise draws and KV prefill/commit.
6. Start new prediction history and countdown for every denoising chunk. Causal
   observation KV has a different lifetime and remains across control cycles.

The native baseline already computes a fixed eight of sixteen DiT slots. Dynamic
selection replaces that mask; it is not a filter applied on top of the mask.
Disabling native dynamic scheduling restores fixed8; disabling Omni's step-cache
backend computes all sixteen. A cross-framework `cache off` label alone does
not establish matching computation.
For a synthetic always-aligned video signal, the upstream rule computes indices
`[0, 1, 6, 11]`. An unstable signal can compute all sixteen, making it **more
expensive than the shipped eight-slot mask**. There is no forced-final-step rule.
The [CPU probe](../../eval/dynamic_step_cache_2026-09-14/cpu_probe.json) checks
these boundaries against extracted upstream and Omni functions. It measures
decision logic, not real-model latency or quality.

At the current DreamZero CFG5 single-GPU route, one denoising evaluation has two
transformer branches. A skipped slot therefore avoids two denoising forwards.
Observation prefill/commit forwards still execute and must be counted separately.

For a fixed configuration, a useful cost model is
`T = T_pre/post/KV + N_computed * T_denoiser + T_decisions`.
Changing eight computed slots to four approximately halves only the denoising
term. It does not establish a 2x E2E gain, and an unchanged two-forward CFG pair
must not be counted as a second, additional 2x gain. Profile the current Thor
checkpoint before estimating a target latency.

The paper reports reducing average DiT evaluations from sixteen to four in its
study. The current Thor Omni receipt records `step_cache=true`, but does not
record the realized skip mask. It cannot establish four evaluations on our
requests, nor isolate how much of the Flash/Omni gap comes from caching.
[DreamZero paper, sections 3.2 and D.1](https://arxiv.org/html/2602.15922v1),
[upstream source](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py),
[Omni source](https://github.com/vllm-project/vllm-omni/blob/f7d9deb45ab56e6a2ccc1690279bd9e6bdefbfe3/vllm_omni/diffusion/cache/stepcache/state.py).

## Current InstinctFlash gaps

- `examples/dreamzero/step_cache.py` provides an environment overlay. The native
  adapter already accepts `DYNAMIC_CACHE_SCHEDULE=true` or a checkpoint
  declaration, with an explicit BEHAVIORAL permission requirement.
- This option is resolved during lazy model construction. Declaration-only
  planning and `execution_policy` before loading do not fully describe it.
- The dynamic plan records `computed_steps` from the fixed-mask environment knob,
  not an observed dynamic count. Replace that ambiguous field with separate
  fixed-mask configuration and per-request execution statistics.
- Native construction mutates the process environment. Resolve the option once
  and pass it to the owned head; do not let a later environment change alter a
  lazy-loaded runtime's policy or leak into another runtime.
- `build_fp8` currently calls native construction with `plan=None` and rejects
  every dynamic schedule. Enabling FP8 plus dynamic caching requires explicit
  plan propagation, installed-head checks and new paired qualification, rather
  than removal of the existing guard alone.
- The current four-framework table measured fixed8/16 Flash against dynamic
  Omni. Its receipts are preserved; do not rewrite them as a matched cache study.

## Public configuration and reporting

Proposed convenience API, following adapter qualification:

```python
runtime = Runtime.from_pretrained(
    "GEAR-Dreams/DreamZero-DROID",
    precision="native",
    tier_ceiling="behavioral",
    step_cache="dynamic",
)
```

`step_cache=None` preserves the checkpoint's existing execution policy, including
DreamZero's shipped fixed mask. It does not mean sixteen uncached evaluations.
`"dynamic"` must resolve to a named, versioned adapter profile, initially
`dreamzero_velocity_v1`, with its exact thresholds, countdowns and signal named
in the execution receipt. Unsupported families fail preflight instead of silently
falling back. A typed configuration may override profile parameters for research;
those overrides become a distinct operating point in the receipt.

| Selected arithmetic | Additional dynamic reuse | Permission / category |
| --- | --- | --- |
| Native | None | Existing checkpoint-relative transformation policy |
| Native | Explicit | BEHAVIORAL / OPERATING-POINT |
| FP8 | None | Explicit FP8, at least NUMERIC |
| FP8 | Explicit | Explicit FP8 plus BEHAVIORAL / OPERATING-POINT |

FP8 and NUMERIC permission alone never authorize dynamic skipping. Quality
evidence remains independent of permission. Changes to a shipped dynamic policy
also need their own comparison; a fine-tuned checkpoint cannot inherit its
family's quality result automatically.

Normalize and validate in both `Runtime.from_pretrained` and `plan_declaration`,
before device allocation. Thread the immutable selection through `choose_backend`,
lazy `InProcessBackend`, `EngineBackend` and supported workers. Extend the typed
CLI with `--runtime.step_cache`; unsupported worker combinations must reject.
Legacy environment options should resolve once through the same path, with
explicit precedence/conflict handling, and remain visible in preflight metadata.

Record requested, resolved and installed configuration separately. Keep
`execution_policy.nfe` as the scheduler grid. Per-request backend statistics add:

- scheduler slots, computed slots, reused slots and realized compute mask;
- conditional/unconditional denoiser forwards and separate prefill/commit calls;
- selected profile, thresholds, decision signal and forced-refresh reasons;
- arithmetic, checkpoint/source hashes, cache reset generation and bounds;
- whether statistics came from a real execution or only a planned configuration.

No nominal `nfe=4` label for a sixteen-slot solver with four observed forwards.

## Shared controller and adapter boundary

Implement `instinctflash/runtime/step_cache.py` with a Torch-lazy configuration
layer and a per-generation controller. The controller decides compute/reuse,
owns bounded prediction history and aggregates telemetry. The adapter supplies
the exact prediction tuple and decision signal. It continues to own CFG combine,
solver updates, RNG, action decoding and any KV side effects.

The initial lifecycle is `begin_generation -> decide -> record_prediction or
reuse -> end_generation`; reset and exceptions discard approximate history.
Create or reset it at the actual denoising-loop entry, not only at `episode()`.
No reuse across observations, prompts, robot states, samples, checkpoints or
precision modes. Concurrent requests require separate state; the existing
`Episode` convenience handle does not itself isolate a shared backend.

Keep at most two full predictions and a bounded trace for the current generation;
aggregate older telemetry. Own retained storage or explicitly lease stable
buffers: compiled/CUDA-graph outputs can alias replay buffers, and a later replay
must not overwrite either the previous similarity input or the cached action.
Cover device/stream lifetime and in-place solver consumers explicitly.

Validate finite ordered thresholds, positive bounded integer countdowns, at least
two history entries, matching output shape/dtype/device and sequential slot
indices. Invalid/nonfinite comparison signals or cached predictions cannot arm
reuse. This is an added input-domain guard: the upstream countdown branch does
not inspect finiteness. Reject invalid payloads explicitly rather than claiming
upstream parity outside the validated finite domain. Refresh invalidates the
cached prediction and countdown together. Do not silently substitute a different
skip policy under the same profile name.

Keep decisions outside captured regions; reuse the existing compiled **one-step
denoiser** when computation is required. Whole-loop graphs with fixed unrolled
steps are incompatible unless a separately qualified conditional execution path
exists. Do not execute the denoiser and discard its output while claiming a cache
hit. First measure scalar GPU-to-host decision overhead; a C++/CUDA reduction can
be considered later if it matters, with the threshold decisions checked again.

## Robotics-specific extension

`dreamzero_velocity_v1` should reproduce the upstream video-only policy for
comparison. A later, separately named action-aware profile can require both
video and action agreement, relative magnitude stability, a bounded skip budget,
and selected forced-compute slots. If actions are padded or contain mixed units,
the adapter must supply the valid action dimensions and appropriate normalization.

These checks address concrete weaknesses: cosine ignores magnitude; a stable
video direction need not imply stable robot actions; batch-mean similarity can
hide one divergent sample. They remain heuristics. They cannot guarantee the
uncomputed next velocity or certify task success, and adding final-step refresh
would deliberately depart from the upstream policy.

## Rollout and regression gates

1. **DreamZero native parity.** Extract the upstream rule for a CPU oracle, then
   compare the integrated controller against the original native dynamic path on
   Thor using identical weights, noise, requests and full multi-cycle episodes.
   Compare masks, real denoiser/solver/commit calls, both prediction streams, full
   actions and retained KV state. Test prompt/reset boundaries and failure cleanup.
2. **Matched Thor ablation.** Run a fresh 2x2: native fixed8/16, native dynamic,
   FP8 fixed8/16, FP8 dynamic. Keep the same request archive, exact request seeds,
   compilation policy, camera transforms and warm/continuation regimes. Verify
   the FP8 variant's actual mask independently: rounding can change threshold
   decisions. Compare Omni dynamic separately with its configured inputs, cache
   policy, source and execution environment all recorded.
3. **Performance and quality.** Record E2E p50/p95/p99, first-cycle/continuation
   latency, peak memory and actual call reduction; bracket candidates with
   reference runs. Separate compiler warmup. Use each model/checkpoint's existing
   quality admission standard and paired seeds. Action deltas and the CPU oracle
   are screens only. Preserve failed candidates; publish the fastest admitted
   candidate rather than the smallest latency among unqualified arms.
4. **Generalize adapters.** Reuse the controller and receipts, not thresholds or
   quality conclusions. Add a family only after its sampler and state boundaries
   pass their own tests and its measured benefit justifies the overhead.

| Family | Next integration investigation |
| --- | --- |
| DreamZero | First: native upstream parity; then FP8 interaction and Thor ablation |
| pi05, LingBot-VLA-4B, LingBot-VLA-V2 | Ten-step action flow provides a potential reuse window; retain per-step graphs and separately calibrate action signals |
| Cosmos3 Edge / Nano | Four-step UniPC has little warmup headroom; preserve multistep solver state and all prefix work; a four-to-two reuse policy needs its own study |
| GR00T N1.7 | Four-step action flow; measure graph-boundary and decision overhead before expanding |
| LingBot-VA | Separate video/action clocks, guidance and deferred/ring KV commits require a dedicated adapter; never skip a commit-bearing call |

This order makes the first release useful for the largest observed gap while
keeping the common infra reusable. It does not predict that every model will
benefit or that dynamic Flash will necessarily beat Omni after policy alignment.

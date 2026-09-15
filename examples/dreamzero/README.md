# DreamZero (GEAR-Dreams), in InstinctFlash

`GEAR-Dreams/DreamZero-DROID` is a causal video-action world-action model: 2 exterior + 1 wrist
camera with checkpoint-native resizing, a prompt, and a KV cache carried **across control cycles** within an episode
(first call warms it with one frame per camera; later calls append four) → a `(24, 8)` action
chunk from 16 scheduler steps at CFG 5.0, of which the shipped fixed mask computes 8 DiT
forwards. The adapter (backbone `dreamzero`) wraps the official serving wrapper in-process —
`DreamZeroWan225BPolicy` over `GrootSimPolicy` from the GEAR-Dreams checkout.

The wrapper's `5B` name does not identify the loaded architecture. The audited
release `96ad344138c66e82536422432ad742f015784942` has a 40-layer, width-5120 DiT,
880 latent tokens per frame and a Wan2.1 VAE. Model geometry comes from the
checkpoint; the original policy owns image transforms and action decoding.

## Run it

```bash
pip install ./examples/dreamzero
export DREAMZERO_ROOT=/path/to/dreamzero-repo      # the GEAR-Dreams checkout
```

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained("GEAR-Dreams/DreamZero-DROID")
with runtime.episode(prompt="pick up the banana and place it in the bowl") as episode:
    action = episode.predict(observation)           # -> {"action": (24, 8) float32}
```

For explicit FP8 on Thor, add `precision="fp8"` to the same Runtime call, or
`--fp8` to `instinctflash serve`. This quantizes the audited causal Q/K/V
and DiT FFN projections on Thor and preserves the selected step schedule. The
shipped default is the fixed eight-of-sixteen mask; FP8 alone does not enable
dynamic step skipping. Omit it for native precision.

Episode boundaries are load-bearing here: the KV cache outlives a cycle, so a new rollout must
be a new `episode()` (the adapter clears upstream's frame buffers and `current_start_frame`).
The native wrapper does not support `executed_action` overrides. Both native and
FP8 paths reject a non-`None` override before inference; supply the native observed
state and camera history through `observation`.
The frozen text/image encoders and VAE resolve through upstream's `ensure_file`
and must be reachable in the HF cache. The audited full checkpoint contains ten
shards totaling about 45.85 GB; loading can also read separate frozen-component
weights and requires more memory than checkpoint size alone suggests. Install the
native repository's pinned dependencies, including `av==15.0.0`.

The expanded Thor FFN recipe has no closed-loop quality certificate. The following
interface checks used the earlier Q/K/V-only recipe. The full-architecture loader avoids the discarded FP32 DiT
initialization, and all 1317 loaded DiT tensors have been verified against the
original checkpoint on Thor. Two three-cycle FP8 episodes have finite, exactly
repeated actions and verified execution of all 120 causal Q/K/V projections.
The adapter manages this checkpoint view's lifetime. Both native and explicit
FP8 public Runtime paths now pass two three-cycle episodes and view cleanup on
Thor. Repeated episodes within each measured process match byte for byte.
Cross-process reproducibility remains unresolved: later native runs differ from
an earlier run even when the earlier source is restored. A same-process control
of old/new FP8 scale reduction preserves action bytes for the tested inputs;
it does not explain the cross-process drift. Current task quality and a controlled
speedup remain unqualified. See the
[current evidence](../../eval/thor_precision_completion_2026-09-09/README.md).

## Optional dynamic step cache

Select the shared, source-gated DreamZero integration explicitly:

```python
runtime = Runtime.from_pretrained(
    "GEAR-Dreams/DreamZero-DROID",
    precision="native",
    step_cache="dynamic",
    tier_ceiling="behavioral",
)
```

Use `precision="fp8"` with the same explicit behavioral ceiling to select FP8
plus dynamic reuse. Precision and schedule permissions are independent. Dynamic
reuse is off in the shipped declaration, and the default retains the fixed mask
unless legacy environment options override it. `step_cache="checkpoint"`
restores the declaration and ignores `DYNAMIC_CACHE_SCHEDULE` and `NUM_DIT_STEPS`;
omitting the option retains those legacy overrides. Selection is frozen before
loading without modifying the process environment.

The upstream rule uses similarity between recent **video** velocities to reuse
both video and action predictions. The sixteen solver updates still execute.
Dynamic DiT counts vary and can exceed the baseline's eight; existing timings
do not establish how much of the Omni gap comes from this mechanism. This is an
**OPERATING-POINT with BEHAVIORAL permission and SCREEN evidence**, without a
task-quality certificate. The final combined CPU suite has 215 passing tests.
The [Thor screen](../../eval/dynamic_step_cache_integration_2026-09-14/results.md)
measured 1.78× native and 1.76× FP8 cache speedups, with same-dynamic-policy action/KV
parity checks. Task-quality qualification relative to the fixed baseline is pending.
The [runtime guide](../../docs/dynamic-step-cache.md) covers the implemented API,
reset boundaries and actual-call telemetry. The
[source audit and design RFC](../../docs/rfc/dynamic-step-cache.md) retain the
upstream comparison and future integration proposals.

`cfg_batch.py` / `verify_cfg_batch.py` / `diag_batch.py` are the CFG-batching research arm
(gates and honest negative results); they are not part of the served path.

## Historical H100 reproduction protocol

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> examples/dreamzero/reproduce_h100.sh
```

Both arms are the official websocket server — the only difference between them is the env var —
measured by a byte-identical client (12 calls x 3 warmup, p50, 1-then-4-frame causal protocol).
This historical protocol is separate from the current Thor four-framework table.

## Attribution

DreamZero and GEAR-Dreams are © their authors (see the upstream checkout's LICENSE).
The adapter imports the native checkout; dynamic integration installs owned hooks
after verifying the native source. The environment helper remains available for
the historical upstream-server reproduction protocol.

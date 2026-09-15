# Dynamic step cache

`step_cache="dynamic"` enables approximate reuse of DreamZero denoiser predictions.
The public selection, shared controller and native integration have CPU and
Thor validation. A bounded matched screen measured 1.78× with native precision
and 1.76× with FP8. Actions and retained-state hashes matched the upstream
**same dynamic policy** in each arithmetic route. Task-quality qualification
relative to the fixed-mask baseline remains pending. See the
[measurements and audit](../eval/dynamic_step_cache_integration_2026-09-14/results.md).

The first implementation supports only DreamZero's audited sixteen-slot,
single-GPU native loop. Construction and loop installation verify pinned native
source methods and reject an unrecognized implementation. pi05, LingBot-VLA-4B,
LingBot-VLA-V2, GR00T N1.7, LingBot-VA and Cosmos3 Edge/Nano each need a separate
adapter integration and qualification. Selecting `"dynamic"` for those families
fails preflight. Their existing caches remain independent of this option.

Select dynamic reuse with explicit BEHAVIORAL permission:

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained(
    "GEAR-Dreams/DreamZero-DROID",
    precision="native",
    tier_ceiling="behavioral",
    step_cache="dynamic",
)
print(runtime.execution_policy)
# Use runtime.predict(...) or runtime.episode(...) with the checkpoint's inputs.
runtime.close()
```

FP8 is a separate arithmetic selection. Where the checkpoint and device have a
supported FP8 backend, request both explicitly:

```python
runtime = Runtime.from_pretrained(
    "GEAR-Dreams/DreamZero-DROID",
    precision="fp8",
    tier_ceiling="behavioral",
    step_cache="dynamic",
)
```

FP8 alone grants at most its default NUMERIC permission and cannot enable
dynamic reuse. Native arithmetic with dynamic reuse is still an
`OPERATING-POINT` with a BEHAVIORAL transform. These are execution permissions;
CPU parity and action screens do not provide a task-quality certificate.

Declaration-only planning accepts the same options:

```python
from instinctflash.runtime.facade import plan_declaration

checkpoint, adapter, plan, device = plan_declaration(
    "GEAR-Dreams/DreamZero-DROID",
    step_cache="dynamic",
    tier_ceiling="behavioral",
    probe_device=False,
)
print(plan.explain())
```

The CLI forwards the selection through preflight and loading:

```bash
instinctflash serve GEAR-Dreams/DreamZero-DROID \
  --runtime.step_cache=dynamic \
  --runtime.tier_ceiling=behavioral \
  --serve.dry_run=true
```

Add `--runtime.precision=fp8` when requesting FP8. Remove `--serve.dry_run=true`
to load and serve. A changed DreamZero schedule is unsupported in worker
placement; an automatic fallback to a worker refuses it too.

The selection is resolved once before device probing and retained for lazy
loading. No process environment variables are written.

| `step_cache` | DreamZero selection |
| --- | --- |
| `None` / omitted | Read `execution.dynamic_cache_schedule`, defaulting to false, then apply legacy `DYNAMIC_CACHE_SCHEDULE` and `NUM_DIT_STEPS` if present. |
| `"checkpoint"` | Restore the declaration's dynamic flag and shipped eight-slot fixed mask; ignore both legacy environment variables. |
| `"dynamic"` | Select `dreamzero_velocity_v1` explicitly, regardless of the legacy environment. Fixed-mask provenance remains eight slots; dynamic decisions replace that mask. |

With the shipped declaration and no legacy overrides, the default remains a
fixed eight-of-sixteen schedule. It does not mean sixteen uncached denoiser
evaluations. A declaration that enables dynamic scheduling still requires a
behavioral ceiling, including when selected with `"checkpoint"`. An explicit
false legacy flag can override such a declaration when `step_cache` is omitted.
Legacy `NUM_DIT_STEPS` accepts 5, 6, 7 or 8; a reduced fixed mask also requires
BEHAVIORAL permission. FP8 supports the shipped fixed mask or the installed
dynamic policy, and continues to refuse altered fixed masks.

Use `step_cache="checkpoint"` to ignore inherited legacy schedule options.
Later environment changes cannot change an existing runtime's frozen selection.
The owned DreamZero constructor applies the selection to its two legacy
environment reads through an instance-local factory, and the loaded head is
checked and bound before serving. This does not modify the original checkpoint
or native class. A changed native constructor requires a new source audit.
The constructor gate applies to default and fixed selections as well as dynamic
reuse; the supported native source scope is intentionally pinned for each.

Standalone `Plan.without()` and `Plan.bitexact_subset()` clones do not retain the
runtime's attached resolved selection and do not reconfigure a loaded runtime.
Create a new runtime with the desired `step_cache` selection instead of using a
plan clone as a cache-off switch. Public `exclude_passes` is handled during
planning; excluding an explicitly changed `dreamzero_schedule` is refused.

The profile compares the last two **computed CFG-combined video velocities**:
flatten from dimension 1, convert the comparison to FP32, calculate cosine
similarity per batch sample, then average. Strict `>0.95` starts four skipped
slots; otherwise strict `>0.93` starts two. Each burst includes its triggering
slot. Expiry forces one computation if another slot remains, and the next slot
may start a new burst. The first two slots compute; the final slot is not forced.

A reuse decision holds both the latest CFG video prediction and the latest
conditional action prediction. Action similarity and magnitude are not part of
the gate. Both original schedulers still advance through all sixteen slots, and
the native loop retains CFG, noise draws, action processing and observation KV
updates. Dynamic mode can compute all sixteen slots for an unstable signal,
which is more work than the shipped eight-slot mask. Matching thresholds between
native and FP8 does not imply matching realized masks.

The shared implementation lives in
[`instinctflash/runtime/step_cache.py`](../instinctflash/runtime/step_cache.py).
`StepCacheConfig` is immutable and Torch-lazy; `StepCacheController` owns at most
two predictions within its configured byte limit, with bounded generation
traces. The default profile uses thresholds `(0.95, 0.93)`, skip counts `(4, 2)`
and a 64 MiB retained-prediction limit. The public Runtime option selects this
profile; custom thresholds are not exposed through that option.

Adapter authors use the sequential lifecycle
`begin_generation -> decide -> record_prediction or reuse -> end_generation`.
Every compute decision must be followed by recording its outputs, and every
reuse decision must consume a retained output. Retained tensors and returned
reuse tensors have owned storage to protect history from replay buffers and
in-place consumers. Shape, dtype, device, finite payloads, sequential indices
and storage limits are checked. Rejecting invalid payloads extends the validated
input domain guard; upstream parity is claimed only for the tested finite domain.

DreamZero starts a new cache generation at each denoising chunk, clears history
on completion or failure, and resets it with the runtime's episode reset. Causal
KV and prompt state have their own native lifetimes. One head permits only one
active generation; `Episode` handles do not provide concurrent backend isolation.
The controller requires inference/no-grad execution and the same CUDA device
and stream throughout a generation. Decisions run outside CUDA graph capture;
the computed denoiser may retain its own compiled execution. Cleanup remains
available after an exception or stream-context change.

Before loading, `runtime.execution_policy` reports the resolved selection and
`schedule_options["dreamzero_schedule"]`: sixteen grid slots, the separate
`fixed_mask_steps`, named profile and controller configuration. For dynamic
selection, `computed_steps` is `None`. This describes requested execution and
cannot establish that a hook was installed or that any forward was saved.

Read observed statistics through the public snapshot property:

```python
snapshot = runtime.backend_stats
if snapshot["status"] == "available":
    stats = snapshot["stats"]
    native_stats = stats.get("native_backend", stats)  # H100 retains this nesting
    cache = native_stats.get("step_cache")
    if cache is not None:
        print(cache["last_generation"])
```

The envelope has `status`, `backend` (the execution backend class name), and
`stats`. `available` contains a deep copy of the loaded backend's statistics.
`not_loaded` has `stats=None` when no local model is loaded, including after
close; `unsupported` has `stats=None` for workers or providers without statistics.
Unavailable snapshots include a reason. Reading the property never loads a
model or contacts a worker. Errors from a loaded statistics provider propagate.

DreamZero's `stats["step_cache"]` reports installation, source-method hashes,
comparison signal, coupled outputs and controller state. After an executed
generation, its `last_generation` includes:

| Telemetry | Meaning |
| --- | --- |
| `computed_steps`, `reused_steps`, `compute_mask`, `trace` | Consumed controller decisions and their reasons, similarity and countdown state. |
| `denoiser_calls`, `denoiser_branch_forwards` | Observed denoising calls and CFG branch forwards. |
| `kv_update_calls`, `kv_update_branch_forwards` | Separately observed native KV-update calls and branch forwards. |
| `status`, `peak_cache_bytes` | Completion state and retained storage for that generation. |

The hook checks that consumed compute decisions match observed denoising calls.
For the current CFG5 route, one computed slot executes two branches; KV-update
forwards remain separate work. An aborted-generation report contains partial
call counts and must not be treated as a completed sixteen-slot result. The
controller also reports cumulative counters and the current or last generation.
These remain observed adapter statistics; `runtime.execution_policy` separately
describes selected execution. Startup websocket metadata does not provide live
statistics, and worker placement currently has no statistics RPC.

This approximate cache is separate from the
[exact tensor cache](../instinctflash/runtime/tensor_cache.py), which reuses a
pure result only when its dependencies match. Here the latent and timestep have
changed. A cosine gate, a source hash or native arithmetic cannot make that reuse
exact or certify the uncomputed action.

The final combined CPU suite has **215 passing tests** covering controller logic,
public policy/dispatch, integration contracts and cleanup. The Thor audit passed
all four native/FP8 × fixed/dynamic processes, with 72 finite action arrays and
24 exact paired action comparisons. Dynamic requests computed four slots in
these episodes; both schedulers retained all sixteen updates. Four measured
continuation samples per arm establish a bounded latency SCREEN, with no
closed-loop quality certificate. The retained Omni receipt has no actual DiT-call
counters, so its historical gap cannot be wholly attributed to this cache.

The public statistics snapshot was added after the Thor inference bundle was
frozen on 2026-09-14. Its [CPU tests](../tests/test_runtime_telemetry.py) validate
observation and copying behavior; it changes no inference path or frozen receipt.
The [post-freeze source review](../eval/dynamic_step_cache_integration_2026-09-14/post_freeze_review.json)
also records metadata-copy and failure-cleanup refinements; successful prediction
and sampling functions are unchanged from the GPU-tested bundle.

The [source audit](../eval/dynamic_step_cache_2026-09-14/source_audit.md) records
the pinned upstream and Omni behavior. The [controller](../instinctflash/runtime/step_cache.py)
and [Runtime policy](../instinctflash/runtime/step_cache_policy.py) implement the
current DreamZero integration. Action-aware profiles and other family integrations
remain future work.

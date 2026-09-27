# LingBot-VLA-V2 Runtime integration

This integration serves `robbyant/lingbot-vla-v2-6b-robotwin` through both public APIs.
It needs the upstream checkout (`LINGBOT_VLA_V2_ROOT`) and the checkpoint; there are no
baked-in machine paths.

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained(
    "/path/to/lingbot-vla-v2-package",     # or the HF id; the known-release declaration applies
    placement="in_process",
    device="cuda:0",
)
runtime.reset(prompt="Use the left arm to pick up the block")
result = runtime.predict({
    "observation.images.cam_high": top_hwc_uint8,
    "observation.images.cam_left_wrist": left_hwc_uint8,
    "observation.images.cam_right_wrist": right_hwc_uint8,
    "observation.state": state_float32_14,
})
actions = result["action"]  # (50, 14)
```

Install the external Runtime adapter in the model environment:

```bash
uv pip install -e .
uv pip install -e ./examples/lingbot_vla_v2
```

`IFL_VLA2_BACKEND=static` is the default. `compile` selects upstream `torch.compile`; `eager`
selects the uncompiled reference.

Two optional LingBot-specific Triton kernels exist and DEFAULT OFF (`IFL_VLA2_CUDA_KERNELS=0`).
They were gated on H100 under the same 6-case null-control protocol as the published row
(`verify_moe_kernel.py`, results committed in `moe_kernel_results.json`):

- **sparse top-4 MoE** with the exact per-expert route bound (`T`, not `T * top_k`) and an
  expert-sorted, atomics-free deterministic output reduction — **accuracy PASS**
  (3.84e-02 vs the 5.08e-02 stock-vs-stock envelope) and **self-consistency 0.0**: with the
  vendor's atomics replaced, identical seeds reproduce identical actions, which the stock model
  cannot do (envelope up to 5.1e-02 against itself). It stays off by default because it measured
  ~2% slower than vendor robby_moe on the eager path and its interaction with the captured
  serving path is ungated; set `IFL_VLA2_MOE_KERNEL=1` when deterministic replay
  matters more than the last 2%.
- **fused RMSNorm/AdaRMSNorm** for the 73 action-expert normalization sites — **accuracy FAIL**
  (6.10e-02, outside the envelope). It remains available for measurement via
  `IFL_VLA2_RMSNORM_KERNEL=1` but is NOT RECOMMENDED until it passes.

Both experimental kernels are refused on Thor (SM110). The supported native
Thor route uses SDPA; FP8 is a separate explicit precision choice through the
same Runtime interface.

GPU image processing is enabled by default. `IFL_VLA2_GPU_PREPROCESS_MODE=processor` batches
the three upstream CPU resizes without changing their values, then uses reusable pinned staging
buffers so the unchanged Qwen processor runs normalization and patchification on CUDA. Set
`IFL_VLA2_GPU_PREPROCESS=0` for the original all-CPU path. The opt-in `full` mode also moves resize
to CUDA, but is experimental because its interpolation produced a small action drift beyond the
current numerical gate.

The fixed-grid vision encoder and 286-token prefix prefill are CUDA Graphs as well. Disable only
these two graphs with `IFL_VLA2_PREFIX_GRAPH=0`; the denoise static-KV graph remains active.

## What the backend is

The static route is a BF16 backend with static-KV CUDA Graph replay (plus the optional kernels
above). It retains the official Qwen3-VL processor, `FeatureTransform`, and action
un-normalisation. Thor also has a native SDPA route and an opt-in FP8 engine
(`instinctflash/engines/vla2`: FP8 prefix and dense-routed MoE experts, native BF16 vision).
Select the engine with `precision="fp8"` or CLI `--fp8`; native precision remains the default.

The graph owns fixed `[prefix | suffix]` K/V allocations for all 36 layers. At each observation it
refills the 286-token prefix and Qwen3-VL 3D-RoPE position ids. At each flow step it overwrites the
51-token state/action suffix, noisy action, and timestep, then replays the same graph. MoE routing
remains data-dependent and is recomputed on device during replay.

## Thor evidence

On 2026-09-09, with the same native processing, ten denoise steps and 50-action chunks,
short-run median latency measured 764.29 ms native eager and 423.50 ms native
NUMERIC capture; that native comparison explicitly permits NUMERIC capture.
The FP8 engine's prefix and ten denoise steps, MoE experts included, take a
129.5 ms engine-level median on Thor. On twelve recorded observations with identical
inputs and noise, its mean absolute action error against the upstream reference
was 0.0276 in the checkpoint's normalized action space.

The complete 40-pair RoboTwin screen of the earlier FP8 engine measured clean
17/20 → 18/20 and randomized 16/20 → 18/20 successes; the current engine has no
RoboTwin screen yet. These small samples establish neither superiority nor
non-inferiority.

## Graph capture requires the NUMERIC tier

Select `tier_ceiling="numeric", placement="in_process"` to allow the native capture arm.
The default BITEXACT ceiling declines it because its self-check accepts nonzero differences.
Device eligibility is measured separately: Thor uses its V2 evidence, not pi05 timings.
Fresh fine-tunes inherit the adapter, not a performance or quality certificate.
The first capture is gated by a runtime
**self-check**: replay vs upstream eager `predict_velocity` (through the stock concat-per-step
KV path) on staged inputs the capture never saw, including a synthetically *refilled* prefix so
a graph that baked K/V values cannot pass. The gate is NOT `atol=0`, and that is a statement
about repeatability: upstream can disagree with itself under fixed inputs, but that does
not establish the graph's error. The unchanged **5.08e-02** guard came from controller
actions in `moe_kernel_results.json`; startup compares denoise velocity. This domain
mismatch is printed as a legacy guard, not a calibrated accuracy margin. Each staged
input is compared three times. PASS admits replay and records the verdict. FAIL removes
denoise/prefix graphs and GPU preprocessing; the in-flight call is discarded and rerun
upstream after restoring RNG and server counters. Experimental custom kernels refuse
further serving when complete fallback cannot be guaranteed.

Kill-switch: `IFL_VLA2_NO_CAPTURE=1` disables the whole capture arm (recorded on the plan,
printed). `IFL_VLA2_SELFCHECK_FAULT=1` is the drill switch: it rebinds the x buffer between
capture and check so the loud-fallback path stays demonstrable on demand.

## Validation and measured artifacts

`reproduce_h100.sh` is the protocol artifact behind the published H100 row
(671.1 -> 127.5 ms p50, 5.26x — H100 re-sweep 2026-08-28, 4xH100 box): upstream eager vs what
the historical capture arm (denoise graph + vision/prefill graphs + GPU preprocessing;
the reproduction script now requests NUMERIC explicitly), one
fresh process per arm, with the 6-case ours-vs-stock gate judged against the model's own
nondeterminism envelope (the fused-MoE null control — upstream was not bit-repeatable under that H100 protocol; measured max |d| 2.57e-02 vs the 5.08e-02 envelope).
Its output is committed as `reproduce_h100_results.json`.

`verify_static_capture.py` is the module-level 6-case gate for the denoise static-KV graph in
isolation (same box: stock 667.0 -> 165.0 ms; the further drop to 127.5 is the vision/prefill
graphs plus GPU preprocessing); its output is committed as `static_capture_results.json`.
`profile_infer.py` is the stock-cost decomposition behind the original 840 ms stock figure.
These artifacts are load-bearing for the README results table — do not replace them with a
weaker protocol.

Correctness tier claimed: `NUMERIC`.

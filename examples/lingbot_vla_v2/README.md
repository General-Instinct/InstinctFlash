# LingBot-VLA-V2 Runtime + FlashRT backend

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

```python
import flash_rt

model = flash_rt.load_model(
    "/path/to/hf_ckpt",
    config="lingbot_vla_v2",
    framework="torch",
    source_root="/path/to/lingbot-vla-v2",   # or set LINGBOT_VLA_V2_ROOT
    qwen3vl_path="Qwen/Qwen3-VL-4B-Instruct",  # or set QWEN3VL_PATH
)
actions = model.predict(
    [top_hwc_uint8, left_hwc_uint8, right_hwc_uint8],
    prompt="Use the left arm to pick up the block",
    state=state_float32_14,
)
```

Install the external Runtime adapter in the model environment:

```bash
uv pip install -e .
uv pip install -e ./examples/lingbot_vla_v2
uv pip install -e ./serving
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
  182.5 ms serving path is ungated; set `IFL_VLA2_MOE_KERNEL=1` when deterministic replay
  matters more than the last 2%.
- **fused RMSNorm/AdaRMSNorm** for the 73 action-expert normalization sites — **accuracy FAIL**
  (6.10e-02, outside the envelope). It remains available for measurement via
  `IFL_VLA2_RMSNORM_KERNEL=1` but is NOT RECOMMENDED until it passes.

Both are REFUSED on Thor (SM110): Triton codegen is measured-dead there and the vendor fallback
path crashes; that device class is served by a separate engine tier available under commercial
access.

GPU image processing is enabled by default. `IFL_VLA2_GPU_PREPROCESS_MODE=processor` batches
the three upstream CPU resizes without changing their values, then uses reusable pinned staging
buffers so the unchanged Qwen processor runs normalization and patchification on CUDA. Set
`IFL_VLA2_GPU_PREPROCESS=0` for the original all-CPU path. The opt-in `full` mode also moves resize
to CUDA, but is experimental because its interpolation produced a small action drift beyond the
current numerical gate.

The fixed-grid vision encoder and 286-token prefix prefill are CUDA Graphs as well. Disable only
these two graphs with `IFL_VLA2_PREFIX_GRAPH=0`; the denoise static-KV graph remains active.

## What the backend is

The FlashRT route is a BF16 backend with static-KV CUDA Graph replay (plus the optional kernels
above). It retains the official Qwen3-VL processor, `FeatureTransform`, and action
un-normalisation. It is the upstream-BF16 datacenter graft; the from-scratch Thor SM110 engine
(fp8 experts via cuBLASLt, fp16 prefill) is a separate arm, not part of this repository.

The graph owns fixed `[prefix | suffix]` K/V allocations for all 36 layers. At each observation it
refills the 286-token prefix and Qwen3-VL 3D-RoPE position ids. At each flow step it overwrites the
51-token state/action suffix, noisy action, and timestep, then replays the same graph. MoE routing
remains data-dependent and is recomputed on device during replay.

## Validation and measured artifacts

`verify_static_capture.py` is the hardware validation entry point behind the published H100 row
(829.1 -> 182.5 ms p50, 4.54x, 6-case gate): it compares replay against stock under fixed seeds,
judged against the model's own nondeterminism envelope (the fused-MoE null control — BITEXACT is
unattainable for any serving of this model, including upstream's own). Its output is committed as
`static_capture_results.json`; `profile_infer.py` is the stock-cost decomposition behind the
840 ms stock figure. These artifacts are load-bearing for the README results table — do not
replace them with a weaker protocol.

Correctness tier claimed: `NUMERIC`.

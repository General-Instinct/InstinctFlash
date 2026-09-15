# Shared BF16 kernel integration — Thor, 2026-09-13

The former isolated CUTLASS BF16/ReLU² experiment is now a shared backend with
versioned ABI, explicit NUMERIC permission, shape/device/activation dispatch,
per-call Torch-managed workspace, and ordinary Cosmos adapter integration.
The default remains disabled; FP8 and schedule permissions are unchanged.

| Paired arm | Request p50 |
|---|---:|
| Reference A: compiled GEN + split prefill | 1158.33 ms |
| Shared BF16 fused linear + ReLU² | 1089.44 ms |
| Reference B | 1160.69 ms |

Latency reduction: **5.95–6.14%**. All 16 action chunks (6 warmup + 10 measured)
are byte-identical across all arms. Each measured request has 224 compiled GEN
and 56 compiled prefill calls; 84 native prefill/extraction checks passed outside
measurement. All 28 GEN layers report an eligible fused-linear backend.
UniPC4 / CFG3 / horizon32 / BF16 are unchanged. These are action/numerical and
latency screens against the NUMERIC baseline, not upstream BITEXACT or task quality.

`compare_e2e.py` exits nonzero if action equality fails or if either paired
reference shows less than **3%** latency improvement. It additionally verifies
schedule/environment, finite action data, action file hashes and region coverage.
This threshold preserves a margin below the earlier isolated ~6% observation.
The integrated result passed. Raw outputs and logs are in `e2e-results/`.

`gpu_regression.py` passed 16 checks using the prior real operands: eager and
fullgraph output equality, independent CUDA streams, original short-shape
fallback, CUDA Graph replay after changing inputs, and invalid C ABI arguments.
The GPU tests use the actual shared component and built ABI, not the prototype.
81 relevant CPU tests passed (permission, dispatch, gradient fallback, lifecycle,
Cosmos adapter/backend/cache/precision regressions).

`applicability.json` records the eight-model source audit. Only Cosmos Edge's
ungated ReLU² block directly matches this kernel; other families retain their
original GELU/gated paths. No acceleration is claimed for those unchanged paths.
See `instinctflash/native/bf16/README.md` for build/use and the compatibility table.

Native build used CUDA13 nvcc / SM110a with CUTLASS
`da5e086dab31d63815acafdac9a9c5893b1c69e2`. `build_nvcc.log` is the successful
build log. `build.log` retains the initial CMake attempt: this Thor environment
had no cmake executable, so the supplied nvcc build script was used. The CMake
configuration is provided but was not executed successfully in this environment.

Remote run: `/home/guanming/ifl_eval/shared_bf16_linear_20260913`.
GPU calls were serialized under `/tmp/thor_gpu.lock`; no GPU worker remains.
The native binary hash and source/result inventory are bound in `completion.json`.

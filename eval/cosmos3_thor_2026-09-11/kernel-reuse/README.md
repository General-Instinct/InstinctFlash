# Direct reuse of existing engine CUDA kernels on Thor

This experiment compiles existing engine `.cu` sources with a thin C ABI. No
kernel arithmetic was rewritten. Build flags include `--fmad=false`; source
hashes and flags are in `kernel-sources.json`. No production default changes.

## Contract screen

`screen.json` tests four synthetic BF16 geometries, not model action accuracy.
CUDA-event timings include Python enqueue gaps and are diagnostic, not isolated
kernel execution times or full-model speedups.

| Existing kernel | Finding | Cosmos decision |
| --- | --- | --- |
| `activation.cu:gate_silu_mul` | Actually computes approximate GELU × up | Not a Nano SwiGLU replacement |
| `silu_mul_qwen36.cu:silu_mul_qwen36_bf16` | All four synthetic cases match eager SiLU × up bytes; faster on small shapes, slower on large shapes | Tested in full compiled Nano below |
| `norm.cu:rms_norm` | Qwen's intermediate BF16 rounding differs; even Nemotron-style reference differs at larger tested dimensions | Not a general BITEXACT replacement |
| `elementwise.cu:residual_add` | All tested cases byte-match; larger cases slower than Torch | No measured reason to replace the existing add |
| `elementwise.cu:gate_mul_residual` | Matches tested FP32-intermediate semantics, not BF16-intermediate multiply | Only reusable at a matching operator site; Cosmos decoder primarily uses plain residual adds |
| `qwen3_qkv_post_proc.cu` | Single-token, head_dim=128 contract | Needs multi-token/geometry adaptation |
| `flash_wm/csrc/kernels_bf16.cu:qk_rmsnorm_rope_fused_bf16` | Supports multiple tokens but omits Cosmos's intermediate BF16 casts; full-head RoPE | Fusion structure reusable; not a direct BITEXACT drop-in |

## Full Nano experiment

`engine_swiglu.py` exposes the **unchanged** Qwen3.6 CUDA SwiGLU kernel through
`torch.library.custom_op`, so it remains callable inside compiled decoder
blocks. It replaces all 72 compatible understanding/generation MLPs.

Pinned upstream and workload match the stronger-baseline audit: native BF16,
four UniPC steps, full CFG 3, 32×8 actions, six warmup and ten timed varied
requests. The reference is the previously recorded compiled arm; this is an
exploratory single-arm comparison, not a fresh A/current/B certificate.

| Path | p50 |
| --- | ---: |
| Upstream compiled | 7404.12 ms |
| Compiled + existing CUDA SwiGLU | 7498.13 ms |

The candidate is about 1.27% slower. Its full action arrays also **do not match**
the compiled reference (max absolute delta 0.63684; mean absolute delta 0.08723).
Synthetic equality to eager activation does not imply whole-model equality to
compiled execution. The cause of the full-model difference has not been
isolated. All outputs were finite; no competing process was observed. This
candidate is **not promoted**. It does not establish a task-quality result.

Raw JSON/NPZ and `model-comparison.json` preserve the result. The next target
must follow profiling of the stronger compiled reference, rather than the older
eager hot spots.

## Compiled-reference profile

`nano-compiled-profile.json` profiles upstream compilation, with the candidate
kernel disabled. The SM80 BF16 NATTEN FMHA kernel accounts for **2884.6 ms across
360 launches** in the recorded request. Three prominent SM110 GEMM kernel
families account for 1482.5, 1338.7 and 434.4 ms respectively. Compiler-region
and ATen rows overlap these GPU events and must not be added to them.

This prioritizes attention and large matrix/layout work above a blanket
activation-kernel swap. Existing reuse candidates include the strided Blackwell
FMHA path in `serving/csrc/attention/fmha_fp16_strided.cu` and the Thor setup /
preallocated-workspace pattern in `serving/flash_wm/csrc/cutlass_fp8_fmha.cu`.
Neither is currently a native BF16 Cosmos drop-in: the former instantiates FP16,
the latter FP8, and both use `NoMask`. BF16 instantiation, exact mask/layout
eligibility and numerical qualification are required before integration. A
different attention algorithm must not inherit the BITEXACT label by dtype alone.

## Reproduction

On Thor, with the Cosmos Python environment:

```bash
bash eval/cosmos3_thor_2026-09-11/kernel-reuse/build.sh
flock -n /tmp/thor_gpu.lock python eval/cosmos3_thor_2026-09-11/kernel-reuse/probe.py /absolute/new-screen.json
```

For the full-model experiment put the pinned upstream checkout, this repo and
`examples/cosmos3_policy` on `PYTHONPATH`, then run the audit under the GPU lock
with `AUDIT_COMPILE=1` and `AUDIT_ENGINE_SWIGLU` set to the absolute built
`kernels.so` path. Use the `nano baseline_a OUTPUT --iterations 10` arguments.
The backend is explicitly experimental; it does not alter Runtime admission.

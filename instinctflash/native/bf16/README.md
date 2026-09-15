# Shared BF16 linear–activation fusion

`instinctflash.backends.bf16_linear.BF16LinearActivation` is the shared adapter
interface. Its first native implementation fuses a BF16 linear projection and
ReLU² on Thor. It preserves the intermediate BF16 rounding boundary. This is
an explicit **NUMERIC** optimization, independent of FP8 and sampling schedules.

Build on Thor (CUDA 13 and a compatible CUTLASS checkout):

```bash
bash instinctflash/native/bf16/build.sh /path/to/cutlass /path/to/build
export IFL_BF16_KERNEL_LIBRARY=/path/to/build/libinstinctflash_bf16.so
export IFL_COSMOS3_GEN_REGIONS=1
export IFL_COSMOS3_SPLIT_PREFILL=1
export IFL_BF16_LINEAR_RELU2=1
```

Use the normal Cosmos Runtime interface with `precision="native"` and
`tier_ceiling="numeric"`. The existing pinned native conditioning-cache and
attention gates still apply. A BITEXACT ceiling rejects this path. A missing or
incorrect explicitly selected library is an error, not an unreported fallback.
CMake is also supported through `instinctflash/native/bf16/CMakeLists.txt` and
`-DCUTLASS_ROOT=/path/to/cutlass`.

The native implementation currently admits SM110, BF16, no bias, and
`(M,N,K)=(3093,9216,2048)`. The public dispatcher uses the original linear and
activation for other shapes, devices, activations, or gradient-enabled calls.
Kernel launch errors propagate. No model weights or activation functions are
rewritten globally. Cosmos retains a separate unmodified native eager callable
for admission checks. Backend metadata lists fused-linear eligibility.

The raw-pointer ABI is versioned and independent of the PyTorch C++ ABI. Each
call allocates scratch through Torch on the caller's device and stream, so
concurrent streams do not share scratch. Torch owns output/scratch lifetimes,
including graph capture. Shared library caching retains CPU handles only.

## Coverage across the eight benchmark models

| Model | Relevant MLP structure | This ReLU² kernel |
|---|---|---|
| Cosmos Edge | Ungated ReLU² | Integrated for the measured GEN shape |
| Cosmos Nano | Qwen gated SiLU/SwiGLU | Incompatible; original path |
| pi05 | Gemma gated GELU | Incompatible; original path |
| LingBot-VLA-4B | Qwen gated SiLU | Incompatible; original path |
| LingBot-VLA-V2 | Gated SiLU and routed experts | Incompatible; original path |
| GR00T N1.7 | GELU-approximate action MLP | Incompatible; original path |
| LingBot-VA | GELU-tanh feed-forward | Incompatible; original path |
| DreamZero | GELU-tanh feed-forward | Incompatible; original path |

This table concerns the target feed-forward blocks, not every activation in each
model. Source locations and hashes are in the accompanying applicability audit.
The shared interface can serve any adapter with the same declared operation and
qualified shape; a family name alone never enables it. GELU and gated activations
need their own epilogues, rounding analysis, real operands and paired model
regressions before extending accelerated coverage. No speedup is claimed for
an unchanged model.

## Regression evidence

The original isolated experiment reduced Edge request p50 from 1155–1161 to
1087 ms and matched 16 action chunks against the compiled NUMERIC baseline.
See `eval/edge_fused_up_2026-09-13/`.

The integrated-component run and stream/compile/graph tests are recorded in
`eval/shared_bf16_linear_2026-09-13/`. CPU tests cover permission ordering,
unsupported activation/device/shape dispatch and exact original training
fallback. Input-level equality is not a closed-loop task-quality certificate.

Integrated Thor validation reproduced **1158–1161 → 1089 ms** (about **6%**),
with byte-identical actions across the 16-call paired corpus. The automatic
paired regression requires identical actions and at least 3% latency reduction
against both reference runs. All 81 relevant CPU tests and 16 real-operand GPU
checks passed. See [integration results](../../../eval/shared_bf16_linear_2026-09-13/README.md).

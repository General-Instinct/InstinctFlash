# Thor FP8 execution audit — 2026-09-10

The public precision selector is uniform; the implementations behind it are not equally complete. The latency table measures selectable Runtime implementations, not an isolated FP8 arithmetic speedup or the hardware's FP8 limit.

## Measurement correction

The original pi05 v044 benchmark supplied `observation.images.wrist_image`, but this checkpoint declares `observation.images.image2`. Its FP8 receipt consequently contained `_frontends.1`: only one camera was active. The same fixture also supplied uint8 0–255 images, whereas the checkpoint visual normalizer is IDENTITY and native `_preprocess_images` expects float32 0–1 before mapping to −1…1. The original 322.29 → 43.49 ms / 7.41× measurement is valid only for that missing-camera, unscaled-image diagnostic workload and must not represent normal two-camera v044 deployment. The original cell was withdrawn and replaced after the corrected pair passed validation. The current result is **321.68 → 54.26 ms, 5.928×**, with exactly two active cameras verified in the FP8 frontend. Both arms still had the same supplied inputs; the error was the benchmark's input contract, not a mismatched checkpoint or step count.

The correction supplies `image` and `image2` as float32 CHW images divided by 255, preserves the native 50-action chunk and ten denoise steps, and asserts that the FP8 frontend actually runs two active cameras. Its receipts and validation are separate from the retained initial sweep. Timing remains three warmups plus twelve measured full generations; queue resets remain outside the timer.

## What each precision route actually executes

| Family | Active FP8 route | Retained computation / integration gap |
|---|---|---|
| pi05 | Dedicated fused SigLIP, language-prefix and action-expert pipeline | **Native vision tower and multimodal projector stay FP32 despite the BF16 config; TF32 was disabled.** Native robot preprocessing/decoding; masked camera tokens are omitted by the engine. Native static capture primarily covers denoise steps, so the engine comparison includes much more than quantization. |
| VLA-4B | Fused language model and action expert | Native BF16 Qwen vision, eager; the old low-level FP16 vision path can overflow and is deliberately bypassed. |
| VLA-V2 | FP8 action expert / checkpoint MoE integration | Native BF16 vision; **language prefill is explicitly FP16** via `load_lm_fp16_stack` and `lm_prefill_precision='fp16'`. |
| GR00T N1.7 | Four-layer fused FP8 VLSA, six projections per layer | Native camera/language backbone; BF16 action DiT. The frontend loads additional FP8 weights that this public route does not execute. |
| LingBot-VA | Fused FP8 DiT projection pipeline | Native T5/VAE, higher-precision attention, embeddings, scheduling and specified projection fallbacks; this already compares against an optimized native path. |
| Cosmos Edge / Nano | BF16 native service with Q/K/V replacement in both MoT towers (168 / 216 projections) | Native attention, output projections, MLP/experts, encoders and orchestration. Every replacement dynamically quantizes its input. This is not a complete fused FP8 policy pipeline. |
| DreamZero | BF16 native policy with 120 causal self-attention Q/K/V replacements | Native attention, output projections, FFN, cross-attention, T5/CLIP/VAE, history and scheduling. This is not a complete fused FP8 policy pipeline. |

The initial sweep's E4M3 tensor inventory proves presence, not executed FLOP coverage. It includes runtime buffers and, for some frontends, loaded but unused weights. In particular, the GR00T public runner verifies four layers of Q/K/V/O/FC1/FC2 FP8 weights; a larger frontend inventory must not be advertised as full-model FP8 coverage.

Source: [dispatch](../../instinctflash/runtime/engine_backend.py), [pi05 integration](../../instinctflash/runtime/pi05_engine.py), [VLA-4B](../../instinctflash/runtime/vla4_engine.py), [VLA-V2](../../instinctflash/runtime/vla2_engine.py), [GR00T](../../instinctflash/runtime/groot_engine.py), [GR00T VLSA runner](../../serving/flash_rt/models/groot_n17/vlsa_runner.py), [Cosmos recipe](../../instinctflash/runtime/cosmos_fp8.py), [DreamZero recipe](../../instinctflash/runtime/dreamzero_fp8.py).

## New stage measurements

Fresh synchronized diagnostic runs use the frozen runtime source from the latency sweep. Four measured calls follow three warmups. Synchronization changes timing overhead; these are attribution measurements, not replacement main-table cells. Parent and graph times overlap and must not be added.

| Route | Stage | Median diagnostic time |
|---|---|---:|
| VLA-4B FP8 | Native BF16 vision | 138.82 ms |
| VLA-4B FP8 | Fused language + action | 54.72 ms |
| VLA-V2 FP8 | Native BF16 vision | 47.36 ms |
| VLA-V2 FP8 | Joint language + action graph | 176.78 ms |
| GR00T FP8 | Native camera/language backbone | 57.82 ms |
| GR00T FP8 | VLSA runner, including its handoff work | 15.38 ms |
| GR00T FP8 | VLSA graph alone (inside preceding row) | 3.28 ms |
| GR00T FP8 | BF16 DiT/state frontend | 45.83 ms |

VLA-4B spends roughly two thirds of its current ~217 ms call in BF16 vision. Accelerating its already-fused language/action stage alone cannot deliver a pi05-sized end-to-end gain. VLA-V2 has a different bottleneck: most measured time is in the joint FP16-language/FP8-expert graph; this diagnostic does not yet isolate those two components.

GR00T's actual FP8 graph is a small part of the public call. The native comparison measured ~54.84 ms in camera/language and ~66.36 ms in the whole action head; its FP8 action head measured ~62.20 ms with synchronization. Small local savings do not establish an end-to-end win; the uninstrumented sweep was slower for FP8.

On the **superseded one-camera, unscaled-image pi05 workload**, the native model region measured 326.08 ms, with approximately 67.29 ms summed over the denoise graph replays. The FP8 vision and language/action graphs measured 3.92 and 35.01 ms respectively. The installed LeRobot `to_bfloat16_for_selected_params` first casts the model to BF16, then explicitly recasts `vision_tower`, `multi_modal_projector` and named norms to FP32. The adapter subsequently moves the model to the device without changing those dtypes. Thus this is not a uniform BF16-versus-FP8 comparison: pi05 also replaces an FP32 vision path with FP8, while other families generally retain or start from BF16/FP16. TF32 is false in both pi05 receipts. This supports a large difference in baseline precision and pipeline coverage, not a claim that all the residual native time is one specific kernel. The native implementation captures `denoise_step` in [static_capture.py](../../examples/pi05_vla/pi05_iwm/static_capture.py); its prefix work remains outside that graph.

## Dynamic conversion can erase GEMM savings

[ThorFP8Linear](../../instinctflash/runtime/torch_fp8_linear.py) performs activation `abs → amax → scale`, E4M3 packing and `_scaled_mm` on every invocation, with `use_fast_accum=False`. Separate Q, K and V modules repeat this work even when they receive the same input. The current recipe does not share that conversion or fuse it into the preceding normalization.

A fresh CUDA-graph microbenchmark used representative synthetic BF16 inputs and the actual production projection class:

| M × K → N | BF16 linear | FP8 prepacked GEMM | Full dynamic FP8 linear |
|---|---:|---:|---:|
| 40 × 2048 → 2048 | 22.69 μs | 14.59 μs | 45.56 μs |
| 256 × 2048 → 2048 | 53.58 μs | 32.94 μs | 71.07 μs |
| 256 × 5120 → 5120 | 230.01 μs | 53.72 μs | 69.90 μs |

Thus some shapes genuinely benefit while others regress despite a faster FP8 GEMM. These are illustrative projection shapes, not an extracted model workload distribution. Independently captured component times need not sum to the whole operator (kernel selection, launch and cache behavior can differ). They do not quantify the model-wide percentage of time spent converting.

## BF16 vision graph candidate: measured improvement without additional quantization

A same-loaded-policy A/B/A test on VLA-4B retained the existing FP8 language/action weights and **kept vision BF16**. The first capture attempt failed because the precomputed window index was on CPU. Moving that index to CUDA exposed a second problem: GPU scalar sequence boundaries were used as Python slice bounds while building attention masks. The successful diagnostic stages the window index on CUDA and materializes the immutable sequence boundaries as host integers before capture.

| Phase | Whole Runtime p50 |
|---|---:|
| Original execution, before | 211.91 ms |
| BF16 vision with static metadata + CUDA Graph | 160.02 ms |
| Original execution, restored | 212.77 ms |

This is **24.5% lower latency / 1.324×** relative to the same-process before arm. All three action archives are byte-equal across eleven calls (three warmups, eight measured), independently verified locally. This does not establish all-input bit-exactness or a simulator certificate. The experiment is a candidate, not the implementation behind the published 220 ms VLA-4B FP8 cell. Production integration still needs graph admission/self-check, shape handling and cleanup; the key result is that additional quantization is not required to recover this portion of overhead.

Evidence: [candidate and independent verification](audit/vision-graph-candidate.json), [diagnostic script](audit_vla4_vision_graph.py). The scope includes static metadata placement **and** graph capture; it does not isolate the gain from graph replay alone.

## Priorities

1. Correct and validate model-specific camera contracts before publishing speed claims. Treat presence of FP8 tensors as an implementation diagnostic, not coverage.
2. Finish BF16 vision/preprocessing graph integration where possible before adding numerical loss. The VLA-4B capture probe immediately exposed a CPU-resident precomputed window index; moving static metadata onto the device is a concrete integration task. The experimental probe retains BF16 and compares action bytes before/after on the same loaded policy.
3. GR00T needs meaningful backbone/DiT optimization coverage; VLA-V2 needs a separate FP16-prefill/expert profile. Quantizing more loaded weights without routing execution through them achieves nothing.
4. Cosmos/DreamZero need shared Q/K/V packing and fused conversion/normalization, followed by shape-aware kernel selection and profiling of unquantized MLP/attention. Broader quantization, static scales or fast accumulation change the numerical recipe and need their own paired quality evaluation.
5. Preserve native precision as default. Keep execution implementation, quantization recipe, operating point, measured latency and closed-loop quality separate under the same public interface.

These findings do not prove a promised speedup for unfinished optimizations. The current numbers characterize the existing implementations; they do not establish that FP8 itself is ineffective on these models.

Evidence: [stage summary](audit/stage-summary.json), [operator timings](audit/linear-cost.json), [initial vision capture failure](audit/vision-graph-ablation.json). Diagnostics are retained under `/home/guanming/ifl_eval/thor_fp8_audit_20260910` on Thor and mirrored locally.

Final validation: [unchanged frozen runtime/library receipt](audit/audit-source-final.json), [pi05 native precision source](audit/pi05-native-precision-source.json), [corrected pi05 receipts](pi05-two-camera/pi05-two-camera-comparison.json).

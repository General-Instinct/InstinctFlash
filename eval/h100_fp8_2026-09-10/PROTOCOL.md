# H100 native / FP8 Runtime comparison

Every pair uses the same checkpoint revision, recorded camera archive, synthetic state, prompt, original denoise schedule, and synchronized public `Runtime.predict` timing. Precision is selected through `Runtime.from_pretrained(..., precision="native"|"fp8")`. The 2V/4A VA row is a separate, explicitly requested operating point.

H100 uses an E4M3 projection executor implemented with PyTorch scaled matrix multiplication. It is **not** the Thor fused-kernel engine and does not inherit its simulator results. General H100 recipes quantize eligible native BF16 attention Q/K/V projections, retaining the rest of the model. Cosmos and DreamZero use their existing explicit Q/K/V module lists. Projection inventories are recorded in each FP8 receipt. No native-precision request enables this executor, and an unavailable FP8 path raises an error.

Measurements use H100 GPUs 4–7 on the eight-H100 host, one process per GPU, sequential native then FP8 within each model pair. Other users' occupied GPUs 0–3 are untouched. Source is frozen before the measured sweep; smoke tests and failed setup attempts are excluded from published numbers.

For stateless policies, discard five warmup generations and report the median of 20 generations. For VA and DreamZero, run three nine-cycle episodes: discard the first episode and the first cycle of each remaining episode, then report the median of 16 cycles. These are early-history measurements, not saturated latency. pi05 resets its action queue outside each timer so the timed call always generates a full chunk rather than returning a buffered action. VA receives identical recorded executed-action feedback in both arms.

Model loading, reset, observation construction, transport, simulator execution, and result serialization are outside the timer. Native image processing, model inference, and returning CPU actions are inside. Calls are bracketed by CUDA synchronization. TF32 and cuDNN benchmark settings are recorded; no inference-step reduction is bundled with FP8. Raw actions must be finite.

The old README table uses several historical timing protocols (for example Cosmos 16-action guidance-1 versus the current native DROID 32-action guidance-3 contract). The new column therefore reports its **own matched Runtime native → FP8 pair**, not a speedup against an unrelated historical cell. Ratios below 1 mean FP8 is slower.

This is a latency comparison, not H100 closed-loop accuracy certification. FP8 changes arithmetic and may reduce task success. Thor quality results remain separately scoped to their device, implementation, checkpoint, and simulator.

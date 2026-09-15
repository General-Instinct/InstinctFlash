# What transfers from existing InstinctFlash caches

This experiment targets the installed Thor Cosmos source `f734253f`, full
four-step CFG 3, BF16, and complete 32 × 8 DROID actions. That source has no
`_make_inference_text_kv_cache` hook. The existing Cosmos persistent-prompt
installer requires a newer source and only admits guidance 1; enabling its flag
does not supply a CFG 3 cache on this stack.

| Existing infrastructure | Reused idea | Cosmos constraint |
|---|---|---|
| LingBot `ConditioningPrefill` / `install_conditioning_prefill` | Compute invariant conditioning once; publish only after complete prefill | Keep positive and negative branches separate. Do not assume the LingBot cross-attention call structure exists in Cosmos. |
| Adapter `KVLifetime.CHUNK`, pi05's already-hoisted prefix | Explicit request lifetime | Every generation call refills conditioning before any reuse, including resets and changed observations. |
| Cosmos `PersistentTextKV` | Bounded LRU and transactional state | Slots retain storage/graphs across requests, but payload validity ends with the request. Token content and skip-text flag identify branches. |
| Cosmos `LayerGraph`, GR00T shared graph pool | Bounded graph capture, actual-output checks, owned outputs | Each conditioning slot owns its graph instances. Cached tensor storage is independent of the shared graph pool and is refilled at stable addresses. |
| LingBot-VLA-V2 `prefix_capture.py` | Stable prefill buffers with explicit refill of downstream KV on a new observation | Reusing a graph or buffer address must not imply the cached observation is still valid. Its fixed image/prefix contract is model-specific. |
| BAGEL engine `prefill_text`, `set_kv_cache` | Separate BF16 text prefill and owned post-RoPE KV; fixed `[prefix | generation]` buffers | A useful next layout adaptation. Cosmos has its own rotary positions, GQA geometry and potentially separately normalized text K for generation. BAGEL's SDPA/normalization cannot replace those under a BITEXACT claim. |
| LingBot `RingKV` | Distinguish committed state from provisional state | Its temporal ring and eviction policy do not directly apply: this is static text conditioning, not growing video history. |

The first experiment caches understanding-tower Q/K/V projections, norms,
output projection and MLP results. It leaves attention kernels, token packing,
RoPE, generation-tower math and CFG arithmetic unchanged. It is therefore a
conditioning-cache experiment rather than a wholesale replacement of Cosmos's
attention with a new KV-aware kernel.

The old two-way attention source computes the understanding path using causal
text Q/K/V only; noisy generation tokens are not its operands. This motivates
reuse within a branch, but does not replace measured validation. The `verify`
arm recomputes every cached module output on every decode forward and compares
finite tensor bytes, with one collective check at the end of each request.
The speed arm skips that diagnostic recomputation and uses the existing graph
admission checks. A separate full-action comparison to uncached native Runtime
is required; graph admission alone compares already-cached execution to itself.

Cross-request payload reuse, packed-input templates, gen-only attention and
cache-aware RoPE are separate follow-ups. Each can change layouts or numerical
execution and needs its own paired comparison. No default Runtime installation
or precision-policy promotion is introduced by this experiment.

The old `_get_velocity` also rebuilds packed metadata and calls `timestep.cpu()`
on each velocity call. New upstream's `_update_inference_pack_template` reuses
the packet and updates noisy tokens/timesteps. That is a separate candidate:
condition masks, modality positions, CFG skip-text semantics and buffer mutation
must all stay correct. It should be tested independently before combining it
with conditioning reuse, so a change in action bytes has an identifiable cause.

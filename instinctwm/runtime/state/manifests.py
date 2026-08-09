"""State manifests for the corpus.

Facts only, cited to file:line. A manifest never says what to optimize; passes derive that.

These live together rather than beside each adapter so they can be diffed against each other —
the whole point of L3 is that one mechanism serves all of them, and the fastest way to catch an
abstraction that has quietly become LingBot-shaped is to read the manifests side by side.
"""

from __future__ import annotations

from instinctwm.runtime.state.types import (
    Addressing, ArenaSpec, CommitMode, Discovery, Extent, LiveSet, Order, Residency, Scope,
    Segment, SliceSpec, StateManifest,
)

# =============================================================================================
# LingBot-VA — the model L3 was born from. Two co-equal episode streams in one ring pool.
# =============================================================================================

_LB_LAYERS = 30
_LB_CAP = 9792          # (attn_window//2)*240 + (attn_window//2)*32, model.py:664-665
_LB_COMMIT = 272        # 240 video + 32 action per control step
# k and v, [B=2, 9792, 24, 128] bf16, x2 tensors x30 layers
_LB_KV_BYTES = 2 * _LB_CAP * 24 * 128 * 2 * 2 * _LB_LAYERS


def lingbot_va_manifest() -> StateManifest:
    return StateManifest(
        model_id="lingbot-va-posttrain-robotwin",
        forwards_per_step=79,
        # post-ring: only the action D2H remains. Stock was 9,480 (model.py:370, :373, :391-392, :451).
        sync_budget=1,
        arenas=(
            ArenaSpec(name="kv", scope=Scope.EPISODE, extent_tokens=_LB_CAP,
                      bytes_per_token=_LB_KV_BYTES // _LB_CAP,
                      slices=("kv_pool",)),
        ),
        slices=(
            SliceSpec(
                name="kv_pool", scope=Scope.EPISODE, residency=Residency.MANAGED,
                bytes_per_episode=_LB_KV_BYTES, commit_mode=CommitMode.CONFIRMED,
                extent=Extent(tokens=_LB_CAP, bounding_field="attn_window"),
                discovery=Discovery.D1_BOOLEAN_SCAN,
                note="model.py:448-453 -- FIXED by ring_kv_addressing (1.44x, bit-exact)"),
            SliceSpec(
                name="prompt_embeds", scope=Scope.EPISODE, residency=Residency.MANAGED,
                bytes_per_episode=2 * 512 * 4096 * 2,
                note="wan_va_server.py:424,:426 are the only writers, yet :257/:290/:314 clone and "
                     "cat both branches EVERY forward = ~632 MiB of D2D copy per cycle for a "
                     "constant. D2 on the read path."),
            SliceSpec(
                name="cross_attn_text_kv", scope=Scope.EPISODE, residency=Residency.RECOMPUTED,
                bytes_per_episode=0, recompute_ms=0.0,
                note="model.py:331 withholds the cache from cross-attention. 0 bytes today; "
                     "materialising costs +360 MiB and removes 89 of 226 TFLOP/cycle. This is why "
                     "the budget needs an E_recompute term -- a byte-only budget scores it zero."),
            SliceSpec(
                name="vae_feat_cache", scope=Scope.EPISODE, residency=Residency.MANAGED,
                bytes_per_episode=int((93.12 + 46.56) * 1024 * 1024),
                note="two streaming causal-VAE instances (cam_high + the half-res wrist pair), "
                     "modules/utils.py:67-97. Fixed named slots; no indexing, no eviction."),
            SliceSpec(
                name="write_receipt", scope=Scope.FORWARD, residency=Residency.MANAGED,
                bytes_per_episode=0, commit_mode=CommitMode.TRANSIENT,
                note="model.py:444-459 -- lives for exactly one forward, of which there are 79 "
                     "per control step. This is why Scope.FORWARD exists."),
        ),
        live_sets=(
            LiveSet(
                name="kv_live", addressing=Addressing.RING_INTERVAL,
                order=Order.PHYSICAL_ASCENDING, backing="kv",
                commit_period=_LB_COMMIT, capacity=_LB_CAP,
                segments=(
                    Segment("video", reads_from=("video", "action"), causal=False),
                    Segment("action", reads_from=("video", "action"), causal=False),
                )),
        ),
    )


# =============================================================================================
# GR00T N — the no-op case. Not "a model with a small manifest": a model with NO state.
# If L3 installs anything here, the abstraction is wrong.
# =============================================================================================

def gr00t_manifest() -> StateManifest:
    return StateManifest(
        model_id="gr00t-n",
        forwards_per_step=5,
        sync_budget=0,
        arenas=(),
        slices=(
            SliceSpec(
                name="rtc_action_continuity", scope=Scope.STEP,
                residency=Residency.CALLER_OWNED, bytes_per_episode=0,
                note="arrives as action_input['action'] per request. CALLER_OWNED means excluded "
                     "from accounting, parking, hoisting and caching, and NEVER content-validated "
                     "(invariant I8)."),
        ),
        live_sets=(),
    )


# =============================================================================================
# pi-0 / pi-0.5 — the second model to validate on. A chunk-scoped prefix that is REBUILT.
# =============================================================================================

_PI0_LAYERS = 18
_PI0_PREFIX = 867          # VLM prefix tokens
_PI0_SUFFIX = 1            # action expert
_PI0_KV_BYTES = 2 * _PI0_LAYERS * (_PI0_PREFIX + _PI0_SUFFIX) * 1 * 256 * 2


def pi0_manifest() -> StateManifest:
    return StateManifest(
        model_id="pi-0",
        forwards_per_step=10,
        sync_budget=1,
        arenas=(
            ArenaSpec(name="prefix_kv", scope=Scope.STEP,
                      extent_tokens=_PI0_PREFIX + _PI0_SUFFIX,
                      bytes_per_token=_PI0_KV_BYTES // (_PI0_PREFIX + _PI0_SUFFIX),
                      slices=("prefix_kv",)),
        ),
        slices=(
            SliceSpec(
                name="prefix_kv", scope=Scope.STEP, residency=Residency.MANAGED,
                bytes_per_episode=_PI0_KV_BYTES, commit_mode=CommitMode.CONFIRMED,
                extent=Extent(tokens=_PI0_PREFIX + _PI0_SUFFIX, bounding_field="max_prefix_len"),
                discovery=Discovery.D2_GROWING_CAT,
                note="I previously recorded this as 'built once, read 10 times, nothing to do'. "
                     "That was wrong: pi0_pytorch.py:453-455 passes past_key_values with "
                     "use_cache=False, so modeling_gemma.py:309-310 rebuilds the set by torch.cat "
                     "on the READ path -- 304.8 MiB of transient copy per control step."),
        ),
        live_sets=(
            LiveSet(
                name="prefix_live", addressing=Addressing.PREFIX,
                order=Order.LOGICAL_POSITION, backing="prefix_kv",
                capacity=_PI0_PREFIX + _PI0_SUFFIX,
                segments=(
                    Segment("prefix", reads_from=("prefix",), causal=True),
                    Segment("suffix", reads_from=("prefix", "suffix"), causal=False),
                )),
        ),
    )


#: SUPPORTED: validated end to end on real hardware with real weights.
#:   lingbot-va    primary target; full correctness gates and episode-mode benchmarks
#:   cosmos3-edge  second reference model; engine generality. Real cuDNN attention, no shim;
#:                 GraphExecutor 2.33x bit-exact. No accuracy claim -- random weights.
#:
#: UNVALIDATED DESIGN ENTRIES: `gr00t` and `pi-0` below are design sketches written from published
#: architecture descriptions. There are NO checkpoints for them on this box and NOTHING has been
#: measured on them. They are kept because they shaped the lifetime abstraction, and they are
#: segregated so nothing can mistake them for support. Do not cite them in a claim.
REGISTRY = {
    "lingbot-va": lingbot_va_manifest,
}

UNVALIDATED_DESIGNS = {
    "gr00t": gr00t_manifest,
    "pi-0": pi0_manifest,
}


# =============================================================================================
# Cosmos3-Edge — the second model, and deliberately the one that looks LEAST like LingBot-VA.
# Two towers, no boolean mask anywhere, and a defect L3's first detector could not have seen.
# =============================================================================================

_C3_GEN_LAYERS = 28


def cosmos3_edge_manifest() -> StateManifest:
    return StateManifest(
        model_id="cosmos3-edge",
        forwards_per_step=16,          # NFE 16 is the served rung
        sync_budget=2,                 # split_lens.tolist() and packed_und_token_indexes.tolist()
        arenas=(
            # The GEN sequence is a PACKED token buffer, not a KV pool. Declaring it as an arena
            # with a live set whose backing is that buffer is what lets the same descriptor cover
            # a model with no persistent KV at all.
            ArenaSpec(name="gen_packed", scope=Scope.STEP, extent_tokens=567,
                      bytes_per_token=2048 * 2, slices=("gen_sequence",)),
        ),
        slices=(
            SliceSpec(
                name="gen_sequence", scope=Scope.STEP, residency=Residency.MANAGED,
                bytes_per_episode=567 * 2048 * 2,
                extent=Extent(tokens=567, bounding_field="max_position_embeddings"),
                discovery=Discovery.D3_STATIC_INDEX,
                note="sequence_packing/runtime.py:60-83 builds the und/gen index tensors from a "
                     "Python range then torch.tensor(..., device=cuda) EVERY pack; :253 .tolist()s "
                     "a device tensor whose only product is `assert len(...) == 0`; get_all_seq "
                     ":435-441 allocates fresh zeros and scatters TWICE to reassemble what it just "
                     "split, called from attention.py:203-204 i.e. 2x per layer per forward "
                     "(2 x 28 x 16 = 896 times per control step). The indices are a pure function "
                     "of (sample_lens, split_lens, attn_modes) -- declared geometry."),
            SliceSpec(
                name="attn_reassembly_scratch", scope=Scope.FORWARD,
                residency=Residency.MANAGED,
                bytes_per_episode=567 * 2048 * 2,
                commit_mode=CommitMode.TRANSIENT,
                extent=Extent(tokens=567, bounding_field="max_position_embeddings"),
                note="get_all_seq (runtime.py:430-441) allocates new_zeros([N_all, H, D]) and "
                     "scatters twice, per call. Called from attention.py:203-204, i.e. 2x per "
                     "layer per forward = 2 x 28 x 16 = 896 allocations per control step, "
                     "~2.08 GB of alloc-and-scatter traffic. The two results are LIVE "
                     "SIMULTANEOUSLY as the K and V arguments of one attention() call. "
                     "set_all_seq (runtime.py:445) is defined but never called anywhere in the "
                     "tree, so the `all_seq` memo at :430 never fires and every call takes the "
                     "allocating path."),
            SliceSpec(
                name="reasoner_kv", scope=Scope.STEP, residency=Residency.UNMANAGED,
                bytes_per_episode=0,
                note="ReasonerKVCache exists in the tree but build_memory_state returns None on "
                     "the served path (omni_mot_model.py:689-708) and no MemoryState subclass "
                     "exists, so E_materialized is 0 here. UNMANAGED means L3 reports it and does "
                     "not manage it -- an earlier estimate summed a cache the server never builds."),
            SliceSpec(
                name="text_cond_kv", scope=Scope.STEP, residency=Residency.RECOMPUTED,
                bytes_per_episode=0,
                note="chunk-scoped and rebuilt every control step although the JSON prompt is "
                     "episode-constant. A candidate for P7, declared-only."),
        ),
        live_sets=(
            # ONE buffer, TWO causalities. This is why Segment carries causality rather than the
            # arena: two_way_attention runs is_causal=True over und-only keys and a full pass whose
            # KV is und UNION gen (attention.py:133-206), inside the same packed sequence.
            LiveSet(
                name="gen_live", addressing=Addressing.STATIC_PARTITION,
                order=Order.LOGICAL_POSITION, backing="gen_packed", capacity=567,
                segments=(
                    Segment("und", reads_from=("und",), causal=True),
                    Segment("gen", reads_from=("und", "gen"), causal=False),
                )),
        ),
    )


REGISTRY["cosmos3-edge"] = cosmos3_edge_manifest

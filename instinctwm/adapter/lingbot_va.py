"""LingBot-VA backend adapter — declarations only.

Every field below is a fact read out of `/home/ubuntu/lingbot-va`, cited to `file:line`. Nothing
here is an optimization; the optimizer derives those. If you find yourself wanting to add a
`use_fast_path=True` field, it belongs in a pass instead.
"""

from __future__ import annotations

from instinctwm.adapter.base import (
    AdapterSpec, CommitMode, GuidanceMode, GuidanceRule, KVLifetime, KVStreamSpec,
    PhaseSpec, PurityKey,
)

# --- geometry, from wan_va/configs/va_robotwin_cfg.py --------------------------------------
_HEIGHT, _WIDTH = 256, 320
_FRAME_CHUNK = 2            # frame_chunk_size
_ACTION_PER_FRAME = 16      # action_per_frame
_PATCH = (1, 2, 2)          # patch_size
# env_type 'robotwin_tshape': cam_high full res, the two wrist cams at half res, composited into
# a T. latent grid = ((256//16)*3)//2 x (320//16) = 24 x 20, patched by (2,2) -> 12 x 10 = 120.
_VIDEO_TOKENS_PER_FRAME = ((_HEIGHT // 16) * 3 // 2) * (_WIDTH // 16) // (_PATCH[1] * _PATCH[2])
_ACTION_TOKENS_PER_FRAME = _ACTION_PER_FRAME


def lingbot_va_spec() -> AdapterSpec:
    return AdapterSpec(
        model_id="lingbot-va-posttrain-robotwin",
        param_bytes=10_179_017_396,  # transformer safetensors, bf16

        # Two CO-EQUAL committed streams. Both are written by the same `update_cache` path
        # (model.py:444-447) and both are persisted with update_cache=2 in _compute_kv_cache
        # (wan_va_server.py:593-601), so both are attended in every later control step. The
        # action stream is NOT scratch — this is the fact that vLLM-Omni's single
        # `tokens_per_frame` + `max_scratch_tokens_per_branch` spec cannot express.
        streams=(
            KVStreamSpec(
                name="video",
                tokens_per_frame=_VIDEO_TOKENS_PER_FRAME,     # 120
                lifetime=KVLifetime.EPISODE,
                commit_mode=CommitMode.CONFIRMED,
                window_frames=72,                              # attn_window
                supports_provisional=True,                     # update_cache=1 writes is_pred
            ),
            KVStreamSpec(
                name="action",
                tokens_per_frame=_ACTION_TOKENS_PER_FRAME,     # 16
                lifetime=KVLifetime.EPISODE,
                commit_mode=CommitMode.CONFIRMED,
                window_frames=72,
                supports_provisional=True,
            ),
        ),

        # Three phases per control step. The video loop must fully complete before the action
        # loop starts: the 26th video forward writes provisional K/V that all 51 action forwards
        # read (wan_va_server.py:504-508 then :542-548). That is a hard barrier, declared here
        # via depends_on so the scheduler plans around it instead of discovering it.
        phases=(
            PhaseSpec(
                name="kv_refresh", nfe=2,
                reads=frozenset({"video", "action"}),
                writes=frozenset({"video", "action"}),
                commit_steps=frozenset({0, 1}),     # BOTH forwards run update_cache=2
            ),
            PhaseSpec(
                name="video", nfe=26,               # num_inference_steps 25, +1 padded timestep
                reads=frozenset({"video", "action"}),
                writes=frozenset({"video"}),
                commit_steps=frozenset({25}),       # last_step -> update_cache=1
                truncatable=True,                   # video_exec_step already exists, set to -1
                min_nfe=1,
                depends_on=("kv_refresh",),
            ),
            PhaseSpec(
                name="action", nfe=51,              # action_num_inference_steps 50, +1 padded
                reads=frozenset({"video", "action"}),
                writes=frozenset({"action"}),
                commit_steps=frozenset({50}),
                truncatable=True,
                min_nfe=1,
                depends_on=("video",),
            ),
        ),

        # guidance_scale=5 on video, action_guidance_scale=1 on action. The action stream's
        # combine takes the else branch and keeps [:1] (wan_va_server.py:552-555), i.e. its
        # negative branch is computed and discarded. Declared as a FACT; CFGBranchElision is
        # what turns it into an optimization.
        guidance={
            "video": GuidanceRule(mode=GuidanceMode.CFG, scale=5.0, batchable=True),
            "action": GuidanceRule(mode=GuidanceMode.POSITIVE_ONLY, scale=1.0, batchable=True),
        },

        # Episode-constant conditioning. The instruction is encoded once in _reset
        # (wan_va_server.py:421-435) and never changes within an episode, yet cross-attention
        # gets attn_caches=None (model.py:331) so its k/v projections over the 512-token text
        # embedding are recomputed in all 30 layers on all 77 forwards.
        purity=(
            PurityKey(artifact="text_kv", fields=("prompt",), scope=KVLifetime.EPISODE),
            PurityKey(artifact="negative_text_kv", fields=("negative_prompt",),
                      scope=KVLifetime.EPISODE),
        ),

        # The predicted video is never consumed by the RoboTwin client — it asks only for
        # actions. `_infer` returns latents that the caller drops (wan_va_server.py:623-624).
        obs_decode_modules=("vae.decoder",),

        notes={
            "attn_mode": "torch (custom_sdpa); forced by the server and by transformer/config.json",
            "kv_pool": "9792 slots, grows 272 tokens/cycle, saturates ~cycle 36, 6.72 GiB",
            "measured_stock_cycle_ms": "8881 on idle H100 = 32 actions = 3.6 Hz",
            "known_sync": "model.py:451 mask.nonzero() per layer per forward",
            "known_gather": "model.py:452-453 key_pool[:, valid] re-gathers the whole pool",
        },
    )

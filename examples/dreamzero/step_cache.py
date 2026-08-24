"""Step-caching for DreamZero serving: the row's real lever, absorbed as an opt-in.

WHERE THE MECHANISM COMES FROM. DreamZero's own action head ships a velocity-cosine dynamic
step-skipper (`should_run_model`, groot/vla/model/dreamzero/action_head/
wan_flow_matching_action_tf.py:943-970): after 2 history steps, if successive CFG-combined
velocity predictions have cosine similarity > 0.95 the next 4 DiT forwards are skipped (> 0.93
skips 2), reusing the last velocity for both the video and action streams. vLLM-Omni's
`stepcache` backend (vllm_omni/diffusion/cache/stepcache/) is this exact algorithm with the same
thresholds — it is where their 2.77x on this row comes from, together with compiled kernels.

Upstream gates it behind `DYNAMIC_CACHE_SCHEDULE` (env, default off, read at model __init__:207)
and otherwise uses a FIXED 16-slot mask selected by `NUM_DIT_STEPS` (default 8 computed steps —
the stock baseline already skips half the schedule; the dynamic path skips more, adaptively).

WHAT THIS MODULE ADDS. Nothing algorithmic — the value is operational: a declared serving
configuration with a measured latency/delta trade-off instead of an undocumented env var.
`serving_env()` returns the environment overlay for the official server; the measurement
protocol and deltas live in /home/ubuntu/iwm_distill/bench_dreamzero_h100/ours_stepcache.json.

TIER: SCREEN, not a certificate. Step-caching skips compute, so outputs differ from stock by
construction; the action-delta statistics quantify by how much, and a closed-loop success-rate
gate would be required before shipping this as a default.
"""

from __future__ import annotations

import os


def serving_env(dynamic: bool = True) -> dict:
    """Environment overlay for `eval_utils/serve_dreamzero_wan22.py` enabling dynamic step-cache."""
    env = dict(os.environ)
    env["DYNAMIC_CACHE_SCHEDULE"] = "true" if dynamic else "false"
    return env

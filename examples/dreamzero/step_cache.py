"""Environment overlay for upstream DreamZero's optional dynamic step cache.

The pinned native and vLLM-Omni implementations share the decision rule: compare
the last two COMPUTED CFG video velocities in FP32; cosine > 0.95 skips four
slots, > 0.93 skips two, including the decision slot. Both the video velocity and
the conditional action velocity are reused, although action is not part of the
similarity test. Every solver slot and the observation KV updates still run.

Dynamic scheduling replaces the shipped fixed eight-of-sixteen mask. It can
compute MORE than eight steps on unstable signals; realized counts need tracing.
Existing cross-framework timings do not isolate this mechanism's speedup.

This helper changes no algorithm. Runtime now supports step_cache="dynamic" with
native or explicitly selected FP8 precision; both require tier_ceiling="behavioral".
The option changes computation and has SCREEN evidence, not a quality certificate.
See INSTALL.rst for the API and
eval/dynamic_step_cache_integration_2026-09-14/audit.json for the bounded
CPU/GPU integration evidence; no task-quality certificate.
"""

from __future__ import annotations

import os


def serving_env(dynamic: bool = True) -> dict:
    """Environment overlay for `eval_utils/serve_dreamzero_wan22.py` enabling dynamic step-cache."""
    env = dict(os.environ)
    env["DYNAMIC_CACHE_SCHEDULE"] = "true" if dynamic else "false"
    return env

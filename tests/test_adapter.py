"""Tests for the declaration layer and the loader.

The LingBot-VA numbers asserted here are the ones the write-ups quote. They are pinned so a
refactor cannot quietly change a published figure — the eval README's results chapter and the adapter have to move
together or this fails.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from instinctflash import (  # noqa: E402  (importable only after the sys.path insert above)
    KVLifetime,
    available_models,
    load,
    register,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = load("lingbot-va-posttrain-robotwin").spec()


# --- declarations match the published numbers ---------------------------------------------

def test_forward_counts():
    assert SPEC.phase("kv_refresh").nfe == 2
    assert SPEC.phase("video").nfe == 26      # num_inference_steps 25, +1 padded timestep
    assert SPEC.phase("action").nfe == 51     # action_num_inference_steps 50, +1 padded
    assert SPEC.total_forwards() == 79        # all phases; the docs' "77" is denoise only
    assert SPEC.phase("video").nfe + SPEC.phase("action").nfe == 77


def test_forwards_breakdown_shows_its_terms():
    # total_forwards() alone (79) contradicts the 77 quoted in the published table, so anything
    # user-facing has to say which count it means.
    assert SPEC.forwards_breakdown() == "kv_refresh=2 + video=26 + action=51"


def test_token_geometry():
    # env_type 'robotwin_tshape': latent grid ((256//16)*3)//2 x (320//16) = 24 x 20,
    # patched by (2,2) -> 12 x 10 = 120.
    video = next(s for s in SPEC.streams if s.name == "video")
    action = next(s for s in SPEC.streams if s.name == "action")
    assert video.tokens_per_frame == 120
    assert action.tokens_per_frame == 16


def test_both_streams_are_co_equal_and_episode_scoped():
    # The finding that shaped the abstraction: a boolean `is_stateful` would have excluded
    # chunk-scoped models, and a single `tokens_per_frame` cannot express two committed
    # streams with different token densities.
    assert {s.lifetime for s in SPEC.streams} == {KVLifetime.EPISODE}
    assert len(SPEC.streams) == 2


def test_the_action_phase_depends_on_the_video_phase():
    # The 26th video forward writes provisional K/V that all 51 action forwards read: a hard
    # barrier, declared rather than discovered by the scheduler.
    assert SPEC.phase("action").depends_on == ("video",)
    assert SPEC.phase("video").depends_on == ("kv_refresh",)


def test_kv_refresh_commits_on_both_forwards():
    # A single commit index would silently mark one of the two streams' commits elidable,
    # corrupting the episode several chunks later.
    assert SPEC.phase("kv_refresh").commit_steps == frozenset({0, 1})


def test_phase_lookup_raises_on_an_unknown_name():
    try:
        SPEC.phase("nope")
    except KeyError:
        return
    raise AssertionError("AdapterSpec.phase should raise on an unknown phase")


# --- loader -------------------------------------------------------------------------------

def test_registry_lists_the_builtin_backend():
    assert "lingbot-va-posttrain-robotwin" in available_models()


def test_load_rejects_an_unknown_model_with_a_useful_message():
    try:
        load("not-a-model")
    except KeyError as exc:
        assert "Registered:" in str(exc)
        return
    raise AssertionError("load() should raise on an unknown model id")


def test_register_refuses_to_overwrite():
    # Two adapters claiming one id means one is silently unused, decided by import order.
    try:
        register("lingbot-va-posttrain-robotwin", lambda: None)
    except KeyError:
        return
    raise AssertionError("register() should refuse to overwrite an existing model id")


def test_load_passes_kwargs_to_the_backend():
    model = load("lingbot-va-posttrain-robotwin", lingbot_root="/somewhere/else")
    assert model.lingbot_root == "/somewhere/else"


# --- the analysis path must not need torch -------------------------------------------------

def test_planning_does_not_import_torch():
    """Deciding what is legal has to work on a laptop with no CUDA.

    Run in a subprocess because another test in this process may already have imported torch.
    """
    code = (
        "import sys; import instinctflash; "
        "from instinctflash import load, Optimizer, Tier; "
        "Optimizer(tier_ceiling=Tier.BITEXACT)"
        ".compile(load('lingbot-va-posttrain-robotwin').spec()).explain(); "
        "assert 'torch' not in sys.modules, 'planning pulled in torch'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


if __name__ == "__main__":
    # Script-style entry, matching the rest of this directory. pytest still collects the
    # test_* functions above directly.
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))

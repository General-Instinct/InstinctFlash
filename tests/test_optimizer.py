"""Tests for the planning layer.

These lock down the invariants the source argues are load-bearing — the ones whose failure
mode is a plausible wrong claim rather than a crash. In tier order of how expensive they are
to get wrong:

  * a plan containing a lossy pass must never report itself bit-exact
  * a pass must never elide a forward that commits K/V
  * a guard on a deployment fact must be reachable
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from instinctwm import (  # noqa: E402  (importable only after the sys.path insert above)
    AdapterSpec,
    CommitMode,
    DeploymentSpec,
    GuidanceMode,
    GuidanceRule,
    KVLifetime,
    KVStreamSpec,
    Optimizer,
    PassResult,
    PhaseSpec,
    Tier,
    default_passes,
    load,
)
from instinctwm.optimizer.passes.cfg_elision import CFGBranchElision

LINGBOT = load("lingbot-va-posttrain-robotwin").spec()
SINGLE_GPU = DeploymentSpec()


def _spec(**overrides) -> AdapterSpec:
    """A minimal two-stream spec, so a test can vary one declaration at a time."""
    base = dict(
        model_id="synthetic",
        param_bytes=1,
        streams=(
            KVStreamSpec("video", 120, KVLifetime.EPISODE, CommitMode.CONFIRMED),
            KVStreamSpec("action", 16, KVLifetime.EPISODE, CommitMode.CONFIRMED),
        ),
        phases=(
            PhaseSpec("video", nfe=4, writes=frozenset({"video"}), commit_steps=frozenset({3})),
            PhaseSpec("action", nfe=6, writes=frozenset({"action"}), commit_steps=frozenset({5})),
        ),
        guidance={
            "video": GuidanceRule(GuidanceMode.CFG, 5.0),
            "action": GuidanceRule(GuidanceMode.POSITIVE_ONLY, 1.0),
        },
    )
    base.update(overrides)
    return AdapterSpec(**base)


# --- tiers do not compose upward ---------------------------------------------------------

def test_plan_tier_is_the_weakest_applied_claim():
    plan = Optimizer(tier_ceiling=Tier.NUMERIC).compile(LINGBOT)
    assert any(r.tier is Tier.NUMERIC for r in plan.applied), "expected a NUMERIC pass to fire"
    assert plan.tier() is Tier.NUMERIC, "one lossy pass must make the whole plan lossy"


def test_bitexact_subset_is_always_bitexact():
    plan = Optimizer(tier_ceiling=Tier.NUMERIC).compile(LINGBOT)
    subset = plan.bitexact_subset()
    assert subset.tier() is Tier.BITEXACT
    assert all(r.tier is Tier.BITEXACT for r in subset.results)


def test_empty_plan_is_bitexact():
    assert Optimizer(passes=[]).compile(LINGBOT).tier() is Tier.BITEXACT


def test_ceiling_blocks_a_legal_but_lossy_pass():
    plan = Optimizer(tier_ceiling=Tier.BITEXACT).compile(LINGBOT)
    cfg = next(r for r in plan.results if r.name == "cfg_branch_elision")
    assert not cfg.applies
    assert "exceeds ceiling" in cfg.reason
    # The pass is still recorded at its true tier, so explain() can show what was withheld.
    assert cfg.tier is Tier.NUMERIC


def test_explain_warns_when_the_plan_is_lossy():
    text = Optimizer(tier_ceiling=Tier.NUMERIC).compile(LINGBOT).explain()
    assert "plan tier: NUMERIC" in text
    assert "non-inferiority" in text


# --- deployment facts reach the passes that guard on them --------------------------------

def test_fsdp_elision_declines_when_sharding_is_real():
    plan = Optimizer().compile(LINGBOT, DeploymentSpec(world_size=8))
    fsdp = next(r for r in plan.results if r.name == "fsdp_elision")
    assert not fsdp.applies, "eliding FSDP at world_size=8 would delete real sharding"
    assert "world_size=8" in fsdp.reason


def test_obs_decode_elision_declines_when_the_caller_wants_pixels():
    plan = Optimizer().compile(LINGBOT, DeploymentSpec(want_pixels=True))
    obs = next(r for r in plan.results if r.name == "obs_decode_elision")
    assert not obs.applies


def test_single_gpu_actions_only_is_the_default_deployment():
    assert Optimizer().compile(LINGBOT).applied == Optimizer().compile(LINGBOT, SINGLE_GPU).applied


def test_every_default_pass_accepts_a_deployment():
    # Regression: FSDPElision and ObsDecodeElision once took `world_size` / `want_pixels` as
    # extra keyword arguments that Optimizer.compile had no way to supply, so both guards
    # were dead code that always took the default branch.
    for p in default_passes():
        result = p.evaluate(LINGBOT, SINGLE_GPU)
        assert isinstance(result, PassResult)
        assert result.name == p.name


# --- CFG branch elision -------------------------------------------------------------------

def test_cfg_elision_never_elides_a_committing_forward():
    result = CFGBranchElision().evaluate(LINGBOT, SINGLE_GPU)
    assert result.applies
    action = result.params["targets"]["action"]
    # LingBot-VA's 51st action forward commits K/V that the video stream's NEGATIVE branch
    # later attends. Eliding it corrupts the episode several chunks later.
    assert action["elide_except_steps"] == [50]
    assert action["n_elided"] == 50
    assert action["n_total"] == 51


def test_cfg_elision_skips_a_phase_where_every_forward_commits():
    result = CFGBranchElision().evaluate(LINGBOT, SINGLE_GPU)
    # kv_refresh writes the action stream but commits on both of its forwards.
    assert "kv_refresh" not in result.params["targets"]


def test_cfg_elision_reports_per_phase_counts():
    # A bare "N of <total_forwards()>" reads as a claim about the whole control step and its
    # denominator (79) contradicts the 77-forward denoise count quoted in the write-ups.
    reason = CFGBranchElision().evaluate(LINGBOT, SINGLE_GPU).reason
    assert "action: 50 of 51" in reason


def test_cfg_elision_declines_without_a_cfg_stream():
    spec = _spec(guidance={
        "video": GuidanceRule(GuidanceMode.NONE),
        "action": GuidanceRule(GuidanceMode.POSITIVE_ONLY),
    })
    assert not CFGBranchElision().evaluate(spec, SINGLE_GPU).applies


def test_cfg_elision_declines_when_every_stream_consumes_both_branches():
    spec = _spec(guidance={
        "video": GuidanceRule(GuidanceMode.CFG, 5.0),
        "action": GuidanceRule(GuidanceMode.CFG, 3.0),
    })
    assert not CFGBranchElision().evaluate(spec, SINGLE_GPU).applies


def test_cfg_elision_declines_when_branches_are_not_batched():
    # The DreamZero case: a positive-only stream whose branches run as separate forwards, so
    # there is no duplicated batch to shrink.
    spec = _spec(guidance={
        "video": GuidanceRule(GuidanceMode.CFG, 5.0, batchable=False),
        "action": GuidanceRule(GuidanceMode.POSITIVE_ONLY, 1.0, batchable=False),
    })
    assert not CFGBranchElision().evaluate(spec, SINGLE_GPU).applies


# --- Plan.without -------------------------------------------------------------------------

def test_without_demotes_but_keeps_the_record():
    plan = Optimizer(tier_ceiling=Tier.NUMERIC).compile(LINGBOT)
    trimmed = plan.without("cfg_branch_elision")
    assert trimmed.tier() is Tier.BITEXACT
    assert "cfg_branch_elision" not in [r.name for r in trimmed.applied]
    dropped = next(r for r in trimmed.results if r.name == "cfg_branch_elision")
    assert "dropped by caller" in dropped.reason


def test_without_rejects_an_unknown_pass():
    plan = Optimizer().compile(LINGBOT)
    try:
        plan.without("no_such_pass")
    except KeyError:
        return
    raise AssertionError("Plan.without should reject a pass that is not in the plan")


if __name__ == "__main__":
    # Script-style entry, matching the rest of this directory. pytest still collects the
    # test_* functions above directly.
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))

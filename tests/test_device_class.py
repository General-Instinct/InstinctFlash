#!/usr/bin/env python3
"""Device hints must not generalize one model's capture timings to every model."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instinctflash.descriptors.deployment import DeploymentSpec  # noqa: E402
from instinctflash.passes.contract import DeviceProfile  # noqa: E402
from instinctflash.passes.generic.graph_capture import GraphCaptureApplicable  # noqa: E402
from instinctflash.planners.planner import Optimizer, Tier
from types import SimpleNamespace


def _dev(cap, name="StubGPU"):
    return DeviceProfile(name=name, capability=cap, total_memory=64 << 30,
                         features=frozenset({"cuda", "cuda_graphs", "triton"}))


class _StaticSpec:
    """The minimum a declaration needs to say 'my shapes repeat across cycles'."""

    model_id = "stub/static-shapes"
    notes = {}

    def phase(self, name):
        return SimpleNamespace(nfe=10)

    def operating_point(self):
        return "action=10"

    def shapes_static_across_cycles(self):
        return True, "one chunk-lifetime prefix rebuilt per chunk"

    def total_forwards(self):
        return 10

    def forwards_breakdown(self):
        return "10 action"


def test_measured_classes_and_the_honest_unmeasured():
    cls, why = _dev((9, 0)).device_class()
    assert cls == "launch-bound" and "1.65-4.54x" in why
    cls, why = _dev((11, 0)).device_class()
    assert cls == "bandwidth-bound-edge" and "family/operating-point" in why and "Thor" in why
    cls, why = _dev((0, 0), name="CPU (x86_64)").device_class()
    assert cls == "cpu"
    for cap in ((8, 9), (10, 0), (12, 0)):
        cls, why = _dev(cap).device_class()
        assert cls == "unmeasured", f"{cap} must not inherit a measured class"
        assert f"sm{cap[0]}{cap[1]}" in why


def test_pi05_measurement_does_not_decide_other_families():
    p = GraphCaptureApplicable()
    spec = _StaticSpec()
    spec.notes = {"backbone": "pi05"}
    r = p.evaluate(spec, DeploymentSpec(device=_dev((11, 0))))
    assert r.applies and "2026-09-09" in r.reason and "pi05" in r.reason
    assert r.tier == Tier.BITEXACT
    spec.notes = {"backbone": "lingbot_vla"}
    r = p.evaluate(spec, DeploymentSpec(device=_dev((11, 0))))
    assert r.applies and r.tier == Tier.BITEXACT
    spec.phase = lambda name: SimpleNamespace(nfe=4)
    assert not p.evaluate(spec, DeploymentSpec(device=_dev((11, 0)))).applies
    spec.notes = {"backbone": "unmeasured-family"}
    r = p.evaluate(spec, DeploymentSpec(device=_dev((11, 0))))
    assert not r.applies and "unmeasured" in r.reason
    assert "1.04x" not in r.reason


def test_v2_capture_uses_its_own_device_evidence_and_numeric_ceiling():
    spec = _StaticSpec()
    spec.notes = {"backbone": "lingbot_vla_v2", "capture_tier": "NUMERIC"}
    deployment = DeploymentSpec(device=_dev((11, 0)))
    p = GraphCaptureApplicable()
    legal = p.evaluate(spec, deployment)
    assert legal.applies and legal.tier == Tier.NUMERIC
    assert "LingBot-VLA-V2" in legal.params["device_evidence"]
    strict = Optimizer(passes=[p]).compile(spec, deployment)
    assert not strict.applied
    numeric = Optimizer(passes=[p], tier_ceiling=Tier.NUMERIC).compile(spec, deployment)
    assert numeric.tier() == Tier.NUMERIC and len(numeric.applied) == 1
    # The same numeric self-check must never masquerade as BITEXACT on H100 either.
    h100 = Optimizer(passes=[p]).compile(spec, DeploymentSpec(device=_dev((9, 0))))
    assert not h100.applied
    spec.phase = lambda name: SimpleNamespace(nfe=4)
    assert not p.evaluate(spec, deployment).applies


def test_groot_thor_capture_is_scoped_to_measured_four_step_schedule():
    p = GraphCaptureApplicable()
    spec = _StaticSpec()
    spec.notes = {"backbone": "groot_n17"}
    deployment = DeploymentSpec(device=_dev((11, 0)))
    assert not p.evaluate(spec, deployment).applies  # unmeasured 10-step schedule
    spec.phase = lambda name: SimpleNamespace(nfe=4)
    result = p.evaluate(spec, deployment)
    assert result.applies and result.tier == Tier.BITEXACT
    assert "existing DiT graph" in result.params["device_evidence"]
    spec.shapes_static_across_cycles = lambda: (False, "changing shape")
    assert not p.evaluate(spec, deployment).applies


def test_legacy_numeric_adapter_cannot_bypass_the_ceiling():
    spec = _StaticSpec()
    spec.notes = {"backbone": "lingbot_vla_v2",
                  "numeric_tier": "NUMERIC (upstream fused-MoE is nondeterministic)"}
    for capability in ((9, 0), (11, 0)):
        plan = Optimizer(passes=[GraphCaptureApplicable()]).compile(
            spec, DeploymentSpec(device=_dev(capability)))
        assert not plan.applied
        assert plan.results[0].tier == Tier.NUMERIC


def test_capture_still_applies_where_launch_bound_or_unmeasured():
    p = GraphCaptureApplicable()
    r90 = p.evaluate(_StaticSpec(), DeploymentSpec(device=_dev((9, 0))))
    assert r90.applies, r90.reason
    # an UNMEASURED sm keeps the launch-bound default -- the class surface says it is a default,
    # and flipping behaviour on silicon nobody measured would itself be an extrapolation.
    r120 = p.evaluate(_StaticSpec(), DeploymentSpec(device=_dev((12, 0))))
    assert r120.applies, r120.reason
    # no probed device: unchanged behaviour, the planner annotates hardware as unchecked
    rnone = p.evaluate(_StaticSpec(), DeploymentSpec())
    assert rnone.applies


def test_the_shape_reason_still_wins_after_the_device_gate():
    class _Dynamic(_StaticSpec):
        def shapes_static_across_cycles(self):
            return False, "episode-lifetime KV ring grows every cycle"

    p = GraphCaptureApplicable()
    r = p.evaluate(_Dynamic(), DeploymentSpec(device=_dev((9, 0))))
    assert not r.applies and "invalidated every cycle" in r.reason


if __name__ == "__main__":
    from run_tests import run_module_tests
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(run_module_tests(globals()))

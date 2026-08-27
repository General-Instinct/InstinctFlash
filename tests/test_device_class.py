#!/usr/bin/env python3
"""The device-class law: measured classes only, and the capture pass obeys it.

Two GPUs were measured and they flip the optimization landscape in opposite directions:
H100 (sm90) is launch-bound -- static-KV capture collects 1.65-4.54x -- and Thor (sm110) is
bandwidth-bound -- the same capture measured 1.04x, no battlefield. `device_class()` states
exactly that and calls everything else 'unmeasured', because assigning an unmeasured device to
either side would be the extrapolation the field exists to replace.

The consequence pinned here: `graph_capture` declares itself launch-bound-devices-only. On the
measured bandwidth-bound edge class the plan DECLINES it with the measured reason, before the
shape question is even asked -- the law holds for every model on the device. No GPU, no torch.

    python tests/test_device_class.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instinctflash.descriptors.deployment import DeploymentSpec  # noqa: E402
from instinctflash.passes.contract import DeviceProfile  # noqa: E402
from instinctflash.passes.generic.graph_capture import GraphCaptureApplicable  # noqa: E402


def _dev(cap, name="StubGPU"):
    return DeviceProfile(name=name, capability=cap, total_memory=64 << 30,
                         features=frozenset({"cuda", "cuda_graphs", "triton"}))


class _StaticSpec:
    """The minimum a declaration needs to say 'my shapes repeat across cycles'."""

    model_id = "stub/static-shapes"

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
    assert cls == "bandwidth-bound-edge" and "1.04x" in why and "Thor" in why
    cls, why = _dev((0, 0), name="CPU (x86_64)").device_class()
    assert cls == "cpu"
    for cap in ((8, 9), (10, 0), (12, 0)):
        cls, why = _dev(cap).device_class()
        assert cls == "unmeasured", f"{cap} must not inherit a measured class"
        assert f"sm{cap[0]}{cap[1]}" in why


def test_capture_declines_on_the_bandwidth_bound_edge_class():
    p = GraphCaptureApplicable()
    r = p.evaluate(_StaticSpec(), DeploymentSpec(device=_dev((11, 0))))
    assert not r.applies, "capture must decline on the measured bandwidth-bound edge class"
    assert "1.04x" in r.reason and "launch-bound-device" in r.reason, r.reason
    assert "bandwidth-bound-edge" in r.reason


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

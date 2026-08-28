"""graph_capture is the DEFAULT for LingBot-VLA-4B, LingBot-VLA-V2 and GR00T-N1.7, gated by the
family-generic startup self-check (instinctflash/runtime/capture_self_check.py).

The pi05 pattern (8c83599), generalized. The gaps these pins close, per family:

  * 4B installed the verified static-KV capture plan-gated but WITHOUT a runtime self-check —
    the trust was a historical measurement on other checkpoints, not a per-process proof;
  * V2 same (and its tier is NUMERIC: the fused-MoE kernel disagrees with itself on identical
    seeds, so its gate is the recorded stock-vs-stock envelope, never atol=0);
  * GR00T kept capture opt-in through IFL_GROOT_STATIC_CAPTURE=1 — a release policy that
    predates the self-check and is superseded by it: a fresh fine-tune served EAGER.

These pins say: default-on when the plan applies on a CUDA build; kill-switches follow the
IFL_PI05_NO_CAPTURE naming (IFL_VLA4B_NO_CAPTURE / IFL_VLA2_NO_CAPTURE / IFL_GROOT_NO_CAPTURE),
loud and recorded on the plan; the retired GROOT opt-in is a no-op with a notice ("0" honored
as an explicit opt-out); a rejected capture releases its graphs, rebinds upstream, and serving
continues. The GPU truth (checks passing on real checkpoints, the fault drill failing loudly)
is measured by scripted checks on H100 — these are the weight-free halves.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for plugin in ("lingbot_vla", "lingbot_vla_v2", "groot_n17"):
    sys.path.insert(0, str(ROOT / "examples" / plugin))

import torch  # noqa: E402  (the family capture modules import it at module scope)

from instinctflash.planners.planner import Plan, PassResult, Tier  # noqa: E402
from instinctflash.runtime.capture_self_check import (  # noqa: E402
    record_self_check_on_plan,
    run_capture_self_check,
)

FAMILY_FLAGS = (
    "IFL_VLA4B_NO_CAPTURE", "IFL_VLA2_NO_CAPTURE", "IFL_GROOT_NO_CAPTURE",
    "IFL_VLA4B_BACKEND", "IFL_VLA2_BACKEND", "IFL_GROOT_STATIC_CAPTURE",
    "IFL_VLA4B_SELFCHECK_FAULT", "IFL_VLA2_SELFCHECK_FAULT", "IFL_GROOT_SELFCHECK_FAULT",
    "IFL_VLA2_GPU_PREPROCESS", "IFL_VLA2_PREFIX_GRAPH", "IFL_VLA2_CUDA_KERNELS",
    "IFL_VLA2_MOE_KERNEL", "IFL_VLA2_RMSNORM_KERNEL",
)


@contextlib.contextmanager
def _clean_env(env: dict | None = None):
    with mock.patch.dict(os.environ):
        for flag in FAMILY_FLAGS:
            os.environ.pop(flag, None)
        os.environ.update(env or {})
        yield


def _plan() -> Plan:
    return Plan("test/fresh-finetune", [
        PassResult("graph_capture", True, Tier.BITEXACT, "shapes repeat")])


def _fake_cuda_torch():
    fake = ModuleType("torch")
    fake.cuda = SimpleNamespace(is_available=lambda: True)
    fake.is_tensor = torch.is_tensor
    return fake


# ── the generic helper: verdict math and printed lines ────────────────────────────────────────


def _cases(deltas):
    a = torch.zeros(4)
    for d in deltas:
        yield ("captured-chunk", lambda a=a: a, lambda a=a, d=d: a + d)


def test_helper_bitexact_verdicts():
    out = io.StringIO()
    with contextlib.redirect_stderr(out):
        good = run_capture_self_check(family="Fam", cases=_cases([0.0, 0.0]), tolerance=0.0)
    assert good["passed"] and good["bitexact"] and good["n"] == 2
    assert out.getvalue() == "", "a PASS prints via the plan recorder, not the helper"

    with contextlib.redirect_stderr(out):
        bad = run_capture_self_check(family="Fam", cases=_cases([0.0, 0.25]), tolerance=0.0)
    assert not bad["passed"] and bad["max_abs_delta"] == 0.25
    text = out.getvalue()
    assert "SELF-CHECK FAILED" in text and "2.500e-01" in text and "[Fam static_capture]" in text
    assert "exact equality" in text


def test_helper_envelope_verdicts():
    out = io.StringIO()
    with contextlib.redirect_stderr(out):
        good = run_capture_self_check(family="Fam", cases=_cases([0.03]),
                                      tolerance=5.08e-02, tolerance_provenance="the null")
    assert good["passed"] and not good["bitexact"]

    with contextlib.redirect_stderr(out):
        bad = run_capture_self_check(family="Fam", cases=_cases([0.06]),
                                     tolerance=5.08e-02, tolerance_provenance="the null")
    assert not bad["passed"]
    text = out.getvalue()
    assert "5.080e-02" in text and "the null" in text, text


def test_recorder_writes_the_promised_lines():
    capture = PassResult("graph_capture", True, Tier.BITEXACT, "shapes repeat")
    record = record_self_check_on_plan(capture, "LingBot-VLA-4B")
    out = io.StringIO()
    with contextlib.redirect_stderr(out):
        record({"n": 6, "passed": True, "bitexact": True, "max_abs_delta": 0.0,
                "tolerance": 0.0, "tolerance_provenance": "", "seconds": 2.31,
                "cases": [{"stage": "captured-chunk"}] * 3 + [{"stage": "refilled"}] * 3})
    text = out.getvalue()
    assert "InstinctFlash LingBot-VLA-4B: graph_capture self-check bit-exact on 6 inputs" in text
    assert "3 refilled" in text and "2.3 s startup" in text
    assert any("bit-exact on 6 inputs" in line for line in capture.params["decision"])
    assert capture.params["self_check"]["seconds"] == 2.31

    numeric = PassResult("graph_capture", True, Tier.BITEXACT, "shapes repeat")
    record = record_self_check_on_plan(numeric, "LingBot-VLA-V2")
    with contextlib.redirect_stderr(out):
        record({"n": 6, "passed": True, "bitexact": False, "max_abs_delta": 1.2e-02,
                "tolerance": 5.08e-02, "tolerance_provenance": "the stock-vs-stock null",
                "seconds": 1.0, "cases": [{"stage": "refilled"}]})
    text = out.getvalue()
    assert "within the recorded envelope" in text
    assert "1.200e-02" in text and "5.080e-02" in text and "stock-vs-stock null" in text

    with contextlib.redirect_stderr(out):
        record({"n": 6, "passed": False, "bitexact": False, "max_abs_delta": 3.2e-01,
                "tolerance": 5.08e-02, "tolerance_provenance": "the stock-vs-stock null",
                "seconds": 1.0, "cases": [{"stage": "refilled"}]})
    text = out.getvalue()
    assert "self-check FAILED" in text and "3.200e-01" in text and "serve continues" in text
    assert any("FAILED" in line for line in numeric.params["decision"])


def test_v2_envelope_is_the_recorded_null_control():
    """The NUMERIC gate's threshold is the artifact's, not a hand-picked number."""
    from lingbot_vla_v2_iwm.static_capture import NULL_ENVELOPE
    artifact = json.loads(
        (ROOT / "examples" / "lingbot_vla_v2" / "moe_kernel_results.json").read_text())
    assert NULL_ENVELOPE == max(artifact["null_control_deltas"]) == artifact["envelope"]


# ── LingBot-VLA-4B: default install, kill-switch, eager selector ─────────────────────────────


def _install_4b(env=None):
    import lingbot_vla_iwm.static_capture as sc4b
    from lingbot_vla_iwm.adapter import LingBotVLA4BAdapter
    installed = {}

    def fake_install(model, on_self_check=None, self_check=True):
        installed.update(model=model, on_self_check=on_self_check)
        return SimpleNamespace(self_check=None, rejected=False)

    plan = _plan()
    out = io.StringIO()
    with _clean_env(env), \
            mock.patch.object(sc4b, "install_static_capture", fake_install), \
            contextlib.redirect_stdout(out):
        driver = LingBotVLA4BAdapter().install(
            SimpleNamespace(vla=SimpleNamespace(model=object())), plan, device="cuda:0")
    return driver, installed, plan, out.getvalue()


def test_4b_fresh_finetune_gets_capture_by_default():
    driver, installed, plan, printed = _install_4b()
    assert driver is not None and installed["model"] is not None
    assert callable(installed["on_self_check"]), "the gate must be WIRED, not just described"
    assert "the family default on capture-capable devices" in printed
    assert "self-check" in printed and "IFL_VLA4B_NO_CAPTURE" in printed


def test_4b_kill_switch_is_loud_and_recorded():
    driver, installed, plan, printed = _install_4b({"IFL_VLA4B_NO_CAPTURE": "1"})
    assert driver is None and "model" not in installed
    assert "IFL_VLA4B_NO_CAPTURE=1" in printed and "running eager" in printed
    assert any("IFL_VLA4B_NO_CAPTURE" in line
               for line in plan.results[0].params.get("decision", ()))


def test_4b_eager_backend_selector_stays_honored_and_names_the_kill_switch():
    driver, installed, plan, printed = _install_4b({"IFL_VLA4B_BACKEND": "eager"})
    assert driver is None and "model" not in installed
    assert "IFL_VLA4B_BACKEND=eager" in printed and "IFL_VLA4B_NO_CAPTURE" in printed
    assert any("IFL_VLA4B_BACKEND=eager" in line
               for line in plan.results[0].params.get("decision", ()))


# ── LingBot-VLA-V2: default install with the envelope gate, kill-switch kills the whole arm ──


def _install_v2(env=None):
    import lingbot_vla_v2_iwm.prefix_capture as pc
    import lingbot_vla_v2_iwm.static_capture as sc
    from lingbot_vla_v2_iwm.adapter import LingBotVLAV2Adapter
    installed = {}

    def fake_install(model, on_self_check=None, self_check=True):
        installed.update(model=model, on_self_check=on_self_check)
        return SimpleNamespace(self_check=None, rejected=False, graph=None, replays=0)

    def fake_prefix(model):
        installed["prefix"] = True
        return SimpleNamespace(close=lambda: installed.update(prefix_closed=True))

    plan = _plan()
    server = SimpleNamespace(vla=SimpleNamespace(model=object()))
    out = io.StringIO()
    with _clean_env(env), \
            mock.patch.object(sc, "install_static_capture", fake_install), \
            mock.patch.object(pc, "install_prefix_capture", fake_prefix), \
            contextlib.redirect_stdout(out):
        driver = LingBotVLAV2Adapter().install(server, plan, mode="static", device="cuda:0")
    return driver, installed, plan, server, out.getvalue()


def test_v2_fresh_finetune_gets_capture_by_default_with_the_envelope_gate():
    driver, installed, plan, server, printed = _install_v2()
    assert driver is not None and installed["model"] is not None
    assert callable(installed["on_self_check"])
    assert installed.get("prefix"), "the vision/prefill graphs stay in the default arm"
    assert "the family default on capture-capable devices" in printed
    assert "stock-vs-stock envelope" in printed and "5.084e-02" in printed
    assert "NUMERIC, not BITEXACT" in printed and "IFL_VLA2_NO_CAPTURE" in printed


def test_v2_kill_switch_kills_the_whole_capture_arm():
    driver, installed, plan, server, printed = _install_v2({"IFL_VLA2_NO_CAPTURE": "1"})
    assert driver is None and "model" not in installed and "prefix" not in installed
    assert "IFL_VLA2_NO_CAPTURE=1" in printed and "vision/prefill" in printed
    assert any("IFL_VLA2_NO_CAPTURE" in line
               for line in plan.results[0].params.get("decision", ()))


def test_v2_failed_self_check_releases_the_prefix_graphs_too():
    from lingbot_vla_v2_iwm.adapter import _release_prefix_graphs_on_fail
    closed = []
    server = SimpleNamespace(
        _instinctflash_prefix_capture=SimpleNamespace(close=lambda: closed.append(True)))
    recorded = []
    hook = _release_prefix_graphs_on_fail(recorded.append, server)
    out = io.StringIO()
    with contextlib.redirect_stderr(out):
        hook({"passed": True})
        assert not closed, "a PASS must not touch the prefix graphs"
        hook({"passed": False})
    assert closed == [True]
    assert server._instinctflash_prefix_capture is None
    assert "vision/prefill graphs released" in out.getvalue()
    assert len(recorded) == 2, "the plan recorder must still see every verdict"


# ── GR00T N1.7: the opt-in is superseded — capture is the default, self-check gated ──────────


def _install_groot(env=None):
    import groot_n17_iwm.static_capture as sc
    from groot_n17_iwm.adapter import GR00TN17Adapter
    installed = {}

    def fake_install(model, on_self_check=None, self_check=True):
        installed.update(model=model, on_self_check=on_self_check)
        return SimpleNamespace(captured=False, captures=0, replays=0)

    plan = _plan()
    policy = SimpleNamespace(model=object())
    out = io.StringIO()
    with _clean_env(env), \
            mock.patch.dict(sys.modules, {"torch": _fake_cuda_torch()}), \
            mock.patch.object(sc, "install_static_capture", fake_install), \
            contextlib.redirect_stdout(out):
        got = GR00TN17Adapter.install(policy, plan, device="cuda:0")
    return got, installed, plan, out.getvalue()


def test_groot_fresh_finetune_gets_capture_by_default():
    got, installed, plan, printed = _install_groot()
    assert got == ["graph_capture"] and installed["model"] is not None
    assert callable(installed["on_self_check"])
    assert "superseding the retired IFL_GROOT_STATIC_CAPTURE opt-in" in printed
    assert "self-check" in printed and "IFL_GROOT_NO_CAPTURE" in printed


def test_groot_retired_opt_in_is_a_noop_with_notice():
    got, installed, plan, printed = _install_groot({"IFL_GROOT_STATIC_CAPTURE": "1"})
    assert got == ["graph_capture"] and installed["model"] is not None
    assert "IFL_GROOT_STATIC_CAPTURE=1 is a no-op" in printed
    assert "IFL_GROOT_NO_CAPTURE=1 disables it" in printed


def test_groot_explicit_opt_out_is_honored_with_notice():
    got, installed, plan, printed = _install_groot({"IFL_GROOT_STATIC_CAPTURE": "0"})
    assert got == [] and "model" not in installed
    assert "honored" in printed and "IFL_GROOT_NO_CAPTURE=1" in printed
    assert any("IFL_GROOT_STATIC_CAPTURE=0" in line
               for line in plan.results[0].params.get("decision", ()))


def test_groot_kill_switch_is_loud_and_recorded():
    got, installed, plan, printed = _install_groot({"IFL_GROOT_NO_CAPTURE": "1"})
    assert got == [] and "model" not in installed
    assert "IFL_GROOT_NO_CAPTURE=1" in printed and "running eager" in printed
    assert any("IFL_GROOT_NO_CAPTURE" in line
               for line in plan.results[0].params.get("decision", ()))


# ── the FAIL arm, weight-free: graphs released, upstream rebound, serving continues ──────────


class _Expert4B:
    def handle_kv_cache(self, *args, **kwargs):
        return ("stock", args, kwargs)


class _FM4B:
    def __init__(self):
        self.qwenvl_with_expert = _Expert4B()

    def predict_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        return ("eager", x_t)


def test_4b_rejection_releases_the_graph_and_rebinds_upstream():
    from lingbot_vla_iwm.static_capture import install_static_capture
    fm = _FM4B()
    upstream = type(fm).predict_velocity
    d = install_static_capture(fm)
    assert d._orig_predict is upstream
    assert type(fm).predict_velocity is not upstream, "install must have routed the step"
    d._graph, d._out = object(), object()
    out = io.StringIO()
    with contextlib.redirect_stderr(out):
        d._release_and_fall_back()
    assert "Graph released" in out.getvalue() and "serving continues" in out.getvalue()
    assert d.rejected and d._graph is None and d._out is None
    # the model's own step now IS upstream again
    assert fm.predict_velocity(None, None, None, 7, None) == ("eager", 7)
    # and a caller still holding the old binding lands on eager too
    assert d(None, None, None, 7, None) == ("eager", 7)
    type(fm).predict_velocity = upstream          # undo the class patch for other tests


class _ExpertV2:
    def handle_kv_cache(self, *args, **kwargs):
        return ("stock", args, kwargs)


class _FMV2:
    def __init__(self):
        self.qwenvl_with_expert = _ExpertV2()

    def predict_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep,
                         prefix_position_ids=None):
        return ("eager", x_t)


def test_v2_rejection_releases_the_graph_and_rebinds_upstream():
    from lingbot_vla_v2_iwm.static_capture import install_static_capture
    fm = _FMV2()
    upstream = fm.predict_velocity
    d = install_static_capture(fm)
    assert fm.predict_velocity is not upstream
    d.graph, d._out = object(), object()
    out = io.StringIO()
    with contextlib.redirect_stderr(out):
        d._release_and_fall_back()
    assert "Graph released" in out.getvalue()
    assert d.rejected and d.graph is None
    assert fm.predict_velocity(None, None, None, 7, None) == ("eager", 7)
    assert d(None, None, None, 7, None, prefix_position_ids=None) == ("eager", 7)


def test_groot_rejection_routes_every_call_to_upstream():
    from groot_n17_iwm.static_capture import StaticDiT
    d = StaticDiT(lambda **kwargs: ("eager", kwargs))
    d._graphs["sig"] = object()
    out = io.StringIO()
    with contextlib.redirect_stderr(out):
        d._release_and_fall_back()
    assert "Graphs released" in out.getvalue() and "serving continues" in out.getvalue()
    assert d.rejected and not d._graphs
    assert d(sample=1)[0] == "eager"


def test_kill_switch_names_follow_the_pi05_convention():
    from groot_n17_iwm.adapter import GR00TN17Adapter
    from lingbot_vla_iwm import adapter as a4b
    from lingbot_vla_v2_iwm import adapter as av2
    assert a4b.CAPTURE_KILL_SWITCH == "IFL_VLA4B_NO_CAPTURE"
    assert av2.CAPTURE_KILL_SWITCH == "IFL_VLA2_NO_CAPTURE"
    assert GR00TN17Adapter.CAPTURE_KILL_SWITCH == "IFL_GROOT_NO_CAPTURE"


if __name__ == "__main__":
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))

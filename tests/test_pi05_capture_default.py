"""graph_capture is the DEFAULT for pi05-class checkpoints, gated by the bit-exact self-check.

The gap under test, historically: `Pi05Adapter.install` took the verified static-KV capture only
when a published compile_model=true had been superseded or an env opt-in was set — things a fresh
lerobot-train fine-tune lacks — so exactly the checkpoint the one-command story is about served
EAGER (~207 ms/chunk) while the same weights measured bit-exact at ~73 ms captured. These pins say:

  * a plain plan with graph_capture APPLY on a CUDA build installs the static-KV capture, period;
  * the first capture is gated by the self-check, whose verdict lands on the PLAN's own
    graph_capture entry (the line explain() renders) in the promised phrasing;
  * IFL_PI05_NO_CAPTURE=1 is the kill-switch — loud, recorded, and refused outright on a
    checkpoint that DECLARES the TF32 static-KV operating point;
  * the old opt-ins are no-ops with a notice, never silent;
  * a rejected capture releases the graphs and rebinds upstream, and serving continues.

The GPU truth (capture applying on real fine-tunes, the self-check passing, the fault drill
failing loudly) is measured by scripted checks on H100 — these are the weight-free halves.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "pi05_vla"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PLUGIN))

from instinctflash.planners.planner import Plan, PassResult, Tier  # noqa: E402

from pi05_iwm.adapter import (  # noqa: E402
    CAPTURE_KILL_SWITCH,
    Pi05Adapter,
    _record_self_check_on_plan,
)
from pi05_iwm.surface import Pi05Surface  # noqa: E402

CAPTURE_FLAGS = (CAPTURE_KILL_SWITCH, Pi05Surface.STATIC_CAPTURE_OPT_IN,
                 Pi05Surface.CAPTURE_OPT_IN, "IFL_PI05_SELFCHECK_FAULT")


def _fake_cuda_torch():
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    torch.get_float32_matmul_precision = lambda: "high"
    torch.backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)))
    torch.device = lambda spec: spec
    return torch


def _plan(*extra: PassResult) -> Plan:
    return Plan("test/pi05-fresh-finetune", [
        PassResult("graph_capture", True, Tier.BITEXACT, "shapes repeat"), *extra])


@contextlib.contextmanager
def _install_env(env: dict | None = None):
    """Fake CUDA torch + stubbed surface/static-capture, with the capture env vars controlled."""
    installed = {}

    def fake_install(model, step_tables=None, on_self_check=None, self_check=True):
        installed["model"] = model
        installed["step_tables"] = step_tables
        installed["on_self_check"] = on_self_check
        return SimpleNamespace(self_check=None, rejected=False)

    class FakeSurface:
        def __init__(self, model):
            pass

        def hoist_loop_constants(self):
            return ["attn hoist", "mask hoist"]

    import pi05_iwm.adapter as adapter_module
    import pi05_iwm.static_capture as static_module
    import pi05_iwm.surface as surface_module
    out = io.StringIO()
    with mock.patch.dict(sys.modules, {"torch": _fake_cuda_torch()}), \
            mock.patch.dict(os.environ), \
            mock.patch.object(static_module, "install_static_capture", fake_install), \
            mock.patch.object(surface_module, "Pi05Surface",
                              type("S", (FakeSurface,), {
                                  "STATIC_CAPTURE_OPT_IN": Pi05Surface.STATIC_CAPTURE_OPT_IN,
                                  "CAPTURE_OPT_IN": Pi05Surface.CAPTURE_OPT_IN})), \
            contextlib.redirect_stdout(out):
        for flag in CAPTURE_FLAGS:
            os.environ.pop(flag, None)
        os.environ.update(env or {})
        yield installed, out
    _ = adapter_module  # imported for symmetry with what install() resolves at call time


def test_a_fresh_finetune_gets_static_capture_by_default():
    """No compile_model=true, no env flag — the plan applies, so the capture installs."""
    plan = _plan()
    with _install_env() as (installed, out):
        got = Pi05Adapter.install(SimpleNamespace(model=object()), plan, device="cuda:0")
    assert got == ["loop_constant_hoist", "graph_capture_static_kv"], got
    printed = out.getvalue()
    assert "the pi05-family default on capture-capable devices" in printed, printed
    assert "self-check" in printed and CAPTURE_KILL_SWITCH in printed
    # the gate is WIRED, not just described: the verdict recorder reaches the installer
    assert callable(installed["on_self_check"])
    assert installed["model"] is not None


def test_the_superseded_compile_model_line_is_unchanged():
    """v044's path: same install, and the because-clause still names the neutralized flag."""
    plan = _plan()
    plan.results[0].params["compile_model_superseded"] = True
    with _install_env() as (installed, out):
        got = Pi05Adapter.install(SimpleNamespace(model=object()), plan, device="cuda:0")
    assert got == ["loop_constant_hoist", "graph_capture_static_kv"]
    assert "installed in place of the checkpoint's neutralized compile_model" in out.getvalue()


def test_the_kill_switch_is_loud_and_recorded_on_the_plan():
    plan = _plan()
    with _install_env({CAPTURE_KILL_SWITCH: "1"}) as (installed, out):
        got = Pi05Adapter.install(SimpleNamespace(model=object()), plan, device="cuda:0")
    assert got == ["loop_constant_hoist"], got
    assert "model" not in installed, "the kill-switch must not install capture"
    printed = out.getvalue()
    assert f"{CAPTURE_KILL_SWITCH}=1" in printed and "running eager" in printed
    decision = plan.results[0].params.get("decision", ())
    assert any(CAPTURE_KILL_SWITCH in line for line in decision), decision
    assert any(CAPTURE_KILL_SWITCH in line for line in plan.explain().splitlines())


def test_the_kill_switch_refuses_a_declared_tf32_operating_point():
    """static_kv_graph:true is the checkpoint's declared semantics — eager is a different one."""
    try:
        from pi05_iwm.passes import PASS_NAME
    except ImportError:
        print("SKIP: the TF32 operating-point pass is not shipped in this repo")
        return
    plan = _plan(PassResult(PASS_NAME, True, Tier.NUMERIC, "declared operating point"))
    with _install_env({CAPTURE_KILL_SWITCH: "1"}):
        try:
            Pi05Adapter.install(SimpleNamespace(model=object()), plan, device="cuda:0")
        except RuntimeError as e:
            assert CAPTURE_KILL_SWITCH in str(e) and "FP32 checkpoint" in str(e)
        else:
            raise AssertionError("a declared static-KV operating point served eager is a lie")


def test_the_old_opt_ins_are_noops_with_notice():
    for flag in (Pi05Surface.STATIC_CAPTURE_OPT_IN, Pi05Surface.CAPTURE_OPT_IN):
        plan = _plan()
        with _install_env({flag: "1"}) as (installed, out):
            got = Pi05Adapter.install(SimpleNamespace(model=object()), plan, device="cuda:0")
        assert got == ["loop_constant_hoist", "graph_capture_static_kv"], (flag, got)
        assert f"{flag}=1 is a no-op" in out.getvalue(), (flag, out.getvalue())


def test_the_recorder_writes_the_promised_lines():
    capture = PassResult("graph_capture", True, Tier.BITEXACT, "shapes repeat")
    record = _record_self_check_on_plan(capture)
    out = io.StringIO()
    # STDERR, because the verdict lands during serving and cli_config.execute defers stdout
    # until the command returns — which for a persistent `serve` is never.
    with contextlib.redirect_stderr(out):
        record({"n": 6, "bitexact": True, "max_abs_delta": 0.0, "seconds": 2.31,
                "cases": [{"prefix": "captured-chunk"}] * 3 + [{"prefix": "refilled"}] * 3})
    assert "self-check bit-exact on 6 inputs" in out.getvalue()
    lines = capture.params["decision"]
    assert any("self-check bit-exact on 6 inputs" in line for line in lines), lines
    assert capture.params["self_check"]["seconds"] == 2.31

    with contextlib.redirect_stderr(out):
        record({"n": 6, "bitexact": False, "max_abs_delta": 3.2e-01, "seconds": 2.0,
                "cases": []})
    text = out.getvalue()
    assert "self-check FAILED" in text and "3.200e-01" in text and "serve continues" in text
    assert any("FAILED" in line for line in capture.params["decision"])


# ── the denoiser's FAIL arm, weight-free ──────────────────────────────────────────────────────


class _Upstream:
    def denoise_step(self, prefix_pad_masks=None, past_key_values=None, x_t=None, timestep=None):
        return ("eager", x_t)


def test_install_binds_the_upstream_reference_arm():
    from pi05_iwm.static_capture import install_static_capture
    model = _Upstream()
    upstream = model.denoise_step
    d = install_static_capture(model, step_tables=False)
    assert d._orig_denoise is not None
    assert d._orig_denoise.__func__ is upstream.__func__
    assert model.denoise_step is not upstream, "install must have routed the step"
    # and the check can be declined explicitly (standalone measurement scripts)
    bare = install_static_capture(_Upstream(), step_tables=False, self_check=False)
    assert bare._orig_denoise is None


def test_rejection_releases_the_graph_and_rebinds_upstream():
    from pi05_iwm.static_capture import install_static_capture
    model = _Upstream()
    d = install_static_capture(model, step_tables=False)
    d._graph, d._out = object(), object()
    out = io.StringIO()
    with contextlib.redirect_stderr(out):    # the running server's live stream — see recorder test
        d._release_and_fall_back(3.2e-01)
    printed = out.getvalue()
    assert "SELF-CHECK FAILED" in printed and "3.200e-01" in printed
    assert "Graphs released" in printed and "serving continues" in printed
    assert d.rejected is True and d._graph is None and d._out is None
    # the model's own step now IS upstream again
    assert model.denoise_step(x_t=7)[0] == "eager"
    # and a caller still holding the old binding lands on eager too
    assert d(None, None, 7, None) == ("eager", 7)


if __name__ == "__main__":
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))

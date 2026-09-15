#!/usr/bin/env python3
"""The action pred-commit elision: its two fail-closed gates and its bookkeeping, without a GPU.

What is covered here, and what is not:

  * the message-pattern gate (controller state machine) with fakes: valid closed-loop order elides and
    reserves; a forward before clear_pred_cache MATERIALIZES the skipped forward first and disables the
    pass for good; a reset drops the stash; nothing is elided outside an armed `_infer`.
  * the config gate, on the REAL vendor classes: `video_exec_step != -1` disables, `-1` arms.
  * the vendor-structure invariants refuse a server whose source no longer carries the trace.
  * `reserve_provisional_slots` on a module without a self-attention cache is a no-op ("none").
  * the pass declares itself applicable on the LingBot-VA spec, and `evaluate()` keeps it opt-in
    until the A/B is recorded (PROVEN_BITEXACT).

The allocator-level proof through the ring wrap is tests/test_ring_allocator.py::test_elision_bookkeeping;
the GPU tier decision is eval/lingbot_va_robotwin/probe_action_terminal_elision.py.

Run:  python tests/test_action_terminal_elision.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va"), "wan_va"))
sys.path.insert(0, "/home/ubuntu/iwm_shims")

from instinctflash.passes.lingbot.action_terminal_elision import (  # noqa: E402
    ActionTerminalForwardElision, ElisionController, action_tokens, check_vendor_invariants,
    controller_of, reserve_provisional_slots,
)
from tests.run_tests import run_module_tests  # noqa: E402


class _Trace:
    """Fake forward plumbing: records what ran, in order."""

    def __init__(self):
        self.ran: list[str] = []
        self.logs: list[str] = []

    def log(self, *a, **k):
        self.logs.append(" ".join(str(x) for x in a))

    def forward(self, ctl, update_cache, action_mode, tag):
        def run(*a, **kw):
            self.ran.append(f"{tag}{'(materialized)' if a or kw else ''}")
            return f"out:{tag}"

        def stash():
            return ((f"args:{tag}",), {"update_cache": update_cache, "action_mode": action_mode})

        return ctl.on_forward(update_cache, action_mode, run, stash)


def _closed_loop_cycle(t, ctl, c):
    """2V/4A: video 0,1(pred) ; action 0,0,0,1(pred) ; clear ; commits 2,2."""
    ctl.armed = True
    outs = [t.forward(ctl, 0, False, f"v0.{c}"), t.forward(ctl, 1, False, f"v1.{c}")]
    for i in range(3):
        outs.append(t.forward(ctl, 0, True, f"a{i}.{c}"))
    outs.append(t.forward(ctl, 1, True, f"a3.{c}"))
    ctl.armed = False
    reserved = []
    ctl.on_clear_pred_cache(lambda args, kwargs: (reserved.append((args, kwargs)), "stock")[1])
    t.forward(ctl, 2, False, f"kv_v.{c}")
    t.forward(ctl, 2, True, f"kv_a.{c}")
    return outs, reserved


def test_valid_pattern_elides_and_reserves():
    t, ctl = _Trace(), ElisionController(log=lambda *a, **k: None)
    for c in range(3):
        outs, reserved = _closed_loop_cycle(t, ctl, c)
        assert outs[-1] is None, "the action pred-commit forward must return None (its consumer is dead)"
        assert all(o is not None for o in outs[:-1])
        assert reserved == [(("args:a3.%d" % c,), {"update_cache": 1, "action_mode": True})]
        assert not ctl.pending
    assert f"a3.0" not in t.ran and "a3.2" not in t.ran, "the elided forward must never run"
    assert ctl.stats()["elided"] == 3 and ctl.stats()["reserved"] == 3
    assert ctl.stats()["materialized"] == 0 and not ctl.disabled_reason


def test_forward_before_clear_materializes_then_disables():
    t, ctl = _Trace(), ElisionController(log=lambda *a, **k: None)
    ctl.armed = True
    t.forward(ctl, 1, False, "v1")
    assert t.forward(ctl, 1, True, "a3") is None and ctl.pending
    # vendor generate(): the next _infer starts with a video forward, no clear_pred_cache in between
    out = t.forward(ctl, 0, False, "v0.next")
    assert out == "out:v0.next"
    assert t.ran[-2:] == ["a3(materialized)", "v0.next"], t.ran
    assert not ctl.pending and ctl.disabled_reason and ctl.stats()["materialized"] == 1
    # sticky: from now on the action pred-commit forward RUNS
    assert t.forward(ctl, 1, True, "a3.next") == "out:a3.next"
    assert ctl.stats()["elided"] == 1
    ctl.on_clear_pred_cache(lambda a, k: (_ for _ in ()).throw(AssertionError("nothing to reserve")))


def test_reset_drops_the_stash():
    t, ctl = _Trace(), ElisionController(log=lambda *a, **k: None)
    ctl.armed = True
    assert t.forward(ctl, 1, True, "a3") is None
    ctl.on_cache_dropped()
    assert not ctl.pending
    called = []
    ctl.on_clear_pred_cache(lambda a, k: called.append(1) or "stock")
    assert called == [], "a reset wiped the pool; there is nothing to reserve"
    assert not ctl.disabled_reason


def test_nothing_elided_when_not_armed_or_not_requested():
    t, ctl = _Trace(), ElisionController(log=lambda *a, **k: None)
    assert t.forward(ctl, 1, True, "a3") == "out:a3"          # not armed: outside _infer
    ctl.armed, ctl.requested = True, False
    assert t.forward(ctl, 1, True, "a3b") == "out:a3b"        # operator switch off
    ctl.requested = True
    ctl.disable("test")
    assert t.forward(ctl, 1, True, "a3c") == "out:a3c"        # sticky disabled
    assert ctl.stats()["elided"] == 0


def test_reserve_is_a_noop_without_a_self_attention_cache():
    class NoCache:
        attn_caches = None
    assert reserve_provisional_slots(NoCache(), "pos", 32) == "none"

    class Uninit:
        attn_caches = {"pos": {"k": None}}
    assert reserve_provisional_slots(Uninit(), "pos", 32) == "none"


def test_action_tokens_follows_the_embed_rearrange():
    import torch
    x = torch.zeros(2, 30, 2, 16, 1)      # b c f h w -> b (f h w) c  =>  32 slots
    assert action_tokens({"noisy_latents": x}) == 32


def test_applicability_and_opt_in():
    from instinctflash.adapters.lingbot_va import lingbot_va_spec
    from instinctflash.descriptors.deployment import DeploymentSpec
    p = ActionTerminalForwardElision()
    app = p.applicability(lingbot_va_spec(), None)
    assert app.applies, app.reason
    assert app.claimed_tier.name == "BITEXACT"
    r = p.evaluate(lingbot_va_spec(), DeploymentSpec())
    if not ActionTerminalForwardElision.PROVEN_BITEXACT:
        assert not r.applies and "opt-in" in r.reason
    else:
        assert r.applies and r.tier.name == "BITEXACT"


def test_vendor_invariants_refuse_a_foreign_server():
    src = "def _infer(self):\n    return 1\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fake_server.py")
        open(path, "w").write(src)
        spec = importlib.util.spec_from_file_location("fake_server", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        try:
            check_vendor_invariants(mod)
        except RuntimeError as e:
            assert "no longer matches the trace" in str(e)
        else:
            raise AssertionError("a server without the traced structure was accepted")


def test_real_vendor_invariants():
    if __name__ != "__main__":
        import pytest
        pytest.importorskip("diffusers", reason="Requires the LingBot vendor environment")
        pytest.importorskip("transformers", reason="Requires the LingBot vendor environment")
    from instinctflash.runtime.lingbot_install import import_lingbot_server
    check_vendor_invariants(import_lingbot_server())


def test_config_gate_on_the_real_classes():
    """Install on the real vendor module + server class; drive the `_infer` wrapper with a fake self."""
    if __name__ != "__main__":
        import pytest
        pytest.importorskip("diffusers", reason="Requires the LingBot vendor environment")
        pytest.importorskip("transformers", reason="Requires the LingBot vendor environment")
    from instinctflash.runtime.lingbot_install import import_lingbot_server
    S = import_lingbot_server()
    import modules.model as M

    seen = {}

    def stub_infer(self, obs, frame_st_id=0):
        seen["armed"] = controller_of(self.transformer).armed
        return "stub"

    S.VA_Server._infer = stub_infer
    ActionTerminalForwardElision().install(S, S.VA_Server)
    assert M.WanTransformer3DModel._iwm_ate_installed
    # idempotent: a second install (e.g. plan + CLI flag) must not double-wrap
    ActionTerminalForwardElision().install(S, S.VA_Server)

    class FakeModel:
        pass

    class FakeSelf:
        transformer = FakeModel()
        job_config = types.SimpleNamespace(video_exec_step=-1)

    fs = FakeSelf()
    assert S.VA_Server._infer(fs, {}) == "stub"
    ctl = controller_of(fs)
    assert seen["armed"] is True and not ctl.disabled_reason and ctl.armed is False

    fs.job_config.video_exec_step = 10
    S.VA_Server._infer(fs, {})
    assert ctl.disabled_reason and "video_exec_step" in ctl.disabled_reason
    assert seen["armed"] is True and not ctl.active, "armed for scoping, but never active once disabled"


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))

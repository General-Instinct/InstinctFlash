"""A published ``compile_model: true`` yields to the plan's own graph_capture, and only to it.

The rule under test lives in pi05_iwm/adapter.py: when the plan's graph_capture pass APPLIES,
the runtime's static-KV capture serves the same denoise loop bit-exactly at 72.8 ms/chunk vs
torch.compile's 173.3 (measured, H100/v044) with seconds of warmup instead of 171 s+ of
first-start autotune — so the checkpoint's compile flag is neutralized, printed, and recorded
on the plan for explain(). Everywhere capture has no case (plan declines, no capture in the
plan, CPU build), the publisher's key must stand untouched: a checkpoint author's deployment
choice is only overridden by something measured to be strictly better, never by silence.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "pi05_vla"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PLUGIN))

from instinctflash.planners.planner import Plan, PassResult, Tier

import pi05_iwm.adapter as adapter_module
from pi05_iwm.adapter import (
    COMPILE_SUPERSEDED_REASON,
    Pi05Adapter,
    _neutralize_compile_model_for_planned_capture,
)


def _plan(applies: bool) -> Plan:
    return Plan("test/pi05", [PassResult("graph_capture", applies, Tier.BITEXACT,
                                         "shapes repeat" if applies else "declined for test")])


def _fire(cfg, plan, device="cuda:0"):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fired = _neutralize_compile_model_for_planned_capture(cfg, plan, device)
    return fired, out.getvalue()


def test_planned_capture_neutralizes_the_published_flag_and_says_why():
    cfg = SimpleNamespace(compile_model=True)
    plan = _plan(True)
    fired, printed = _fire(cfg, plan)
    assert fired is True
    assert cfg.compile_model is False
    assert COMPILE_SUPERSEDED_REASON in printed, printed
    assert "compile_model neutralized" in printed
    # the installer's contract rides on the same plan object build hands to install()
    assert plan.results[0].params["compile_model_superseded"] is True


def test_the_decision_reaches_plan_explain_and_runtime_explain():
    cfg = SimpleNamespace(compile_model=True)
    plan = _plan(True)
    _fire(cfg, plan)
    text = plan.explain()
    assert "decision:" in text
    assert COMPILE_SUPERSEDED_REASON in text

    from instinctflash.runtime.facade import Runtime
    checkpoint = SimpleNamespace(
        model_id="test/pi05", path="/tmp/nowhere",
        execution=SimpleNamespace(model_id="test/pi05", backbone="pi05", servable=True),
        capabilities=lambda: frozenset({"backbone:pi05"}),
    )
    runtime = Runtime(checkpoint, SimpleNamespace(), plan,
                      SimpleNamespace(close=lambda: None), placement_reason="test")
    assert COMPILE_SUPERSEDED_REASON in runtime.explain()


def test_declining_capture_keeps_the_publishers_key():
    """The regression pin: where capture has no case, compile_model stays honored."""
    cfg = SimpleNamespace(compile_model=True)
    plan = _plan(False)
    fired, printed = _fire(cfg, plan)
    assert fired is False
    assert cfg.compile_model is True, "capture declined: the checkpoint author's choice stands"
    assert printed == ""
    assert "compile_model_superseded" not in plan.results[0].params
    assert "decision:" not in plan.explain()


def test_a_plan_without_capture_keeps_the_publishers_key():
    cfg = SimpleNamespace(compile_model=True)
    fired, printed = _fire(cfg, Plan("test/pi05", []))
    assert fired is False and cfg.compile_model is True and printed == ""


def test_a_cpu_build_keeps_the_key_even_when_the_plan_applies():
    # an APPLICABILITY-UNCHECKED plan can carry graph_capture APPLY onto a CPU build; capture
    # cannot install there, and neutralizing compile_model there would replace the author's
    # choice with nothing at all
    cfg = SimpleNamespace(compile_model=True)
    fired, _ = _fire(cfg, _plan(True), device="cpu")
    assert fired is False and cfg.compile_model is True


def test_an_absent_or_false_flag_is_never_touched_or_narrated():
    cfg = SimpleNamespace(compile_model=False)
    plan = _plan(True)
    fired, printed = _fire(cfg, plan)
    assert fired is False and printed == ""
    assert "compile_model_superseded" not in plan.results[0].params


# ── the whole ordering claim, through the real build path with a stubbed lerobot ─────────────


class _FakeParameter:
    dtype = "torch.float32"

    def is_floating_point(self):
        return True


def _fake_cuda_torch():
    torch = ModuleType("torch")
    torch.float32 = "torch.float32"
    torch.uint8 = "torch.uint8"
    torch.cuda = SimpleNamespace(is_available=lambda: True,
                                 get_device_capability=lambda _dev=None: (9, 0))
    torch.device = lambda spec: spec
    return torch


def test_build_neutralizes_before_construction_and_the_installer_sees_the_promise():
    """The flag is consumed by PI05Policy.from_pretrained, so the decision must precede it —
    and the promise must reach install() on the same plan object, not vanish after the print."""
    fake_torch = _fake_cuda_torch()
    seen = {}

    class FakeConfig:
        dtype = "bfloat16"
        compile_model = True
        num_inference_steps = 10

        @classmethod
        def from_pretrained(cls, _repo):
            return cls()

    class FakePolicy:
        def __init__(self, config):
            self.config = config
            self.model = object()

        @classmethod
        def from_pretrained(cls, _repo, *, config):
            seen["compile_at_construction"] = config.compile_model
            return cls(config)

        def eval(self):
            return self

        def to(self, _device):
            return self

        def named_parameters(self):
            return iter((("weight", _FakeParameter()),))

        def reset(self):
            pass

    factory = ModuleType("lerobot.policies.factory")
    factory.make_pre_post_processors = lambda *_args, **_kwargs: (lambda x: x, lambda x: x)
    config_mod = ModuleType("lerobot.policies.pi05.configuration_pi05")
    config_mod.PI05Config = FakeConfig
    model_mod = ModuleType("lerobot.policies.pi05.modeling_pi05")
    model_mod.PI05Policy = FakePolicy
    fake_modules = {
        "torch": fake_torch,
        "lerobot": ModuleType("lerobot"),
        "lerobot.policies": ModuleType("lerobot.policies"),
        "lerobot.policies.factory": factory,
        "lerobot.policies.pi05": ModuleType("lerobot.policies.pi05"),
        "lerobot.policies.pi05.configuration_pi05": config_mod,
        "lerobot.policies.pi05.modeling_pi05": model_mod,
    }
    checkpoint = SimpleNamespace(
        model_id="test/pi05-finetune",
        execution=SimpleNamespace(
            extra={"base_weights": "test/pi05-finetune-weights"},
            nfe={"prefix": 1, "action": 10},
        ),
    )
    plan = SimpleNamespace(results=[PassResult("graph_capture", True, Tier.BITEXACT, "test")])

    def record_install(_policy, install_plan, *, device=None, **_kw):
        capture = next(r for r in install_plan.results if r.name == "graph_capture")
        seen["superseded_at_install"] = capture.params.get("compile_model_superseded")
        return []

    out = io.StringIO()
    with mock.patch.dict(sys.modules, fake_modules), \
            mock.patch.object(adapter_module, "_require_processor_steps", lambda _repo: None), \
            mock.patch.object(Pi05Adapter, "install", record_install), \
            contextlib.redirect_stdout(out):
        loop = Pi05Adapter().build_in_process(checkpoint, plan, device="cuda:0")
        loop.close()

    assert seen == {"compile_at_construction": False, "superseded_at_install": True}
    assert COMPILE_SUPERSEDED_REASON in out.getvalue()


if __name__ == "__main__":
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))

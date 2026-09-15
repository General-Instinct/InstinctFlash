"""The engine tier honors the operating point or declines — never a silent mismatch.

THE REPORTED DEFECT (Thor probe, 2026-09-02, iwm_distill/fewstep/cert_pi05_nfe1_thor.md §2):
``Runtime.from_pretrained(v044, nfe={"action": 1}, placement="engine")`` accepted the request
and served the 10-step graph — same digest, same 49.5 ms as nfe10 — while the plan printed
``schedule {prefix=1 + action=1}`` and ``explain()`` promised 2 forwards. ``EngineBackend``
took ``nfe`` and never read it; the pi05 frontend bakes ``steps = 10`` into ``action_out_proj``
(pre-scaled by −1/steps at weight load), the per-step time tables and six ``'steps': 10`` dims
dicts. The T3 certificate covers that engine at 10 steps only.

THE RULE, one fact with two enforcement surfaces (the geometry gate's shape, other axis):
  * ``ENGINE_BAKED_OPERATING_POINTS`` declares, per engine build, the schedule it bakes and
    the guidance it serves; ``engine_operating_point_problem`` compares a requested point to it;
  * plan time — ``engine_offload`` DECLINES (no ``params['backend']``) when the plan's operating
    point (``spec.with_nfe`` / ``with_guidance``) is not the baked one, so 'auto' serves the
    declared point on the torch chain and the placement line quotes the reason;
  * build time — ``EngineBackend.__init__`` refuses the same mismatch from the checkpoint's
    declaration plus the ``nfe=`` override, which is what guards explicit ``placement='engine'``.
No frontend in serving/ is step-parameterized; none may be declared so without a proving test.

No GPU, no torch: declaration checks by construction; the engine seam is stubbed exactly like
tests/test_engine_geometry_gate.py does.
"""

from __future__ import annotations

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for plugin in ("pi05_vla", "lingbot_vla", "lingbot_vla_v2", "groot_n17"):
    p = ROOT / "examples" / plugin
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import instinctflash.runtime.engine_backend as eb  # noqa: E402
from instinctflash.descriptors.deployment import DeploymentSpec  # noqa: E402
from instinctflash.passes.contract import DeviceProfile  # noqa: E402
from instinctflash.passes.generic.engine_offload import EngineOffloadApplicable  # noqa: E402
from instinctflash.planners.planner import Optimizer  # noqa: E402
from instinctflash.runtime.engine_backend import (  # noqa: E402
    ENGINE_BACKBONES,
    ENGINE_BAKED_OPERATING_POINTS,
    PI05_ENGINE_ACTION_STEPS,
    EngineBackend,
    baked_step_literals,
    engine_baked_operating_point,
    engine_operating_point_problem,
    requested_operating_point,
)
from instinctflash.runtime.execution import InProcessBackend, choose_backend  # noqa: E402
from pi05_iwm.adapter import Pi05Adapter  # noqa: E402


def _thor() -> DeploymentSpec:
    return DeploymentSpec(device=DeviceProfile(
        name="thor-test", capability=(11, 0), total_memory=64 << 30,
        features=frozenset({"cuda", "fp8"})))


def _h100() -> DeploymentSpec:
    return DeploymentSpec(device=DeviceProfile(
        name="h100-test", capability=(9, 0), total_memory=80 << 30,
        features=frozenset({"cuda", "fp8"})))


def _checkpoint(nfe=None, guidance=None, action_dim=7):
    """A LIBERO-shaped pi05 checkpoint (the geometry the engine serves), at a declared point."""
    return SimpleNamespace(
        path="/nonexistent",
        execution=SimpleNamespace(
            backbone="pi05", model_id="test/pi05-operating-point",
            extra={"action_dim": action_dim},
            nfe=dict({"prefix": 1, "action": 10} if nfe is None else nfe),
            guidance=dict({"action": "none"} if guidance is None else guidance)),
    )


def _pi05_spec(nfe=None, guidance=None):
    spec = Pi05Adapter().spec_for_checkpoint(_checkpoint())
    if nfe:
        spec = spec.with_nfe(nfe)
    if guidance is not None:
        spec = spec.with_guidance(guidance)
    return spec


def _evaluate(spec, deployment=None):
    return EngineOffloadApplicable().evaluate(spec, deployment or _thor())


# ── the declaration: every engine build states its baked point ──────────────────────────────


def test_every_engine_frontend_declares_its_baked_point_and_none_claims_any_nfe():
    frontends = {cap.frontend for cap in ENGINE_BAKED_OPERATING_POINTS}
    # the frontends this repo's runtime or adapters name — each must have a declared point
    for expected in ("pi05_thor.py", "vla4b_thor.py", "vla2_thor.py",
                     "groot_n17_thor.py", "groot_thor.py"):
        assert any(f.endswith(expected) for f in frontends), f"{expected} declares no baked point"
    for cap in ENGINE_BAKED_OPERATING_POINTS:
        assert (ROOT / cap.frontend).is_file(), cap.frontend
        assert cap.steps and all(int(n) >= 1 for n in cap.steps.values())
        assert cap.guidance, "the served guidance is part of the baked point"
        assert cap.steps_parameterized is False, (
            f"{cap.frontend} claims to serve any nfe; that claim needs a test proving it "
            f"(digest differs from the baked count's) before it is declared")
    # the one backbone the runtime wires must be declared; the pi05 mirror is the report's 10
    for backbone in ENGINE_BACKBONES:
        if backbone in ("wan_va", "cosmos3_policy", "dreamzero"):
            assert engine_baked_operating_point(backbone) is None  # Requires a built instance.
        else:
            assert engine_baked_operating_point(backbone) is not None
    assert engine_baked_operating_point("pi05").steps == {"action": PI05_ENGINE_ACTION_STEPS}
    assert PI05_ENGINE_ACTION_STEPS == 10
    assert engine_baked_operating_point("wam") is None
    assert engine_baked_operating_point(None) is None, "a frontend without a backbone is unreachable"


def test_the_mirrors_match_the_engine_sources():
    # The runtime cross-checks the LIVE frontend at build time (Thor); this is the torch-less
    # equivalent for CI — every declared count must be exactly the literal set in serving/.
    for cap in ENGINE_BAKED_OPERATING_POINTS:
        src = (ROOT / cap.frontend).read_text()
        found = baked_step_literals(src, cap.source_patterns)
        assert found == set(cap.steps.values()), (
            f"{cap.frontend}: source bakes {sorted(found)}, the mirror says "
            f"{sorted(set(cap.steps.values()))} — update ENGINE_BAKED_OPERATING_POINTS")
    # the pi05 report named the sites: two `steps = 10` and six `'steps': 10`
    import re
    pi05_src = (ROOT / "serving/flash_rt/frontends/torch/pi05_thor.py").read_text()
    assert len(re.findall(r"(?m)^\s*steps\s*=\s*(\d+)\b", pi05_src)) == 2
    assert len(re.findall(r"['\"]steps['\"]\s*:\s*(\d+)\b", pi05_src)) == 6
    # a parameterized literal (the Thor scratch build's shape) is caught, not accepted
    assert baked_step_literals("steps = int(os.environ.get('STEPS', 10))",
                               engine_baked_operating_point("pi05").source_patterns) == set()


# ── plan time: engine_offload declines a plan priced at any other point ─────────────────────


def test_pi05_engine_declines_nfe1_with_the_reason():
    r = _evaluate(_pi05_spec(nfe={"action": 1}))
    assert not r.applies
    assert "backend" not in r.params, \
        "an operating-point decline must not carry params['backend'] — that key is what 'auto' reads"
    assert "10-step action schedule" in r.reason
    assert "requests action=1" in r.reason
    assert "torch placement" in r.reason
    assert "action_out_proj" in r.reason, "the reason names where the count is baked"
    # the whole frontier the sweep certified on the torch chain is declined on the engine
    for n in (2, 3, 5, 9, 11):
        r = _evaluate(_pi05_spec(nfe={"action": n}))
        assert not r.applies and f"requests action={n}" in r.reason, n


def test_pi05_engine_accepts_the_baked_nfe10_and_names_the_served_point():
    r = _evaluate(_pi05_spec())
    assert r.applies
    assert r.params.get("backend") == "engine", "the auto-placement signal must survive"
    assert "action=10 steps" in r.reason and "action=10 steps" in r.params["engine_operating_point"]
    # an explicit nfe equal to the baked one is the same operating point
    r = _evaluate(_pi05_spec(nfe={"prefix": 1, "action": 10}))
    assert r.applies and r.params.get("backend") == "engine"


def test_pi05_engine_declines_a_cfg_operating_point_but_not_cfg_at_scale_1():
    # the frontend HAS an RL-mode CFG pipeline; EngineBackend never enables it, so w>1 declines
    r = _evaluate(_pi05_spec(guidance={"action": {"mode": "cfg", "scale": 1.5}}))
    assert not r.applies and "backend" not in r.params
    assert "guidance none@1" in r.reason and "requests action=cfg@1.5" in r.reason
    # cfg@1, positive_only and none are the SAME computation (no negative branch combined)
    for same in ({"action": 1.0}, {"action": "positive_only"}, {"action": "none"}):
        r = _evaluate(_pi05_spec(guidance=same))
        assert r.applies and r.params.get("backend") == "engine", same


def test_unverifiable_operating_point_is_declined_fail_closed():
    spec = SimpleNamespace(notes={"backbone": "pi05", "action_dim": "7", "action_dim_source": "t"},
                           phases=(), guidance={})
    r = _evaluate(spec)
    assert not r.applies and "backend" not in r.params
    assert "could not be verified" in r.reason


def test_the_geometry_decline_still_comes_first():
    spec = _pi05_spec(nfe={"action": 1})
    import dataclasses
    spec = dataclasses.replace(spec, notes={**spec.notes, "action_dim": "33"})
    r = _evaluate(spec)
    assert not r.applies and "declares 33" in r.reason


def test_h100_uses_native_schedule_executor_and_non_engine_placements_are_unaffected():
    # H100 uses a separate projection executor; it does not bake the Thor NFE.
    for nfe in (None, {"action": 1}):
        r = _evaluate(_pi05_spec(nfe=nfe), _h100())
        assert r.applies and r.params["executor"] == "h100_torch_fp8"
        assert "engine_operating_point" not in r.params
    # a spec no engine build declares keeps today's behaviour
    r = _evaluate(SimpleNamespace(notes={"family": "wam"}))
    assert r.applies and r.params.get("backend") == "engine"
    # explicit torch placement builds the torch chain whatever the plan says about the engine
    plan = Optimizer().compile(_pi05_spec(nfe={"action": 1}), deployment=_thor())
    backend, why = choose_backend("in_process", _Adapter(), _checkpoint(nfe={"prefix": 1, "action": 1}),
                                  plan, nfe={"action": 1})
    assert isinstance(backend, InProcessBackend)
    assert why == "placement='in_process' (explicit)"


# ── the printed plan and the placement line name the reason ────────────────────────────────


def test_the_printed_plan_carries_the_decline_reason():
    plan = Optimizer().compile(_pi05_spec(nfe={"action": 1}), deployment=_thor())
    text = plan.explain()
    assert "operating point: schedule {prefix=1 + action=1}" in text
    line = next(l for l in text.splitlines() if "engine_offload" in l)
    assert line.lstrip().startswith("skip"), line
    assert "requests action=1" in line and "torch placement" in line, line
    # and at the baked point the same printout says the engine serves it
    plan10 = Optimizer().compile(_pi05_spec(), deployment=_thor())
    line10 = next(l for l in plan10.explain().splitlines() if "engine_offload" in l)
    assert "baked denoise operating point: action=10 steps" in line10, line10
    assert "action-horizon and observation/control compatibility still require validation" in line10, line10


def _local_declaration(td: str, nfe) -> str:
    Path(td, "instinctflash.json").write_text(json.dumps({
        "instinctflash_schema": 1,
        "execution": {"model_id": "test/pi05-libero-shaped", "backbone": "pi05",
                      "servable": True, "guidance": {"action": "none"},
                      "nfe": nfe, "action_dim": 7,
                      "base_weights": "lerobot/pi05_libero_finetuned_v044"},
    }))
    return td


@contextmanager
def _probe_returns_thor():
    orig = DeviceProfile.__dict__["probe"]
    DeviceProfile.probe = staticmethod(lambda device=None: _thor().device)
    try:
        yield
    finally:
        DeviceProfile.probe = orig


def test_the_preflight_printout_names_the_decline_reason():
    # the CLI's `plan` / `serve --dry_run` surface: declaration-only, device probed, no weights
    from unittest.mock import patch
    from instinctflash.cli import _serve_preflight
    from instinctflash.cli_config import RuntimeConfig
    from instinctflash.runtime import loader
    from instinctflash.runtime.facade import plan_declaration

    # This source-level plan test does not require an installed plugin entry point.
    with (patch.dict(loader._REGISTRY, {"pi05": Pi05Adapter}),
          patch.object(loader, "_DISCOVERED", True),
          tempfile.TemporaryDirectory() as td, _probe_returns_thor()):
        _local_declaration(td, {"prefix": 1, "action": 10})
        # the nfe= override is the request the Thor probe made
        _, _, plan, probed = plan_declaration(td, nfe={"action": 1})
        assert probed.capability == (11, 0)
        assert "requests action=1" in plan.explain()
        result, text = _serve_preflight(td, RuntimeConfig(nfe={"action": 1}))
        assert "requests action=1" in text and "torch placement" in text
        assert "schedule {prefix=1 + action=1}" in result["operating_point"]
        # a declaration that itself declares nfe1 declines the same way with no override
        _local_declaration(td, {"prefix": 1, "action": 1})
        _, _, plan, _ = plan_declaration(td)
        assert "requests action=1" in plan.explain()
        # and the baked point still routes to the engine on this device
        _local_declaration(td, {"prefix": 1, "action": 10})
        _, _, plan, _ = plan_declaration(td)
        eng = next(r for r in plan.results if r.name == "engine_offload")
        assert eng.params.get("backend") == "engine", eng.reason


class _Adapter:
    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
        raise AssertionError("test never builds the impl")


@contextmanager
def _engine_looks_available():
    orig = eb.engine_available
    eb.engine_available = lambda: (True, "test stub: pretend SM110 + engine kernels")
    try:
        yield
    finally:
        eb.engine_available = orig


def test_auto_falls_back_to_torch_and_quotes_the_reason_on_the_placement_line():
    plan = Optimizer().compile(_pi05_spec(nfe={"action": 1}), deployment=_thor())
    with _engine_looks_available():
        backend, why = choose_backend(
            "auto", _Adapter(), _checkpoint(), plan, nfe={"action": 1})
    assert isinstance(backend, InProcessBackend), why
    assert "auto -> in_process" in why
    eng = next(r for r in plan.results if r.name == "engine_offload")
    assert not eng.applies and "requests action=1" in eng.reason


# ── build time: EngineBackend refuses before touching the device ────────────────────────────


def test_explicit_engine_placement_is_refused_at_nfe1_before_touching_the_device():
    # the Thor probe's exact request: declaration at 10, nfe={"action": 1} override
    try:
        EngineBackend(Pi05Adapter(), _checkpoint(), None, nfe={"action": 1})
    except RuntimeError as e:
        msg = str(e)
        assert msg.startswith("engine placement refused"), msg
        assert "10-step action schedule" in msg and "requests action=1" in msg, msg
        assert "nfe= override" in msg, "the refusal says where the request came from"
        assert "unavailable" not in msg, "refused on the declaration, not on the missing device"
    else:
        raise AssertionError("nfe={'action': 1} reached the 10-step engine frontend")
    # a declaration that declares nfe1 itself, no override
    try:
        EngineBackend(Pi05Adapter(), _checkpoint(nfe={"prefix": 1, "action": 1}), None)
    except RuntimeError as e:
        assert "requests action=1" in str(e) and "execution.nfe" in str(e), str(e)
    else:
        raise AssertionError("a declared nfe1 reached the 10-step engine frontend")
    # a declared CFG scale the backend does not enable
    try:
        EngineBackend(Pi05Adapter(), _checkpoint(guidance={"action": 1.5}), None)
    except RuntimeError as e:
        assert "requests action=cfg@1.5" in str(e), str(e)
    else:
        raise AssertionError("a declared CFG scale reached the no-CFG engine frontend")
    # through choose_backend, placement='engine' is a refusal, never a 10-step service
    plan = Optimizer().compile(_pi05_spec(nfe={"action": 1}), deployment=_thor())
    with _engine_looks_available():
        try:
            choose_backend("engine", Pi05Adapter(), _checkpoint(), plan, nfe={"action": 1}, precision="fp8")
        except RuntimeError as e:
            assert "requests action=1" in str(e), str(e)
        else:
            raise AssertionError("placement='engine' served nfe1 on a 10-step engine")


def test_the_baked_point_passes_the_gate_and_reaches_the_device_check():
    # at nfe10 / no CFG the operating-point gate is silent; the next gate (engine_available)
    # is what stops this CPU box — proving the refusal above was the declaration, not the device
    try:
        EngineBackend(Pi05Adapter(), _checkpoint(), None, nfe={"action": 10})
    except RuntimeError as e:
        assert str(e).startswith("engine backend unavailable"), str(e)
    else:
        raise AssertionError("EngineBackend built on a box with no engine")


def test_plan_time_and_build_time_share_one_rule():
    baked = engine_baked_operating_point("pi05")
    plan_reason = _evaluate(_pi05_spec(nfe={"action": 1})).reason
    try:
        EngineBackend(Pi05Adapter(), _checkpoint(), None, nfe={"action": 1})
    except RuntimeError as e:
        build_reason = str(e)
    # same rule text up to the requester clause
    head = plan_reason.split("; this plan's operating point")[0]
    assert head in build_reason, (head, build_reason)
    assert engine_operating_point_problem(baked, {"action": 10}, {"action": ("none", 1.0)}) is None
    assert "requests action=1" in engine_operating_point_problem(
        baked, {"action": 1}, {"action": ("none", 1.0)})


def test_requested_operating_point_resolves_override_over_declaration_over_family():
    steps, guidance, source = requested_operating_point(Pi05Adapter(), _checkpoint(), {"action": 1})
    assert steps == {"prefix": 1, "action": 1}
    assert guidance == {"action": ("none", 1.0)}
    assert "nfe= override" in source and "execution.nfe" in source and "family adapter" in source
    steps, _, source = requested_operating_point(Pi05Adapter(), _checkpoint(nfe={"action": 3}))
    assert steps == {"prefix": 1, "action": 3}
    assert "nfe= override" not in source
    # no adapter: declaration + override only, guidance parsed from the declaration
    steps, guidance, _ = requested_operating_point(None, _checkpoint(guidance={"action": 1.5}), None)
    assert steps == {"prefix": 1, "action": 10}
    assert guidance["action"][0] == "cfg" and guidance["action"][1] == 1.5


# ── the other engine families: same check, their own baked counts ───────────────────────────


def test_the_other_engine_families_decline_any_nfe_but_their_baked_one():
    from groot_n17_iwm.adapter import GR00TN17Adapter
    from lingbot_vla_iwm.adapter import LingBotVLA4BAdapter
    from lingbot_vla_v2_iwm.adapter import LingBotVLAV2Adapter

    for adapter, backbone, frontend in (
        (LingBotVLA4BAdapter(), "lingbot_vla", "vla4b_thor.py"),
        (LingBotVLAV2Adapter(), "lingbot_vla_v2", "vla2_thor.py"),
        (GR00TN17Adapter(), "groot_n17", "groot_n17_thor.py"),
    ):
        spec = adapter.spec()
        assert spec.notes["backbone"] == backbone, "the pass keys on notes['backbone']"
        baked = engine_baked_operating_point(backbone)
        assert baked is not None and baked.frontend.endswith(frontend)
        n = baked.steps["action"]
        assert spec.phase("action").nfe == n, (
            f"{backbone}: the family default ({spec.phase('action').nfe}) is not the engine's "
            f"baked count ({n}) — the mirror or the adapter is wrong")
        r = _evaluate(spec)
        assert r.applies and r.params.get("backend") == "engine", (backbone, r.reason)
        assert f"action={n} steps" in r.reason
        for other in (1, n - 1, n + 1):
            if other < 1:
                continue
            r = _evaluate(spec.with_nfe({"action": other}))
            assert not r.applies and "backend" not in r.params, (backbone, other, r.reason)
            assert f"engine {backbone} frontend bakes a {n}-step action schedule" in r.reason
            assert f"requests action={other}" in r.reason
        # VLA4 and V2 have native-processor/FP8 Runtime bridges. The same
        # schedule gate remains mandatory when a family becomes executable.
        if backbone in {"lingbot_vla", "lingbot_vla_v2", "groot_n17"}:
            assert backbone in ENGINE_BACKBONES
        else:
            assert backbone not in ENGINE_BACKBONES


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:                                # noqa: BLE001
                failures += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)

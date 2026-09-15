"""Checkpoint dimensions are preserved by the normalized pi05 engine bridge.

The old standalone engine returned seven dimensions. Runtime now accepts native
checkpoint dimensions 1..32, delegates decoding to the checkpoint processor,
and refuses unverified or larger output dimensions before loading the model.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "examples" / "pi05_vla"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import instinctflash.runtime.engine_backend as eb  # noqa: E402
from instinctflash.descriptors.deployment import DeploymentSpec  # noqa: E402
from instinctflash.passes.contract import DeviceProfile  # noqa: E402
from instinctflash.passes.generic.engine_offload import EngineOffloadApplicable  # noqa: E402
from instinctflash.planners.planner import Plan, PassResult, Tier  # noqa: E402
from instinctflash.runtime.engine_backend import (  # noqa: E402
    PI05_ENGINE_ACTION_DIM,
    PI05_ENGINE_MAX_ACTION_DIM,
    EngineBackend,
    declared_action_dim,
)
from instinctflash.runtime.execution import InProcessBackend, choose_backend  # noqa: E402


def _checkpoint(extra=None, path="/nonexistent", backbone="pi05"):
    return SimpleNamespace(
        path=path,
        execution=SimpleNamespace(
            backbone=backbone, model_id="test/pi05-geometry", extra=dict(extra or {}),
            nfe={"prefix": 1, "action": 10}),
    )


def _config_json(td: str, action_shape) -> str:
    doc = {"type": "pi05", "output_features": {"action": {"type": "ACTION",
                                                          "shape": action_shape}}}
    Path(td, "config.json").write_text(json.dumps(doc))
    return td


def _thor() -> DeploymentSpec:
    return DeploymentSpec(device=DeviceProfile(
        name="thor-test", capability=(11, 0), total_memory=64 << 30,
        features=frozenset({"cuda", "fp8"})))


# ── declared_action_dim: the one resolution rule ────────────────────────────────────────────


def test_declaration_extra_action_dim_wins():
    dim, src = declared_action_dim(_checkpoint(extra={"action_dim": 32}))
    assert dim == 32
    assert "execution.action_dim" in src


def test_config_json_output_features_is_the_fallback():
    with tempfile.TemporaryDirectory() as td:
        dim, src = declared_action_dim(_checkpoint(path=_config_json(td, [7])))
        assert dim == 7
        assert "output_features.action.shape" in src


def test_unresolvable_geometry_says_why():
    dim, src = declared_action_dim(_checkpoint())
    assert dim is None
    assert "no execution.action_dim" in src and "config.json" in src

    with tempfile.TemporaryDirectory() as td:
        # config.json present but with no readable action shape — still unresolved, still says so
        Path(td, "config.json").write_text(json.dumps({"type": "pi05"}))
        dim, src = declared_action_dim(_checkpoint(path=td))
        assert dim is None
        assert "output_features.action.shape" in src

        Path(td, "config.json").write_text("{broken")
        dim, src = declared_action_dim(_checkpoint(path=td))
        assert dim is None
        assert "unreadable" in src


# ── plan time: engine_offload declines mismatched or unverified geometry ────────────────────


def _pass_result(notes) -> PassResult:
    # The pi05 family's own spec (10 action steps, no CFG -- the engine's baked operating
    # point) with the notes under test, so these cases isolate the GEOMETRY rule. The pass also
    # gates the operating point now, and a notes-only stub has no schedule to verify, which it
    # declines fail-closed; that rule has its own file, tests/test_engine_operating_point_gate.py.
    import dataclasses

    from pi05_iwm.adapter import Pi05Adapter
    spec = dataclasses.replace(Pi05Adapter().spec(), notes=dict(notes))
    return EngineOffloadApplicable().evaluate(spec, _thor())


def test_matching_geometry_still_routes_to_the_engine():
    r = _pass_result({"backbone": "pi05", "action_dim": str(PI05_ENGINE_ACTION_DIM),
                      "action_dim_source": "test"})
    assert r.applies
    assert r.params.get("backend") == "engine", "the auto-placement signal must survive"


def test_mismatched_geometry_is_declined_with_the_structural_reason():
    r = _pass_result({"backbone": "pi05", "action_dim": "33", "action_dim_source": "test"})
    assert not r.applies
    assert "declares 33" in r.reason
    assert f"action_dim=1..{PI05_ENGINE_MAX_ACTION_DIM}" in r.reason
    assert "torch placement" in r.reason
    assert "backend" not in r.params, \
        "a geometry decline must not carry params['backend'] — that key is what 'auto' reads"


def test_unverified_geometry_is_declined_fail_closed():
    r = _pass_result({"backbone": "pi05",
                      "action_dim_unresolved": "no execution.action_dim and no config.json"})
    assert not r.applies
    assert "could not be verified" in r.reason
    assert "no execution.action_dim" in r.reason
    assert "backend" not in r.params


def test_non_pi05_specs_keep_todays_behaviour():
    # choose_backend never builds the engine for them (ENGINE_BACKBONES); the pass result
    # itself is unchanged so existing plans keep their shape.
    r = _pass_result({"family": "wam"})
    assert r.applies and r.params.get("backend") == "engine"


def test_the_pi05_adapter_puts_the_geometry_facts_on_the_spec():
    from pi05_iwm.adapter import Pi05Adapter

    adapter = Pi05Adapter()
    with tempfile.TemporaryDirectory() as td:
        spec = adapter.spec_for_checkpoint(_checkpoint(path=_config_json(td, [32])))
    assert spec.notes["backbone"] == "pi05"
    assert spec.notes["action_dim"] == "32"
    assert "output_features.action.shape" in spec.notes["action_dim_source"]

    spec = adapter.spec_for_checkpoint(_checkpoint(extra={"action_dim": 7}))
    assert spec.notes["action_dim"] == "7"

    spec = adapter.spec_for_checkpoint(_checkpoint())
    assert "action_dim" not in spec.notes
    assert "config.json" in spec.notes["action_dim_unresolved"]


def test_the_planner_and_runtime_gates_cannot_drift_from_the_engine_source():
    source = (ROOT / "serving/flash_rt/frontends/torch/pi05_thor.py").read_text()
    m = re.search(r"^\s+MODEL_ACTION_DIM\s*=\s*(\d+)", source, re.M)
    assert m and int(m.group(1)) == PI05_ENGINE_MAX_ACTION_DIM


def test_native_base_geometry_is_eligible_without_silent_truncation():
    r = _pass_result({"backbone": "pi05", "action_dim": "32", "action_dim_source": "test"})
    assert r.applies and r.params["backend"] == "engine"


# ── build time: EngineBackend refuses before anything expensive ─────────────────────────────


def test_engine_backend_refuses_a_mismatched_checkpoint_before_touching_torch():
    try:
        EngineBackend(adapter=None, checkpoint=_checkpoint(extra={"action_dim": 33}), plan=None)
    except RuntimeError as e:
        msg = str(e)
        assert f"action_dim=1..{PI05_ENGINE_MAX_ACTION_DIM}" in msg and "declares 33" in msg, msg
        assert "truncat" in msg
    else:
        raise AssertionError("a 33-dim checkpoint reached the engine frontend")


def test_engine_backend_refuses_unverifiable_geometry():
    try:
        EngineBackend(adapter=None, checkpoint=_checkpoint(), plan=None)
    except RuntimeError as e:
        assert "cannot be verified" in str(e), str(e)
    else:
        raise AssertionError("the engine was built blind to the checkpoint's action geometry")


# ── placement: 'auto' falls back to torch with the decline reason on the line ───────────────


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


def _geometry_declined_plan() -> Plan:
    return Plan("pi05-test", [
        PassResult(
            name="engine_offload", applies=False, tier=Tier.NUMERIC,
            reason=(f"engine pi05 frontend supports action_dim={PI05_ENGINE_ACTION_DIM} as "
                    f"built, exactly (...); this checkpoint declares 32 (source: test) — "
                    f"falling back to the torch placement, which serves the declared geometry. "
                    f"Never silently truncate."),
            params={}),
    ])


def test_auto_falls_back_to_torch_and_says_why():
    plan = _geometry_declined_plan()
    with _engine_looks_available():
        backend, why = choose_backend(
            "auto", _Adapter(), _checkpoint(extra={"action_dim": 32}), plan)
    assert isinstance(backend, InProcessBackend), why
    assert "auto -> in_process" in why
    assert not plan.results[0].applies
    assert "declares 32" in plan.results[0].reason


def test_explicit_engine_placement_is_refused_not_truncated():
    plan = _geometry_declined_plan()
    with _engine_looks_available():
        try:
            choose_backend("engine", _Adapter(), _checkpoint(extra={"action_dim": 32}), plan, precision="fp8")
        except RuntimeError as e:
            assert "declares 32" in str(e), str(e)
        else:
            raise AssertionError("placement='engine' served a checkpoint the frontend truncates")


# ── the second parked finding: compile_model=true is dead on sm_110a ────────────────────────


def test_compile_model_is_neutralized_on_sm110a_only():
    from pi05_iwm.adapter import _neutralize_compile_model_on_sm110a

    cfg = SimpleNamespace(compile_model=True)
    assert _neutralize_compile_model_on_sm110a(cfg, (11, 0)) is True
    assert cfg.compile_model is False

    cfg = SimpleNamespace(compile_model=True)
    assert _neutralize_compile_model_on_sm110a(cfg, (9, 0)) is False
    assert cfg.compile_model is True, "off sm_110a the publisher's key must stand"

    cfg = SimpleNamespace(compile_model=False)
    assert _neutralize_compile_model_on_sm110a(cfg, (11, 0)) is False
    assert cfg.compile_model is False

    cfg = SimpleNamespace(compile_model=True)
    assert _neutralize_compile_model_on_sm110a(cfg, None) is False, \
        "no probed capability -> not sm_110a -> leave the key alone"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:                                # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)

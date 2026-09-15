"""Regression: a caller's exclude of engine_offload must never resurrect the engine.

The reported scenario (confirmed end to end on the shipped Thor + pi05 + flash_rt deployment):

  1. The optimizer compiles with its default BITEXACT ceiling, which demotes engine_offload
     (NUMERIC by construction) to applies=False while KEEPING params['backend']=='engine'.
  2. The caller explicitly opts out: plan.without('engine_offload') (the exclude_passes knob).
     Plan.without also preserves params, so explain() can show what was dropped.
  3. choose_backend('auto') keyed only on params['backend']=='engine' — it could not tell a
     caller exclusion from the ceiling demotion it exists to overrule — and built the engine.
  4. _mark_plan_engine_executed then rewrote the caller's exclusion back to applies=True, so
     even the plan stopped saying it had happened.

The fix is the `excluded` flag on PassResult: set only by Plan.without, respected by both
choose_backend (auto declines; explicit placement='engine' refuses loudly) and
_mark_plan_engine_executed (never rewrites an excluded entry). Ceiling demotion carries
excluded=False. Precision policy now additionally requires explicit FP8 permission and an eligible plan.

No GPU, no torch: engine availability is stubbed at the module seam choose_backend imports from.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import instinctflash.runtime.engine_backend as eb  # noqa: E402
from instinctflash.planners.planner import Plan, PassResult, Tier  # noqa: E402
from instinctflash.runtime.execution import (  # noqa: E402
    InProcessBackend,
    _mark_plan_engine_executed,
    choose_backend,
)


class _FakeEngine:
    """Stands in for EngineBackend; counts constructions so a resurrection is visible."""

    built = 0

    def __init__(self, adapter, checkpoint, plan, **kw):
        type(self).built += 1


@contextmanager
def _engine_looks_available():
    orig = eb.engine_available, eb.EngineBackend
    eb.engine_available = lambda: (True, "test stub: pretend SM110 + engine kernels")
    eb.EngineBackend = _FakeEngine
    try:
        yield
    finally:
        eb.engine_available, eb.EngineBackend = orig


def _pi05_checkpoint():
    return SimpleNamespace(execution=SimpleNamespace(backbone="pi05"))


class _Adapter:
    """Hostable in-process, so 'auto' has a legitimate non-engine landing spot."""

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
        raise AssertionError("test never builds the impl")


def _ceiling_demoted_plan() -> Plan:
    """engine_offload exactly as the optimizer's default BITEXACT ceiling leaves it:
    applies=False, but params['backend']=='engine' preserved — the auto path's signal."""
    return Plan("pi05-test", [
        PassResult(
            name="engine_offload", applies=False, tier=Tier.NUMERIC,
            reason="legal but tier NUMERIC exceeds ceiling BITEXACT: SM110 device, fused fp8 "
                   "pipeline is the measured winner",
            params={"backend": "engine",
                    "tier_note": "NUMERIC (uncertified — cosine-level only)"}),
    ])


def test_default_native_never_bypasses_the_ceiling():
    plan = _ceiling_demoted_plan()
    before = _FakeEngine.built
    with _engine_looks_available():
        backend, why = choose_backend("auto", _Adapter(), _pi05_checkpoint(), plan)
    assert isinstance(backend, InProcessBackend), why
    assert _FakeEngine.built == before
    assert not plan.results[0].applies


def test_explicit_fp8_reaches_engine_only_with_an_eligible_plan():
    from dataclasses import replace
    plan = _ceiling_demoted_plan()
    plan.results[0] = replace(plan.results[0], applies=True, reason="eligible NUMERIC engine")
    with _engine_looks_available():
        backend, why = choose_backend("auto", _Adapter(), _pi05_checkpoint(), plan, precision="fp8")
    assert isinstance(backend, _FakeEngine), why
    assert plan.results[0].applies


def test_caller_exclusion_is_honored_on_auto():
    # THE REPORTED SCENARIO. exclude_passes=['engine_offload'] == plan.without(...): 'auto'
    # must not build the engine, and the exclusion must survive in the plan.
    plan = _ceiling_demoted_plan().without("engine_offload")
    eng = next(r for r in plan.results if r.name == "engine_offload")
    assert eng.excluded and not eng.applies
    assert eng.params.get("backend") == "engine", \
        "without() keeps params by design — that is what made the bypass possible"

    before = _FakeEngine.built
    with _engine_looks_available():
        backend, why = choose_backend("auto", _Adapter(), _pi05_checkpoint(), plan)
    assert _FakeEngine.built == before, f"excluded engine was built anyway: {why}"
    assert isinstance(backend, InProcessBackend), why
    eng = next(r for r in plan.results if r.name == "engine_offload")
    assert not eng.applies, "the exclusion was rewritten back to APPLY"
    assert eng.excluded


def test_caller_exclusion_refuses_explicit_engine_placement():
    # placement='engine' plus an excluded engine_offload are contradictory instructions;
    # the same override occurred here too. Refuse loudly instead of picking a side silently.
    plan = _ceiling_demoted_plan().without("engine_offload")
    before = _FakeEngine.built
    with _engine_looks_available():
        try:
            choose_backend("engine", _Adapter(), _pi05_checkpoint(), plan, precision="fp8")
        except RuntimeError as e:
            assert "exclud" in str(e).lower()
        else:
            raise AssertionError("placement='engine' overrode the caller's exclusion silently")
    assert _FakeEngine.built == before


def test_mark_plan_engine_executed_never_rewrites_an_exclusion():
    # Defence in depth: even called directly, the rewrite must not resurrect an exclusion.
    plan = _ceiling_demoted_plan().without("engine_offload")
    eng = next(r for r in plan.results if r.name == "engine_offload")
    _mark_plan_engine_executed(plan, eng)
    eng = next(r for r in plan.results if r.name == "engine_offload")
    assert not eng.applies and eng.excluded


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))

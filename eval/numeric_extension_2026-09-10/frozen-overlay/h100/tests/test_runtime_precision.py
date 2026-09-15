"""Arithmetic permission must not leak through placement or a numeric capture ceiling."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instinctflash.runtime.facade import Runtime
from instinctflash.runtime.execution import choose_backend, InProcessBackend
from instinctflash.planners.planner import Plan, PassResult, Tier
from instinctflash.runtime.engine_backend import ENGINE_BACKBONES


class Adapter:
    def build_in_process(self, checkpoint, plan, **kw):
        raise AssertionError("No model construction in dispatch tests")


def plan(applies=True):
    return Plan("test", [PassResult("engine_offload", applies, Tier.NUMERIC,
                                  reason="eligible" if applies else "ceiling refused",
                                  params={"backend": "engine"})])


def checkpoint(backbone="pi05"):
    return SimpleNamespace(execution=SimpleNamespace(backbone=backbone))


def test_native_numeric_does_not_probe_or_build_fp8():
    import instinctflash.runtime.facade as facade
    p = plan()
    with patch.object(facade, "_load_package", return_value=checkpoint()), \
         patch.object(facade, "_compile_declaration", return_value=(Adapter(), p, None)) as compile_, \
         patch("instinctflash.runtime.engine_backend.engine_available", side_effect=AssertionError("FP8 probed")):
        rt = Runtime.from_pretrained("test", tier_ceiling="numeric")
    assert compile_.call_args.kwargs["tier_ceiling"] == "numeric"
    assert rt._precision == "native" and isinstance(rt._backend, InProcessBackend)
    assert not p.results[0].applies and "not authorized" in p.results[0].reason


def test_fp8_is_explicit_and_compiles_numeric_without_changing_schedule():
    import instinctflash.runtime.facade as facade
    engine = object()
    p = plan()
    with patch.object(facade, "_load_package", return_value=checkpoint()), \
         patch.object(facade, "_compile_declaration", return_value=(Adapter(), p, None)) as compile_, \
         patch("instinctflash.runtime.engine_backend.engine_available", return_value=(True, "test")), \
         patch("instinctflash.runtime.engine_backend.EngineBackend", return_value=engine) as build:
        rt = Runtime.from_pretrained("test", precision="fp8", nfe={"action": 10})
    assert compile_.call_args.kwargs["tier_ceiling"] == "numeric"
    assert build.call_args.kwargs["nfe"] == {"action": 10}
    assert rt._backend is engine and rt._precision == "fp8"


def test_conflicting_precision_is_refused_before_loading_weights():
    import instinctflash.runtime.facade as facade
    with patch.object(facade, "_load_package", side_effect=AssertionError("weights touched")):
        for kwargs in ({"precision": "fp8", "tier_ceiling": "bitexact"},
                       {"placement": "engine"}, {"precision": "unknown"},
                       {"precision": "fp8", "placement": "worker"}):
            try:
                Runtime.from_pretrained("test", **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(kwargs)


def test_fp8_never_falls_back_or_overrides_a_decline():
    cases = [(checkpoint("unsupported"), plan(), "frontend"),
             (checkpoint(), plan(False), "planner declined"),
             (checkpoint(), plan().without("engine_offload"), "excluded"),
             (checkpoint(), plan(), "unavailable")]
    with patch("instinctflash.runtime.engine_backend.engine_available", return_value=(False, "unavailable")), \
         patch("instinctflash.runtime.engine_backend.EngineBackend", side_effect=AssertionError("engine built")):
        for ckpt, p, expected in cases:
            try:
                choose_backend("auto", Adapter(), ckpt, p, precision="fp8")
            except RuntimeError as error:
                assert expected in str(error), error
            else:
                raise AssertionError("FP8 silently fell back")


def test_fp8_still_refuses_unsupported_seed():
    with patch("instinctflash.runtime.engine_backend.engine_available", return_value=(True, "test")):
        try:
            choose_backend("auto", Adapter(), checkpoint(), plan(), precision="fp8", seed=1)
        except RuntimeError as error:
            assert "seed=" in str(error)
        else:
            raise AssertionError("FP8 ignored seed")


def test_public_precision_and_default_server_metadata_agree():
    from instinctflash.serving.ws_server import default_metadata
    for precision in ('native', 'fp8'):
        runtime = object.__new__(Runtime)
        runtime._precision = precision
        assert runtime.precision == precision
        assert default_metadata(runtime)['precision'] == precision
    assert 'precision' not in default_metadata(SimpleNamespace())


def test_each_family_propagates_fp8_build_failure_without_native_fallback():
    for backbone in ENGINE_BACKBONES:
        failure = RuntimeError("actual FP8 construction failed")
        p = plan()
        with patch("instinctflash.runtime.engine_backend.engine_available", return_value=(True, "Thor")), \
             patch("instinctflash.runtime.engine_backend.EngineBackend", side_effect=failure), \
             patch("instinctflash.runtime.execution.InProcessBackend", side_effect=AssertionError("native fallback")), \
             patch("instinctflash.runtime.execution.WorkerBackend", side_effect=AssertionError("worker fallback")), \
             patch("instinctflash.runtime.execution._mark_plan_engine_executed") as mark:
            try:
                choose_backend("auto", Adapter(), checkpoint(backbone), p, precision="fp8")
            except RuntimeError as caught:
                assert caught is failure, backbone
            else:
                raise AssertionError(f"{backbone} ignored FP8 construction failure")
        mark.assert_not_called()


def test_each_family_native_permission_blocks_engine_even_when_plan_eligible():
    for backbone in ENGINE_BACKBONES:
        p = plan()
        sentinel = object()
        with patch("instinctflash.runtime.engine_backend.engine_available", side_effect=AssertionError("FP8 probe")), \
             patch("instinctflash.runtime.execution.can_host_in_process", return_value=(True, "installed")), \
             patch("instinctflash.runtime.execution.InProcessBackend", return_value=sentinel):
            backend, _ = choose_backend("auto", Adapter(), checkpoint(backbone), p, precision="native")
        assert backend is sentinel, backbone
        assert not p.results[0].applies and "not authorized" in p.results[0].reason


def test_cli_preflight_forwards_precision_placement_and_device():
    from instinctflash.cli import _serve_preflight
    from instinctflash.cli_config import RuntimeConfig
    ckpt = SimpleNamespace(execution=SimpleNamespace(model_id="test", backbone="pi05", servable=True),
                           path="test", capabilities=lambda: set())
    with patch("instinctflash.runtime.facade.plan_declaration", return_value=(ckpt, Adapter(), plan(), None)) as preflight:
        result, rendered = _serve_preflight("test", RuntimeConfig(precision="fp8", device="cuda:1"))
    assert preflight.call_args.kwargs["precision"] == "fp8"
    assert preflight.call_args.kwargs["placement"] == "auto"
    assert preflight.call_args.kwargs["device"] == "cuda:1"
    assert result["precision"] == "fp8" and "fp8" in rendered


def test_preflight_rejects_fp8_without_verified_thor_device():
    import tempfile
    from test_cli_preflight import _write_declaration
    from instinctflash.runtime.facade import plan_declaration
    with tempfile.TemporaryDirectory() as td:
        _write_declaration(td)
        try:
            plan_declaration(td, precision="fp8", probe_device=False)
        except RuntimeError as error:
            assert "SM110" in str(error), error
        else:
            raise AssertionError("preflight accepted FP8 without a verified Thor device")


if __name__ == "__main__":
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))


def test_native_default_requests_bitexact_before_checkpoint_planning():
    from instinctflash.runtime.precision import resolve_precision
    assert resolve_precision("native", None) == "bitexact"
    assert resolve_precision("native", "numeric") == "numeric"
    assert resolve_precision("fp8", None) == "numeric"


def test_native_does_not_import_quantization_code():
    import builtins
    from instinctflash.runtime.precision import install_requested_fp8
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.endswith(("h100_fp8", "torch_fp8_linear", "fp8_pack")):
            raise AssertionError("native path imported quantization")
        return original(name, *args, **kwargs)
    with patch("builtins.__import__", side_effect=guarded):
        install_requested_fp8(object(), Plan("test", []), "pi05")


def test_step_override_is_reported_separately_from_exact_transforms():
    from instinctflash.serving.ws_server import default_metadata
    ckpt = SimpleNamespace(model_id="test", execution=SimpleNamespace(nfe={"action": 10}))
    rt = Runtime(ckpt, None, Plan("test", []), None, nfe={"action": 4})
    assert rt.execution_policy["category"] == "OPERATING-POINT"
    assert rt.execution_policy["transform_tier"] == "BITEXACT"
    assert rt.execution_policy["changed_schedule"] == {"action": {"checkpoint": 10, "selected": 4}}
    assert default_metadata(rt)["execution_policy"] == rt.execution_policy
    native = Runtime(ckpt, None, Plan("test", []), None)
    assert native.execution_policy["category"] == "BITEXACT"


def test_environment_transform_requires_declared_permission():
    from instinctflash.runtime.precision import require_transform_permission
    import pytest
    with pytest.raises(ValueError, match="tier_ceiling='behavioral'"):
        require_transform_permission(Plan("test", []), Tier.BEHAVIORAL, "step skipping")
    allowed = Plan("test", [], tier_ceiling=Tier.BEHAVIORAL)
    require_transform_permission(allowed.without(), Tier.BEHAVIORAL, "step skipping")
    with pytest.raises(ValueError):
        require_transform_permission(allowed.bitexact_subset(), Tier.BEHAVIORAL, "step skipping")



def test_h100_fp8_reporting_preserves_installed_native_passes():
    from instinctflash.runtime.execution import _mark_plan_engine_executed
    from instinctflash.planners.planner import PassResult, Tier
    from types import SimpleNamespace
    for executor, expected in [('h100_torch_fp8', True), ('thor_fused', False)]:
        engine = PassResult('engine_offload', True, Tier.NUMERIC, 'selected',
                            params={'executor': executor})
        native = PassResult('conditioning_prefill', True, Tier.BITEXACT, 'installed')
        excluded = PassResult('graph_capture', False, Tier.BITEXACT, 'not selected')
        plan = SimpleNamespace(results=[engine, native, excluded])
        _mark_plan_engine_executed(plan, engine)
        assert plan.results[0].applies
        assert plan.results[1].applies is expected
        assert not plan.results[2].applies
        if expected:
            assert plan.results[1] is native

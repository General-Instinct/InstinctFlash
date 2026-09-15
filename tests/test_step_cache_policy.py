"""CPU-only policy/dispatch tests: no model construction or real CUDA device probes."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instinctflash.adapters.base import AdapterSpec, PhaseSpec
from instinctflash.descriptors.checkpoint import ExecutionDeclaration
from instinctflash.descriptors.package import Checkpoint
from instinctflash.planners.planner import Plan, PassResult, Tier
from instinctflash.runtime.execution import InProcessBackend, choose_backend, _mark_plan_engine_executed
from instinctflash.runtime.facade import Runtime, _compile_declaration, plan_declaration
from instinctflash.runtime.step_cache_policy import (
    ResolvedStepCache, annotate_step_cache_plan, resolve_step_cache,
)

raises = unittest.TestCase().assertRaisesRegex


def checkpoint(backbone="dreamzero", *, dynamic=False):
    return Checkpoint("unused", ExecutionDeclaration(
        model_id="test", backbone=backbone, servable=True,
        nfe={"video_action": 16}, extra={"dynamic_cache_schedule": dynamic}))


class Adapter:
    def __init__(self, backbone="dreamzero"):
        self.backbone = backbone
        self.builds = []

    def spec(self):
        return AdapterSpec("test", 0, (), (PhaseSpec("video_action", 16),), {},
                           notes={"backbone": self.backbone})

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None, step_cache=None):
        self.builds.append(step_cache)
        return SimpleNamespace(predict=lambda obs: obs)


@contextmanager
def environment(**values):
    with patch.dict(os.environ, values):
        for key in ("DYNAMIC_CACHE_SCHEDULE", "NUM_DIT_STEPS"):
            if key not in values:
                os.environ.pop(key, None)
        yield


def test_explicit_selection_overrides_environment_without_mutation():
    with environment(DYNAMIC_CACHE_SCHEDULE="true", NUM_DIT_STEPS="5"):
        before = dict(os.environ)
        selected = resolve_step_cache(checkpoint(), step_cache="checkpoint")
        assert selected == ResolvedStepCache(False, 8, None, "runtime.step_cache=checkpoint")
        dynamic = resolve_step_cache(checkpoint(), step_cache="dynamic", tier_ceiling="behavioral")
        assert dynamic.dynamic and dynamic.fixed_steps == 8
        assert dynamic.profile == "dreamzero_velocity_v1"
        assert os.environ == before
    with raises(FrozenInstanceError, "cannot assign"):
        selected.dynamic = True
    public = selected.to_dict()
    public["dynamic"] = True
    assert not selected.dynamic


def test_resolved_values_cannot_claim_unsupported_profiles_or_masks():
    for dynamic, fixed_steps, profile in [(1, 8, "dreamzero_velocity_v1"),
                                          (True, 4, "dreamzero_velocity_v1"),
                                          (False, True, None),
                                          (False, 8.0, None),
                                          (True, 8, "invented"),
                                          (True, 8, None),
                                          (False, 8, "dreamzero_velocity_v1")]:
        with raises(ValueError, "Resolved"):
            ResolvedStepCache(dynamic, fixed_steps, profile, "test")


def test_legacy_env_false_and_mask_are_resolved_once():
    with environment(DYNAMIC_CACHE_SCHEDULE="false", NUM_DIT_STEPS="6"):
        selected = resolve_step_cache(checkpoint(dynamic=True), tier_ceiling="behavioral")
        assert not selected.dynamic and selected.fixed_steps == 6
        assert selected.source == "environment:DYNAMIC_CACHE_SCHEDULE,NUM_DIT_STEPS"
    with environment(DYNAMIC_CACHE_SCHEDULE="true", NUM_DIT_STEPS="5"):
        selected = resolve_step_cache(checkpoint(), tier_ceiling="behavioral")
        p = Plan("test", [], tier_ceiling=Tier.BEHAVIORAL)
        annotate_step_cache_plan(p, selected)
        schedule = p.applied[0].params["execution_schedule"]
        assert schedule["computed_steps"] is None
        assert schedule["fixed_mask_steps"] == 5


def test_fp8_legacy_fixed_masks_refused_by_public_metadata_and_execution_gates():
    from instinctflash.runtime import facade
    from instinctflash.passes.contract import DeviceProfile

    adapter, ckpt = Adapter(), checkpoint()
    device = DeviceProfile("mock Thor", (11, 0), 128 * 1024**3,
                           frozenset({"cuda", "fp8"}))
    for steps in (5, 6, 7):
        for dynamic in ("false", "true"):
            with environment(NUM_DIT_STEPS=str(steps), DYNAMIC_CACHE_SCHEDULE=dynamic), \
                 patch.object(facade, "_load_package", return_value=ckpt), \
                 patch.object(facade, "load_declaration_ref", return_value=(ckpt.execution, {}, "unused")), \
                 patch("instinctflash.runtime.loader.load", return_value=adapter), \
                 patch.object(DeviceProfile, "probe", return_value=device), \
                 patch("instinctflash.runtime.engine_backend.EngineBackend") as backend:
                with raises(ValueError, "fixed 8-of-16"):
                    plan_declaration(".", precision="fp8", tier_ceiling="behavioral")
                with raises(ValueError, "fixed 8-of-16"):
                    Runtime.from_pretrained("unused", precision="fp8", tier_ceiling="behavioral")
                backend.assert_not_called()
    assert not adapter.builds


def test_fp8_tensorrt_environment_refused_by_public_metadata_and_execution_gates():
    from instinctflash.runtime import facade
    from instinctflash.passes.contract import DeviceProfile

    adapter, ckpt = Adapter(), checkpoint()
    device = DeviceProfile("mock Thor", (11, 0), 128 * 1024**3,
                           frozenset({"cuda", "fp8"}))
    for values in ({"ENABLE_TENSORRT": "true"},
                   {"ENABLE_TENSORRT": "false", "LOAD_TRT_ENGINE": "engine.plan"},
                   {"ENABLE_TENSORRT": "false", "LOAD_TRT_ENGINE": ""}):
        with environment(**values), \
             patch.object(facade, "_load_package", return_value=ckpt), \
             patch.object(facade, "load_declaration_ref", return_value=(ckpt.execution, {}, "unused")), \
             patch("instinctflash.runtime.loader.load", return_value=adapter), \
             patch.object(DeviceProfile, "probe", return_value=device), \
             patch("instinctflash.runtime.engine_backend.EngineBackend") as backend:
            with raises(ValueError, "native PyTorch path"):
                plan_declaration(".", precision="fp8")
            with raises(ValueError, "native PyTorch path"):
                Runtime.from_pretrained("unused", precision="fp8")
            backend.assert_not_called()
    assert not adapter.builds


def test_checkpoint_dynamic_still_requires_permission():
    with environment(DYNAMIC_CACHE_SCHEDULE="false"):
        with raises(ValueError, "tier_ceiling='behavioral'"):
            resolve_step_cache(checkpoint(dynamic=True), step_cache="checkpoint")
        selected = resolve_step_cache(checkpoint(dynamic=True), step_cache="checkpoint",
                                      tier_ceiling="behavioral")
        assert selected.dynamic


def test_invalid_options_refuse_instead_of_guessing():
    with environment():
        for choice in (False, True, "off", "enabled", 1):
            with raises(ValueError, "step_cache must"):
                resolve_step_cache(checkpoint(), step_cache=choice)
        with raises(ValueError, "not supported"):
            resolve_step_cache(checkpoint("cosmos3_policy"), step_cache="dynamic",
                               tier_ceiling="behavioral")
        assert resolve_step_cache(checkpoint("cosmos3_policy"), step_cache="checkpoint") is None
    for values in ({"DYNAMIC_CACHE_SCHEDULE": "maybe"},
                   {"NUM_DIT_STEPS": "4"}, {"NUM_DIT_STEPS": "five"}):
        with environment(**values), raises(ValueError, "boolean flag|NUM_DIT_STEPS"):
            resolve_step_cache(checkpoint(), tier_ceiling="behavioral")


def test_public_refusals_precede_device_probe_and_backend_selection():
    from instinctflash.runtime import facade
    cases = [("dreamzero", {"step_cache": "dynamic"}),
             ("dreamzero", {"step_cache": "dynamic", "precision": "fp8"}),
             ("cosmos3_policy", {"step_cache": "dynamic", "tier_ceiling": "behavioral"}),
             ("dreamzero", {"step_cache": "dynamic", "tier_ceiling": "behavioral",
                            "placement": "worker"})]
    with environment():
        for backbone, kwargs in cases:
            with patch.object(facade, "_load_package", return_value=checkpoint(backbone)), \
                 patch("instinctflash.runtime.loader.load", return_value=Adapter(backbone)), \
                 patch("instinctflash.passes.contract.DeviceProfile.probe") as probe, \
                 patch.object(facade, "choose_backend") as choose:
                with raises(ValueError, "behavioral|not supported"):
                    Runtime.from_pretrained("unused", **kwargs)
                probe.assert_not_called()
                choose.assert_not_called()


def test_runtime_freezes_default_false_before_lazy_load():
    from instinctflash.runtime import facade
    adapter, ckpt = Adapter(), checkpoint()
    with environment(), \
         patch.object(facade, "_load_package", return_value=ckpt), \
         patch("instinctflash.runtime.loader.load", return_value=adapter), \
         patch("instinctflash.passes.contract.DeviceProfile.probe", return_value=None):
        runtime = Runtime.from_pretrained("unused")
        assert not adapter.builds
        assert runtime.execution_policy["step_cache"]["dynamic"] is False
        os.environ["DYNAMIC_CACHE_SCHEDULE"] = "true"
        os.environ["NUM_DIT_STEPS"] = "5"
        ckpt.execution.extra["dynamic_cache_schedule"] = True
        assert runtime.predict({"test": 1}) == {"test": 1}
        assert adapter.builds == [runtime._step_cache]
        assert not adapter.builds[0].dynamic and adapter.builds[0].fixed_steps == 8


def test_preflight_and_runtime_agree_and_preserve_requested_nfe():
    from instinctflash.runtime import facade
    adapter, ckpt = Adapter(), checkpoint()
    with environment(), \
         patch.object(facade, "_load_package", return_value=ckpt), \
         patch.object(facade, "load_declaration_ref", return_value=(ckpt.execution, {}, "unused")), \
         patch("instinctflash.runtime.loader.load", return_value=adapter), \
         patch("instinctflash.passes.contract.DeviceProfile.probe", return_value=None):
        runtime = Runtime.from_pretrained("unused", step_cache="dynamic", tier_ceiling="behavioral")
        _, _, planned, _ = plan_declaration("unused", step_cache="dynamic",
                                            tier_ceiling="behavioral", probe_device=False)
    assert not adapter.builds
    assert planned.resolved_step_cache == runtime._step_cache
    policy = runtime.execution_policy
    assert policy["category"] == "OPERATING-POINT"
    assert policy["transform_tier"] == "BEHAVIORAL"
    assert policy["nfe"] == {"video_action": 16}
    assert policy["schedule_options"]["dreamzero_schedule"]["computed_steps"] is None
    assert policy["schedule_options"]["dreamzero_schedule"]["fixed_mask_steps"] == 8
    from instinctflash.runtime.step_cache import StepCacheConfig
    assert policy["schedule_options"]["dreamzero_schedule"]["config"] == StepCacheConfig().to_dict()


def test_policy_and_preflight_metadata_are_detached_snapshots():
    from instinctflash.cli import _serve_preflight
    from instinctflash.cli_config import RuntimeConfig
    from instinctflash.runtime.step_cache import StepCacheConfig

    ckpt, plan = checkpoint(), Plan("test", [], tier_ceiling=Tier.BEHAVIORAL)
    selected = ResolvedStepCache(True, 8, "dreamzero_velocity_v1", "test")
    annotate_step_cache_plan(plan, selected)
    runtime = Runtime(ckpt, None, plan, None, step_cache=selected)
    public = runtime.execution_policy
    public["schedule_options"]["dreamzero_schedule"]["config"]["thresholds"][0] = 0.1
    public["schedule_options"]["dreamzero_schedule"]["config"]["skip_counts"].append(100)
    expected = StepCacheConfig().to_dict()
    assert runtime.execution_policy["schedule_options"]["dreamzero_schedule"]["config"] == expected
    with patch("instinctflash.runtime.facade.plan_declaration", return_value=(ckpt, Adapter(), plan, None)):
        reported, _ = _serve_preflight("unused", RuntimeConfig(step_cache="dynamic"))
        reported["schedule_options"]["dreamzero_schedule"]["config"]["thresholds"].clear()
        again, _ = _serve_preflight("unused", RuntimeConfig(step_cache="dynamic"))
    assert again["schedule_options"]["dreamzero_schedule"]["config"] == expected
    assert plan.applied[0].params["execution_schedule"]["config"] == expected


def test_selection_survives_other_exclusions_and_refuses_hidden_schedule():
    with environment(), patch("instinctflash.runtime.loader.load", return_value=Adapter()):
        _, plan, _ = _compile_declaration(checkpoint(), step_cache="dynamic",
                                          tier_ceiling="behavioral", probe_device=False,
                                          exclude_passes=("engine_offload",))
        assert plan.resolved_step_cache.dynamic
        assert any(r.name == "dreamzero_schedule" and r.applies for r in plan.results)
        with patch("instinctflash.passes.contract.DeviceProfile.probe") as probe:
            with raises(ValueError, "excluded dreamzero_schedule"):
                _compile_declaration(checkpoint(), step_cache="dynamic", tier_ceiling="behavioral",
                                     exclude_passes=("dreamzero_schedule",))
            probe.assert_not_called()
        _, baseline, _ = _compile_declaration(checkpoint(), probe_device=False,
                                               exclude_passes=("dreamzero_schedule",))
        annotate_step_cache_plan(baseline, baseline.resolved_step_cache)
        assert not next(r for r in baseline.results if r.name == "dreamzero_schedule").applies


def test_auto_worker_refuses_changed_schedule_before_worker_creation():
    selected = ResolvedStepCache(True, 8, "dreamzero_velocity_v1", "test")
    with patch("instinctflash.runtime.execution.can_host_in_process", return_value=(False, "missing")), \
         patch("instinctflash.runtime.execution.WorkerBackend") as worker:
        with raises(ValueError, "worker placement"):
            choose_backend("auto", Adapter(), checkpoint(), Plan("test", []), step_cache=selected)
        worker.assert_not_called()


def test_unrelated_adapter_retains_strict_build_signature():
    class LegacyAdapter:
        def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
            return SimpleNamespace(predict=lambda obs: obs)
    backend, _ = choose_backend("in_process", LegacyAdapter(), checkpoint("pi05"),
                                 Plan("test", []), step_cache=None)
    assert backend.predict({"legacy": True}) == {"legacy": True}


def engine_plan(*, h100=False):
    params = {"backend": "engine"}
    if h100:
        params["executor"] = "h100_torch_fp8"
    return Plan("test", [PassResult("engine_offload", True, Tier.NUMERIC, "eligible", params)],
                tier_ceiling=Tier.BEHAVIORAL)


def test_fp8_selection_reaches_backend_and_retains_behavioral_category():
    selected = ResolvedStepCache(True, 8, "dreamzero_velocity_v1", "test")
    plan = engine_plan()
    annotate_step_cache_plan(plan, selected)
    with patch("instinctflash.runtime.engine_backend.engine_available", return_value=(True, "fake")), \
         patch("instinctflash.runtime.engine_backend.EngineBackend", return_value=object()) as build:
        _, why = choose_backend("auto", Adapter(), checkpoint(), plan,
                                 precision="fp8", step_cache=selected)
    assert build.call_args.kwargs["step_cache"] is selected
    assert plan.tier() == Tier.BEHAVIORAL and "BEHAVIORAL" in why
    assert next(r for r in plan.results if r.name == "dreamzero_schedule").applies


def test_engine_rewrite_preserves_schedule_but_demotes_unexecuted_passes():
    plan = engine_plan()
    selected = ResolvedStepCache(True, 8, "dreamzero_velocity_v1", "test")
    annotate_step_cache_plan(plan, selected)
    plan.results.append(PassResult("torch_only", True, Tier.BITEXACT, "torch"))
    _mark_plan_engine_executed(plan, plan.results[0])
    assert plan.tier() == Tier.BEHAVIORAL
    assert not next(r for r in plan.results if r.name == "torch_only").applies


def test_thor_and_h100_preserve_dreamzero_and_cosmos_build_contracts():
    import torch
    from instinctflash.runtime.engine_backend import EngineBackend
    from instinctflash.runtime.h100_fp8 import build_h100_loop

    assert not torch.cuda.is_initialized()
    selected = ResolvedStepCache(True, 8, "dreamzero_velocity_v1", "test")
    for family in ("dreamzero", "cosmos3_policy"):
        ckpt, plan, seen = checkpoint(family), engine_plan(), []
        loop = SimpleNamespace(
            backend_stats={"fp8_recipe": {"precision": "fp8", "projections": [{"path": "q"}]}},
            declaration=lambda: {"precision": "fp8"}, close=Mock())

        class DreamZero:
            def build_fp8(self, checkpoint, *, device=None, nfe=None, plan=None, step_cache=None):
                seen.append((plan, step_cache))
                return loop

        class Cosmos:
            def build_fp8(self, checkpoint, *, device=None, nfe=None):
                seen.append("unchanged")
                return loop

        adapter = DreamZero() if family == "dreamzero" else Cosmos()
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_capability", return_value=(11, 0)), \
             patch("instinctflash.runtime.engine_backend.requested_operating_point",
                   return_value=({}, {}, "test")), \
             patch("instinctflash.runtime.engine_backend.engine_baked_operating_point", return_value={}), \
             patch("instinctflash.runtime.engine_backend.engine_operating_point_problem", return_value=None):
            EngineBackend(adapter, ckpt, plan, step_cache=selected if family == "dreamzero" else None)
        build_h100_loop(adapter, ckpt, plan,
                        step_cache=selected if family == "dreamzero" else None)
        assert seen == ([(plan, selected)] * 2 if family == "dreamzero" else ["unchanged"] * 2)
    assert not torch.cuda.is_initialized()


def test_h100_engine_dispatch_forwards_the_same_selection():
    import torch
    from instinctflash.runtime.engine_backend import EngineBackend
    selected = ResolvedStepCache(True, 8, "dreamzero_velocity_v1", "test")
    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), \
         patch("instinctflash.runtime.h100_fp8.build_h100_loop", return_value=object()) as build:
        EngineBackend(Adapter(), checkpoint(), engine_plan(h100=True), step_cache=selected)
    assert build.call_args.kwargs["step_cache"] is selected
    assert not torch.cuda.is_initialized()


def test_cli_parses_step_cache_and_forwards_it_to_preflight_and_load():
    import instinctflash.cli as cli
    from instinctflash.cli_config import ConfigError, RuntimeConfig, parse_config
    assert parse_config(cli.ServeConfig, []).runtime.step_cache is None
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "config.yaml"
        cfg_path.write_text("runtime:\n  step_cache: dynamic\n")
        cfg = parse_config(cli.ServeConfig, [f"--config_path={cfg_path}",
                                             "--runtime.step_cache=checkpoint"])
        assert cfg.runtime.step_cache == "checkpoint"
    with raises(ConfigError, "step_cache"):
        parse_config(cli.ServeConfig, ["--runtime.step_cache=unknown"])
    plan = Plan("test", [], tier_ceiling=Tier.BEHAVIORAL)
    selected = ResolvedStepCache(True, 8, "dreamzero_velocity_v1", "test")
    annotate_step_cache_plan(plan, selected)
    with patch("instinctflash.runtime.facade.plan_declaration",
               return_value=(checkpoint(), Adapter(), plan, None)) as preflight:
        result, _ = cli._serve_preflight("unused", RuntimeConfig(step_cache="dynamic"))
        assert preflight.call_args.kwargs["step_cache"] == "dynamic"
        assert result["step_cache"] == selected.to_dict()
        assert result["schedule_options"]["dreamzero_schedule"]["computed_steps"] is None
    from instinctflash.cli_config import CommandReport
    with patch.object(cli, "_serve_autoscaffold", return_value=(None, "", [])), \
         patch.object(cli, "_serve_package_gate", return_value=None), \
         patch.object(cli, "_serve_preflight", return_value=({}, "preflight")), \
         patch.object(Runtime, "from_pretrained", return_value=Mock()) as load, \
         patch.object(cli, "_serve_smoke", return_value=CommandReport({}, "ok")):
        assert cli.cmd_serve(["unused", "--serve.smoke=true", "--runtime.step_cache=dynamic",
                              "--runtime.tier_ceiling=behavioral"]) == 0
        assert load.call_args.kwargs["step_cache"] == "dynamic"


if __name__ == "__main__":
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))

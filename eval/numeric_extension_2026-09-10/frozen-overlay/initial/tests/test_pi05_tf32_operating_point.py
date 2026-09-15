"""The TF32 operating point is declaration-driven, truthfully NUMERIC, and lifecycle-safe."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "pi05_vla"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PLUGIN))

from instinctflash.descriptors.checkpoint import load_declaration
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.descriptors.package import Checkpoint, validate_package
from instinctflash.passes.contract import DeviceProfile
from instinctflash.passes.generic.graph_capture import GraphCaptureApplicable
from instinctflash.planners.planner import Optimizer, PassResult, Tier
from instinctflash.runtime.facade import _adapter_spec_for_checkpoint

import pi05_iwm.adapter as adapter_module
from pi05_iwm.adapter import (
    Pi05Adapter,
    _require_all_floating_parameters_fp32,
    _validate_tf32_plan,
)
from pi05_iwm.passes import CAPABILITY, PASS_NAME, Pi05TF32Numeric
from pi05_iwm.precision import Pi05PrecisionLease, _state_for_tests

TF32_PACKAGE = ROOT / "examples" / "checkpoint" / "pi05-base-tf32-h100"
V044_TF32_PACKAGE = ROOT / "examples" / "checkpoint" / "pi05-libero-v044-tf32-h100"
FP32_PACKAGE = ROOT / "examples" / "pi05_vla"
H100 = DeviceProfile(
    name="NVIDIA H100 80GB HBM3", capability=(9, 0), total_memory=80_000_000_000,
    features=frozenset({"cuda", "cuda_graphs", "cublas", "fp8", "wgmma", "tma"}),
)


def _register_pi05_plugin():
    """What `pip install -e examples/pi05_vla` does, for this in-tree test environment."""
    from instinctflash.passes.registry import register_passes
    from instinctflash.runtime import loader

    if "pi05" not in loader._REGISTRY:
        loader.register("pi05", Pi05Adapter)
    import pi05_iwm.passes as pi05_passes
    register_passes("pi05", pi05_passes.default_passes)


_register_pi05_plugin()


def compile_package(path, tier_ceiling=None, profile=H100):
    """THROUGH THE REAL PATH. The PR's original tests built the plan with a hand-picked pass
    list and `_adapter_spec_for_checkpoint` directly, bypassing `_compile_declaration` -- which
    is exactly why they stayed green while the declared-id plan-header override broke both
    pointer packages at serve/build time. Everything here plans the way `serve` does."""
    from instinctflash.runtime.facade import _compile_declaration

    checkpoint = Checkpoint(str(path), load_declaration(path))
    old_probe = DeviceProfile.probe
    DeviceProfile.probe = staticmethod(lambda device=None: profile)
    try:
        _adapter, plan, _probed = _compile_declaration(checkpoint, tier_ceiling=tier_ceiling)
    finally:
        DeviceProfile.probe = old_probe
    return checkpoint, plan


def test_pointer_package_is_valid_and_plan_is_numeric_by_declaration():
    report = validate_package(TF32_PACKAGE)
    assert report.ok, report.explain()
    checkpoint, plan = compile_package(TF32_PACKAGE, tier_ceiling="numeric")
    result = next(r for r in plan.results if r.name == PASS_NAME)
    assert result.applies and result.tier is Tier.NUMERIC
    assert plan.tier() is Tier.NUMERIC
    assert CAPABILITY in checkpoint.capabilities()
    text = plan.explain()
    assert "plan tier: NUMERIC" in text
    assert result.params["required_by_checkpoint"] == CAPABILITY
    assert "2.482e-3" in text
    assert "there is no valid BITEXACT" in text
    try:
        plan.bitexact_subset()
    except RuntimeError as exc:
        assert "Select the FP32 checkpoint" in str(exc)
    else:
        raise AssertionError("checkpoint-required NUMERIC pass produced a fake BITEXACT subset")


def test_v044_pointer_plan_has_its_own_identity_and_measured_evidence():
    report = validate_package(V044_TF32_PACKAGE)
    assert report.ok, report.explain()
    checkpoint, plan = compile_package(V044_TF32_PACKAGE, tier_ceiling="numeric")
    result = next(r for r in plan.results if r.name == PASS_NAME)
    assert result.applies and result.tier is Tier.NUMERIC
    # The plan header names the CHECKPOINT being planned (main's declared-id rule); the
    # measured upstream weights are the pass's evidence key and appear in its reason line.
    assert plan.model_id == "instinctflash/pi05-libero-v044-tf32-h100"
    assert result.params["evidence"].endswith("tf32_v044_static_h100_results.json")
    assert result.params["max_abs_action_delta"] == 0.0008987784385681152
    text = plan.explain()
    assert "plan tier: NUMERIC" in text
    assert "lerobot/pi05_libero_finetuned_v044" in text
    assert "185.960 -> 72.407 ms (2.568x)" in text
    assert "8.988e-4" in text


def test_checkpoint_aware_spec_hook_preserves_legacy_adapter_fallback():
    checkpoint = Checkpoint(str(V044_TF32_PACKAGE), load_declaration(V044_TF32_PACKAGE))
    spec = _adapter_spec_for_checkpoint(Pi05Adapter(), checkpoint)
    # model_id is NOT rewritten by the hook: the plan header is owned by the facade's
    # declared-id override; the evidence key rides in notes.
    assert spec.model_id == "lerobot/pi05_base"
    assert spec.notes["base_weights"] == "lerobot/pi05_libero_finetuned_v044"
    # Geometry comes from observation_contract / the declared obs_features -- the 4-field v044
    # contract including the model-owned empty camera, NOT a hook-local hardcode.
    assert [(field.key, field.shape) for field in spec.observation.fields] == [
        ("observation.images.image", (3, 256, 256)),
        ("observation.images.image2", (3, 256, 256)),
        ("observation.images.empty_camera_0", (3, 224, 224)),
        ("observation.state", (8,)),
    ]
    contract, source = Pi05Adapter().observation_contract(checkpoint)
    assert tuple(contract.fields) == tuple(spec.observation.fields)
    assert "obs_features" in source

    sentinel = object()
    legacy = SimpleNamespace(spec=lambda: sentinel)
    assert _adapter_spec_for_checkpoint(legacy, checkpoint) is sentinel


def test_plain_pi_package_remains_bitexact():
    checkpoint, plan = compile_package(FP32_PACKAGE)
    result = next(r for r in plan.results if r.name == PASS_NAME)
    assert not result.applies
    assert CAPABILITY not in checkpoint.capabilities()
    assert plan.tier() is Tier.BITEXACT
    assert "plan tier: BITEXACT" in plan.explain()


class _FakeParameter:
    def __init__(self, dtype, floating=True):
        self.dtype = dtype
        self._floating = floating

    def is_floating_point(self):
        return self._floating


def _fake_torch_module():
    torch = ModuleType("torch")
    torch.float32 = "torch.float32"
    torch.float16 = "torch.float16"
    torch.precision = "medium"
    torch.backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))
    )
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.get_float32_matmul_precision = lambda: torch.precision

    def set_precision(value):
        torch.precision = value

    torch.set_float32_matmul_precision = set_precision
    return torch


def test_tf32_build_overrides_v044_config_before_model_construction_and_gates_dtype():
    fake_torch = _fake_torch_module()
    seen = {}

    class FakeConfig:
        dtype = "bfloat16"
        compile_model = True
        num_inference_steps = 10

        @classmethod
        def from_pretrained(cls, repo):
            seen["config_repo"] = repo
            return cls()

    class FakePolicy:
        def __init__(self, config):
            self.config = config
            self.model = object()

        @classmethod
        def from_pretrained(cls, repo, *, config):
            seen["model_repo"] = repo
            seen["dtype_at_construction"] = config.dtype
            seen["compile_at_construction"] = config.compile_model
            return cls(config)

        def eval(self):
            return self

        def to(self, _device):
            return self

        def named_parameters(self):
            return iter((("weight", _FakeParameter(fake_torch.float32)),))

        def reset(self):
            pass

    lerobot = ModuleType("lerobot")
    policies = ModuleType("lerobot.policies")
    factory = ModuleType("lerobot.policies.factory")
    config_mod = ModuleType("lerobot.policies.pi05.configuration_pi05")
    model_mod = ModuleType("lerobot.policies.pi05.modeling_pi05")
    factory.make_pre_post_processors = lambda *_args, **_kwargs: (lambda x: x, lambda x: x)
    config_mod.PI05Config = FakeConfig
    model_mod.PI05Policy = FakePolicy
    fake_modules = {
        "torch": fake_torch,
        "lerobot": lerobot,
        "lerobot.policies": policies,
        "lerobot.policies.factory": factory,
        "lerobot.policies.pi05": ModuleType("lerobot.policies.pi05"),
        "lerobot.policies.pi05.configuration_pi05": config_mod,
        "lerobot.policies.pi05.modeling_pi05": model_mod,
    }
    op = {
        "enabled": True, "tier": "NUMERIC", "math_mode": "tf32",
        "parameter_dtype": "float32", "compile_model": False,
        "static_kv_graph": True, "hardware": "sm90",
        "max_abs_action_delta": 0.0008987784385681152,
        "evidence": "examples/pi05_vla/tf32_v044_static_h100_results.json",
    }
    checkpoint = SimpleNamespace(
        model_id="instinctflash/pi05-libero-v044-tf32-h100",
        execution=SimpleNamespace(
            extra={"base_weights": "lerobot/pi05_libero_finetuned_v044",
                   "pi05_tf32_numeric": op},
            nfe={"prefix": 1, "action": 10},
        ),
    )
    plan = SimpleNamespace(results=[
        PassResult("graph_capture", True, Tier.BITEXACT, "test"),
        PassResult(PASS_NAME, True, Tier.NUMERIC, "test",
                   params={
                       "hardware": "sm90",
                       "max_abs_action_delta": 0.0008987784385681152,
                       "evidence": "examples/pi05_vla/tf32_v044_static_h100_results.json",
                   }),
    ])

    with mock.patch.dict(sys.modules, fake_modules), \
            mock.patch.object(adapter_module, "_require_processor_steps", lambda _repo: None), \
            mock.patch.object(Pi05Adapter, "install", lambda *_args, **_kwargs: []):
        loop = Pi05Adapter().build_in_process(checkpoint, plan, device="cpu")
        loop.close()

    assert seen == {
        "config_repo": "lerobot/pi05_libero_finetuned_v044",
        "model_repo": "lerobot/pi05_libero_finetuned_v044",
        "dtype_at_construction": "float32",
        "compile_at_construction": False,
    }
    assert fake_torch.precision == "medium"
    assert not fake_torch.backends.cuda.matmul.allow_tf32


def test_tf32_parameter_dtype_gate_rejects_any_non_fp32_parameter():
    torch = _fake_torch_module()
    policy = SimpleNamespace(named_parameters=lambda: iter((
        ("good", _FakeParameter(torch.float32)),
        ("bad", _FakeParameter(torch.float16)),
    )))
    try:
        _require_all_floating_parameters_fp32(policy, torch)
    except RuntimeError as exc:
        assert "bad=torch.float16" in str(exc)
        assert "requires every floating parameter" in str(exc)
    else:
        raise AssertionError("TF32 operating point accepted an FP16 parameter")


def test_forged_required_param_without_manifest_cap_cannot_escape_ceiling():
    class Forged:
        name = "forged_numeric"
        def evaluate(self, _spec, _deployment):
            return PassResult(
                self.name, True, Tier.NUMERIC, "forged",
                params={"required_by_checkpoint": "declares:not_present"},
            )
    checkpoint = Checkpoint(str(FP32_PACKAGE), load_declaration(FP32_PACKAGE))
    plan = Optimizer(passes=[Forged()]).compile(
        Pi05Adapter().spec(), capabilities=checkpoint.capabilities()
    )
    assert not plan.results[0].applies
    assert "exceeds ceiling BITEXACT" in plan.results[0].reason

    class ForgedGenericCapability:
        name = "forged_servable"
        def evaluate(self, _spec, _deployment):
            return PassResult(
                self.name, True, Tier.NUMERIC, "forged",
                params={"required_by_checkpoint": "servable"},
            )
    generic = Optimizer(passes=[ForgedGenericCapability()]).compile(
        Pi05Adapter().spec(), capabilities=checkpoint.capabilities()
    )
    assert not generic.results[0].applies


def test_explicit_bitexact_ceiling_refuses_the_tf32_package_loudly():
    """The conservative ceiling policy (team decision, 2026-08-27).

    A caller who EXPLICITLY demands tier_ceiling='bitexact' and selects a TF32 package gets a
    refusal that names both facts -- never PR#4's silent NUMERIC override, and never a silently
    dropped pass that fails later at build time. Omitting the ceiling is the runtime's default
    policy, which also requires explicit permission for numeric transforms.
    """
    try:
        compile_package(TF32_PACKAGE, tier_ceiling="bitexact")
    except RuntimeError as exc:
        assert "required by the selected checkpoint" in str(exc)
        assert "tier_ceiling=BITEXACT" in str(exc)
        assert "select a compatible checkpoint" in str(exc)
    else:
        raise AssertionError("explicit bitexact ceiling must refuse a TF32 package")

    # explicit ceilings at or above the declared tier still plan normally
    _checkpoint, plan = compile_package(TF32_PACKAGE, tier_ceiling="numeric")
    assert next(r for r in plan.results if r.name == PASS_NAME).applies


def test_default_policy_never_admits_a_behavioral_operating_point():
    """'Never widen beyond the runtime's default policy': the hatch is capped at NUMERIC."""
    class DeclaredBehavioral:
        name = "declared_behavioral"

        def evaluate(self, _spec, _deployment):
            return PassResult(
                self.name, True, Tier.BEHAVIORAL, "declared",
                params={"required_by_checkpoint": "declares:pi05_tf32_numeric"},
            )

    checkpoint = Checkpoint(str(TF32_PACKAGE), load_declaration(TF32_PACKAGE))
    try:
        Optimizer(passes=[DeclaredBehavioral()]).compile(
            Pi05Adapter().spec(), capabilities=checkpoint.capabilities())
    except RuntimeError as exc:
        assert "never widen the default BITEXACT" in str(exc)
    else:
        raise AssertionError("a BEHAVIORAL operating point must need an explicit ceiling")


def test_serve_preflight_dry_run_path_applies_tf32_for_the_pointer_package():
    """plan_declaration -- the exact call behind `serve --serve.dry_run=true` -- must show the
    pass APPLYING at tier NUMERIC for the pointer package, and BITEXACT for the plain one."""
    from instinctflash.runtime.facade import plan_declaration

    old_probe = DeviceProfile.probe
    DeviceProfile.probe = staticmethod(lambda device=None: H100)
    try:
        _ckpt, _adapter, plan, probed = plan_declaration(str(TF32_PACKAGE), tier_ceiling="numeric")
        assert probed is H100
        result = next(r for r in plan.results if r.name == PASS_NAME)
        assert result.applies and plan.tier() is Tier.NUMERIC
        _ckpt2, _adapter2, plain, _ = plan_declaration("lerobot/pi05_base")
        assert plain.tier() is Tier.BITEXACT
        assert not any(r.applies and r.name == PASS_NAME for r in plain.results)
    finally:
        DeviceProfile.probe = old_probe


def test_excluding_required_pass_is_refused_not_silently_fp32():
    checkpoint, plan = compile_package(TF32_PACKAGE, tier_ceiling="numeric")
    excluded = plan.without(PASS_NAME)
    try:
        _validate_tf32_plan(checkpoint, excluded)
    except RuntimeError as exc:
        assert "refuses to silently substitute" in str(exc)
    else:
        raise AssertionError("excluded mandatory TF32 pass was accepted")


class _FakeTorch:
    def __init__(self, precision="medium", allow=False):
        self.precision = precision
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=allow))
        )
    def get_float32_matmul_precision(self):
        return self.precision
    def set_float32_matmul_precision(self, value):
        self.precision = value


def test_precision_lease_refcounts_rejects_mixing_and_restores_on_exception():
    torch = _FakeTorch(precision="medium", allow=True)
    one = Pi05PrecisionLease(torch, "tf32")
    two = Pi05PrecisionLease(torch, "tf32")
    assert _state_for_tests() == ("tf32", 2)
    assert torch.precision == "high" and torch.backends.cuda.matmul.allow_tf32
    one.close()
    assert _state_for_tests() == ("tf32", 1)
    try:
        Pi05PrecisionLease(torch, "fp32")
    except RuntimeError as exc:
        assert "process-global" in str(exc)
    else:
        raise AssertionError("mixed FP32/TF32 runtimes were accepted")
    two.close()
    assert _state_for_tests() == (None, 0)
    assert torch.precision == "medium" and torch.backends.cuda.matmul.allow_tf32
    try:
        with Pi05PrecisionLease(torch, "fp32"):
            assert torch.precision == "highest"
            raise ValueError("injected build failure")
    except ValueError:
        pass
    assert _state_for_tests() == (None, 0)
    assert torch.precision == "medium" and torch.backends.cuda.matmul.allow_tf32


def test_inprocess_backend_close_reaches_impl_close():
    from instinctflash.runtime.execution import InProcessBackend
    closed = []
    class Impl:
        def close(self):
            closed.append(True)
    class Adapter:
        def build_in_process(self, *_args, **_kwargs):
            return Impl()
        def install(self, *_args, **_kwargs):
            pass
    backend = InProcessBackend(Adapter(), SimpleNamespace(execution=SimpleNamespace(backbone="pi05")),
                               SimpleNamespace(results=[]))
    backend._ensure()
    backend.close()
    backend.close()
    assert closed == [True]


def test_runtime_explain_carries_numeric_plan():
    from instinctflash.runtime.facade import Runtime
    checkpoint, plan = compile_package(TF32_PACKAGE, tier_ceiling="numeric")
    runtime = Runtime(checkpoint, Pi05Adapter(), plan, SimpleNamespace(close=lambda: None),
                      placement_reason="test in-process")
    text = runtime.explain()
    assert "pi05_tf32_numeric" in text
    assert "plan tier: NUMERIC" in text


if __name__ == "__main__":
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))


def test_default_refuses_checkpoint_required_numeric_before_execution():
    for path in (TF32_PACKAGE, V044_TF32_PACKAGE):
        try:
            compile_package(path, tier_ceiling=None)
        except RuntimeError as error:
            assert "never widen the default BITEXACT" in str(error)
            assert "tier_ceiling='numeric'" in str(error)
        else:
            raise AssertionError("checkpoint silently widened default arithmetic policy")


THOR_TF32_PACKAGE = ROOT / "examples/checkpoint/pi05-libero-v044-tf32-thor"
THOR = DeviceProfile(name="NVIDIA Thor", capability=(11, 0), total_memory=128_000_000_000,
                     features=frozenset({"cuda", "cuda_graphs", "cublas", "fp8"}))


def test_thor_tf32_is_explicit_and_does_not_inherit_h100_margin():
    checkpoint, plan = compile_package(THOR_TF32_PACKAGE, "numeric", profile=THOR)
    assert validate_package(THOR_TF32_PACKAGE).ok
    assert _validate_tf32_plan(checkpoint, plan)
    result = next(r for r in plan.results if r.name == PASS_NAME)
    assert result.applies and result.tier is Tier.NUMERIC
    assert result.params["hardware"] == "sm110"
    assert result.params["max_abs_action_delta"] is None
    assert result.params["qualification"] == "unqualified"
    assert "H100" not in result.params["evidence"]


def test_tf32_hardware_declarations_cannot_cross_devices():
    import pytest
    for package, device in [(THOR_TF32_PACKAGE, H100), (V044_TF32_PACKAGE, THOR)]:
        with pytest.raises((RuntimeError, ValueError)):
            checkpoint, plan = compile_package(package, "numeric", profile=device)
            _validate_tf32_plan(checkpoint, plan)


def test_thor_tf32_cannot_spend_bitexact_or_import_a_margin():
    import pytest
    with pytest.raises((RuntimeError, ValueError)):
        compile_package(THOR_TF32_PACKAGE, "bitexact", profile=THOR)
    checkpoint, plan = compile_package(THOR_TF32_PACKAGE, "numeric", profile=THOR)
    checkpoint.execution.extra["pi05_tf32_numeric"]["max_abs_action_delta"] = 0.001
    with pytest.raises(RuntimeError, match="no inherited action margin"):
        _validate_tf32_plan(checkpoint, plan)

"""SM120 CPU contracts; mocked device identities do not qualify CUDA arithmetic."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import DeviceProfile
from instinctflash.passes.generic.engine_offload import EngineOffloadApplicable
from instinctflash.planners.planner import PassResult, Plan, Tier
from instinctflash.runtime import sm89_fp8, sm120_fp8
from instinctflash.runtime.desktop_fp8 import backend_for_capability
from instinctflash.runtime.engine_backend import EngineBackend, engine_available
from instinctflash.runtime.execution import _mark_plan_engine_executed, choose_backend
from instinctflash.runtime.precision import install_requested_fp8
from instinctflash.runtime.torch_fp8_linear import (
    SM89FP8Linear,
    SM120FP8Linear,
    ThorFP8Linear,
)


class GemmaAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, torch.nn.Linear(16, 16, dtype=torch.bfloat16))


def plan(family="pi05", backend=sm120_fp8):
    return Plan("test", [PassResult(
        "engine_offload", True, Tier.NUMERIC, "test candidate",
        params={"backend": "engine", "executor": backend.EXECUTOR,
                "recipe_id": backend.recipe_for(family).recipe_id})])


def receipt(family="pi05"):
    return {"executor": sm120_fp8.EXECUTOR, "precision": "fp8", "family": family,
            "recipe_id": sm120_fp8.recipe_for(family).recipe_id, "capability": [12, 0],
            "projections": [{"path": "attention.q_proj"}]}


@pytest.mark.parametrize("family", sm89_fp8.RECIPES)
def test_architecture_specific_recipe_keeps_existing_sites_and_numeric_tier(family):
    old, new = sm89_fp8.recipe_for(family), sm120_fp8.recipe_for(family)
    assert old.recipe_id == f"sm89_{family}_attention_qkv_v1"
    assert new.recipe_id == f"sm120_{family}_attention_qkv_v1"
    assert old.attention_types == new.attention_types
    assert old.projection_names == new.projection_names
    assert old.requires_owned_builder == new.requires_owned_builder
    device = DeviceProfile(name="RTX 5090", capability=(12, 0), total_memory=32 << 30,
                           features=frozenset({"cuda", "fp8"}))
    result = EngineOffloadApplicable().evaluate(
        SimpleNamespace(notes={"backbone": family}), DeploymentSpec(device=device))
    assert result.applies and result.tier == Tier.NUMERIC
    assert result.params["executor"] == "sm120_torch_fp8"
    assert result.params["recipe_id"] == new.recipe_id
    assert result.params["certification"] == "uncertified"
    assert "qualification" in result.reason


def test_dependency_gate_checks_exact_device_and_callable_without_launch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (8, 9))
    assert not sm120_fp8.available("pi05")[0]
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (12, 0))
    monkeypatch.setattr(sm89_fp8, "find_spec", lambda name: object())
    assert engine_available("pi05")[0]
    assert not engine_available("unknown")[0]
    monkeypatch.setattr(torch, "_scaled_mm", None)
    assert not engine_available("pi05")[0]
    monkeypatch.setattr(torch, "_scaled_mm", Mock())
    monkeypatch.setattr(sm89_fp8, "find_spec", lambda name: None)
    assert not engine_available("pi05")[0]
    torch._scaled_mm.assert_not_called()
    assert backend_for_capability((8, 9)) is sm89_fp8
    assert backend_for_capability((12, 0)) is sm120_fp8
    with pytest.raises(ValueError, match="No desktop"):
        backend_for_capability((10, 0))


def test_native_install_and_native_passes_remain_explicit():
    model = GemmaAttention()
    selected = plan()
    with patch.object(sm120_fp8, "maybe_install_sm120_fp8") as install:
        install_requested_fp8(model, SimpleNamespace(results=[]), "pi05")
        install.assert_not_called()
        install_requested_fp8(model, selected, "pi05")
        install.assert_called_once_with(model, selected, "pi05")
    native = PassResult("graph_capture", True, Tier.BITEXACT, "native graph")
    selected.results.append(native)
    _mark_plan_engine_executed(selected, selected.results[0])
    assert selected.results[1] is native and native.applies
    checkpoint = SimpleNamespace(execution=SimpleNamespace(backbone="pi05"))
    with patch("instinctflash.runtime.engine_backend.engine_available", return_value=(True, "test")) as probe, \
            patch("instinctflash.runtime.engine_backend.EngineBackend") as backend:
        actual, _ = choose_backend("auto", object(), checkpoint, selected, precision="fp8")
    probe.assert_called_once_with("pi05")
    assert actual is backend.return_value


def test_cpu_packing_preserves_arithmetic_and_keeps_architecture_receipts_separate(monkeypatch):
    source = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.fill_(0.0361328125)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (8, 9))
    old = SM89FP8Linear(source, device="cuda", storage_device="cpu")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (12, 0))
    new = SM120FP8Linear(source, device="cuda", storage_device="cpu")
    for a, b in zip(old.buffers(), new.buffers()):
        assert a.device.type == b.device.type == "cpu"
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert new.weight_scale.view(torch.int32).item() == 0x38A92493
    model = GemmaAttention()
    output = model.o_proj
    result = sm120_fp8.install_sm120_fp8(model, "pi05", storage_device="cpu")
    assert model.o_proj is output
    assert isinstance(model.q_proj, SM120FP8Linear)
    assert result["capability"] == [12, 0] and result["quality_status"] == "unverified"
    assert result["quality_certificate"] is None
    assert len(result["projections"]) == 3
    assert not hasattr(model, "_sm89_fp8_recipe")
    assert model._sm120_fp8_recipe is result
    with pytest.raises(ValueError, match="already installed"):
        sm89_fp8.install_sm89_fp8(model, "pi05")


@pytest.mark.parametrize("linear_type,capability", [
    (SM120FP8Linear, (8, 9)), (SM89FP8Linear, (12, 0)), (ThorFP8Linear, (12, 0)),
])
def test_projection_refuses_cross_architecture_before_tensor_placement(monkeypatch, linear_type, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)
    with pytest.raises(ValueError, match="capability"):
        linear_type(torch.nn.Linear(16, 16, dtype=torch.bfloat16), device="cuda", storage_device="cpu")


def test_cross_recipe_and_partial_conversion_refused_before_model_mutation(monkeypatch):
    model = GemmaAttention()
    originals = dict(model.named_modules())
    with pytest.raises(ValueError, match="explicit executor"):
        sm120_fp8._require_requested_recipe(plan(backend=sm89_fp8), "pi05")
    with pytest.raises(ValueError, match="explicit executor"):
        sm120_fp8.maybe_install_sm120_fp8(model, plan("wan_va"), "pi05")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (12, 0))
    with torch.no_grad():
        model.v_proj.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        sm120_fp8.install_sm120_fp8(model, "pi05", storage_device="cpu")
    assert dict(model.named_modules()) == originals
    assert not hasattr(model, "_sm120_fp8_recipe")


@pytest.mark.parametrize("field,value", [("capability", [8, 9]), ("capability", None),
    ("executor", "sm89_torch_fp8"), ("recipe_id", sm89_fp8.recipe_for("pi05").recipe_id),
    ("family", "wan_va"), ("projections", [])])
def test_receipts_cannot_relabel_sm89_as_sm120(field, value):
    with pytest.raises(ValueError):
        sm120_fp8.validate_receipt({**receipt(), field: value}, "pi05")


@pytest.mark.parametrize("family", ["cosmos3_policy", "dreamzero"])
def test_owned_builder_route_validates_installed_recipe_and_closes_on_failure(monkeypatch, family):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (12, 0))
    checkpoint = SimpleNamespace(execution=SimpleNamespace(backbone=family))
    installed = receipt(family)
    installed["recipe_id"] += "_dense_mlp"
    loop = SimpleNamespace(backend_stats={"fp8_recipe": installed}, close=Mock())
    adapter = SimpleNamespace(build_sm120_fp8=Mock(return_value=loop), build_sm89_fp8=Mock())
    selected = plan(family)
    backend = EngineBackend(adapter, checkpoint, selected, device="cuda:0")
    assert backend._loop.declaration()["executor"] == "sm120_torch_fp8"
    assert selected.results[0].params["recipe_id"] == installed["recipe_id"]
    adapter.build_sm89_fp8.assert_not_called()
    loop.close.assert_not_called()
    installed["capability"] = [8, 9]
    with pytest.raises(ValueError, match="capability"):
        EngineBackend(adapter, checkpoint, plan(family), device="cuda:0")
    loop.close.assert_called_once()


def test_missing_memory_builder_and_wrong_device_refuse_before_loading(monkeypatch):
    adapter = SimpleNamespace(build_in_process=Mock())
    checkpoint = SimpleNamespace(execution=SimpleNamespace(backbone="dreamzero"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (8, 9))
    with pytest.raises(RuntimeError, match="SM120 CUDA device"):
        sm120_fp8.build_sm120_loop(adapter, checkpoint, plan("dreamzero"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (12, 0))
    with pytest.raises(RuntimeError, match="owned FP8"):
        sm120_fp8.build_sm120_loop(adapter, checkpoint, plan("dreamzero"))
    adapter.build_in_process.assert_not_called()


def test_qualifier_refuses_foreign_device_and_preserves_failure_receipt(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("sm120_qualifier_test", scripts / "qualify_sm120_fp8.py")
    qualifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (8, 9))
    set_device = Mock(side_effect=AssertionError("must not select a GPU"))
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    output = tmp_path / "refused.json"
    assert qualifier.main(["--device", "cuda:0", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["schema"] == "instinctflash.sm120_fp8_primitive_qualification.v1"
    assert report["passed"] is False and report["model_constructed"] is False
    assert "SM120" in report["error"]["message"]
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        qualifier.main(["--output", str(output)])
    assert output.read_bytes() == original
    set_device.assert_not_called()

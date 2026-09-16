"""An explicit upstream VA schedule must not become a Runtime baseline."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.regression import native_reference, user_e2e
from instinctflash.descriptors.checkpoint import ExecutionDeclaration
from instinctflash.descriptors.package import Checkpoint


def checkpoint():
    return Checkpoint(
        path="/unchanged/pinned/checkpoint",
        execution=ExecutionDeclaration(
            model_id="test-va", nfe={"video": 25, "action": 50},
            guidance={"video": "cfg", "action": "positive_only"},
            extra={"va_config": "robotwin"}))


@pytest.fixture
def upstream_stub(monkeypatch, tmp_path):
    calls = []
    original_config = SimpleNamespace(
        num_inference_steps=25, action_num_inference_steps=50,
        guidance_scale=5.0, action_guidance_scale=1.0,
        obs_cam_keys=("high", "left", "right"), frame_chunk_size=8)

    def construct(config):
        calls.append(config)
        return SimpleNamespace(job_config=config)

    server_module = SimpleNamespace(
        VA_CONFIGS={"robotwin": original_config},
        init_distributed=lambda *args: None, VA_Server=construct)

    def apply_guidance(config, guidance):
        assert guidance == {"video": "cfg", "action": "positive_only"}
        assert config.guidance_scale == 5.0 and config.action_guidance_scale == 1.0

    adapter = SimpleNamespace(lingbot_root="/unmodified/vendor",
                              materialize=lambda value: value.path)
    monkeypatch.setitem(sys.modules, "instinctflash.adapters.lingbot_va", SimpleNamespace(
        LingBotVA=lambda: adapter,
        _ControlLoop=lambda server, *args, **kwargs: SimpleNamespace(_server=server),
        resolve_observation_geometry=lambda *args, **kwargs: ({}, None),
        apply_declared_guidance=apply_guidance))
    monkeypatch.setitem(sys.modules, "instinctflash.runtime.lingbot_install", SimpleNamespace(
        import_lingbot_server=lambda root: server_module))
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "debug"))
    for key in ("LINGBOT_CKPT", "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"):
        monkeypatch.setenv(key, "test-owned")
    return calls, original_config


def test_existing_native_constructor_call_is_unchanged(upstream_stub, tmp_path):
    calls, original = upstream_stub
    declared = checkpoint()
    result = native_reference.build("va", declared, output_dir=tmp_path)
    assert result._checkpoint is declared
    assert result.execution_policy == {
        "reference": "upstream eager native policy; shared I/O translation only"}
    assert result.plan.results == []
    assert calls[0].num_inference_steps == original.num_inference_steps == 25
    assert calls[0].action_num_inference_steps == original.action_num_inference_steps == 50


def test_rtx4090_routes_to_declared_residency_without_changing_schedule(upstream_stub, tmp_path):
    from benchmarks.regression.hardware import target_record

    calls, original = upstream_stub
    target = target_record("rtx4090")
    hardware = {"target": target, "status": "passed", "name": "NVIDIA GeForce RTX 4090",
                "capability": [8, 9], "uuid": "CPU-route-test-only", "total_memory_bytes": 24 << 30}
    installer = sys.modules["instinctflash.runtime.lingbot_install"]
    residency_calls = []

    def native_residency(module, config, *, device, expected_device):
        residency_calls.append((device, deepcopy(expected_device)))
        return module.VA_Server(config), {"schema": "instinctflash.native_va_residency.v1",
            "device": {key: expected_device[key] for key in
                       ("name", "capability", "uuid", "total_memory_bytes")}, "successful_resets": 0}

    installer.build_native_reference_server = native_residency
    declared = checkpoint()
    api = native_reference.build("va", declared, output_dir=tmp_path, target=target, hardware=hardware)
    assert residency_calls == [("cuda:0", hardware)]
    assert calls[0].num_inference_steps == original.num_inference_steps == 25
    assert calls[0].action_num_inference_steps == original.action_num_inference_steps == 50
    assert calls[0].guidance_scale == original.guidance_scale == 5.0
    assert calls[0].frame_chunk_size == original.frame_chunk_size
    assert api.execution_policy["native_residency"]["device"]["uuid"] == hardware["uuid"]
    assert "residency changes" in api.execution_policy["reference"]


def test_groot_local_nfe_is_not_misread_as_a_native_override(monkeypatch, tmp_path):
    policy = SimpleNamespace(model=SimpleNamespace(action_head=SimpleNamespace(), config=SimpleNamespace()))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "groot_n17_iwm.adapter", SimpleNamespace(
        _source_root=lambda: tmp_path, _resolve_model_path=lambda value: value.path,
        DEFAULT_EMBODIMENT="test", _GR00TN17Loop=lambda *args, **kwargs: policy))
    tokenizer = SimpleNamespace(PreTrainedTokenizerBase=type("Tokenizer", (), {}))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(tokenization_utils_base=tokenizer))
    monkeypatch.setitem(sys.modules, "transformers.tokenization_utils_base", tokenizer)
    monkeypatch.setitem(sys.modules, "gr00t.policy.gr00t_policy", SimpleNamespace(
        Gr00tPolicy=lambda **kwargs: policy))
    declared = Checkpoint("/unchanged/groot", ExecutionDeclaration(nfe={"backbone": 1, "action": 4}))
    result = native_reference.build("groot", declared, output_dir=tmp_path)
    assert result._checkpoint is declared
    assert result.execution_policy == {
        "reference": "upstream eager native policy; shared I/O translation only"}
    assert policy.model.action_head.num_inference_timesteps == 4


def test_override_changes_only_fresh_upstream_config(upstream_stub, tmp_path):
    calls, original = upstream_stub
    declared = checkpoint()
    before = deepcopy(declared)
    override = {"video": 2, "action": 4}
    result = native_reference.build("va", declared, output_dir=tmp_path, nfe=override)
    assert declared == before and result._checkpoint is declared
    assert calls[0].num_inference_steps == 2
    assert calls[0].action_num_inference_steps == 4
    assert original.num_inference_steps == 25 and original.action_num_inference_steps == 50
    assert result.execution_policy["checkpoint_nfe"] == {"video": 25, "action": 50}
    assert result.execution_policy["nfe"] == {"video": 2, "action": 4}
    assert result.execution_policy["native_nfe_override"] == override
    assert result.execution_policy["schedule_changed"] is True
    assert result.plan.results == []
    override["video"] = 99
    assert result.execution_policy["native_nfe_override"]["video"] == 2


@pytest.mark.parametrize("family,nfe", [
    ("pi05", {"action": 4}), ("dreamzero", {"video": 2}),
    ("va", {}), ("va", {"video": True}), ("va", {"action": False}),
    ("va", {"video": 0}), ("va", {"action": -1}),
    ("va", {"video": 2.0}), ("va", {"video": "2"}),
    ("va", {"prefix": 1}), ("va", [2, 4]),
])
def test_invalid_overrides_rejected_before_upstream_construction(family, nfe, upstream_stub, tmp_path):
    calls, _ = upstream_stub
    with pytest.raises(ValueError):
        native_reference.build(family, checkpoint(), output_dir=tmp_path, nfe=nfe)
    assert calls == []


def cell():
    return {"family": "va", "arm": "eager_native",
            "native_nfe": {"video": 2, "action": 4},
            "effective_schedule": {"nfe": {"video": 2, "action": 4}}}


def test_capture_override_does_not_mutate_frozen_cell():
    value = cell()
    original = deepcopy(value)
    result = user_e2e.native_schedule_override(value)
    result["video"] = 99
    assert value == original
    assert user_e2e.native_schedule_override({"arm": "runtime_default"}) is None


@pytest.mark.parametrize("update", [
    {"arm": "runtime_default"}, {"arm": "runtime_selected"},
    {"arm": "operating_point"}, {"family": "edge"},
    {"native_nfe": None}, {"native_nfe": {}}, {"native_nfe": {"video": True}},
    {"effective_schedule": {"nfe": {"video": 25, "action": 50}}},
    {"effective_schedule": {}},
])
def test_capture_rejects_wrong_backend_or_frozen_schedule(update):
    with pytest.raises(ValueError):
        user_e2e.native_schedule_override({**cell(), **update})


def test_new_single_cell_recipe_preserves_public_history_protocol():
    root = Path(__file__).resolve().parents[1]
    matrix = json.loads((root / "eval/va_native_2v4a_2026-09-15/matrix_v1.json").read_text())
    value, = matrix["cells"]
    assert value["native_nfe"] == user_e2e.native_schedule_override(value) == {"video": 2, "action": 4}
    assert value["expected_runtime_kwargs"] == value["expected_optimizer_environment"] == {}
    assert value["default_schedule"] == {"video": 25, "action": 50}
    assert matrix["protocol"]["history"] == {
        "warmup_episodes": 1, "measured_episodes": 6, "cycles_per_episode": 3}
    assert matrix["require_observed_schedule"] is True
    assert matrix["quality_certified"] is False


@pytest.fixture
def synthetic_validation_capture(tmp_path):
    import numpy as np

    parent = Path(__file__).resolve().parents[1] / "eval/va_native_2v4a_2026-09-15"
    spec = importlib.util.spec_from_file_location("native_va_validator_test", parent / "validate_native_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrix_path, recipe_path = parent / "matrix_v1.json", parent / "recipe_review_v1.json"
    matrix, recipe = json.loads(matrix_path.read_text()), json.loads(recipe_path.read_text())
    value, = matrix["cells"]
    receipt_path = tmp_path / value["receipt"]
    receipt_path.parent.mkdir(parents=True)
    np.savez(receipt_path.with_suffix(".npz"), actions=np.zeros((21, 16, 2, 16), np.float32))
    policy = native_reference.Reference(checkpoint(), None, native_nfe=value["native_nfe"]).execution_policy
    receipt = {
        "schema": 1, "ok": True, "family": "va", "arm": "eager_native",
        "cell_id": value["id"], "model_id": value["model_id"], "revision": value["revision"],
        "runtime_kwargs": {}, "optimizer_environment": {}, "precision": "native",
        "default_schedule": {"video": 25, "action": 50},
        "effective_schedule": value["effective_schedule"], "guidance": {"video": 5, "action": 1},
        "native_nfe_override": value["native_nfe"], "execution_policy": policy,
        "device": "synthetic Thor", "torch": "synthetic", "scope": "synthetic CPU test only",
        "setup_seconds": 1.0, "input_contract": {"synthetic_test": True},
        "observed_nfe_before": value["native_nfe"], "observed_nfe_after": value["native_nfe"],
        "numeric_environment": {"matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False},
        "plan": "Upstream eager; no optimization passes installed",
        "applied_passes": [], "graph_stats": {}, "backend_stats": {},
        "matrix_sha256": module.sha(matrix_path), "input_archive_sha256": recipe["fixture"]["sha256"],
        "cases": recipe["request_cases"],
        "calls": [{"i": i, "ms": i + 1.0, "shape": [16, 2, 16]} for i in range(21)],
        "actions_sha256": module.sha(receipt_path.with_suffix(".npz")),
    }
    receipt_path.write_text(json.dumps(receipt))
    return module, tmp_path, matrix_path, recipe_path, receipt_path


def test_single_cell_validator_keeps_default_and_effective_separate(synthetic_validation_capture):
    module, root, matrix, recipe, _ = synthetic_validation_capture
    result = module.validate(root, matrix, recipe)
    assert result["status"] == "passed_recorded_input_native_latency"
    assert result["primary_p50_ms"] == 13.0
    assert result["native"]["total_calls"] == 21
    assert result["quality_certified"] is False


@pytest.mark.parametrize("field,value", [
    ("default_schedule", {"video": 2, "action": 4}),
    ("native_nfe_override", {"video": 25, "action": 50}),
    ("graph_stats", {"replays": 1}),
    ("numeric_environment", {"matmul_tf32": True, "cudnn_tf32": False, "cudnn_benchmark": False}),
])
def test_single_cell_validator_rejects_changed_native_contract(synthetic_validation_capture, field, value):
    module, root, matrix, recipe, receipt_path = synthetic_validation_capture
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        module.validate(root, matrix, recipe)

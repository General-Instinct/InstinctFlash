"""Target/receipt bindings only; the hardware records below are CPU test fixtures."""
from __future__ import annotations

import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmarks.regression import native_reference
from benchmarks.regression.hardware import target_record


def hardware():
    return {"target": target_record("rtx4090"), "status": "passed",
            "name": "NVIDIA GeForce RTX 4090", "capability": [8, 9],
            "uuid": "CPU-receipt-unit-test-only", "total_memory_bytes": 24 << 30}


@pytest.mark.parametrize("change", [
    {"name": "NVIDIA L40S"}, {"capability": [12, 0]},
    {"target": target_record("jetson_thor")}, {"status": "failed"},
])
def test_wrong_actual_target_fails_before_any_vendor_constructor(tmp_path, change):
    observed = {**hardware(), **change}
    # None checkpoint would fail on access if the target guard did not run first.
    with pytest.raises(ValueError, match="actual GPU"):
        native_reference.build("va", None, output_dir=tmp_path,
                               target=target_record("rtx4090"), hardware=observed)


def test_rtx4090_native_constructor_requires_device_receipt(tmp_path):
    with pytest.raises(ValueError, match="actual GPU"):
        native_reference.build("va", None, output_dir=tmp_path, target=target_record("rtx4090"))


def test_reference_binds_device_residency_and_original_schedule():
    observed = hardware()
    residency = {"schema": "instinctflash.native_va_residency.v1", "successful_resets": 0,
                 "device": {key: observed[key] for key in
                            ("name", "uuid", "capability", "total_memory_bytes")}}
    checkpoint = SimpleNamespace(execution=SimpleNamespace(nfe={"video": 25, "action": 50}))
    api = native_reference.Reference(checkpoint, None, native_nfe={"video": 2, "action": 4},
        target=target_record("rtx4090"), hardware=observed, native_residency=residency)
    assert api.execution_policy["checkpoint_nfe"] == {"video": 25, "action": 50}
    assert api.execution_policy["nfe"] == {"video": 2, "action": 4}
    assert api.execution_policy["native_residency"] is residency
    residency["successful_resets"] += 1
    assert api.execution_policy["native_residency"]["successful_resets"] == 1
    assert "residency changes" in api.execution_policy["reference"]
    observed["uuid"] = "mutated-input"
    assert api.execution_policy["hardware"]["uuid"] == "CPU-receipt-unit-test-only"
    assert api.plan.results == []


def test_reference_refuses_cross_device_residency_receipt():
    observed = hardware()
    residency = {"device": deepcopy(observed)}
    residency["device"]["uuid"] = "another-device"
    with pytest.raises(ValueError, match="different device"):
        native_reference.Reference(None, None, target=target_record("rtx4090"),
                                   hardware=observed, native_residency=residency)
    with pytest.raises(ValueError, match="requires the RTX 4090"):
        native_reference.Reference(None, None, native_residency=residency)


@pytest.mark.parametrize("family", ["edge", "nano"])
def test_cosmos_native_target_uses_original_service_args_and_no_fp8(monkeypatch, tmp_path, family):
    calls = []

    class NativeService:
        def __init__(self, args):
            self.args = args
            calls.append(("service", vars(args)))

    def resident_service(factory, *, nano, device):
        calls.append(("residency", nano, device))
        return factory()

    monkeypatch.setitem(sys.modules, "cosmos3_iwm.adapter", SimpleNamespace(
        _resolve_model_path=lambda checkpoint: checkpoint.path))
    monkeypatch.setitem(sys.modules, "cosmos3_iwm.sm89_residency", SimpleNamespace(
        build_native_service=resident_service))
    monkeypatch.setitem(sys.modules, "cosmos_framework.scripts.action_policy_server_robolab", SimpleNamespace(
        RobolabPolicyService=NativeService, RobolabServerArgs=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "instinctflash.runtime.cosmos_droid", SimpleNamespace(
        CosmosDROIDLoop=lambda service: SimpleNamespace(service=service)))
    fields = dict(domain_name="DROID", action_chunk_size=32, conditioning_fps=8,
                  action_dim=7, image_height=256, image_width=256, format_prompt_as_json=False)
    checkpoint = SimpleNamespace(path="/original/checkpoint", execution=SimpleNamespace(
        extra=fields, nfe={"action": 4}))
    result = native_reference.build(family, checkpoint, output_dir=tmp_path,
        target=target_record("rtx4090"), hardware=hardware())
    assert calls[0] == ("residency", family == "nano", "cuda:0")
    expected = dict(checkpoint_path=checkpoint.path, **fields, num_steps=4,
                    guidance=3.0, shift=5.0, seed=0)
    assert calls[1] == ("service", expected)
    assert result.execution_policy["native_residency"]["precision"] == "native"
    assert not result.execution_policy["native_residency"]["schedule_changed_by_residency"]


def test_dreamzero_native_residency_keeps_lazy_load_full_value_gate_and_fixed_mask(monkeypatch, tmp_path):
    import torch.distributed.device_mesh as device_mesh

    calls = []
    loop = SimpleNamespace(_loading_receipt={}, close=lambda: None)
    checkpoint = SimpleNamespace(path=tmp_path, execution=SimpleNamespace(
        extra={}, nfe={"video_action": 16, "kv_commit": 1}))

    def policy_factory(**kwargs):
        calls.append(("policy", kwargs))
        return SimpleNamespace()

    def owned_builder(path, create_policy, create_wrapper, *, step_cache, residency_precision):
        calls.append(("builder", path, step_cache.to_dict(), residency_precision))
        policy = create_policy(path)
        create_wrapper(policy)
        return loop

    def verify_loaded(policy, original, *, output_dir):
        calls.append(("value_gate", original, output_dir))
        return {"status": "passed", "receipt": "CPU-fixture-only.json"}

    def wrap(**kwargs):
        calls.append(("wrapper", kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(device_mesh, "init_device_mesh", lambda *args, **kwargs: "CPU-test-mesh-only")
    monkeypatch.setitem(sys.modules, "dreamzero_iwm.adapter", SimpleNamespace(
        _source_root=lambda: tmp_path, _resolve_model_path=lambda checkpoint: checkpoint.path,
        _DreamZeroLoop=None, _head_declaration=None, _build_owned_native_loop=owned_builder))
    monkeypatch.setitem(sys.modules, "eval_utils.serve_dreamzero_wan22", SimpleNamespace(
        DreamZeroWan225BPolicy=wrap, _get_expected_video_resolution=lambda policy: (256, 320),
        _maybe_init_distributed=lambda: None))
    monkeypatch.setitem(sys.modules, "groot.vla.data.schema", SimpleNamespace(EmbodimentTag=lambda tag: tag))
    monkeypatch.setitem(sys.modules, "groot.vla.model.n1_5.sim_policy", SimpleNamespace(GrootSimPolicy=policy_factory))
    monkeypatch.setitem(sys.modules, "benchmarks.regression.native_loaded", SimpleNamespace(verify_loaded=verify_loaded))
    result = native_reference.build("dreamzero", checkpoint, output_dir=tmp_path,
        target=target_record("rtx4090"), hardware=hardware())
    assert [call[0] for call in calls] == ["builder", "policy", "value_gate", "wrapper"]
    assert calls[0][2] == dict(dynamic=False, fixed_steps=8, profile=None, source="native_reference_checkpoint")
    assert calls[0][3] == "native"
    assert calls[1][1]["lazy_load"] is True
    assert calls[2][1] == checkpoint.path
    assert loop._loading_receipt["loaded_value_gate"]["status"] == "passed"
    assert result.execution_policy["native_residency"]["family"] == "dreamzero"

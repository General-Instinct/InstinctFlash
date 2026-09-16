import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/cosmos3_policy"))
from cosmos3_iwm.sm89_residency import cpu_construction


class NativeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((8, 8), device="meta"))
        self.native_materialization_devices = []

    def to_empty(self, *, device, recurse=True):
        self.native_materialization_devices.append(str(device))
        return super().to_empty(device=device, recurse=recurse)


def test_only_owned_network_changes_first_materialization_and_constructor_is_restored():
    original = NativeNetwork.__init__
    with cpu_construction(network_type=NativeNetwork) as construction:
        net = NativeNetwork().to(dtype=torch.bfloat16)
        net.to_empty(device="cuda")
        construction.verify(SimpleNamespace(net=net))
        assert net.weight.device.type == "cpu"
        assert net.weight.dtype == torch.bfloat16
        assert net.native_materialization_devices == ["cpu"]
        assert "to_empty" not in net.__dict__
    assert NativeNetwork.__init__ is original
    unowned = NativeNetwork()
    assert "to_empty" not in unowned.__dict__
    assert unowned.weight.device.type == "meta"


def test_constructor_failure_cleans_instance_override_and_does_not_poison_next_load():
    original = NativeNetwork.__init__
    with pytest.raises(ValueError, match="checkpoint failed"):
        with cpu_construction(network_type=NativeNetwork):
            net = NativeNetwork()
            raise ValueError("checkpoint failed")
    assert NativeNetwork.__init__ is original
    assert "to_empty" not in net.__dict__
    with cpu_construction(network_type=NativeNetwork) as construction:
        NativeNetwork().to_empty(device="cuda")
        assert construction.materializations == 1


def test_foreign_thread_network_is_not_claimed():
    foreign = []
    with cpu_construction(network_type=NativeNetwork) as construction:
        worker = threading.Thread(target=lambda: foreign.append(NativeNetwork()))
        worker.start()
        worker.join()
        assert construction.net is None
        owned = NativeNetwork()
        owned.to_empty(device="cuda")
    assert "to_empty" not in foreign[0].__dict__
    assert foreign[0].weight.device.type == "meta"


def test_wrong_service_and_unconsumed_materialization_refused():
    with cpu_construction(network_type=NativeNetwork) as construction:
        net = NativeNetwork()
        with pytest.raises(RuntimeError, match="owned CPU construction"):
            construction.verify(SimpleNamespace(net=net))
        net.to_empty(device="cuda")
        with pytest.raises(RuntimeError, match="owned CPU construction"):
            construction.verify(SimpleNamespace(net=object()))


def test_droid_finalizer_receives_original_serving_profile_before_legacy_fp8(tmp_path, monkeypatch):
    import json
    from types import ModuleType
    from instinctflash.runtime import cosmos_droid, cosmos_fp8

    module = ModuleType("cosmos_framework.scripts.action_policy_server_robolab")
    class Args(SimpleNamespace):
        pass
    class NativeService:
        def __init__(self, args):
            self.args = args
            self.setup = self._build_setup_args(args)

        def _build_setup_args(self, args):
            return SimpleNamespace(model_copy=lambda *, update: update)

    module.RobolabServerArgs = Args
    module.RobolabPolicyService = NativeService
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cosmos_fp8, "install_cosmos_fp8",
                        lambda *args: pytest.fail("legacy FP8 installer executed"))
    (tmp_path / "checkpoint.json").write_text(json.dumps({"policy": {
        "domain_name": "droid_lerobot", "action_chunk_size": 32, "conditioning_fps": 15}}))
    called = []
    def finish(service):
        called.append(service)
        return {"recipe_id": "owned-sm89-test"}
    service, receipt = cosmos_droid.build_droid_service(
        tmp_path, precision="fp8", format_prompt_as_json=False, model_finalizer=finish)
    assert called == [service]
    assert receipt["recipe_id"] == "owned-sm89-test"
    assert service.args.num_steps == 4
    assert service.args.guidance == 3.0
    assert service.args.action_chunk_size == 32
    assert service.args.conditioning_fps == 15
    assert service.args.format_prompt_as_json is False
    assert service.setup["use_torch_compile"] is False


@pytest.mark.parametrize("capability,prefix", [((8, 9), "sm89"), ((12, 0), "sm120")])
@pytest.mark.parametrize("precision", ["native", "fp8"])
def test_finalizer_binds_cpu_storage_to_actual_desktop_recipe(monkeypatch, capability, prefix, precision):
    from cosmos3_iwm.sm89_residency import INFERENCE_RESERVE_BYTES, finalize_service

    from instinctflash.runtime import module_residency, sm89_fp8, sm120_fp8

    class MoTDecoderLayer(torch.nn.Module):
        pass

    monkeypatch.setitem(sys.modules, "cosmos_framework.model.generator.mot.unified_mot",
                        SimpleNamespace(MoTDecoderLayer=MoTDecoderLayer))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    layers = [MoTDecoderLayer()]
    net = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))
    model = SimpleNamespace(net=net, config=SimpleNamespace(
        compile=SimpleNamespace(enabled=False, use_cuda_graphs=False), lora_enabled=False))
    service, construction = SimpleNamespace(model=model), SimpleNamespace(verify=Mock())
    installers = {"sm89": Mock(return_value={"executor": "sm89_torch_fp8"}),
                  "sm120": Mock(return_value={"executor": "sm120_torch_fp8"})}
    monkeypatch.setattr(sm89_fp8, "install_sm89_fp8", installers["sm89"])
    monkeypatch.setattr(sm120_fp8, "install_sm120_fp8", installers["sm120"])
    owner = SimpleNamespace(receipt={})
    placement = Mock(return_value=owner)
    monkeypatch.setattr(module_residency, "install_module_residency", placement)
    result = finalize_service(service, construction, precision=precision, device="cuda:0")
    construction.verify.assert_called_once_with(model)
    placement.assert_called_once_with(net, layers, device="cuda:0", reserve_bytes=INFERENCE_RESERVE_BYTES)
    assert service._ifl_sm89_residency is construction.residency is owner
    assert owner.receipt["checkpoint_materialization"] == "cpu"
    if precision == "fp8":
        installers[prefix].assert_called_once_with(model, "cosmos3_policy", device="cuda:0",
                                                   storage_device="cpu", include_mlp=True)
        assert result["executor"] == prefix + "_torch_fp8"
    else:
        assert result is None
        installers[prefix].assert_not_called()
    installers["sm89" if prefix == "sm120" else "sm120"].assert_not_called()

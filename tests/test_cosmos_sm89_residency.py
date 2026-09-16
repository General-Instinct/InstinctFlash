import sys
import threading
from pathlib import Path
from types import SimpleNamespace

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

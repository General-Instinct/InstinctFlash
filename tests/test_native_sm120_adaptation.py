"""Native SM120 policy guards; no experimental GEMM is installed here."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace, ModuleType
import json

import pytest
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
for directory in ("groot_n17", "lingbot_vla_v2", "dreamzero"):
    sys.path.insert(0, str(ROOT / "examples" / directory))

from groot_n17_iwm.adapter import _default_gpu_collate
from lingbot_vla_v2_iwm.sm120_native import NativeSDPA, selected as select_sdpa
from dreamzero_iwm.sm120_native import Stage, selected as select_residency
from dreamzero_iwm.sm120_native import load_streamed_native, prepare_streaming_view
from instinctflash.planners.planner import Tier


def test_existing_gemm_experiment_is_not_selected_or_changed_by_native_profiles():
    experiment = SimpleNamespace(resolved_gemm_backend="sm120_tma_experimental")
    assert not _default_gpu_collate((12, 0), experiment)
    assert not select_sdpa(experiment, (12, 0), "static", enabled=True)
    assert not select_residency((12, 0), experiment)
    assert _default_gpu_collate((12, 0), None)
    assert _default_gpu_collate((11, 0), experiment)
    assert not select_residency((11, 0), None)


def test_sdpa_requires_numeric_permission_and_existing_capture_plan():
    plan = SimpleNamespace(tier_ceiling=Tier.BITEXACT, results=[])
    with pytest.raises(ValueError, match="numeric"):
        select_sdpa(plan, (12, 0), "static", enabled=True)
    plan.tier_ceiling = Tier.NUMERIC
    with pytest.raises(ValueError, match="capture plan"):
        select_sdpa(plan, (12, 0), "static", enabled=True)
    plan.results = [SimpleNamespace(name="graph_capture", applies=True)]
    assert select_sdpa(plan, (12, 0), "static", enabled=True)


def test_sdpa_changes_only_owned_qwen_configuration_and_restores_it():
    config = SimpleNamespace(_attn_implementation="flash_attention_2")
    Attention = type("Qwen3VLVisionAttention", (), {})
    modules = [Attention(), Attention()]
    for module in modules:
        module.config = config
    model = SimpleNamespace(named_modules=lambda: list(enumerate(modules)))
    handle = NativeSDPA(model)
    assert config._attn_implementation == "sdpa"
    assert len(handle.report()["attention_sites"]) == 2
    handle.close()
    handle.close()
    assert config._attn_implementation == "flash_attention_2"


@pytest.mark.parametrize("raise_forward", [False, True])
def test_native_staging_restores_exact_cpu_master_and_releases_on_error(raise_forward):
    class Layer(torch.nn.Linear):
        def forward(self, x):
            if raise_forward:
                raise RuntimeError("test failure")
            return super().forward(x)
    module = Layer(16, 8).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in module.state_dict().items()}
    owner = SimpleNamespace(device=torch.device("cpu"), active=None, closed=False,
                            lock=threading.RLock(), synchronize=lambda: None)
    stage = Stage(owner, module, "test")
    x = torch.randn(2, 16)
    with torch.inference_mode():
        if raise_forward:
            with pytest.raises(RuntimeError, match="test failure"):
                module(x)
        else:
            assert torch.equal(module(x), torch.nn.functional.linear(x, before["weight"], before["bias"]))
    assert owner.active is None and stage.calls == 1
    assert all(torch.equal(v, before[k]) for k, v in module.state_dict().items())
    stage.close()
    stage.close()
    assert not module._forward_hooks and not module._forward_pre_hooks


def test_streamed_checkpoint_assigns_all_values_without_meta_constants(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    class Toy(torch.nn.Module):
        config_class = SimpleNamespace(from_dict=lambda config: config)
        def __init__(self, config):
            super().__init__()
            self.proj = torch.nn.Linear(4, 3)
            self.register_buffer("buffer", torch.ones(1))
            self.native_constant = torch.tensor([.125], dtype=torch.float32)
    native = ModuleType("groot.vla.model.dreamzero.base_vla")
    native.VLA = Toy
    monkeypatch.setitem(sys.modules, native.__name__, native)
    values = {"proj.weight": torch.randn(3, 4).bfloat16(),
              "proj.bias": torch.randn(3).bfloat16(), "buffer": torch.full((1,), 2., dtype=torch.bfloat16)}
    save_file({"proj.weight": values["proj.weight"]}, str(tmp_path / "first.safetensors"))
    save_file({k:v for k,v in values.items() if k != "proj.weight"}, str(tmp_path / "last.safetensors"))
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        k: "first.safetensors" if k == "proj.weight" else "last.safetensors" for k in values}}))
    model = load_streamed_native(tmp_path)
    assert model._instinctflash_streamed_tensors == 3
    assert all(torch.equal(value, values[name]) for name, value in model.state_dict().items())
    assert model.native_constant.device.type == "cpu" and model.native_constant.item() == .125


def test_streaming_view_keeps_original_experiment_and_config_unchanged(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    text = "model:\n  _target_: groot.vla.model.dreamzero.base_vla.VLA\nseed: 1140\n"
    (original / "conf.yaml").write_text(text)
    (original / "metadata.json").write_text("{}")
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "experiment_cfg").symlink_to(original, target_is_directory=True)
    (owned / "config.json").write_text(json.dumps({"action_head_cfg": {"config": {
        "diffusion_model_cfg": {"_target_": "loader", "checkpoint_path": "original", "dim":5120}}}}))
    prepare_streaming_view(owned)
    assert (original / "conf.yaml").read_text() == text
    assert (owned / "experiment_cfg" / "metadata.json").resolve() == original / "metadata.json"
    assert "StreamingVLA" in (owned / "experiment_cfg" / "conf.yaml").read_text()
    assert "checkpoint_path" not in json.loads((owned / "config.json").read_text())["action_head_cfg"]["config"]["diffusion_model_cfg"]

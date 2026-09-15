"""Dense MoT extension excludes vision, sparse experts and custom forwards."""
import sys
from types import ModuleType

import pytest
import torch
from instinctflash.runtime import cosmos_fp8


@pytest.fixture
def model_types(monkeypatch):
    class Dense(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = torch.nn.Linear(16, 32, dtype=torch.bfloat16)
            self.down_proj = torch.nn.Linear(32, 16, dtype=torch.bfloat16)
            self.act_fn = torch.nn.ReLU()
    class Gated(Dense):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(16, 32, dtype=torch.bfloat16)
    class Layer(torch.nn.Module):
        def __init__(self, cls):
            super().__init__()
            self.mlp = cls()
            self.mlp_moe_gen = cls()
    stub = ModuleType('cosmos_framework.model.generator.mot.unified_mot')
    stub.MoTDecoderLayer, stub.Qwen3VLTextMLP, stub.Nemotron3DenseVLMLP = Layer, Gated, Dense
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    return Layer, Gated, Dense


def test_both_dense_architectures_and_towers_only(model_types, monkeypatch):
    Layer, Gated, Dense = model_types
    class Packed(torch.nn.Identity):
        recipe = 'test'
        def __init__(self, source):
            super().__init__()
    monkeypatch.setattr(cosmos_fp8, 'ThorFP8Linear', Packed)
    model = torch.nn.Sequential(Layer(Gated), Layer(Dense), Gated())
    vision_up = model[2].up_proj
    activation = model[0].mlp.act_fn
    receipt = cosmos_fp8.install_cosmos_dense_mlp_fp8(model)
    assert receipt['mlp_modules'] == 4 and len(receipt['projections']) == 10
    assert model[2].up_proj is vision_up and model[0].mlp.act_fn is activation
    assert isinstance(model[1].mlp_moe_gen.down_proj, Packed)


def test_late_pack_failure_is_atomic(model_types, monkeypatch):
    Layer, Gated, _ = model_types
    model = Layer(Gated)
    before = dict(model.named_modules())
    count = 0
    def pack(source):
        nonlocal count
        count += 1
        if count == 5:
            raise ValueError('allocation failed')
        return torch.nn.Identity()
    monkeypatch.setattr(cosmos_fp8, 'ThorFP8Linear', pack)
    with pytest.raises(ValueError, match='allocation failed'):
        cosmos_fp8.install_cosmos_dense_mlp_fp8(model)
    assert dict(model.named_modules()) == before


def test_custom_and_hooked_mlp_refused_before_pack(model_types, monkeypatch):
    Layer, Gated, _ = model_types
    class Custom(Gated):
        pass
    monkeypatch.setattr(cosmos_fp8, 'ThorFP8Linear', lambda _: pytest.fail('unexpected packing'))
    with pytest.raises(ValueError, match='Unsupported'):
        cosmos_fp8.install_cosmos_dense_mlp_fp8(Layer(Custom))
    model = Layer(Gated)
    model.mlp_moe_gen.up_proj.register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match='hooked'):
        cosmos_fp8.install_cosmos_dense_mlp_fp8(model)

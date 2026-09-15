import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from instinctflash.runtime import dreamzero_fp8


@pytest.fixture
def head(monkeypatch):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ('q', 'k', 'v', 'o'):
                setattr(self, name, torch.nn.Linear(16, 16, dtype=torch.bfloat16))
    module = ModuleType('groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk')
    module.CausalWanSelfAttention = Attention
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv('ENABLE_TENSORRT', 'false')
    return SimpleNamespace(model=torch.nn.Sequential(Attention(), Attention()),
                           cpu_offload=False, num_inference_timesteps=4,
                           num_inference_steps=16,
                           dit_step_mask=[True]*8+[False]*8)


def test_projection_install_preserves_native_output_and_schedule(head, monkeypatch):
    class Packed(torch.nn.Identity):
        recipe = 'test_only'
        def __init__(self, source):
            super().__init__()
    monkeypatch.setattr(dreamzero_fp8, 'ThorFP8Linear', Packed)
    outputs = [layer.o for layer in head.model]
    mask = head.dit_step_mask
    receipt = dreamzero_fp8.install_dreamzero_fp8(head)
    assert head.dit_step_mask is mask and head.num_inference_timesteps == 4
    assert receipt['scheduler_steps'] == head.num_inference_steps == 16
    assert receipt['dit_step_mask'] == mask and receipt['attention_layers'] == 2
    for index, layer in enumerate(head.model):
        assert layer.o is outputs[index]
        assert all(isinstance(getattr(layer, n), Packed) for n in ('q', 'k', 'v'))


def test_late_pack_error_does_not_modify_model(head, monkeypatch):
    original = dict(head.model.named_modules())
    calls = []
    def pack(source):
        calls.append(source)
        if len(calls) == 5:
            raise ValueError('bad weight')
        return torch.nn.Identity()
    monkeypatch.setattr(dreamzero_fp8, 'ThorFP8Linear', pack)
    with pytest.raises(ValueError, match='bad weight'):
        dreamzero_fp8.install_dreamzero_fp8(head)
    assert dict(head.model.named_modules()) == original


def test_offload_and_trt_bypass_rejected_before_packing(head, monkeypatch):
    monkeypatch.setattr(dreamzero_fp8, 'ThorFP8Linear', lambda _: pytest.fail('packing reached'))
    head.cpu_offload = True
    with pytest.raises(ValueError, match='GPU-resident'):
        dreamzero_fp8.install_dreamzero_fp8(head)
    head.cpu_offload = False
    monkeypatch.setenv('ENABLE_TENSORRT', 'true')
    with pytest.raises(ValueError, match='TensorRT'):
        dreamzero_fp8.install_dreamzero_fp8(head)


def test_extended_ffn_recipe_is_separate_and_preserves_activation(head, monkeypatch):
    module = sys.modules['groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk']
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = module.CausalWanSelfAttention()
            self.ffn = torch.nn.Sequential(torch.nn.Linear(16, 32, dtype=torch.bfloat16),
                                           torch.nn.GELU(), torch.nn.Linear(32, 16, dtype=torch.bfloat16))
    module.CausalWanAttentionBlock = Block
    head.model = torch.nn.Sequential(Block(), Block())
    class Packed(torch.nn.Identity):
        recipe = 'test'
        def __init__(self, source): super().__init__()
    monkeypatch.setattr(dreamzero_fp8, 'ThorFP8Linear', Packed)
    activation, output = head.model[0].ffn[1], head.model[0].self_attn.o
    receipt = dreamzero_fp8.install_dreamzero_fp8(head, include_ffn=True)
    assert len(receipt['projections']) == 10 and receipt['attention_layers'] == 2
    assert 'qkv_ffn' in receipt['recipe'] and receipt['quality_certificate'] is None
    assert head.model[0].ffn[1] is activation and head.model[0].self_attn.o is output

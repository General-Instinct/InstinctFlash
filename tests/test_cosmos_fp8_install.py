"""Installer mutation safety; actual Thor arithmetic has a separate GPU probe."""
import sys
from types import ModuleType

import pytest
import torch

from instinctflash.runtime import cosmos_fp8


@pytest.fixture
def attention_type(monkeypatch):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in cosmos_fp8.PROJECTIONS:
                setattr(self, name, torch.nn.Linear(16, 16, dtype=torch.bfloat16))
            self.o_proj = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
            self.o_proj_moe_gen = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
    stub = ModuleType('cosmos_framework.model.generator.mot.unified_mot')
    stub.PackedAttentionMoT = Attention
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    return Attention


def test_pack_failure_leaves_all_original_modules(attention_type, monkeypatch):
    model = torch.nn.Sequential(attention_type(), attention_type())
    original = dict(model.named_modules())
    calls = []
    def pack(source):
        calls.append(source)
        if len(calls) == 8:
            raise ValueError('bad later layer')
        return torch.nn.Identity()
    monkeypatch.setattr(cosmos_fp8, 'ThorFP8Linear', pack)
    with pytest.raises(ValueError, match='bad later layer'):
        cosmos_fp8.install_cosmos_fp8(model)
    assert dict(model.named_modules()) == original


def test_only_audited_qkv_replaced(attention_type, monkeypatch):
    class Packed(torch.nn.Identity):
        recipe = 'test_only'
        def __init__(self, source):
            super().__init__()
    monkeypatch.setattr(cosmos_fp8, 'ThorFP8Linear', Packed)
    model = torch.nn.Sequential(attention_type())
    output = model[0].o_proj
    gen_output = model[0].o_proj_moe_gen
    receipt = cosmos_fp8.install_cosmos_fp8(model)
    assert receipt['attention_layers'] == 1
    assert len(receipt['projections']) == 6
    assert all(isinstance(getattr(model[0], n), Packed) for n in cosmos_fp8.PROJECTIONS)
    assert model[0].o_proj is output and model[0].o_proj_moe_gen is gen_output


def test_custom_attention_and_hooked_projections_refused(attention_type):
    class Custom(attention_type):
        pass
    with pytest.raises(ValueError, match='No audited'):
        cosmos_fp8.install_cosmos_fp8(Custom())
    model = attention_type()
    model.v_proj.register_forward_hook(lambda *args: None)
    original = dict(model.named_modules())
    with pytest.raises(ValueError, match='hooked'):
        cosmos_fp8.install_cosmos_fp8(model)
    assert dict(model.named_modules()) == original

import sys
from types import ModuleType
import pytest
import torch
from instinctflash.runtime import groot_fp8


def test_text_only_atomic_install(monkeypatch):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                setattr(self, n, torch.nn.Linear(16, 16, dtype=torch.bfloat16))
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for n in ('gate_proj', 'up_proj', 'down_proj'):
                setattr(self, n, torch.nn.Linear(16, 16, dtype=torch.bfloat16))
    stub = ModuleType('transformers.models.qwen3_vl.modeling_qwen3_vl')
    stub.Qwen3VLTextAttention, stub.Qwen3VLTextMLP = Attention, MLP
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    class Vision(Attention): pass
    model = torch.nn.Sequential(Attention(), MLP(), Vision())
    before = dict(model.named_modules())
    count = 0
    def fail(source):
        nonlocal count
        count += 1
        if count == 7: raise ValueError('bad final weight')
        return torch.nn.Identity()
    monkeypatch.setattr(groot_fp8, 'ThorFP8Linear', fail)
    with pytest.raises(ValueError, match='bad final weight'):
        groot_fp8.install_groot_backbone_fp8(model)
    assert before == dict(model.named_modules())
    class Packed(torch.nn.Identity):
        recipe = 'test'
        def __init__(self, source): super().__init__()
        def forward(self, x): return x * 1
    monkeypatch.setattr(groot_fp8, 'ThorFP8Linear', Packed)
    receipt = groot_fp8.install_groot_backbone_fp8(model)
    assert len(receipt['projections']) == 7 and receipt['quality_certificate'] is None
    assert model[2].q_proj is before['2.q_proj']
    assert isinstance(model[0].o_proj, Packed)
    if torch.cuda.is_available():
        projection = model[0].q_proj
        first = projection(torch.ones(4, 16, device='cuda'))
        second = projection(torch.full((4, 16), 2., device='cuda'))
        assert torch.equal(first, torch.ones_like(first))
        assert torch.equal(second, torch.full_like(second, 2.))
        assert first.data_ptr() != second.data_ptr()
        for graph in model._instinctflash_fp8_graphs: graph.close()

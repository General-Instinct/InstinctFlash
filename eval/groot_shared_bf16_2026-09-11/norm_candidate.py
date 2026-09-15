"""Offline GR00T norm-graph ablation; never installed by the public Runtime."""
import hashlib
import inspect
import torch
from instinctflash.backends.bf16_pointwise import norm
from instinctflash.runtime.static_tensor_graph import StaticTensorGraph


def install(model, *, fused):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm
    expected = '0d702ef8d7d8a1290c99ec04e3727f499f7e4268a7c999e44db99853f6e45e10'
    assert hashlib.sha256(inspect.getsource(Qwen3VLTextRMSNorm.forward).strip().encode()).hexdigest() == expected
    assert torch.cuda.get_device_capability() == (11, 0)
    graphs = []
    for path, module in model.named_modules():
        if type(module) is not Qwen3VLTextRMSNorm or module.weight.shape != (2048,):
            continue
        assert module.weight.dtype == torch.bfloat16
        assert not module.training
        assert not module._forward_hooks and not module._forward_pre_hooks
        assert 'forward' not in vars(module), 'Already replaced norm'
        original = module.forward
        def candidate(x, m=module):
            return norm(x, m.weight, m.variance_epsilon, True)
        graph = StaticTensorGraph(candidate if fused else original, reference=original, name=path)
        def forward(x, g=graph):
            return g(x).clone()
        module.forward = forward
        graphs.append(graph)
    assert graphs, 'No eligible norm modules'
    return graphs

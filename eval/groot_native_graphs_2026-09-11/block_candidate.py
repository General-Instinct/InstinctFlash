"""Offline native BF16 Norm/MLP/residual graph ablation; not a Runtime default."""
import hashlib
import inspect
import torch
from instinctflash.backends.bf16_pointwise import norm, silu_table, swiglu
from instinctflash.runtime.static_tensor_graph import StaticTensorGraph


def install(model, *, fused):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextMLP, Qwen3VLTextRMSNorm
    hashes = {
        Qwen3VLTextDecoderLayer: 'a22b893bb95a805fdb6ecca92486d98b5515d04530ca1a27305c6116adc8b6d6',
        Qwen3VLTextMLP: '3997a4458faa14a51c930437acc7682f55e233831383020675ba9dd3ea17fd8d',
        Qwen3VLTextRMSNorm: '0d702ef8d7d8a1290c99ec04e3727f499f7e4268a7c999e44db99853f6e45e10',
    }
    for cls, expected in hashes.items():
        assert hashlib.sha256(inspect.getsource(cls.forward).strip().encode()).hexdigest() == expected
    assert torch.cuda.get_device_capability() == (11, 0)
    graphs = []
    table = silu_table(next(model.parameters()).device) if fused else None
    for path, layer in model.named_modules():
        if type(layer) is not Qwen3VLTextDecoderLayer:
            continue
        assert type(layer.mlp) is Qwen3VLTextMLP and layer.mlp.config.hidden_act == 'silu'
        assert type(layer.post_attention_layernorm) is Qwen3VLTextRMSNorm
        assert layer.post_attention_layernorm.weight.shape == (2048,)
        for module in layer.modules():
            assert not module.training and not module._forward_hooks and not module._forward_pre_hooks
            assert 'forward' not in vars(module), 'Already patched layer component'
        assert all(p.dtype == torch.bfloat16 for p in layer.parameters())
        def reference(x, m=layer):
            return x + m.mlp(m.post_attention_layernorm(x))
        def candidate(x, m=layer, lut=table):
            n = m.post_attention_layernorm
            y = norm(x, n.weight, n.variance_epsilon, True)
            f = m.mlp
            return x + f.down_proj(swiglu(f.gate_proj(y), f.up_proj(y), lut))
        graph = StaticTensorGraph(candidate if fused else reference, reference=reference, name=path)
        original = layer.forward
        def forward(hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                    past_key_values=None, use_cache=False, cache_position=None,
                    _m=layer, _g=graph, _original=original, **kwargs):
            if _m.training or torch.is_grad_enabled() or 'past_key_value' in kwargs:
                return _original(hidden_states, position_embeddings, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache,
                    cache_position=cache_position, **kwargs)
            residual = hidden_states
            hidden_states = _m.input_layernorm(hidden_states)
            hidden_states, _ = _m.self_attn(hidden_states=hidden_states,
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
            return _g(residual + hidden_states).clone()
        layer.forward = forward
        graphs.append(graph)
    assert graphs, 'No eligible decoder layers'
    return graphs

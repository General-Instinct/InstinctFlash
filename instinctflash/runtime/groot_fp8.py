"""Audited GR00T Qwen3 text projections; native vision and processors retained."""
from .torch_fp8_linear import ThorFP8Linear


def install_groot_backbone_fp8(backbone):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention, Qwen3VLTextMLP
    layouts = {Qwen3VLTextAttention: ('q_proj', 'k_proj', 'v_proj', 'o_proj'),
               Qwen3VLTextMLP: ('gate_proj', 'up_proj', 'down_proj')}
    targets = []
    for path, module in backbone.named_modules():
        for name in layouts.get(type(module), ()):
            source = getattr(module, name)
            if source._forward_hooks or source._forward_pre_hooks or source._backward_hooks:
                raise ValueError(f'Cannot replace hooked GR00T projection {path}.{name}')
            targets.append((path, module, name, source))
    if not targets:
        raise ValueError('No audited Qwen3 text projections found in GR00T backbone')
    packed = [ThorFP8Linear(source) for _, _, _, source in targets]
    receipt = dict(recipe='groot_qwen3_text_' + ThorFP8Linear.recipe,
                   projections=[dict(path=f'{path}.{name}', in_features=source.in_features,
                                     out_features=source.out_features)
                                for path, _, name, source in targets],
                   scope='Qwen3 text attention and MLP, plus existing FP8 VLSA; native vision and BF16 DiT',
                   quality_certificate=None)
    for (_, parent, name, _), replacement in zip(targets, packed):
        setattr(parent, name, replacement)
    # Fixed prompt/image geometry makes these stateless boundaries reusable.
    # Capture each text MLP together; attention projections retain their native
    # surrounding attention and cache logic. Return owned tensors: an upstream
    # cache must never retain a borrowed buffer overwritten on the next replay.
    from .static_tensor_graph import StaticTensorGraph
    graphs = []
    def install_graph(module, name):
        graph = StaticTensorGraph(module.forward, name=name)
        def forward(x):
            return graph(x).clone()
        module.forward = forward
        graphs.append(graph)
    for path, module in list(backbone.named_modules()):
        if type(module) is Qwen3VLTextMLP:
            install_graph(module, 'groot_text_mlp:' + path)
        elif type(module) is Qwen3VLTextAttention:
            for name in layouts[Qwen3VLTextAttention]:
                install_graph(getattr(module, name), 'groot_text_attention:' + path + '.' + name)
    backbone._instinctflash_fp8_graphs = graphs
    receipt['graph_boundaries'] = len(graphs)
    receipt['graph_admission'] = 'component eager/graph action-independent byte checks; owned output copies'
    return receipt

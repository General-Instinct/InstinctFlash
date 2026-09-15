"""Read VLA2 routed-expert weights directly from safetensors, one layer at a time."""
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path

MOE_PREFIX = 'model.qwenvl_with_expert.qwen_expert.model.layers.'
MOE_LAYERS = 36
MOE_SHAPES = {
    'gate.weight': (32,768),
    'e_score_correction_bias': (32,),
    'experts.gate_proj': (32,512,768),
    'experts.up_proj': (32,512,768),
    'experts.down_proj': (32,768,512),
}


@contextmanager
def checkpoint_moe_layers(checkpoint):
    """Validate all expert geometry from headers before allocating engine buffers.

    Tensor materialization is lazy so checkpoint-sized FP32 CPU copies are not
    retained. The consumer quantizes directly from source precision, avoiding
    the extra rounding in from_frontend's per-expert-to-shared conversion.
    """
    from safetensors import safe_open
    root = Path(checkpoint)
    index = root/'model.safetensors.index.json'
    weight_map = json.loads(index.read_text())['weight_map'] if index.exists() else None
    with ExitStack() as stack:
        opened = {}
        def reader(key):
            filename = weight_map[key] if weight_map is not None else 'model.safetensors'
            if filename not in opened:
                opened[filename] = stack.enter_context(safe_open(str(root/filename),framework='pt',device='cpu'))
            return opened[filename]
        for layer in range(MOE_LAYERS):
            for suffix, shape in MOE_SHAPES.items():
                key = f'{MOE_PREFIX}{layer}.mlp.{suffix}'
                try:
                    actual = tuple(reader(key).get_slice(key).get_shape())
                except (KeyError, RuntimeError) as e:
                    raise ValueError(f'VLA2 checkpoint is missing required MoE tensor {key}') from e
                if actual != shape:
                    raise ValueError(f'VLA2 MoE tensor {key} has shape {actual}, expected {shape}')

        class Layers:
            def __len__(self):return MOE_LAYERS
            def __iter__(self):
                for layer in range(MOE_LAYERS):
                    yield {suffix:reader(f'{MOE_PREFIX}{layer}.mlp.{suffix}').get_tensor(
                        f'{MOE_PREFIX}{layer}.mlp.{suffix}') for suffix in MOE_SHAPES}
        yield Layers()

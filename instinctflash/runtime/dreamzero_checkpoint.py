"""Validate full DreamZero DiT coverage before skipping redundant base-DiT loading.

The default temporary view changes only the native skip_component_loading flag.
The optional direct-BF16 candidate also replaces discarded DiT initialization with
exact checkpoint assignment. Both retain T5/CLIP/VAE and final checkpoint loading.
The adapter uses direct BF16 loading for the audited full architecture after
actual Thor loaded-value and FP8 history checks. Public-interface qualification
and the native paired comparison remain separate from these loading checks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile


DIT_TARGET = 'groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk.CausalWanModel'


def read_tensor_headers(root):
    root = Path(root)
    index = root / 'model.safetensors.index.json'
    weight_map = json.loads(index.read_text())['weight_map'] if index.exists() else None
    shards = sorted(set(weight_map.values())) if weight_map else ['model.safetensors']
    headers = {}
    for shard in shards:
        path = root / shard
        if Path(shard).name != shard:
            raise ValueError('DreamZero checkpoint shards must be root-level files')
        with path.open('rb') as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError('Incomplete safetensors header')
            length = struct.unpack('<Q', length_bytes)[0]
            if length > 100_000_000:
                raise ValueError('Oversized safetensors header')
            metadata = json.loads(stream.read(length))
        for name, info in metadata.items():
            if name == '__metadata__':
                continue
            if name in headers:
                raise ValueError(f'Duplicate checkpoint tensor {name}')
            if weight_map is not None and weight_map.get(name) != shard:
                raise ValueError(f'Checkpoint index disagrees with shard for {name}')
            headers[name] = {'shape': info['shape'], 'dtype': info['dtype']}
    if weight_map is not None and set(weight_map) != set(headers):
        raise ValueError('Checkpoint index contains tensors absent from shard headers')
    return headers


def _validated_meta_dit(config, headers, model_class):
    import torch

    head = config['action_head_cfg']['config']
    dit = head['diffusion_model_cfg']
    if head.get('train_architecture') != 'full' or dit.get('_target_') != DIT_TARGET:
        raise ValueError('Redundant DiT loading may only be skipped for the audited full architecture')
    # Meta construction exercises the actual native module hierarchy without
    # allocating its full weights. Preserve the caller's CPU RNG state.
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        model = model_class(**{k: v for k, v in dit.items() if not k.startswith('_')})
    expected = {k: list(v.shape) for k, v in model.state_dict().items()}
    prefix = 'action_head.model.'
    actual = {k[len(prefix):]: value for k, value in headers.items() if k.startswith(prefix)}
    if not expected or set(expected) != set(actual):
        raise ValueError('Full checkpoint does not exactly cover the native DiT state')
    for name, shape in expected.items():
        if actual[name]['shape'] != shape or actual[name]['dtype'] != 'BF16':
            raise ValueError(f'DiT checkpoint shape/dtype mismatch for {name}')
    return model, {'verified_dit_tensors': len(expected), 'expected_shapes': expected}


def validate_dit_coverage(config, headers, model_class):
    return _validated_meta_dit(config, headers, model_class)[1]


def load_full_dit_bf16(root):
    """CPU loader avoiding a discarded full FP32 DiT allocation and conversion.

    It preserves stored tensor values and native RoPE construction. The audited
    full architecture uses it through the adapter's owned checkpoint view.
    It does not reproduce the discarded random initialization's RNG consumption.
    """
    import inspect
    import torch
    from safetensors import safe_open
    from groot.vla.model.dreamzero.modules import wan_video_dit_action_casual_chunk as native

    # The unregistered RoPE tensors below follow this audited constructor. A new
    # native implementation must be rechecked rather than guessed from names.
    source_hash = hashlib.sha256(Path(inspect.getfile(native.CausalWanModel)).read_bytes()).hexdigest()
    if source_hash != 'efc5120f73cbce2e00b67d78e8bf71a561e943a909791f039b435abc27ccd605':
        raise ValueError('DreamZero native constructor changed; requalify direct BF16 loading')
    root = Path(root).resolve()
    config = json.loads((root / 'config.json').read_text())
    headers = read_tensor_headers(root)
    model, receipt = _validated_meta_dit(config, headers, native.CausalWanModel)
    index = root / 'model.safetensors.index.json'
    shards = (sorted(set(json.loads(index.read_text())['weight_map'].values()))
              if index.exists() else ['model.safetensors'])
    weights = {}
    prefix = 'action_head.model.'
    for shard in shards:
        with safe_open(str(root / shard), framework='pt', device='cpu') as file:
            for key in file.keys():
                if key.startswith(prefix):
                    weights[key[len(prefix):]] = file.get_tensor(key)
    model.load_state_dict(weights, strict=True, assign=True)
    # The native constructor intentionally does not register these as buffers,
    # so load_state_dict cannot materialize them from the meta device.
    d = model.dim // model.num_heads
    with torch.device('cpu'):
        model.freqs_action = native.rope_params(1024 * 10, d)
        model.freqs_state = native.rope_params(1024, d)
        model.freqs = [native.rope_params(1024, d - 4 * (d // 6)),
                       native.rope_params(1024, 2 * (d // 6)),
                       native.rope_params(1024, 2 * (d // 6))]

    def has_meta(value):
        if isinstance(value, torch.Tensor):
            return value.is_meta
        if isinstance(value, (list, tuple)):
            return any(has_meta(item) for item in value)
        if isinstance(value, dict):
            return any(has_meta(item) for item in value.values())
        return False

    for module in model.modules():
        if any(has_meta(value) for value in vars(module).values()):
            raise RuntimeError('DreamZero direct load left an unmaterialized meta tensor')
    receipt.update(original_checkpoint=str(root), native_source_sha256=source_hash,
                   parameter_dtype='bfloat16',
                   scope='Candidate DiT construction only; native policy, CPU RNG consumption and Thor inference remain separately qualified')
    return model, receipt


def load_full_dit_for_native(checkpoint_path, **kwargs):
    """Hydra factory for the candidate view; returns the original native class."""
    config = json.loads((Path(checkpoint_path) / 'config.json').read_text())
    expected = config['action_head_cfg']['config']['diffusion_model_cfg']
    expected = {k: v for k, v in expected.items() if not k.startswith('_')}
    if kwargs != expected:
        raise ValueError('DreamZero direct-load view disagrees with the original DiT configuration')
    model, _ = load_full_dit_bf16(checkpoint_path)
    return model


def prepare_full_checkpoint(root, model_class, *, direct_bf16=False):
    """Return an owned TemporaryDirectory and validation receipt; caller cleans up.

    This validates state coverage, not the final loaded values. The first actual
    use must additionally compare loaded DiT tensors with the full checkpoint.
    """
    root = Path(root).resolve()
    config_bytes = (root / 'config.json').read_bytes()
    config = json.loads(config_bytes)
    headers = read_tensor_headers(root)
    receipt = validate_dit_coverage(config, headers, model_class)
    config['action_head_cfg']['config']['skip_component_loading'] = True
    if direct_bf16:
        dit = config['action_head_cfg']['config']['diffusion_model_cfg']
        dit['_target_'] = 'instinctflash.runtime.dreamzero_checkpoint.load_full_dit_for_native'
        dit['checkpoint_path'] = str(root)
    view = tempfile.TemporaryDirectory(prefix='instinctflash-dreamzero-full-')
    try:
        for item in root.iterdir():
            if item.name != 'config.json':
                (Path(view.name) / item.name).symlink_to(item, target_is_directory=item.is_dir())
        (Path(view.name) / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    except Exception:
        view.cleanup()
        raise
    receipt.update(original_checkpoint=str(root), original_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                   direct_bf16_dit=bool(direct_bf16),
                   scope=('Candidate direct BF16 DiT assignment; initialization RNG consumption differs; '
                          'retain frozen components and final checkpoint loading' if direct_bf16 else
                          'Skip base DiT preload only after exact full-checkpoint state coverage; '
                          'retain native initialization, frozen components and final checkpoint loading'))
    return view, receipt

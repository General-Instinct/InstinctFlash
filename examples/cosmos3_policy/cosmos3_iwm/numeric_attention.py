"""Instance-local, explicitly permitted BF16 cuDNN attention for Thor DROID.

This is a numerical transformation, not a task-quality certificate. The narrow
backend route is screened separately from the vendor's general cuDNN admission.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import FunctionType


def _clone(function, replacements):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class NumericAttention:
    def __init__(self, module, owners):
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        self.stats = dict(backend='cudnn', precision='native', dtype='bfloat16',
                          transformation='NUMERIC', quality_status='screen',
                          task_quality_certified=False, eligible_python_calls=0,
                          fallback_python_calls=0, cudnn_version=torch.backends.cudnn.version())
        original = module.attention

        def attention(q, k, v, *args, **kwargs):
            eligible = (not args and set(kwargs).issubset({'is_causal', 'return_lse'})
                        and not kwargs.get('is_causal', False) and not kwargs.get('return_lse', False)
                        and q.ndim == k.ndim == v.ndim == 4
                        and q.shape[0] == k.shape[0] == v.shape[0] == 1
                        and q.dtype == k.dtype == v.dtype == torch.bfloat16
                        and q.device == k.device == v.device and q.is_cuda
                        and k.shape == v.shape and q.shape[-1] == k.shape[-1] == 128
                        and k.shape[2] > 0 and q.shape[2] % k.shape[2] == 0)
            if not eligible:
                self.stats['fallback_python_calls'] += 1
                return original(q, k, v, *args, **kwargs)
            self.stats['eligible_python_calls'] += 1
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    dropout_p=0.0, is_causal=False, scale=128 ** -0.5,
                    enable_gqa=True).transpose(1, 2)

        two_way = _clone(module.two_way_attention, {'attention': attention})
        self.dispatch = _clone(module.dispatch_attention, {'two_way_attention': two_way})
        self.patches = []
        try:
            for owner in owners:
                self.patches.append((owner, owner.dispatch_attention_fn))
                owner.dispatch_attention_fn = self.dispatch
        except Exception:
            self.close()
            raise

    def report(self):
        return dict(self.stats, note='Python call counts exclude CUDA graph replays')

    def close(self):
        for owner, original in reversed(self.patches):
            if owner.dispatch_attention_fn is self.dispatch:
                owner.dispatch_attention_fn = original
        self.patches.clear()


def install(service):
    """Fail before patching unsupported stacks; no global dispatch mutation."""
    import torch
    import cosmos_framework
    import cosmos_framework.model.generator.mot.attention as mot
    from .conditioning_cache import SOURCE_HASHES

    existing = getattr(service, '_ifl_numeric_attention', None)
    if existing is not None:
        return existing
    if torch.cuda.get_device_capability() != (11, 0):
        raise ValueError('Cosmos cuDNN numeric attention is qualified on Thor only')
    if torch.backends.cudnn.version() != 91501:
        raise ValueError('Cosmos numeric attention requires screened cuDNN 9.15.1')
    if service.cfg.num_steps != 4 or float(service.cfg.guidance) != 3.0:
        raise ValueError('Cosmos numeric attention requires four steps and CFG 3')
    model = service.model
    config = model.config
    if config.joint_attn_implementation != 'two_way' or config.video_temporal_causal or config.sound_gen:
        raise ValueError('Unsupported numeric attention configuration')
    parallel = model.parallel_dims
    if parallel is not None and any((parallel.cp_enabled, parallel.cfgp_enabled, parallel.dp_shard_enabled)):
        raise ValueError('Numeric attention requires unsharded inference')
    root = Path(cosmos_framework.__file__).parent
    for name, digest in SOURCE_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Unqualified Cosmos source: {name}')
    layers = model.net.language_model.model.layers
    if len(layers) not in (28, 36):
        raise ValueError('Unqualified numeric attention layer count')
    owners = [layer.self_attn for layer in layers]
    if any(owner.dispatch_attention_fn is not mot.dispatch_attention for owner in owners):
        raise ValueError('Numeric attention must be installed before dispatch wrapping')
    for owner in owners:
        if any(p.dtype != torch.bfloat16 or not p.is_cuda for p in owner.parameters()):
            raise ValueError('Numeric attention requires native BF16 CUDA parameters')
    result = NumericAttention(mot, owners)
    service._ifl_numeric_attention = result
    return result

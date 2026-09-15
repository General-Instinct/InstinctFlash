"""Thor pointwise fusion with explicit upstream rounding boundaries.

Mean reduction and rsqrt stay in PyTorch. No GEMM, attention backend, dtype, or schedule
is changed. Install only on a service instance, never patch process-wide classes.
"""
from __future__ import annotations

import os
from pathlib import Path
import inspect
import hashlib

import torch


from instinctflash.backends.bf16_pointwise import norm, relu2, silu_table, swiglu, rope


def _supported(x):
    return (not torch.is_grad_enabled() and x.is_cuda and x.dtype == torch.bfloat16
            and x.is_contiguous() and 0 < x.numel() < 2**31)


def _exact(a, b):
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(_exact(x, y) for x, y in zip(a, b))
    return (a.shape == b.shape and a.dtype == b.dtype and bool(torch.isfinite(a).all())
            and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


class Checked:
    """Check real first-use inputs for every geometry, fall back permanently on mismatch."""
    def __init__(self, original, candidate, eligible, stats, label):
        self.original, self.candidate, self.eligible = original, candidate, eligible
        self.stats, self.label = stats, label
        self.shapes = set()
        self.rejected = False

    def __call__(self, *args, **kwargs):
        try:
            eligible = not self.rejected and self.eligible(*args, **kwargs)
        except TypeError:
            eligible = False
        if not eligible:
            return self.original(*args, **kwargs)
        tensors = args + tuple(kwargs[k] for k in sorted(kwargs))
        key = tuple((tuple(x.shape), tuple(x.stride()), x.dtype, x.device) for x in tensors if isinstance(x, torch.Tensor))
        if key not in self.shapes:
            reference = self.original(*args, **kwargs)
            try:
                candidate = self.candidate(*args, **kwargs)
                exact = _exact(reference, candidate)
            except Exception as error:
                self.stats['errors'].append({'module': self.label, 'error': repr(error)})
                exact = False
            if not exact:
                self.rejected = True
                self.stats['rejected'].append(self.label)
            else:
                self.shapes.add(key)
                self.stats['checks'] += 1
            return reference
        self.stats['calls'] += 1
        return self.candidate(*args, **kwargs)


# Exact upstream function source hashes; unfamiliar implementations remain eager.
SOURCE_HASHES = {'mlp': ['9da004fb9da792dbc670bf8f7b408d502b7fdd5cb7e7e21370431028f168971d',
         '3997a4458faa14a51c930437acc7682f55e233831383020675ba9dd3ea17fd8d'],
 'norm': ['fc4e5410050696e74e3f4af426b47cae66206699d1849a7694cd92fd38ce2179',
          '0d702ef8d7d8a1290c99ec04e3727f499f7e4268a7c999e44db99853f6e45e10',
          'aaf1b08ebcd06e386686615b2adfb24d15f6730fb80485f811e396b75babb737'],
 'rope': ['bba4ebae1137836a461190723f270e01b5fd7cae0efe48e6036d52cbf0ab8f48',
          '94e6bb1e9ea0b34f1ad10c7802514d732c7f87446d73a9fac30745d3e583a30d',
          'a2f1d400ad6b55c81fcf9a2935671ab68d5fa83781127f5a1a3b6698ec65942c']}


def _source_hash(function):
    try:
        return hashlib.sha256(inspect.getsource(function).strip().encode()).hexdigest()
    except (OSError, TypeError):
        return None


def install(service):
    if torch.cuda.get_device_capability() != (11, 0):
        raise ValueError('Cosmos exact pointwise fusion is qualified only on Thor')
    compiler = Path('/usr/local/cuda/bin/ptxas')
    if compiler.is_file():
        os.environ.setdefault('TRITON_PTXAS_PATH', str(compiler))
        os.environ.setdefault('TRITON_PTXAS_BLACKWELL_PATH', str(compiler))
    if hasattr(service, '_ifl_exact_pointwise'):
        return service._ifl_exact_pointwise
    stats = {'installed': [], 'checks': 0, 'calls': 0, 'rejected': [], 'errors': []}
    for name, module in service.model.net.named_modules():
        cls = type(module).__name__
        if cls in ('Nemotron3DenseVLRMSNorm', 'Qwen3VLTextRMSNorm', 'Qwen3VLMoeTextRMSNorm'):
            source = _source_hash(type(module).forward)
            if source not in SOURCE_HASHES.get('norm', []):
                continue
            round_first = cls != 'Nemotron3DenseVLRMSNorm'
            original = module.forward
            def candidate(x, mod=module, round_first=round_first):
                return norm(x, mod.weight, mod.variance_epsilon, round_first)
            def eligible(x, mod=module):
                return _supported(x) and mod.weight.dtype == x.dtype and mod.weight.is_contiguous() and mod.weight.device == x.device
            module.forward = Checked(original, candidate, eligible, stats, name)
            stats['installed'].append(name)
        elif (cls == 'Nemotron3DenseVLMLP' and module.config.mlp_hidden_act == 'relu2'
              and _source_hash(type(module).forward) in SOURCE_HASHES['mlp']):
            if isinstance(module.act_fn, torch.nn.Module):
                module.act_fn.forward = Checked(module.act_fn.forward, relu2, _supported, stats, name + '.relu2')
            else:
                module.act_fn = Checked(module.act_fn, relu2, _supported, stats, name + '.relu2')
            stats['installed'].append(name + '.relu2')
        if (os.environ.get('IFL_COSMOS3_SWIGLU') == '1' and cls == 'Qwen3VLTextMLP'
                and module.config.hidden_act == 'silu'
                and _source_hash(type(module).forward) in SOURCE_HASHES['mlp']):
            def eager(g, u, mod=module):
                return mod.act_fn(g) * u
            if not hasattr(service, '_ifl_silu_lut'):
                service._ifl_silu_lut = silu_table(module.gate_proj.weight.device)
            def fused(g, u, table=service._ifl_silu_lut):
                return swiglu(g, u, table)
            gate = Checked(eager, fused, lambda g, u: _supported(g) and _supported(u)
                           and g.shape == u.shape, stats, name + '.swiglu')
            def forward(x, mod=module, gate=gate):
                return mod.down_proj(gate(mod.gate_proj(x), mod.up_proj(x)))
            module.forward = forward
            stats['installed'].append(name + '.swiglu')
        if hasattr(module, '_apply_rotary_pos_emb'):
            original = module._apply_rotary_pos_emb
            source = _source_hash(original)
            if source not in SOURCE_HASHES.get('rope', []):
                continue
            def candidate(q, k, cos, sin, **kwargs):
                return rope(q, k, cos, sin)
            def eligible(q, k, cos, sin, **kwargs):
                return (kwargs.get('unsqueeze_dim', 1) == 1 and kwargs.get('position_ids') is None
                        and _supported(q) and _supported(k) and q.ndim == k.ndim == 3
                        and cos.ndim == sin.ndim == 2 and cos.shape == sin.shape
                        and cos.dtype == sin.dtype == q.dtype == k.dtype
                        and cos.device == sin.device == q.device == k.device
                        and cos.shape[0] == q.shape[0] == k.shape[0]
                        and 0 < cos.shape[1] <= q.shape[2] == k.shape[2] and cos.shape[1] % 2 == 0)
            module._apply_rotary_pos_emb = Checked(original, candidate, eligible, stats, name + '.rope')
            stats['installed'].append(name + '.rope')
    service._ifl_exact_pointwise = stats
    return stats

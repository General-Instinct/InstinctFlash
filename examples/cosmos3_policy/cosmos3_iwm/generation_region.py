"""Experimental tensor-only Cosmos GEN blocks; prompt/CFG state stays outside.

Opt-in NUMERIC screen. No checkpoint, sampler, CFG or precision changes. The
existing pinned conditioning-cache admission is required. The extracted eager
block must match native full layer actions on real inputs before compiled use.
"""
from __future__ import annotations

import copy
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from instinctflash.runtime.compiled_region import CompiledRegion


def _original(fn):
    # Exact-pointwise admission wrappers must remain outside compiled regions.
    return getattr(fn, 'original', fn)


class GenerationCompute:
    def __init__(self, layer, *, contiguous_kv=False):
        self.contiguous_kv = contiguous_kv
        self.fused_linear = None
        a = layer.self_attn
        self.heads, self.kv_heads, self.dim = a.num_attention_heads, a.num_key_value_heads, a.head_dim
        self.norm1 = _original(layer.input_layernorm_moe_gen.forward)
        self.norm2 = _original(layer.post_attention_layernorm_moe_gen.forward)
        self.qnorm, self.knorm = _original(a.q_norm_moe_gen.forward), _original(a.k_norm_moe_gen.forward)
        self.q, self.k, self.v, self.out = a.q_proj_moe_gen, a.k_proj_moe_gen, a.v_proj_moe_gen, a.o_proj_moe_gen
        self.rope = _original(a._apply_rotary_pos_emb)
        mlp = layer.mlp_moe_gen
        self.up, self.down = mlp.up_proj, mlp.down_proj
        self.gate = getattr(mlp, 'gate_proj', None)
        activation = mlp.act_fn
        self.activation = _original(activation.forward if isinstance(activation, torch.nn.Module) else activation)

    def __call__(self, hidden, und_k, und_v, cos, sin, causal_indices, full_indices):
        x = self.norm1(hidden)
        q = self.qnorm(self.q(x).view(-1, self.heads, self.dim))
        k = self.knorm(self.k(x).view(-1, self.kv_heads, self.dim))
        v = self.v(x).view(-1, self.kv_heads, self.dim)
        q, k = self.rope(q, k, cos, sin, unsqueeze_dim=1)
        # Preserve the source's exact token presentation order, including any
        # interleaving of understanding/generation tokens. Do not assume cat.
        if self.contiguous_kv:
            keys = torch.cat((und_k, k), dim=0)
            values = torch.cat((und_v, v), dim=0)
        else:
            total = und_k.shape[0] + k.shape[0]
            keys = k.new_zeros((total, self.kv_heads, self.dim))
            values = v.new_zeros((total, self.kv_heads, self.dim))
            keys[causal_indices] = und_k
            keys[full_indices] = k
            values[causal_indices] = und_v
            values[full_indices] = v
        attended = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), keys.transpose(0, 1).unsqueeze(0),
            values.transpose(0, 1).unsqueeze(0), dropout_p=0.0, is_causal=False,
            scale=self.dim ** -0.5, enable_gqa=True)
        residual = hidden + self.out(attended.squeeze(0).transpose(0, 1).reshape(hidden.shape[0], -1))
        x = self.norm2(residual)
        if self.fused_linear is not None:
            activated = self.fused_linear(x)
        else:
            up = self.up(x)
            activated = self.activation(up) if self.gate is None else self.activation(self.gate(x)) * up
        return residual + self.down(activated)


class GenerationRegions:
    def __init__(self, service, plan, *, split_prefill=False, contiguous_kv=False, fused_linear=False):
        self.cache = service._ifl_conditioning_cache
        self.split_prefill = split_prefill
        self.contiguous_kv = contiguous_kv
        self.regions, self.patches = [], []
        self.fused_linears = []
        self.stats = dict(prefills=0, eager_checks=0, compiled_calls=0, rejected=[], closed=False,
                          prefill_checks=0, compiled_prefill_calls=0, layout_checks=0)
        try:
            for index, layer in enumerate(service.model.net.language_model.model.layers):
                compute = GenerationCompute(layer, contiguous_kv=contiguous_kv)
                compiled_compute = compute
                if fused_linear and compute.gate is None:
                    from instinctflash.backends.bf16_linear import BF16LinearActivation
                    config = layer.mlp_moe_gen.config
                    activation_name = getattr(config, 'mlp_hidden_act', getattr(config, 'hidden_act', None))
                    if activation_name == 'relu2':
                        compiled_compute = copy.copy(compute)
                        compiled_compute.fused_linear = BF16LinearActivation(
                            compute.up, compute.activation, activation_name=activation_name, plan=plan)
                        self.fused_linears.append(compiled_compute.fused_linear)
                region = CompiledRegion(compiled_compute, plan=plan, name=f'cosmos.gen.{index}')
                self.regions.append(region)
                self._wrap(layer.self_attn, 'dispatch_attention_fn', self._capture_kv(index))
                from .understanding_prefill import UnderstandingPrefill
                prefill = UnderstandingPrefill(layer) if split_prefill else None
                self._wrap(layer, 'forward', self._forward(index, compute, region, prefill))
        except Exception:
            self.close()
            raise

    def _wrap(self, obj, attr, builder):
        original = getattr(obj, attr)
        replacement = builder(original)
        self.patches.append((obj, attr, original, replacement))
        setattr(obj, attr, replacement)

    def _capture_kv(self, index):
        def wrap(original):
            def dispatch(q, k, v, *args, **kwargs):
                c = self.cache
                if c.active is not None and c.filling and not c.disabled:
                    key_pack = kwargs.get('packed_key_states_normalized')
                    if key_pack is None:
                        key_pack = k
                    n = key_pack['_causal_indices'].shape[0]
                    state = c.active.setdefault('gen_regions', {}).setdefault(index, {})
                    state['key'] = key_pack['causal_seq'][:n].clone()
                    state['value'] = v['causal_seq'][:n].clone()
                    self.stats['prefills'] += 1
                return original(q, k, v, *args, **kwargs)
            return dispatch
        return wrap

    @staticmethod
    def _matches(left, right):
        return (left.shape == right.shape and left.dtype == right.dtype
                and bool(torch.isfinite(left).all())
                and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))

    def _forward(self, index, compute, region, prefill=None):
        def wrap(original):
            def forward(*args, **kwargs):
                c = self.cache
                active = c.active
                if active is None or c.disabled or c.validating or (c.filling and prefill is None):
                    result = original(*args, **kwargs)
                    if active is not None and c.filling and not c.disabled:
                        active['gen_regions'][index]['und_output'] = result[0]['causal_seq'].clone()
                    return result
                if kwargs.get('memory_value') is not None or kwargs.get('natten_metadata') is not None:
                    raise ValueError('GEN regions do not support memory or NATTEN metadata')
                pack = args[0] if args else kwargs['input']
                positions = args[2] if len(args) > 2 else kwargs['packed_position_embeddings']
                if (pack.get('is_sharded', False) or pack['sample_offsets'].shape[0] != 2
                        or pack['full_only_seq'].shape[0] != pack['_full_indices'].shape[0]
                        or (prefill is not None and pack['causal_seq'].shape[0] != pack['_causal_indices'].shape[0])):
                    raise ValueError('GEN regions require one unpadded generation sequence')
                state = active.setdefault('gen_regions', {}).setdefault(index, {})
                if getattr(self, 'contiguous_kv', False):
                    ci, fi = pack['_causal_indices'], pack['_full_indices']
                    expected = active.get('contiguous_kv_indices')
                    if index == 0 or expected is None or ci is not expected[0] or fi is not expected[1]:
                        # Revalidate every denoiser input, not just a shape or
                        # recycled tensor address. Native admission graph outputs
                        # can clone metadata; validate those new objects too.
                        validate_contiguous_layout(ci, fi)
                        active['contiguous_kv_indices'] = (ci, fi)
                        self.stats['layout_checks'] += 1
                reference = None
                if c.filling:
                    if not state.get('prefill_checked', False):
                        # Capture the real mixed-path reference, including the K
                        # normalization used specifically by GEN cross-attention.
                        reference = original(*args, **kwargs)
                    und_output, key, value = prefill(pack['causal_seq'],
                        positions[0]['causal_seq'], positions[1]['causal_seq'])
                    if reference is not None:
                        pairs = ((und_output, reference[0]['causal_seq']),
                                 (key, state['key']), (value, state['value']))
                        if not all(self._matches(a, b) for a, b in pairs):
                            self.stats['rejected'].append(dict(layer=index, stage='prefill'))
                            raise ValueError(f'UND prefill differs from native layer {index}')
                    state.update(key=key, value=value, und_output=und_output)
                inputs = (pack['full_only_seq'], state['key'], state['value'],
                          positions[0]['full_only_seq'], positions[1]['full_only_seq'],
                          pack['_causal_indices'], pack['_full_indices'])
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    if reference is not None:
                        if not self._matches(compute(*inputs), reference[0]['full_only_seq']):
                            self.stats['rejected'].append(dict(layer=index, stage='prefill_gen'))
                            raise ValueError(f'Prefill GEN extraction differs from native layer {index}')
                        state['prefill_checked'] = True
                        self.stats['prefill_checks'] += 1
                        return reference
                    if not state.get('checked', False):
                        reference = original(*args, **kwargs)
                        candidate = compute(*inputs)
                        expected = reference[0]['full_only_seq']
                        if (not torch.isfinite(candidate).all() or candidate.shape != expected.shape
                                or candidate.dtype != expected.dtype
                                or not torch.equal(candidate.contiguous().view(torch.uint8),
                                                   expected.contiguous().view(torch.uint8))):
                            self.stats['rejected'].append(index)
                            raise ValueError(f'GEN extraction differs from native layer {index}')
                        state['checked'] = True
                        self.stats['eager_checks'] += 1
                        return reference
                    generated = region(*inputs)
                self.stats['compiled_calls'] += 1
                if c.filling:
                    self.stats['compiled_prefill_calls'] += 1
                result = dict(pack)
                result['causal_seq'] = state['und_output']
                result['full_only_seq'] = generated
                # Source may memoize a joined tensor in a pack. Never return an
                # input's memoized values after changing either stream.
                result.pop('all_seq', None)
                return result, {}, None
            return forward
        return wrap

    def close(self):
        for obj, attr, original, replacement in reversed(self.patches):
            if getattr(obj, attr) is replacement:
                setattr(obj, attr, original)
        self.patches.clear()
        for region in self.regions:
            region.close()
        for fused in getattr(self, "fused_linears", []):
            # Weights are model-owned; drop this owner's references on close.
            fused.linear = fused.activation = None
        # Caller closes this owner before the underlying conditioning cache.
        for slot in self.cache.slots.values():
            slot.pop('gen_regions', None)
            slot.pop('contiguous_kv_indices', None)
        self.stats['closed'] = True

    def report(self):
        return dict(self.stats, regions=[r.report() for r in self.regions],
                    split_prefill=self.split_prefill, contiguous_kv=self.contiguous_kv,
                    fused_linear=[f.report() for f in getattr(self, 'fused_linears', [])],
                    transformation='NUMERIC', quality_certified=False)


def validate_contiguous_layout(causal_indices, full_indices):
    if (causal_indices.ndim != 1 or full_indices.ndim != 1
            or causal_indices.dtype not in (torch.int32, torch.int64)
            or full_indices.dtype not in (torch.int32, torch.int64)):
        raise ValueError('Contiguous K/V requires one-dimensional token indices')
    n = causal_indices.numel()
    if (not torch.equal(causal_indices, torch.arange(n, device=causal_indices.device))
            or not torch.equal(full_indices, torch.arange(n, n + full_indices.numel(), device=full_indices.device))):
        raise ValueError('Contiguous K/V requires UND tokens followed by GEN tokens')


def install(service, plan, *, split_prefill=False, contiguous_kv=False, fused_linear=False):
    from instinctflash.planners.planner import Tier
    from instinctflash.runtime.precision import require_transform_permission
    require_transform_permission(plan, Tier.NUMERIC, 'Cosmos generation regions')
    if getattr(service, '_ifl_generation_regions', None) is not None:
        return service._ifl_generation_regions
    cache = getattr(service, '_ifl_conditioning_cache', None)
    status = getattr(service, '_ifl_conditioning_cache_status', {})
    if cache is None or cache.disabled or not status.get('admitted'):
        raise ValueError('GEN regions require the pinned admitted conditioning cache')
    if getattr(service, '_ifl_numeric_attention', None) is None:
        raise ValueError('GEN regions require the screened native cuDNN path')
    if cache.stats['requests']:
        raise ValueError('Install GEN regions before the first request')
    # Both pinned native dense MoT families share this UND interface. The
    # conditioning gate above binds source/modules; every real prefill still
    # checks native UND output, GEN key and value before compilation.
    if split_prefill and len(service.model.net.language_model.model.layers) not in (28, 36):
        raise ValueError('Split UND prefill requires pinned Edge/Nano layers')
    result = GenerationRegions(service, plan, split_prefill=split_prefill, contiguous_kv=contiguous_kv,
                               fused_linear=fused_linear)
    service._ifl_generation_regions = result
    return result

"""Offline old-source Cosmos conditioning reuse; not a Runtime precision pass.

Reuse the LingBot prefill/transaction pattern and Cosmos's owned-output layer
graphs. Cache lifetime is one generate call, independently for each CFG branch.
Graph storage survives requests, but every request refills it before decoding.
"""
from collections import OrderedDict
import types

import torch
from cosmos3_iwm.thor_graphs import LayerGraph, _tree, _signature


class ConditioningCache:
    def __init__(self, model, *, verify=False, max_slots=4):
        assert model.config.joint_attn_implementation == 'two_way'
        assert not model.config.video_temporal_causal and not model.config.sound_gen
        parallel = model.parallel_dims
        assert parallel is None or not any((parallel.cp_enabled, parallel.cfgp_enabled, parallel.dp_shard_enabled))
        assert torch.cuda.get_device_capability() == (11, 0)
        self.model, self.verify, self.max_slots = model, verify, max_slots
        self.slots = OrderedDict()
        self.active = None
        self.filling = False
        self.request = False
        self.seen = set()
        self.checks = []
        self.stats = dict(requests=0, prefills=0, decodes=0, module_hits=0,
                          verified_tensors=0, evictions=0, modules=0)
        self.pool = torch.cuda.graph_pool_handle()
        for index, layer in enumerate(model.net.language_model.model.layers):
            original = layer.forward
            if isinstance(original, LayerGraph):
                assert not original.entries, 'Install before the first prediction'
                original = original.original
            assert getattr(original, '__self__', None) is layer
            names = ('input_layernorm', 'post_attention_layernorm', 'mlp',
                     'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                     'self_attn.q_norm', 'self_attn.k_norm', 'self_attn.o_proj')
            for name in names:
                module = layer.get_submodule(name)
                module.forward = self.wrap_module(module.forward, (index, name))
                self.stats['modules'] += 1
            layer.forward = self.wrap_layer(original, index)

    def wrap_module(self, original, key):
        def forward(*args, **kwargs):
            assert self.active is not None and not torch.is_grad_enabled()
            values = self.active['values']
            if self.filling:
                result = original(*args, **kwargs)
                if key not in values or _signature(values[key]) != _signature(result):
                    assert not self.active['graphs'], 'Cached tensor geometry changed; new experiment slot required'
                    def owned(t):
                        out = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
                        out.copy_(t)
                        return out
                    values[key] = _tree(result, owned)
                else:
                    dest, source = [], []
                    _tree(values[key], lambda t: dest.append(t))
                    _tree(result, lambda t: source.append(t))
                    for d, s in zip(dest, source):
                        d.copy_(s)
                self.active['filled'].add(key)
                return result
            cached = values[key]
            self.stats['module_hits'] += 1
            if self.verify:
                result = original(*args, **kwargs)
                assert _signature(result) == _signature(cached), key
                left, right = [], []
                _tree(result, lambda t: left.append(t))
                _tree(cached, lambda t: right.append(t))
                for a, b in zip(left, right):
                    self.checks.append(torch.isfinite(a).all() &
                        (a.contiguous().view(torch.uint8) == b.contiguous().view(torch.uint8)).all())
                    self.stats['verified_tensors'] += 1
            return cached
        return forward

    def wrap_layer(self, original, index):
        def forward(*args, **kwargs):
            assert self.active is not None
            if self.filling or self.verify:
                return original(*args, **kwargs)
            graphs = self.active['graphs']
            if index not in graphs:
                stats = dict(captures=0, checks=0, replays=0, rejected=[])
                self.active['graph_stats'][index] = stats
                graphs[index] = LayerGraph(original, stats, str(index), pool=self.pool)
            return graphs[index](*args, **kwargs)
        return forward

    def velocity(self, original, **kwargs):
        assert self.request and self.active is None
        assert kwargs.get('net') is None or kwargs['net'] is self.model.net
        tokens = kwargs['text_tokens']
        assert len(tokens) == len(kwargs['sequence_plans']) == 1
        # Token identity, not tensor pointers or CFG call order, selects a branch.
        key = (tuple(tuple(x) for x in tokens), bool(kwargs.get('skip_text_tokens', False)))
        if key not in self.slots:
            if len(self.slots) == self.max_slots:
                self.slots.popitem(last=False)
                self.stats['evictions'] += 1
            self.slots[key] = dict(values={}, graphs={}, graph_stats={}, filled=set())
        self.slots.move_to_end(key)
        self.active = self.slots[key]
        self.filling = key not in self.seen
        if self.filling:
            self.active['filled'].clear()
        try:
            result = original(**kwargs)
            if self.filling:
                assert len(self.active['filled']) == self.stats['modules'], 'Incomplete prefill'
                self.seen.add(key)  # Commit only after a complete velocity forward.
                self.stats['prefills'] += 1
            else:
                self.stats['decodes'] += 1
            return result
        finally:
            self.active = None
            self.filling = False

    def generate(self, original, *args, **kwargs):
        assert not self.request, 'Concurrent/reentrant requests are unsupported'
        assert not torch.is_grad_enabled()
        assert kwargs.get('velocity_postprocess_builder') is None
        assert kwargs.get('upsample_task') is None
        self.request = True
        self.seen.clear()
        self.checks.clear()
        try:
            result = original(*args, **kwargs)
            if self.checks:
                assert bool(torch.stack(self.checks).all()), 'Text conditioning changed within a branch'
            self.stats['requests'] += 1
            return result
        finally:
            self.request = False
            self.seen.clear()  # No payload is valid in the next request until refilled.
            self.checks.clear()

    def report(self):
        return dict(self.stats, slots=len(self.slots), verify=self.verify,
            graph_stats=[s for slot in self.slots.values() for s in slot['graph_stats'].values()])


def install(model, *, verify=False):
    cache = ConditioningCache(model, verify=verify)
    generate, velocity = model.generate_samples_from_batch, model._get_velocity
    model.generate_samples_from_batch = lambda *a, **kw: cache.generate(generate, *a, **kw)
    model._get_velocity = lambda **kw: cache.velocity(velocity, **kw)
    return cache

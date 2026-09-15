"""Guarded native request-local text conditioning reuse on Thor.

Reuse the LingBot prefill/transaction pattern and Cosmos's owned-output layer
graphs. Cache lifetime is one generate call, independently for each CFG branch.
Graph storage survives requests, but every request refills it before decoding.
"""
from collections import OrderedDict
import inspect
import hashlib
import threading
from pathlib import Path

import torch
from .thor_graphs import LayerGraph, _tree, _signature


class ConditioningCache:
    def __init__(self, model, *, verify=False, max_slots=4):
        if max_slots < 1:
            raise ValueError('max_slots must be positive')
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
        self.validating = False
        self.disabled = False
        self.closed = False
        self.lock = threading.RLock()
        self.patches = []
        self.stats = dict(requests=0, prefills=0, decodes=0, module_hits=0,
                          verified_tensors=0, evictions=0, modules=0, bypasses=0,
                          admitted_branches=0, rejected=[])
        self.retired_graph_stats = dict(captures=0, checks=0, replays=0,
                                       rejection_count=0, rejection_samples=[])
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
                self.patch(module, 'forward', self.wrap_module(module.forward, (index, name)))
                self.stats['modules'] += 1
            self.patch(layer, 'forward', self.wrap_layer(original, layer.forward, index))

    def patch(self, obj, name, replacement):
        self.patches.append((obj, name, getattr(obj, name), replacement))
        setattr(obj, name, replacement)

    def disable(self, reason):
        self.disabled = True
        self.stats['rejected'].append(str(reason))

    def retire_slot(self, slot):
        # Keep bounded evidence after eviction, without retaining graph tensors.
        total = self.retired_graph_stats
        for stats in slot['graph_stats'].values():
            for key in ('captures', 'checks', 'replays'):
                total[key] += stats.get(key, 0)
            rejected = stats.get('rejected', [])
            total['rejection_count'] += len(rejected)
            room = max(0, 8 - len(total['rejection_samples']))
            total['rejection_samples'].extend(dict(row) for row in rejected[:room])

    def clear_slots(self):
        for slot in self.slots.values():
            self.retire_slot(slot)
        self.slots.clear()

    def wrap_module(self, original, key):
        def forward(*args, **kwargs):
            if self.active is None or self.disabled or torch.is_grad_enabled():
                return original(*args, **kwargs)
            values = self.active['values']
            if self.filling:
                result = original(*args, **kwargs)
                try:
                    if key not in values or _signature(values[key]) != _signature(result):
                        if self.active['graphs']:
                            self.disable('Cached tensor geometry changed')
                            return result
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
                except (TypeError, ValueError, torch.OutOfMemoryError) as error:
                    self.disable(error)
                    return result
                self.active['filled'].add(key)
                return result
            cached = values[key]
            self.stats['module_hits'] += 1
            if self.validating:
                result = original(*args, **kwargs)
                if _signature(result) != _signature(cached):
                    self.disable(f'Cached output signature changed: {key}')
                    return result
                left, right = [], []
                _tree(result, lambda t: left.append(t))
                _tree(cached, lambda t: right.append(t))
                for a, b in zip(left, right):
                    self.checks.append(torch.isfinite(a).all() &
                        (a.contiguous().view(torch.uint8) == b.contiguous().view(torch.uint8)).all())
                    self.stats['verified_tensors'] += 1
                return result  # Propagate the real reference through every validated layer.
            return cached
        return forward

    def wrap_layer(self, original, fallback, index):
        def forward(*args, **kwargs):
            if self.active is None or self.disabled:
                return fallback(*args, **kwargs)
            if self.filling or self.validating:
                return original(*args, **kwargs)
            graphs = self.active['graphs']
            # A handle does not keep the allocator pool alive after its last
            # graph is destroyed. Evicted slots must never revive that handle.
            if 'pool' not in self.active:
                self.active['pool'] = torch.cuda.graph_pool_handle()
            if index not in graphs:
                stats = dict(captures=0, checks=0, replays=0, rejected=[])
                self.active['graph_stats'][index] = stats
                graphs[index] = LayerGraph(original, stats, str(index), pool=self.active['pool'])
            return graphs[index](*args, **kwargs)
        return forward

    def velocity(self, original, **kwargs):
        if not self.request or self.disabled:
            return original(**kwargs)
        if kwargs.get('net') is not None and kwargs['net'] is not self.model.net:
            self.stats['bypasses'] += 1
            return original(**kwargs)
        tokens = kwargs['text_tokens']
        if len(tokens) != 1 or len(kwargs['sequence_plans']) != 1:
            self.stats['bypasses'] += 1
            return original(**kwargs)
        # Token identity, not tensor pointers or CFG call order, selects a branch.
        key = (tuple(tuple(x) for x in tokens), bool(kwargs.get('skip_text_tokens', False)))
        if key not in self.slots:
            if len(self.slots) == self.max_slots:
                _, retired = self.slots.popitem(last=False)
                self.retire_slot(retired)
                self.stats['evictions'] += 1
            self.slots[key] = dict(values={}, graphs={}, graph_stats={}, filled=set(), qualified=False)
        self.slots.move_to_end(key)
        self.active = self.slots[key]
        self.filling = key not in self.seen
        self.validating = not self.filling and (self.verify or not self.active['qualified'])
        self.checks.clear()
        if self.filling:
            self.active['filled'].clear()
        try:
            result = original(**kwargs)
            if self.filling:
                if len(self.active['filled']) != self.stats['modules']:
                    self.disable('Incomplete prefill')
                self.seen.add(key)  # Commit only after a complete velocity forward.
                self.stats['prefills'] += 1
            else:
                self.stats['decodes'] += 1
                if self.validating and not self.disabled:
                    if not self.checks or not bool(torch.stack(self.checks).all()):
                        self.disable('Text conditioning changed within a branch')
                    elif not self.active['qualified']:
                        self.active['qualified'] = True
                        self.stats['admitted_branches'] += 1
            return result
        finally:
            self.active = None
            self.filling = False
            self.validating = False
            self.checks.clear()
            if self.disabled:
                self.clear_slots()

    def generate(self, original, *args, **kwargs):
        with self.lock:
            return self._generate(original, *args, **kwargs)

    def _generate(self, original, *args, **kwargs):
        if self.request:
            raise RuntimeError('Reentrant Cosmos generation is unsupported')
        if (self.disabled or self.closed or torch.is_grad_enabled() or len(args) > 1
                or kwargs.get('velocity_postprocess_builder') is not None
                or kwargs.get('upsample_task') is not None):
            self.stats['bypasses'] += 1
            return original(*args, **kwargs)
        self.request = True
        self.seen.clear()
        self.checks.clear()
        try:
            result = original(*args, **kwargs)
            self.stats['requests'] += 1
            return result
        finally:
            self.request = False
            self.seen.clear()  # No payload is valid in the next request until refilled.
            self.checks.clear()

    def report(self):
        return dict(self.stats, slots=len(self.slots), verify=self.verify, disabled=self.disabled, closed=self.closed,
            retired_graph_stats=dict(self.retired_graph_stats,
                rejection_samples=list(self.retired_graph_stats['rejection_samples'])),
            graph_stats=[s for slot in self.slots.values() for s in slot['graph_stats'].values()])


    def close(self):
        with self.lock:
            for obj, name, original, replacement in reversed(self.patches):
                if getattr(obj, name) is replacement:
                    setattr(obj, name, original)
            self.patches.clear()
            self.clear_slots()
            self.closed = True

SOURCE_HASHES = {'data/generator/sequence_packing/__init__.py': 'c22cb8955903d8eb3109956425d867e38904ce6d4ee5ac148dbd5f15c55df732',
 'data/generator/sequence_packing/modality.py': '8ada9d183128816b9fce2a6e193e147b1e37ce4b31753078ff4e5279829b02a3',
 'data/generator/sequence_packing/mrope.py': '4f0e1dd70183a8834328cda816e921903a9594115fb88839f3356e2830677ef2',
 'data/generator/sequence_packing/natten.py': '921256a0f316a10622294f2d6ace710a998ae7686182f6635fe332ee3ed0067f',
 'data/generator/sequence_packing/packers.py': '8653642bc21394d13cbb41cc72dd4cb10262395c63d84b36441e18cd04b5f7a7',
 'data/generator/sequence_packing/runtime.py': '1ab37d79813e5f24f4e4f178299acb69e62e75ca94011a5e45ae2315c9b008eb',
 'data/generator/sequence_packing/sequence.py': '625dd6bbae28977b7c37b3f2d6d24ae58e1452aea8f1c8b885061a297e76b760',
 'data/generator/sequence_packing/temporal_causal.py': 'cc69a6b01ee63e5ace308c4fdb526b683116e5f3800378865802dfc449e17666',
 'model/generator/mot/__init__.py': 'f0b3bb691d1a6a3168fe9a9d3d7ea1e718c545ce71f6c6f245f38b3071a24075',
 'model/generator/mot/attention.py': 'a86f74acc1bef46f90af04985452a2aef8a38c8b18e16723c2dbec7601ad4392',
 'model/generator/mot/context_parallel_utils.py': '1c9ec80f6a9403db6745135d88646337d9f6b456378bcd4818cebde97735b305',
 'model/generator/mot/cosmos3_vfm_network.py': 'f155aee28da3d7b3164775f2c5c11a1084acdcd756363c92ffeddf97748fbb6b',
 'model/generator/mot/domain_aware_linear.py': 'ece44fdbcbffa94c39e4782c1929f995e38646a9322804891e501cc41de8da16',
 'model/generator/mot/modeling_utils.py': '145725f4ccda2ee5bee1cfd0fb676ad7c3fee4a5b27dbc9f12fdbb08d07e4a48',
 'model/generator/mot/parallelize_unified_mot.py': '2a53fec799dc68dcfc1395619fc4265bf1a10a2f9ed5e522500372138618d67d',
 'model/generator/mot/parallelize_vfm_network.py': '9d81f2fd225af6ca03c9dbef0b56da2b1b0dd5a3df4c92125f16d7baf8d91173',
 'model/generator/mot/unified_mot.py': '92fb455e597f4cdfbbde182dab192f370bebd1068f24d917e05f34268fcf7032',
 'model/generator/omni_mot_model.py': '167e8c012d6ad243187a88e67facba8a681f8d2fa3406672cea0de9dac37c161'}


def install(service, *, verify=False):
    """Opt-in admission; unsupported stacks retain their existing native forwards."""
    if getattr(service, '_ifl_conditioning_cache', None) is not None:
        return service._ifl_conditioning_cache
    status = {'admitted': False, 'reason': None}
    service._ifl_conditioning_cache_status = status
    cache = None
    try:
        if torch.cuda.get_device_capability() != (11, 0):
            raise ValueError('Thor only')
        if service.cfg.num_steps != 4 or float(service.cfg.guidance) != 3.0:
            raise ValueError('Qualified for four steps and CFG 3')
        model = service.model
        config = model.config
        if config.joint_attn_implementation != 'two_way' or config.video_temporal_causal or config.sound_gen:
            raise ValueError('Unsupported attention or modality configuration')
        parallel = model.parallel_dims
        if parallel is not None and any((parallel.cp_enabled, parallel.cfgp_enabled, parallel.dp_shard_enabled)):
            raise ValueError('Unsharded inference only')
        import cosmos_framework
        root = Path(cosmos_framework.__file__).parent
        for name, digest in SOURCE_HASHES.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Unqualified Cosmos source: {name}')
        layers = model.net.language_model.model.layers
        if len(layers) not in (28, 36):
            raise ValueError('Unqualified layer count')
        from .exact_pointwise import Checked
        for layer in layers:
            original = layer.forward
            if isinstance(original, LayerGraph):
                if original.entries:
                    raise ValueError('Install before inference')
                original = original.original
            if getattr(original, '__self__', None) is not layer:
                raise ValueError('Unqualified layer forward wrapper')
            for name in ('input_layernorm', 'post_attention_layernorm', 'mlp',
                         'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                         'self_attn.q_norm', 'self_attn.k_norm', 'self_attn.o_proj'):
                module = layer.get_submodule(name)
                forward = module.forward
                if not isinstance(forward, Checked) and getattr(forward, '__self__', None) is not module:
                    raise ValueError(f'Unqualified module wrapper: {name}')
                for parameter in module.parameters():
                    if parameter.dtype != torch.bfloat16 or not parameter.is_cuda:
                        raise ValueError('BF16 CUDA parameters required')
        cache = ConditioningCache(model, verify=verify)
        generate, velocity = model.generate_samples_from_batch, model._get_velocity
        cache.patch(model, 'generate_samples_from_batch', lambda *a, **kw: cache.generate(generate, *a, **kw))
        cache.patch(model, '_get_velocity', lambda **kw: cache.velocity(velocity, **kw))
        service._ifl_conditioning_cache = cache
        status.update(admitted=True, reason='Pinned native source; real-input admission before cached replay')
        return cache
    except (ValueError, AttributeError, OSError, AssertionError) as error:
        if cache is not None:
            cache.close()
        status['reason'] = str(error)
        return None

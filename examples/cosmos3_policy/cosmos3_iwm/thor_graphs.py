"""Bounded per-layer eager CUDA graphs; actual inputs checked before replay admission."""
from __future__ import annotations
import dataclasses
import copy
import torch


def _tree(value, tensor_fn):
    if isinstance(value, torch.Tensor):
        return tensor_fn(value)
    if type(value).__module__ == "cosmos_framework.model.generator.mot.attention" and type(value).__name__ == "SplitInfo":
        result = copy.copy(value)
        result.__dict__ = {k: _tree(v, tensor_fn) for k, v in vars(value).items()}
        return result
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{f.name: _tree(getattr(value, f.name), tensor_fn) for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {k: _tree(v, tensor_fn) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree(v, tensor_fn) for v in value)
    if isinstance(value, list):
        return [_tree(v, tensor_fn) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported graph input: {type(value)}')


def _signature(value):
    aliases = {}
    def visit(v):
        if isinstance(v, torch.Tensor):
            if not v.is_cuda:
                raise TypeError('CPU tensor graph inputs are not admitted')
            aliases.setdefault(id(v), len(aliases))
            return ('tensor', aliases[id(v)], tuple(v.shape), tuple(v.stride()), str(v.dtype), str(v.device))
        if type(v).__module__ == "cosmos_framework.model.generator.mot.attention" and type(v).__name__ == "SplitInfo":
            return ("SplitInfo", visit(vars(v)))
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return (type(v).__module__, type(v).__name__, tuple((f.name, visit(getattr(v, f.name))) for f in dataclasses.fields(v)))
        if isinstance(v, dict):
            return ('dict', tuple((k, visit(x)) for k, x in v.items()))
        if isinstance(v, (list, tuple)):
            return (type(v).__name__, tuple(visit(x) for x in v))
        if v is None or isinstance(v, (str, int, float, bool)):
            return (type(v).__name__, v)
        raise TypeError(f'Unsupported graph input: {type(v)}')
    return visit(value)


def _equal(left, right):
    if _signature(left) != _signature(right):
        return False
    a, b = [], []
    _tree(left, lambda x: a.append(x))
    _tree(right, lambda x: b.append(x))
    return len(a) == len(b) and all(x.shape == y.shape and x.dtype == y.dtype
        and bool(torch.isfinite(x).all())
        and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)) for x, y in zip(a, b))


class LayerGraph:
    def __init__(self, original, stats, name, max_shapes=4, *, pool=None):
        self.original, self.stats, self.name = original, stats, name
        self.entries = {}
        self.disabled = False
        self.max_shapes = max_shapes
        # Reuse GR00T StaticFullFlow's pool ownership pattern. Unlike that flow's
        # immediately consumed output, layer outputs can remain live in residuals.
        # Copy them out of the pool before another layer/shape replays.
        # Calls are serial on the owning inference stream.
        self.pool = pool

    def _result(self, outputs):
        if self.pool is None:
            return outputs
        memo = {}
        def clone(tensor):
            if id(tensor) not in memo:
                memo[id(tensor)] = tensor.clone()
            return memo[id(tensor)]
        return _tree(outputs, clone)

    def __call__(self, *args, **kwargs):
        if self.disabled or torch.is_grad_enabled() or kwargs.get('memory_value') is not None:
            return self.original(*args, **kwargs)
        inputs = (args, kwargs)
        try:
            weights = tuple((id(p), p.data_ptr(), tuple(p.shape), tuple(p.stride()), p.dtype, p.device)
                            for p in self.original.__self__.parameters())
            numeric_context = (
                torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'),
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
                torch.are_deterministic_algorithms_enabled(),
            )
            key = (_signature(inputs), weights, numeric_context)
        except TypeError as error:
            self.stats.setdefault('bypass', {})[str(error)] = self.stats.setdefault('bypass', {}).get(str(error), 0) + 1
            return self.original(*args, **kwargs)
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_shapes:
                return self.original(*args, **kwargs)
            if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
                self.stats['memory_bypasses'] = self.stats.get('memory_bypasses', 0) + 1
                return self.original(*args, **kwargs)
            reference = self.original(*args, **kwargs)
            try:
                memo = {}
                sources = []
                def clone(x):
                    if id(x) not in memo:
                        y = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
                        y.copy_(x)
                        memo[id(x)] = y
                        sources.append(y)
                    return memo[id(x)]
                static_args, static_kwargs = _tree(inputs, clone)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    self.original(*static_args, **static_kwargs)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool):
                    outputs = self.original(*static_args, **static_kwargs)
                graph.replay()
                if not _equal(reference, outputs):
                    raise ValueError('Capture output is not byte-identical')
                entry = [graph, outputs, sources, 1]
                self.entries[key] = entry
                self.stats['captures'] += 1
                self.stats['checks'] += 1
                return self._result(outputs)
            except Exception as error:
                self.disabled = True
                self.stats['rejected'].append({'module': self.name, 'error': repr(error)})
            return reference
        graph, outputs, static, checks = entry
        current, seen = [], set()
        def collect(x):
            if id(x) not in seen:
                current.append(x)
                seen.add(id(x))
        _tree(inputs, collect)
        for dest, source in zip(static, current):
            dest.copy_(source)
        graph.replay()
        if checks < 2:
            reference = self.original(*args, **kwargs)
            matched = _equal(reference, outputs)
            if not matched:
                self.disabled = True
                self.stats['rejected'].append({'module': self.name, 'error': 'Changed-input replay mismatch'})
            entry[3] += 1
            self.stats['checks'] += 1
            return self._result(outputs) if matched else reference
        self.stats['replays'] += 1
        return self._result(outputs)


def install(service):
    if torch.cuda.get_device_capability() != (11, 0):
        raise ValueError('Cosmos layer graphs target Thor only')
    if hasattr(service, '_ifl_layer_graphs'):
        return service._ifl_layer_graphs
    stats = {'captures': 0, 'checks': 0, 'replays': 0, 'rejected': []}
    for module in service.model.net.modules():
        if isinstance(module, torch.nn.Dropout) and module.training and module.p > 0:
            raise ValueError('CUDA graph admission requires deterministic inference without dropout')
    pool = torch.cuda.graph_pool_handle()
    for name, layer in service.model.net.language_model.model.layers.named_children():
        layer.forward = LayerGraph(layer.forward, stats, name, pool=pool)
    service._ifl_layer_graphs = stats
    return stats

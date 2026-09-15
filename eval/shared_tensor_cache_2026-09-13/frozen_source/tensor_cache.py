"""Bounded, owned tensor results for explicitly pure inference computations.

Adapters supply complete immutable keys and invalidate on dependency changes.
This is not a cache of model actions, RNG, or arbitrary module forwards.
"""
from collections import OrderedDict
from threading import RLock


class TensorResultCache:
    """Serialize cache operations and return caller-owned copies.

    A key must include every dependency of ``compute``. The adapter must prevent
    concurrent mutation of those dependencies. CUDA event ordering and allocator
    stream recording protect cached storage when callers use different streams.
    Cold misses retain the original output; hits clone the retained tensor.
    """

    def __init__(self, *, max_bytes=128 * 1024 * 1024, max_entries=16):
        for value in (max_bytes, max_entries):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('Cache bounds must be positive integers')
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._lock = RLock()
        self._entries = OrderedDict()
        self._bytes = 0
        self._closed = False
        self._stats = dict(hits=0, misses=0, evictions=0, invalidations=0)

    def get_or_compute(self, key, compute):
        import torch
        if torch.is_grad_enabled():
            raise RuntimeError('TensorResultCache requires inference/no_grad')
        hash(key)
        with self._lock:
            if self._closed:
                raise RuntimeError('TensorResultCache is closed')
            if key in self._entries:
                self._stats['hits'] += 1
                value, event = self._entries.pop(key)
                self._entries[key] = (value, event)
                if value.is_cuda:
                    stream = torch.cuda.current_stream(value.device)
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError('TensorResultCache must run outside CUDA capture')
                    stream.wait_event(event)
                    value.record_stream(stream)
                return value.clone()
            self._stats['misses'] += 1
            result = compute()
            if not isinstance(result, torch.Tensor) or result.layout != torch.strided:
                raise TypeError('Cache computations must return a strided Tensor')
            if result.requires_grad:
                raise ValueError('Cannot cache a gradient-bearing result')
            if result.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('TensorResultCache must run outside CUDA capture')
            size = result.numel() * result.element_size()
            if size <= self.max_bytes:
                while self._entries and (len(self._entries) >= self.max_entries
                                         or self._bytes + size > self.max_bytes):
                    _, (old, _) = self._entries.popitem(last=False)
                    self._bytes -= old.numel() * old.element_size()
                    self._stats['evictions'] += 1
                value = result.detach().clone()
                event = None
                if value.is_cuda:
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(value.device))
                self._entries[key] = (value, event)
                self._bytes += size
            return result

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            self._stats['invalidations'] += 1

    def close(self):
        with self._lock:
            self.clear()
            self._closed = True

    def report(self):
        with self._lock:
            return dict(self._stats, entries=len(self._entries), bytes=self._bytes,
                        max_bytes=self.max_bytes, max_entries=self.max_entries,
                        closed=self._closed)

"""Opt-in native Cosmos timestep reuse, backed by the shared owned cache.

The inspected native embedder is pure and row-wise. Preserve its full GEMM
geometry and results. Model mutation must remain serialized with inference.
"""
from threading import RLock

from instinctflash.runtime.tensor_cache import TensorResultCache


class NativeTimestepCache:
    def __init__(self, module):
        import torch
        if (type(module).__name__ != 'TimestepEmbedder'
                or not hasattr(module, 'frequency_embedding_size')
                or [type(m) for m in module.mlp] != [torch.nn.Linear, torch.nn.SiLU, torch.nn.Linear]):
            raise ValueError('Expected the native Linear/SiLU/Linear TimestepEmbedder')
        self.module = module
        self.original = module.forward
        self.cache = TensorResultCache()
        self._lock = RLock()
        self._version = None
        self._fallbacks = 0

    def __call__(self, t):
        import torch
        with self._lock:
            if (torch.is_grad_enabled() or any(m.training for m in self.module.modules())
                    or t.ndim != 1 or t.dtype != torch.float32
                    or t.layout != torch.strided
                    or (t.is_cuda and torch.cuda.is_current_stream_capturing())
                    or any(m._forward_hooks or m._forward_pre_hooks for m in self.module.modules())):
                self._fallbacks += 1
                return self.original(t)
            try:
                version = tuple((name, id(p), p.data_ptr(), p._version, p.dtype,
                                 p.device, tuple(p.shape), tuple(p.stride()))
                                for name, p in (*self.module.named_parameters(), *self.module.named_buffers()))
            except RuntimeError:
                self._fallbacks += 1
                return self.original(t)
            # Include native Python configuration and execution math context.
            context = (self.module.frequency_embedding_size,
                       torch.is_autocast_enabled(t.device.type),
                       torch.get_autocast_dtype(t.device.type),
                       torch.get_float32_matmul_precision(),
                       torch.backends.cuda.matmul.allow_tf32,
                       torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                       torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                       torch.are_deterministic_algorithms_enabled())
            version = (version, context)
            if version != self._version:
                if self._version is not None:
                    self.cache.clear()
                self._version = version
            key = (t.device, tuple(t.shape), tuple(t.stride()),
                   t.detach().cpu().numpy().tobytes())
            return self.cache.get_or_compute(key, lambda: self.original(t))

    def report(self):
        return dict(self.cache.report(), fallbacks=self._fallbacks,
                    kind='native_full_timestep_result', quality_certified=False)

    def close(self):
        with self._lock:
            if self.module.forward is self:
                self.module.forward = self.original
            self.cache.close()


def install(service):
    if getattr(service, '_ifl_timestep_cache', None) is not None:
        return service._ifl_timestep_cache
    module = service.model.net.time_embedder
    cache = NativeTimestepCache(module)
    module.forward = cache
    service._ifl_timestep_cache = cache
    return cache

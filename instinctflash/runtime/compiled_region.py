"""Inference-only tensor regions with explicit numerical permission.

Adapters own cache lookup/commit, hooks, schedule and weight construction outside
these callables. Compilation does not grant FP8 or schedule-change permission and
is not a quality certificate. No process-wide compiler settings are changed.
"""
from __future__ import annotations


class CompiledRegion:
    """Compile a pure tensor callable; report and surface lazy failures.

    CUDA graph capture is disabled here: the owning adapter may manage its own
    storage and graphs. Inputs may be tensors or nested tuple/list/string-keyed
    dictionaries of tensors. Pass changing cache metadata as tensors, not Python
    objects. The caller must serialize calls and keep inference weights fixed.
    """

    def __init__(self, forward, *, plan, name, max_signatures=8, compiler=None):
        from instinctflash.planners.planner import Tier
        from .precision import require_transform_permission

        require_transform_permission(plan, Tier.NUMERIC, name)
        if not callable(forward) or not isinstance(name, str) or not name:
            raise ValueError('A named callable is required')
        if isinstance(max_signatures, bool) or not isinstance(max_signatures, int) or max_signatures < 1:
            raise ValueError('max_signatures must be a positive integer')
        self.name = name
        self.max_signatures = max_signatures
        self._signatures = set()
        self._forward = forward
        self._compiled = None
        self._state = 'ready'
        self._error = None
        self._calls = 0
        import torch
        try:
            self._compiled = (compiler or torch.compile)(
                forward, fullgraph=True, dynamic=False,
                options={'triton.cudagraphs': False})
        except Exception as error:
            self._state, self._error = 'failed', repr(error)
            self._forward = None
            raise

    def __call__(self, *args, **kwargs):
        import torch
        if self._state != 'ready':
            raise RuntimeError(f'{self.name} is {self._state}: {self._error or ""}')
        if torch.is_grad_enabled():
            raise RuntimeError('CompiledRegion requires no_grad/inference_mode')
        signature = self._signature((args, kwargs))
        if signature not in self._signatures:
            if len(self._signatures) >= self.max_signatures:
                raise RuntimeError(f'{self.name} exceeded its input specialization bound')
            self._signatures.add(signature)
        try:
            result = self._compiled(*args, **kwargs)
        except Exception as error:
            # A failed callable may have performed work. Never silently execute
            # it again via an eager fallback (especially with accidental state).
            self._state, self._error = 'failed', repr(error)
            self._compiled = self._forward = None
            raise
        self._calls += 1
        return result

    @staticmethod
    def _signature(value):
        import torch
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise TypeError('CompiledRegion supports strided tensors only')
            return ('tensor', tuple(value.shape), tuple(value.stride()), value.dtype,
                    value.device, value.requires_grad)
        if isinstance(value, (tuple, list)):
            return (type(value).__name__, tuple(CompiledRegion._signature(x) for x in value))
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            return ('dict', tuple((k, CompiledRegion._signature(v)) for k, v in value.items()))
        raise TypeError('CompiledRegion inputs must contain tensors only; keep cache objects outside')

    def close(self):
        self._compiled = self._forward = None
        self._signatures.clear()
        self._state = 'closed'

    def report(self):
        return dict(name=self.name, state=self._state, calls=self._calls,
                    input_signatures=len(self._signatures), max_signatures=self.max_signatures,
                    fullgraph=True, dynamic=False, cudagraphs=False,
                    transformation='NUMERIC', quality_certified=False, error=self._error)

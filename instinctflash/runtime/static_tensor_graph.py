"""Exact-admitted stateless tensor graph with eager fallback and bounded storage."""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

class StaticTensorGraph:
    def __init__(self, forward, *, reference=None, name='tensor_graph'):
        self.forward = forward
        self.reference = reference or forward
        self.name = name
        self.graph = self.input = self.output = self.signature = None
        self.disabled = False
        self.replays = 0
        self.verdict = None

    def _clear_graph(self):
        self.graph = self.input = self.output = self.signature = None

    def close(self):
        """Release graph storage and bound references to the owner's weights."""
        self._clear_graph()
        self.forward = self.reference = None
        self.disabled = True

    def __call__(self, value):
        if self.reference is None:
            raise RuntimeError(f"{self.name} is closed")
        import torch
        from torch.utils._pytree import tree_flatten
        from .capture_self_check import compare_tensors
        signature = (tuple(value.shape), value.dtype, value.device)
        if value.device.type != 'cuda' or self.disabled:
            return self.reference(value)
        if self.signature is not None and signature != self.signature:
            # A graph never replays against a differently shaped/dtyped input.
            self._clear_graph()
        if self.graph is None:
            self.input = value.detach().clone()
            stream = torch.cuda.Stream(device=value.device)
            stream.wait_stream(torch.cuda.current_stream(value.device))
            try:
                with torch.no_grad(), torch.cuda.stream(stream):
                    for _ in range(3):
                        self.forward(self.input)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
                    output = self.forward(self.input)
                torch.cuda.current_stream(value.device).wait_stream(stream)
                checks = []
                for case in (value, value.flip(-1), torch.zeros_like(value)):
                    with torch.no_grad():
                        ref, ref_spec = tree_flatten(self.reference(case))
                        ref = [x.detach().clone() for x in ref]
                        self.input.copy_(case)
                        graph.replay()
                        actual, actual_spec = tree_flatten(output)
                        assert ref_spec == actual_spec and len(ref) == len(actual)
                        checks.extend(compare_tensors(a,b) for a,b in zip(ref,actual))
                self.verdict = dict(passed=bool(checks) and all(c['valid'] and c['bitexact'] for c in checks),
                                    comparisons=len(checks), max_abs_delta=max(c['max_abs_delta'] for c in checks))
                if not self.verdict['passed']:
                    raise RuntimeError(f'exact graph self-check failed: {self.verdict}')
                self.graph, self.output, self.signature = graph, output, signature
            except Exception as exc:
                self._clear_graph()
                self.disabled = True
                self.verdict = dict(passed=False, reason=str(exc))
                log.warning('%s disabled; retaining eager execution: %s',self.name,exc)
                return self.reference(value)
        self.input.copy_(value)
        self.graph.replay()
        self.replays += 1
        return self.output

    @property
    def stats(self):
        return dict(captured=self.graph is not None, replays=self.replays, self_check=self.verdict,
                    disabled=self.disabled)

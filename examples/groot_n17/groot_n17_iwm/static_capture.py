"""Replay-safe CUDA Graph capture for GR00T N1.7's DiT action head.

The four flow-matching calls use the same tensor shapes for a given prompt
shape. Each signature owns static input buffers and one captured DiT forward;
values are copied into those buffers before replay and outputs are cloned out
of the graph pool before returning to upstream code.
"""

from __future__ import annotations

import torch


class StaticDiT:
    """Capture one DiT graph per tensor-shape/non-tensor-argument signature."""

    def __init__(self, dit_forward):
        self._fwd = dit_forward
        self._graphs: dict = {}
        self.replays = 0
        self.captures = 0

    @property
    def captured(self) -> bool:
        return bool(self._graphs)

    def _signature(self, tensors: dict, other: dict):
        return (
            tuple((key, tuple(value.shape), value.dtype)
                  for key, value in sorted(tensors.items())),
            tuple(sorted((key, repr(value)) for key, value in other.items())),
        )

    def __call__(self, *args, **kwargs):
        if args:
            # Upstream's action loop calls with kwargs only. Preserve eager
            # behavior for any other caller rather than capturing a new API.
            return self._fwd(*args, **kwargs)
        tensors = {key: value for key, value in kwargs.items() if torch.is_tensor(value)}
        other = {key: value for key, value in kwargs.items() if not torch.is_tensor(value)}
        signature = self._signature(tensors, other)
        entry = self._graphs.get(signature)
        if entry is None:
            buffers = {key: value.clone() for key, value in tensors.items()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._fwd(**buffers, **other)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._fwd(**buffers, **other)
            entry = (graph, buffers, output)
            self._graphs[signature] = entry
            self.captures += 1
        graph, buffers, output = entry
        for key, value in tensors.items():
            buffers[key].copy_(value)
        graph.replay()
        self.replays += 1
        return output.clone() if torch.is_tensor(output) else output

    def close(self) -> None:
        self._graphs.clear()


def install_static_capture(model) -> StaticDiT:
    """Replace the action-head DiT forward and return its capture counters."""
    dit = model.action_head.model
    if isinstance(dit.forward, StaticDiT):
        return dit.forward
    handle = StaticDiT(dit.forward)
    dit.forward = handle
    return handle


__all__ = ["StaticDiT", "install_static_capture"]

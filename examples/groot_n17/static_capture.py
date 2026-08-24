"""Replay-safe CUDA-graph capture for GR00T-N1.7's DiT action head.

The measured picture on H100 (bench_groot_h100/profile.json): eager get_action 122.5 ms =
backbone 42.2 (1 forward) + DiT 63.9 (4 denoise forwards) + pre/post ~16. The DiT step is
100% submit-bound (wall 15.80 ms vs submit 15.79 ms) — the same launch-bound profile that made
pi05's denoise loop worth capturing, and the same cure applies.

Mechanism, mirroring examples/pi05_vla/pi05_iwm/static_capture.py: the DiT is called four times
per get_action as `model(hidden_states=, encoder_hidden_states=, timestep=, ...)` with tensor
kwargs whose SHAPES repeat forever for a given embodiment and prompt length. Graphs are cached
per shape signature: every tensor kwarg gets a static input buffer, the graph bakes one forward
over those buffers, and each step copies its values in and replays. A new prompt length is a new
signature — it captures its own graph rather than replaying a wrong one. Nothing inside the DiT
allocates per call (plain transformer blocks + AdaLN), which is what makes replay legal; the
verify script proves it bitexact rather than assuming it.

Opt-in:

    from static_capture import install_static_capture
    install_static_capture(policy.model)          # after Gr00tPolicy(...)

or set IFL_GROOT_STATIC_CAPTURE=1 before building the policy and call it from your loader.
"""

from __future__ import annotations

import torch


class StaticDiT:
    """The DiT forward, captured once per input-shape signature and replayed thereafter."""

    def __init__(self, dit_forward):
        self._fwd = dit_forward
        self._graphs: dict = {}
        self.replays = 0
        self.captures = 0

    def _signature(self, tensors: dict, other: dict):
        return (tuple((k, tuple(v.shape), v.dtype) for k, v in sorted(tensors.items())),
                tuple(sorted((k, repr(v)) for k, v in other.items())))

    def __call__(self, *args, **kwargs):
        if args:
            # the loop calls with kwargs only; anything else falls back to eager, correctly
            return self._fwd(*args, **kwargs)
        tensors = {k: v for k, v in kwargs.items() if torch.is_tensor(v)}
        other = {k: v for k, v in kwargs.items() if not torch.is_tensor(v)}
        key = self._signature(tensors, other)
        ent = self._graphs.get(key)
        if ent is None:
            bufs = {k: v.clone() for k, v in tensors.items()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):                       # warm cuBLAS/cuDNN on the side stream
                    self._fwd(**bufs, **other)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._fwd(**bufs, **other)
            self._graphs[key] = (graph, bufs, out)
            self.captures += 1
            ent = self._graphs[key]
        graph, bufs, out = ent
        for k, v in tensors.items():
            bufs[k].copy_(v)
        graph.replay()
        self.replays += 1
        return out.clone() if torch.is_tensor(out) else out


def install_static_capture(model) -> StaticDiT:
    """Swap the DiT's forward for the captured path. Returns the handle (for counters)."""
    dit = model.action_head.model
    handle = StaticDiT(dit.forward)
    dit.forward = handle
    return handle

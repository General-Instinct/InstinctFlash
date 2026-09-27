"""The staged engine the LingBot-VLA-V2 Runtime loop drives: native vision + GI engine."""
from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from instinctflash.engines.vla2.engine import Vla2Engine
from instinctflash.engines.vla2.weights import load_vla2_weights


class GiVla2Staged:
    """``set_prompt(ids)`` / ``infer_staged(patches, state, noise)`` with no calibration pass."""

    calibrations = 0

    def __init__(self, checkpoint_dir, vision, device=None, *, steps=10):
        dev = torch.device(device or "cuda")
        self.vision = vision
        self.engine = Vla2Engine(load_vla2_weights(checkpoint_dir, dev, steps=steps), dev)
        self._tokens = None
        self.replays = 0
        self.prompt_graph_reuses = 0
        self.latency_records = []
        self.frontend = SimpleNamespace(prompt_updates=0, prompt_buffer_reuses=0)

    @property
    def _graph(self):
        return self.engine._graphs.graphs.get(self.engine._rows)

    @property
    def graph_captures(self):
        return len(self.engine._graphs.graphs)

    def set_prompt(self, token_ids):
        ids = tuple(int(i) for i in token_ids)
        if not ids:
            raise ValueError("VLA2 requires active language tokens")
        if ids == self._tokens:
            return
        self.engine.set_prompt(list(ids))
        self.frontend.prompt_updates += 1
        if self._graph is not None:
            self.prompt_graph_reuses += 1
        self._tokens = ids

    def infer_staged(self, patches, state, noise):
        if self._tokens is None:
            raise RuntimeError("set_prompt must precede inference")
        t0 = time.perf_counter()
        with torch.no_grad():
            merged, deepstack = self.vision(patches)
            actions = self.engine.infer(merged, deepstack, state.float(), noise.float()).clone()
        self.replays += 1
        elapsed = (time.perf_counter() - t0) * 1e3
        self.latency_records.append(elapsed)
        return dict(actions=actions, latency_ms=elapsed)

"""The action-generator frontend the LingBot-VLA-4B Runtime loop drives."""
from __future__ import annotations

import time

import torch

from instinctflash.engines.vla4.engine import Vla4Engine
from instinctflash.engines.vla4.weights import load_vla4_weights


class GiVla4Frontend:
    """``set_prompt(ids)`` / ``infer_staged(images, state, noise, vision_embeddings=)`` over the GI engine."""

    def __init__(self, checkpoint_dir, device=None, *, steps=10):
        dev = torch.device(device or "cuda")
        self.engine = Vla4Engine(load_vla4_weights(checkpoint_dir, dev, steps=steps), dev)
        self.latency_records = []
        self.prompt_buffer_reuses = 0

    @property
    def graph_captured(self):
        return self.engine.graph_captured

    @property
    def prompt_updates(self):
        return self.engine.prompt_updates

    @property
    def prompt_graph_captures(self):
        return len(self.engine._graphs.graphs)

    def set_prompt(self, token_ids):
        self.engine.set_prompt(token_ids)

    def infer_staged(self, images, state, noise, vision_embeddings=None):
        if vision_embeddings is None:
            raise ValueError("the GI VLA4 engine consumes the native vision tower's embeddings")
        t0 = time.perf_counter()
        actions = self.engine.infer(vision_embeddings, state.float(), noise.float()).clone()
        self.latency_records.append((time.perf_counter() - t0) * 1e3)
        return {"actions": actions}

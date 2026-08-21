"""A toy world-action model that is deliberately NOT shaped like LingBot-VA.

Structural differences, chosen so the integration exercises the parts of InstinctFlash that LingBot
would never touch:

    LingBot-VA                          gridworld-ar
    two streams (video, action)         ONE stream
    diffusion, 79 forwards per cycle    autoregressive, 1 forward per cycle
    classifier-free guidance            no guidance at all
    ring KV with a commit phase         a plain token history, no commit
    frozen VAE + T5 by pointer          self-contained, no base weights
    multi-phase control cycle           single phase

If InstinctFlash's public surface only fits models that look like LingBot, this file is where that
shows up.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GridworldAR(nn.Module):
    """Embed a short history of discrete observation tokens, predict the next action."""

    def __init__(self, vocab: int = 64, dim: int = 32, action_dim: int = 4, history: int = 8):
        super().__init__()
        self.vocab, self.dim, self.action_dim, self.history = vocab, dim, action_dim, history
        self.embed = nn.Embedding(vocab, dim)
        self.core = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, action_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h, _ = self.core(self.embed(tokens))
        return torch.tanh(self.head(h[:, -1]))


def quantize(obs, vocab: int = 64) -> int:
    """Whatever the caller hands us becomes one discrete token. Deliberately crude."""
    import numpy as np
    a = np.asarray(obs, dtype=np.float64).ravel()
    if a.size == 0:
        return 0
    return int(abs(hash(round(float(a.mean()), 6))) % vocab)

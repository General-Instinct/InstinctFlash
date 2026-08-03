"""Turning a single-output backbone into PDD's multi-head student.

The paper's architecture change is deliberately minimal: "we utilize the same backbone architecture
of the pretrained flow model, but with the final linear layer repeated N times, i.e. one for each
time step in the grid", each copy initialised from the teacher's single final layer. So the trunk is
the teacher's trunk, the heads start as N copies of the teacher's head, and at initialisation every
head predicts the teacher's instantaneous velocity -- a sane starting point, since the mean velocity
over a short interval is close to it.

This module owns only the WRAPPING. Where the final layer lives in a given backbone is a
backbone-specific fact, so it arrives as a callable rather than being discovered by inspection --
guessing at module names is how a wrapper silently attaches to the wrong layer.
"""

from __future__ import annotations

from typing import Any, Callable


class MultiHeadStudent:
    """Adapts a backbone into `MultiHeadVelocityModel`.

    `trunk_fn(x, t, cond) -> features` runs everything up to (not including) the final projection.
    `make_head()` returns a fresh copy of the teacher's final projection.

    Kept as a plain class rather than an `nn.Module` subclass so this file needs no torch import at
    module scope; `parameters()` and `state_dict()` are delegated to the head container, which is a
    real module. That keeps the package importable for schedule/grid work on a machine with no torch.
    """

    def __init__(self, trunk_fn: Callable[..., Any], make_head: Callable[[], Any], n_heads: int):
        import torch

        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        self._trunk = trunk_fn
        self.n_heads = n_heads
        self.head_list = torch.nn.ModuleList([make_head() for _ in range(n_heads)])

    def heads(self, x: Any, t: Any, *, cond: Any = None) -> Any:
        """Trunk once, then every head. Shape (n_heads, *x.shape)."""
        import torch
        feats = self._trunk(x, t, cond)
        return torch.stack([h(feats) for h in self.head_list], dim=0)

    def head(self, x: Any, t: Any, index: int, *, cond: Any = None) -> Any:
        """One head only. Cheaper when just a single interval is being supervised."""
        return self.head_list[index](self._trunk(x, t, cond))

    def parameters(self, recurse: bool = True):
        return self.head_list.parameters(recurse=recurse)

    def state_dict(self, *a, **k):
        return self.head_list.state_dict(*a, **k)

    def load_state_dict(self, *a, **k):
        return self.head_list.load_state_dict(*a, **k)

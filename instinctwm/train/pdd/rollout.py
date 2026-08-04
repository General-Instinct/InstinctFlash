"""Algorithm 3: data-free, on-policy state generation.

The paper trains its video models WITHOUT a dataset. Instead of drawing X_n from the interpolant
against a real sample, it draws noise once and lets the student walk its own trajectory:

    "we sample an initial state X_0 ~ p_0, then for the next N/L-1 iterations, we utilize the
     output of the parallel decoder u_bar^theta_n(.|X_n) both for the optimization step and to
     advance X_bar_n to X_bar_{n+L}"

Two consequences worth being explicit about, because both shape how this is used.

FIRST, it is stateful. One optimisation step corresponds to one block, and the state persists to the
next step; after N/L blocks the trajectory is finished and fresh noise is drawn. A training loop that
assumes steps are independent given a batch will still work -- the state lives here, in the recipe's
own object -- but the batch no longer determines the step, and reproducibility comes from this
object's generator rather than from the dataloader.

SECOND, it removes the dataset from the critical path entirely. For a world-action model that is a
much bigger deal than it is for text-to-video: it means distilling to a new embodiment needs no
collected trajectories, only whatever conditioning the backbone wants. What it does NOT remove is
conditioning -- the paper still feeds real prompts.

The data-based variant (Algorithm 2) remains available as plain `pdd_loss`; it is the easier thing
to validate an implementation against, because a fixed batch makes the target deterministic.
"""

from __future__ import annotations

from typing import Any, Callable

from instinctwm.train.pdd.core import advance
from instinctwm.train.pdd.schedule import Grid


class Rollout:
    """A student trajectory in progress, advanced one block per optimisation step.

    `sample_noise()` returns a fresh X_0. It is a callable rather than a tensor so the rollout can
    restart itself without the caller having to notice that a trajectory finished.
    """

    def __init__(self, grid: Grid, sample_noise: Callable[[], Any], *,
                 l_max: int | None = None, generator: Any = None):
        self.grid = grid
        self._sample_noise = sample_noise
        self.l_max = l_max or grid.block
        if self.l_max < 1:
            raise ValueError("l_max must be >= 1")
        self.generator = generator
        self._x: Any = None
        self._n = 0
        self.trajectories = 0
        self.blocks = 0

    @property
    def x(self) -> Any:
        return self._x

    @property
    def n(self) -> int:
        return self._n

    def _restart(self) -> None:
        self._x = self._sample_noise()
        self._n = 0
        self.trajectories += 1

    def begin_block(self) -> tuple[int, Any]:
        """Return (n, X_n) for the block about to be trained on, restarting if the grid is spent."""
        if self._x is None or self._n >= self.grid.n_intervals:
            self._restart()
        return self._n, self._x

    def pick_k(self, n: int) -> int:
        """Sample the supervised interval k uniformly in [n, min(n + l_max, N)).

        Uniform over the block, per Algorithm 2. Sampling one index rather than supervising all L
        is what keeps the step to a single teacher call; over many steps every head is covered.
        """
        import torch
        hi = min(n + self.l_max, self.grid.n_intervals)
        if hi <= n:
            raise ValueError(f"no interval to supervise at n={n} (hi={hi})")
        return int(torch.randint(n, hi, (1,), generator=self.generator).item())

    def advance(self, heads: Any) -> None:
        """Walk the state forward one full block, detached, and move the pointer.

        Detached because the paper says so -- "we apply the stop-gradient operation when advancing
        the state, preventing additional memory or compute cost". Without it the trajectory would
        accumulate a graph across the whole rollout, and by the last block the backward pass would
        span every earlier step.
        """
        L = min(self.grid.block, self.grid.n_intervals - self._n)
        self._x = advance(self._x, heads, self.grid, self._n, self._n + L).detach()
        self._n += L
        self.blocks += 1

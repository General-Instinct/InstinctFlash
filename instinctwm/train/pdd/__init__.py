"""Parallel Decoding Distillation -- a clean-room implementation from arXiv 2607.26004.

Destined to become a standalone open-source project (`instinct-pdd`). Nothing in this package may
import from `instinctwm`; `tests/test_pdd_core.py` asserts it, so the eventual split is a directory
move rather than an untangling.

Scope: single-stream PDD over an arbitrary backbone. Applying it to several streams with different
schedules -- as a world-action model needs -- is composition, and belongs to the caller.

    from instinctwm.train.pdd import Grid, MultiHeadStudent, block_sample, pdd_loss

    grid = Grid.from_shift(n_intervals=256, block=64, shift=5.0, scale=1000.0)
    loss, metrics = pdd_loss(student, teacher, x_n, grid, n=0, k=3)
    sample = block_sample(student, noise, grid)
"""

from instinctwm.train.pdd.core import (
    SOLVERS, advance, block_sample, mean_velocity_euler, mean_velocity_midpoint, pdd_loss,
)
from instinctwm.train.pdd.heads import MultiHeadStudent
from instinctwm.train.pdd.protocols import MultiHeadVelocityModel, VelocityModel
from instinctwm.train.pdd.rollout import Rollout
from instinctwm.train.pdd.schedule import Grid, shift_time

__all__ = [
    "Grid", "shift_time",
    "VelocityModel", "MultiHeadVelocityModel", "MultiHeadStudent",
    "pdd_loss", "block_sample", "advance", "Rollout",
    "mean_velocity_euler", "mean_velocity_midpoint", "SOLVERS",
]

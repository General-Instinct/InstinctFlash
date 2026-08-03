"""The entire interface PDD needs from a backbone.

THIS PACKAGE IS DESTINED TO BECOME A STANDALONE PROJECT (`instinct-pdd`), so it may not import
anything from `instinctwm`. `tests/test_pdd_core.py` enforces that mechanically rather than by
convention -- a boundary that is only documented is a boundary that erodes.

The design rule that follows from it: PDD knows about states, times and velocities. It does not know
what a video is, what an action is, what a KV cache is, or that a backbone might want a dict of
seventeen tensors. All of that lives behind `VelocityModel.velocity`, which is deliberately the
narrowest thing that can express "sample the probability-flow ODE".

Multi-stream composition is NOT here either. PDD is a single-stream algorithm; running it over a
video stream and an action stream with different schedules is orchestration, and belongs to whoever
is doing the orchestrating. Putting it here would bake a world-action model's shape into a method
that should also serve plain text-to-video.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VelocityModel(Protocol):
    """A teacher: the pretrained flow/diffusion model, frozen.

    `t` is the model's OWN time convention, whatever that is -- PDD passes back exactly the values
    it was given in the `Grid`, and never rescales them. This matters: LingBot-VA's backbone is
    conditioned on `sigma * 1000`, Wan's on something else, and a method that "helpfully" normalised
    times would silently condition every backbone on the wrong one.
    """

    def velocity(self, x: Any, t: Any, *, cond: Any = None) -> Any:
        """Return dx/dt at state `x`, time `t`. Shape must match `x`."""
        ...


@runtime_checkable
class MultiHeadVelocityModel(Protocol):
    """A student: one forward pass, many mean velocities.

    This is the whole architectural claim of PDD. Where a teacher answers "what is the velocity
    here", the student answers "what are the mean velocities of the next L intervals from here",
    which is what lets one network evaluation advance L steps.

    `heads` returns a tensor whose FIRST dimension indexes the grid, i.e. shape (n_heads, *x.shape).
    Returning all heads and slicing is deliberate: the paper's student has one final linear layer per
    grid point over a shared trunk, so the trunk is computed once and the heads are nearly free.
    """

    n_heads: int

    def heads(self, x: Any, t: Any, *, cond: Any = None) -> Any:
        """Return stacked mean-velocity predictions, shape (n_heads, *x.shape)."""
        ...

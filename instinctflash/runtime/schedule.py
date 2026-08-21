"""The sigma schedule a checkpoint is served on, derived from the server's own scheduler.

WHY THIS MODULE EXISTS

Serving a checkpoint whose output projection is a set of per-interval velocity heads needs exactly two
numbers per grid point: the width of each interval, and the conditioning timestep at each block start.
Both are functions of the sampler's sigma list and nothing else.

Before this module, the serving path obtained them by constructing a PDD *training oracle* over the
live server and asking it for an `instinct_pdd.Grid`. That made the training library of one
distillation method a runtime dependency: a checkpoint from any other recipe could not be served
without `instinct_pdd` installed, and the serving path imported it to compute two lists of floats.
See AUDIT.md F1.

Nothing here imports a training package, and nothing here knows what produced the heads.

THE ONLY SUBTLETY IS THE CONVENTION, and it is worth stating because getting it wrong is silent.

instinct-pdd fixes time ASCENDING from 0 (noise) to 1 (data) to match the paper's interpolant.
LingBot-VA integrates sigma DESCENDING 1 -> 0. The bridge is `t = 1 - sigma` in the same index order,
so with `Grid.from_times([1 - s for s in sigmas], scale=-1000, offset=+1000)`:

    h(l)     = t[l+1] - t[l]  = (1 - sigma[l+1]) - (1 - sigma[l])  = sigma[l] - sigma[l+1]
    cond(i)  = t[i] * -1000 + 1000                                 = sigma[i] * 1000

Both reduce to plain statements about sigma. `interval_widths` and `conditioning_timesteps` are those
two lines, and `tests/test_serve_parity.py` asserts they equal the Grid's answers at 0.00e+00 so the
equivalence is verified rather than argued.

`h(l)` is positive because sigma descends. That matters: `fold_heads` normalises by `sum(h)` over a
block, and a sign error there would flip every folded head at once.
"""

from __future__ import annotations

from typing import Sequence

#: Sigmas below this are treated as the terminal clean-data point.
_EPS = 1e-12


def sigmas_from_scheduler(scheduler, n_intervals: int) -> list[float]:
    """The exact sigma schedule an `n_intervals`-step sampler walks, off the live scheduler.

    Taken from the scheduler rather than recomputed. Re-deriving a schedule is how a served grid stops
    matching the sampler -- `Grid.from_shift` once warped the progress fraction instead of sigma, and
    at N=25, shift=5 the second timestep came out 827.6 instead of 991.7. Nothing about that is visible
    in a loss curve.

    `set_timesteps` MUTATES the scheduler, so the server's own inference schedule is saved and
    restored. Leaving it set to a different grid would silently change what the next `_infer` does.
    """
    prev_sigmas = getattr(scheduler, "sigmas", None)
    prev_training = getattr(scheduler, "training", False)
    try:
        scheduler.set_timesteps(n_intervals)
        sigmas = [float(s) for s in scheduler.sigmas]
    finally:
        if prev_sigmas is not None:
            scheduler.sigmas = prev_sigmas
            scheduler.timesteps = prev_sigmas * scheduler.num_train_timesteps
            scheduler.training = prev_training

    # The server pads a terminal sigma = 0 so the trajectory ends on clean data
    # (wan_va_server.py:473). Without it the last interval would be missing.
    if abs(sigmas[-1]) > _EPS:
        sigmas.append(0.0)
    return sigmas


def interval_widths(sigmas: Sequence[float]) -> list[float]:
    """Width of each interval, in the ascending-t convention. Positive, because sigma descends.

    THE SUBTRACTION IS DELIBERATELY ROUND-TRIPPED THROUGH t, AND THAT IS NOT PEDANTRY.

    Algebraically `h(l) = sigma[l] - sigma[l+1]`, and writing it that way is shorter, cheaper and
    slightly more accurate. It is also NOT the same in floating point as what the retired
    `instinct_pdd.Grid` computed, which was `t[l+1] - t[l]` over `t = 1 - sigma`:

        (1 - sigma[l+1]) - (1 - sigma[l])   !=   sigma[l] - sigma[l+1]      at ~5.6e-16

    because subtracting from 1.0 rounds. Every number this project has published -- including the Fast
    operating point's certification at 566 matched pairs -- was measured with the left-hand form baked
    into the folded heads. Switching to the right-hand form would be an improvement that silently
    invalidates the claim that this refactor changed nothing.

    So the arithmetic is preserved exactly, and `tests/test_serve_parity.py` holds it at 0.00e+00
    against the Grid. If the direct form is ever wanted, changing it is a one-line, gated decision --
    not something that happens by accident during a cleanup.
    """
    t = [1.0 - float(s) for s in sigmas]
    return [t[i + 1] - t[i] for i in range(len(t) - 1)]


def conditioning_timesteps(sigmas: Sequence[float], num_train_timesteps: float) -> list[float]:
    """What the backbone is conditioned on at each grid point: `cond(i) = sigma[i] * scale`.

    This is the value `_BlockHead` matches against to decide which folded map is active. Deriving it
    from the same sigma list the sampler uses is what makes the match exact rather than approximate.
    """
    return [float(s) * float(num_train_timesteps) for s in sigmas]


def block_start_timesteps(sigmas: Sequence[float], block: int, n_blocks: int,
                          num_train_timesteps: float) -> list[float]:
    """`cond()` at the start of each block -- one entry per served step."""
    cond = conditioning_timesteps(sigmas, num_train_timesteps)
    return [cond[b * block] for b in range(n_blocks)]


def block_weights(sigmas: Sequence[float], block: int, n_blocks: int,
                  n_intervals: int) -> list[list[float]]:
    """Per-block normalised interval weights: `w_l = h_l / sum_block h`.

    These are the coefficients `fold_heads` folds the L linear heads with. The normalisation is what
    makes the folded map equal a single scheduler step: `sum h_l` over a block is
    `sigma[n] - sigma[n+L]`, which is exactly the `dsigma` the sampler applies, so it cancels.
    """
    h = interval_widths(sigmas)
    out = []
    for b in range(n_blocks):
        n = b * block
        idx = list(range(n, min(n + block, n_intervals)))
        tot = sum(h[l] for l in idx)
        out.append([h[l] / tot for l in idx])
    return out

"""Parallel Decoding Distillation: the teacher target, the loss, and the sampler.

Clean-room from arXiv 2607.26004 (Shaul, Liu, Vahdat, Berner -- NVIDIA / Weizmann). No reference
code existed when this was written; the NVIDIA implementation is announced for NVlabs/FastGen
(Apache-2.0) and this should be diffed against it when it lands.

THE METHOD IN THREE LINES

  * The student predicts the MEAN velocity of each of the next L intervals, all from one forward
    pass:   u_bar^theta_n(k | X_n) ~= u_k(X_k)   for k = n .. n+L-1
  * Sampling composes those L predictions into one jump (Eq 10), so NFE = N/L.
  * Training regresses head k against the teacher's mean velocity over interval k, evaluated at the
    state the STUDENT itself reached (Eq 11), with a stop-gradient on that state.

WHAT MAKES IT ON-POLICY, AND WHY THAT IS THE WHOLE ALGORITHM

The naive target -- average the teacher's field along the straight coupling
x_u = (1-u) data + u noise -- is wrong, and cheaply so. For a fixed (data, noise) pair the
flow-matching target is `noise - data`, constant in time, so that average converges to the straight
line, which the student can reproduce with no teacher at all. What a few-step student must match is
the many-step SAMPLER, which integrates the learned field and is therefore curved. PDD closes that
gap by evaluating the teacher at student-generated states: the student is corrected exactly where it
actually goes, not where an idealised coupling says it should be.
"""

from __future__ import annotations

from typing import Any, Callable

from instinctwm.train.pdd.protocols import MultiHeadVelocityModel, VelocityModel
from instinctwm.train.pdd.schedule import Grid


def mean_velocity_euler(teacher: VelocityModel, x: Any, grid: Grid, k: int, cond=None) -> Any:
    """First-order estimate of the mean velocity over interval k: one teacher evaluation.

    u_k(X_k) ~= v_{t_k}(X_k). Cheapest option, and what the paper uses for its N=256 grids -- with
    a fine grid the interval is short enough that first order is adequate.
    """
    return teacher.velocity(x, grid.times[k], cond=cond)


def mean_velocity_midpoint(teacher: VelocityModel, x: Any, grid: Grid, k: int, cond=None) -> Any:
    """Second-order estimate: two teacher evaluations, at the interval start and its midpoint.

    Step half an interval with the first evaluation, then take the velocity there. O(h^2) against
    Euler's O(h), for twice the teacher cost -- the paper pairs this with a coarser N=128 grid.
    """
    h = grid.h(k)
    v0 = teacher.velocity(x, grid.times[k], cond=cond)
    x_mid = x + v0 * (0.5 * h)
    return teacher.velocity(x_mid, grid.times[k] + 0.5 * h, cond=cond)


SOLVERS: dict[str, Callable[..., Any]] = {
    "euler": mean_velocity_euler,
    "midpoint": mean_velocity_midpoint,
}


def advance(x: Any, heads: Any, grid: Grid, start: int, stop: int) -> Any:
    """Compose student head predictions into a state jump: x + sum_l h_l * u_l.

    Shared by training (advancing within a block to reach the supervised index) and by sampling
    (the block-step rule). One implementation so the two cannot drift -- if training composed heads
    differently from sampling, the student would be optimised for a jump it never performs.
    """
    out = x
    for l in range(start, stop):
        out = out + heads[l] * grid.h(l)
    return out


def block_sample(student: MultiHeadVelocityModel, x: Any, grid: Grid, *, cond=None,
                 block: int | None = None) -> Any:
    """Algorithm 1: generate by taking N/L block steps, one student forward each.

    `block` overrides the grid's block size, which is how variable-NFE sampling works: a student
    trained over a range of block sizes can be asked at serving time for a coarser or finer jump
    without retraining. That is the property that makes PDD worth building a runtime around.
    """
    L = block or grid.block
    if grid.n_intervals % L:
        raise ValueError(f"block={L} does not divide {grid.n_intervals} intervals")
    for n in range(0, grid.n_intervals, L):
        heads = student.heads(x, grid.times[n], cond=cond)
        x = advance(x, heads, grid, n, n + L)
    return x


def pdd_loss(student: MultiHeadVelocityModel, teacher: VelocityModel, x_n: Any, grid: Grid,
             n: int, k: int, *, cond=None, solver: str = "euler", loss: str = "mse"):
    """Algorithm 2 / Eq 11, for one sampled (n, k) pair.

        heads   = student(X_n, t_n)                     one forward, all heads
        X_k     = X_n + sum_{l=n}^{k-1} h_l * heads_l   student's own trajectory
        target  = RK(teacher, sg(X_k), t_k, h_k)        teacher's mean velocity, there
        L_PD    = || heads_k - sg(target) ||^2

    Returns `(loss, metrics)`.

    Two stop-gradients, for different reasons. On the target because the teacher is frozen and we
    are regressing onto it. On X_k because the paper says so explicitly -- and because without it
    the gradient would flow back through every head used to build the state, turning one supervised
    prediction into a backward pass over the whole block for no benefit the objective asks for.
    """
    import torch

    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; have {sorted(SOLVERS)}")
    if not (0 <= n <= k < grid.n_intervals):
        raise ValueError(
            f"need 0 <= n <= k < {grid.n_intervals}, got n={n}, k={k}. k indexes the interval being "
            f"supervised, so it must be a real interval and must lie at or after the block start.")
    # Heads are indexed by ABSOLUTE grid position, not by offset within the block. The paper gives
    # the student "one final linear layer for each time step in the grid", and Algorithm 2 reads
    # u[k], not u[k-n]. Relative indexing looks equivalent and is not: it would supervise head k-n
    # with interval k's target, so every head would be trained on a mixture of intervals and the
    # loss would still fall. Cheap to get wrong, invisible afterwards.
    if k >= student.n_heads:
        raise ValueError(
            f"interval k={k} but the student has {student.n_heads} heads; it needs one per grid "
            f"point (n_heads == grid.n_intervals).")
    if k - n >= grid.block:
        raise ValueError(
            f"k - n = {k - n} exceeds block size {grid.block}; the supervised interval lies outside "
            f"the block this forward pass can predict.")

    heads = student.heads(x_n, grid.times[n], cond=cond)
    x_k = advance(x_n, heads, grid, n, k).detach()          # sg(X_k)

    with torch.no_grad():
        target = SOLVERS[solver](teacher, x_k, grid, k, cond=cond)

    pred = heads[k]
    fn = torch.nn.functional.huber_loss if loss == "huber" else torch.nn.functional.mse_loss
    value = fn(pred, target)

    with torch.no_grad():
        # How far the teacher's own trajectory departs from a constant-velocity jump over this
        # interval. If this collapses to ~0 the target carries no information the student could not
        # get for free, which is the failure mode that makes a plausible-looking objective useless.
        drift = float((target - (x_k - x_n) / max(1e-12, grid.times[k] - grid.times[n])).abs().mean()
                      ) if k > n else float("nan")
    return value, {"pdd/loss": float(value.detach()), "pdd/n": float(n), "pdd/k": float(k),
                   "pdd/span": float(k - n), "pdd/target_drift": drift}

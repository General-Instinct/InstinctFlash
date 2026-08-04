"""The time grid PDD trains and samples on.

PDD fixes a grid of N+1 times and works in blocks of L, so NFE = N/L. Both training and inference
read the same grid, which is the point: the student is only ever asked to jump between times it was
trained to jump between.
"""

from __future__ import annotations

from dataclasses import dataclass


def shift_time(t: float, s: float) -> float:
    """The paper's time warp, Eq 16: shift_s(t) = (t/s) / (1 + (1/s - 1) t).

    Written in the paper's `s` convention. Note this is the SAME curve most flow-matching samplers
    call `shift`, under a reciprocal: with s = 1/shift it reduces to shift*t / (1 + (shift-1) t),
    which is exactly what LingBot-VA's `FlowMatchScheduler` computes. `Grid.from_shift` takes the
    sampler-style `shift` so callers do not have to remember which convention they are in -- getting
    that backwards warps the grid the wrong way and is invisible until accuracy is quietly worse.
    """
    if s <= 0:
        raise ValueError(f"s must be > 0, got {s}")
    return (t / s) / (1.0 + (1.0 / s - 1.0) * t)


@dataclass(frozen=True)
class Grid:
    """N+1 times plus the block size L -- and TWO time axes, which is not a redundancy.

    `times[i]` is the ODE variable: the thing the state is integrated against, so interval widths
    `h` come from here and a step is `x + v * h`.

    `cond(i)` is what the BACKBONE is conditioned on, which is not always the same number. LingBot-VA
    integrates in sigma over [0, 1] -- `FlowMatchScheduler.step` computes
    `sample + model_output * (sigma_next - sigma)` -- but conditions the transformer on
    `sigma * num_train_timesteps`, i.e. sigma * 1000.

    Collapsing the two is a 1000x error in every Euler step, and a silent one: training still runs,
    the loss still moves, and only the samples are wrong. Keeping both axes on the grid is what makes
    it impossible to use the wrong one by accident.
    """
    times: tuple[float, ...]
    block: int
    time_scale: float = 1.0

    def cond(self, i: int):
        """The conditioning value for grid point i."""
        return self.times[i] * self.time_scale

    def cond_at(self, t: float):
        """The conditioning value for an arbitrary ODE time (the midpoint solver needs this)."""
        return t * self.time_scale

    def __post_init__(self):
        if len(self.times) < 2:
            raise ValueError("a grid needs at least two times")
        if self.block < 1:
            raise ValueError(f"block must be >= 1, got {self.block}")
        if self.n_intervals % self.block:
            raise ValueError(
                f"block={self.block} does not divide {self.n_intervals} intervals. NFE = N/L must "
                f"be an integer, otherwise the last block would run off the end of the grid.")

    @property
    def n_intervals(self) -> int:
        return len(self.times) - 1

    @property
    def nfe(self) -> int:
        """Network evaluations per sample: one student forward per block."""
        return self.n_intervals // self.block

    def h(self, k: int) -> float:
        """Width of interval k, signed. Negative when the grid descends (noise -> data)."""
        return self.times[k + 1] - self.times[k]

    def block_starts(self) -> tuple[int, ...]:
        return tuple(range(0, self.n_intervals, self.block))

    @classmethod
    def from_sigmas(cls, sigmas, block: int, *, scale: float = 1.0,
                    append_terminal: bool = True) -> "Grid":
        """Build a grid from a sampler's OWN sigma list. Prefer this over `from_shift`.

        Re-deriving a schedule is how a training grid silently stops matching the sampler's. If the
        deployment scheduler can hand over its sigmas, take them: then there is nothing to get wrong.
        `append_terminal` adds a trailing 0.0 when the list does not already end there, mirroring
        samplers that pad a final step onto clean data.
        """
        vals = [float(s) for s in sigmas]
        if not vals:
            raise ValueError("sigmas is empty")
        if append_terminal and abs(vals[-1]) > 1e-12:
            vals.append(0.0)
        return cls(times=tuple(vals), block=block, time_scale=scale)

    @classmethod
    def from_shift(cls, n_intervals: int, block: int, *, shift: float = 1.0,
                   t_start: float = 1.0, t_end: float = 0.0, scale: float = 1.0) -> "Grid":
        """Build a shifted grid, warping the ODE VARIABLE -- not the progress fraction.

        THE DIRECTION OF THE WARP IS THE WHOLE SUBTLETY, and getting it backwards is silent. The
        first version of this method computed `shift(i/N)` and then mapped that through
        `t_start -> t_end`, i.e. it warped the PROGRESS fraction and inverted. A sampler warps sigma
        itself: `sigma <- shift*sigma / (1 + (shift-1)*sigma)` applied to the descending sigma list.

        Those are different grids, not two spellings of one. Measured against LingBot-VA's real
        FlowMatchScheduler at N=25, shift=5: the correct second grid point is sigma*1000 = 991.7,
        the inverted one gives 827.6. The inverted grid clusters its steps at the DATA end while the
        sampler clusters them at the NOISE end, so a student trained on it would be asked at serving
        time to make jumps it had never practised -- and the loss curve would look fine throughout.

        `t_start`/`t_end` bound the ODE variable (1 -> 0 for flow matching, noise to data). `scale`
        becomes `time_scale`; see the class docstring for why it is not folded into `times`.
        """
        if n_intervals < 1:
            raise ValueError("n_intervals must be >= 1")
        # Uniform in the ODE variable first...
        vals = [t_start + (t_end - t_start) * (i / n_intervals) for i in range(n_intervals + 1)]
        # ...then warp each value in place. shift_time expects [0, 1]; sigma already is.
        if shift != 1.0:
            vals = [shift_time(v, 1.0 / shift) for v in vals]
        return cls(times=tuple(vals), block=block, time_scale=scale)

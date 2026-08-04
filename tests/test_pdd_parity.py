#!/usr/bin/env python3
"""The sigma <-> t convention bridge the LingBot adapter depends on.

HISTORY, because it explains what this file is for. InstinctWM used to carry its own copy of PDD in
`instinctwm/train/pdd/`, integrating sigma DESCENDING 1 -> 0. The algorithm now lives in the
`instinct-pdd` submodule, which fixes time ASCENDING 0 -> 1 to match the paper's interpolant. Before
the old copy was deleted, a parity test compared the two directly across the mapping and found them
identical -- `max|Δ| = 0.00e+00` on grid widths, `advance()`, block sampling at L = 1/2/4/8, and the
loss at four (n, k) pairs, plus identical gradient routing. That test could not survive the deletion
of the thing it compared against.

What survives, and needs to, is the INVARIANT it established: under

    t = 1 - sigma        (same index order, no reversal)
    dt = -dsigma    =>   v_t = -v_sigma

a step in one convention equals a step in the other, because both the width and the velocity flip
sign. `instinctwm/adapter/lingbot_velocity.py` relies on exactly this -- it maps the scheduler's
descending sigmas onto an ascending `Grid` and negates every velocity crossing the boundary. A
one-sided flip would train against a target pointing backwards along the trajectory, and the loss
would fall regardless.

So this file checks the bridge itself, against integration done by hand in sigma. No submodule
internals, no server, no GPU.

    $IWM_SERVER_PY tests/test_pdd_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The submodule is not installed; add it explicitly rather than relying on import order
# through instinctwm/__init__.py, since these tests import instinct_pdd directly.
sys.path.insert(0, str(ROOT / "instinct-pdd" / "src"))

import torch  # noqa: E402

from instinct_pdd import Grid, advance, pdd_loss, sample  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


#: A descending sigma schedule, shaped like one FlowMatchScheduler would emit (terminal 0 included).
SIGMAS = [1.0, 0.82, 0.61, 0.44, 0.29, 0.17, 0.08, 0.03, 0.0]
SCALE = 1000.0


def bridged_grid(block: int) -> Grid:
    """Exactly what the adapter builds: t = 1 - sigma, with cond() still reporting sigma * 1000."""
    return Grid.from_times([1.0 - s for s in SIGMAS], block=block, scale=-SCALE, offset=SCALE)


def integrate_in_sigma(fn, x0, *, block: int):
    """Reference: block-step in the OLD convention, by hand.

    Widths are `sigma_{k+1} - sigma_k` (negative) and `fn` is a sigma-velocity, so this is the exact
    arithmetic the retired implementation performed.
    """
    x = x0
    for n in range(0, len(SIGMAS) - 1, block):
        v = fn(x, SIGMAS[n] * SCALE)                 # one evaluation per block, at its left endpoint
        for k in range(n, min(n + block, len(SIGMAS) - 1)):
            x = x + v * (SIGMAS[k + 1] - SIGMAS[k])
    return x


class NegatedHeads:
    """A student in the NEW convention wrapping a sigma-velocity: the sign flip, once."""

    def __init__(self, fn, n_heads):
        self.fn, self.n_heads = fn, n_heads

    def heads(self, x, t, *, cond=None):
        return torch.stack([-self.fn(x, t) for _ in range(self.n_heads)], dim=0)


class NegatedField:
    def __init__(self, fn):
        self.fn = fn

    def velocity(self, x, t, *, cond=None):
        return -self.fn(x, t)


def main() -> int:
    torch.manual_seed(0)
    N = len(SIGMAS) - 1

    print("=== 1. the mapped grid still describes the sigma schedule ===")
    g = bridged_grid(block=4)
    check(g.n_intervals == N and g.nfe == 2, f"N={g.n_intervals}, NFE={g.nfe}")
    dc = max(abs(g.cond(i) - SIGMAS[i] * SCALE) for i in range(N + 1))
    check(dc < 1e-6, "cond(i) == sigma_i * 1000 -- the backbone sees the sigma it expects",
          f"max|Δ| = {dc:.2e}")
    check(all(g.h(k) > 0 for k in range(N)), "widths are positive on the ascending axis")
    dh = max(abs(g.h(k) - (SIGMAS[k] - SIGMAS[k + 1])) for k in range(N))
    check(dh < 1e-12, "and each width is |dsigma|", f"max|Δ| = {dh:.1e}")

    print("\n=== 2. a step in t equals a step in sigma (both signs flip) ===")
    for name, fn in (("constant", lambda x, t: torch.full_like(x, 0.37)),
                     ("linear in x", lambda x, t: 0.6 * x),
                     ("time varying", lambda x, t: 0.1 * x + float(t) / SCALE)):
        x0 = torch.randn(6, 3)
        hs = NegatedHeads(fn, N).heads(x0, g.cond(0))
        got = advance(x0, hs, g, 0, 4)
        want = x0
        v = fn(x0, SIGMAS[0] * SCALE)
        for k in range(0, 4):
            want = want + v * (SIGMAS[k + 1] - SIGMAS[k])
        d = float((got - want).abs().max())
        check(d < 1e-6, f"advance matches hand-integrated sigma for a {name} field",
              f"max|Δ| = {d:.2e}")

    print("\n=== 3. whole-schedule sampling agrees at every block size ===")
    fn = lambda x, t: 0.45 * x + 0.2                                        # noqa: E731
    for L in (1, 2, 4, 8):
        x0 = torch.randn(5, 3)
        got = sample(NegatedHeads(fn, N), x0, bridged_grid(L))
        want = integrate_in_sigma(fn, x0, block=L)
        d = float((got - want).abs().max())
        check(d < 1e-5, f"L={L} (NFE={N//L}) endpoint agrees", f"max|Δ| = {d:.2e}")

    print("\n=== 4. the flip must be applied to BOTH sides, or the loss changes ===")
    tgt = lambda x, t: 0.55 * x + 0.05                                      # noqa: E731
    stu = lambda x, t: 0.3 * x - 0.15                                       # noqa: E731
    x_n = torch.randn(7, 3)
    both = pdd_loss(NegatedHeads(stu, N), NegatedField(tgt), x_n, g, 0, 3)

    class Unflipped:                     # student negated, teacher NOT -- the one-sided mistake
        def velocity(self, x, t, *, cond=None):
            return tgt(x, t)

    one = pdd_loss(NegatedHeads(stu, N), Unflipped(), x_n, g, 0, 3)
    check(float(both.loss) < float(one.loss),
          "flipping both sides gives a smaller loss than flipping one",
          f"both {float(both.loss):.6f} vs one-sided {float(one.loss):.6f}")
    check(abs(float(both.loss) - float(one.loss)) > 1e-3,
          "the one-sided error is large, not a rounding difference",
          f"Δ = {abs(float(both.loss) - float(one.loss)):.4f}")

    print("\n" + "=" * 72)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the sigma<->t bridge is consistent; the adapter's mapping is sound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

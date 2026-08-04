#!/usr/bin/env python3
"""The PDD grid must equal LingBot-VA's real serving schedule, to the float.

Lives OUTSIDE tests/test_pdd_core.py on purpose: it imports LingBot's scheduler, and the core tests
assert that `instinctwm/train/pdd/**` has no LingBot dependency. Keeping the parity check here is what
lets both statements stay true.

WHY IT EXISTS. `Grid.from_shift` warped the wrong axis -- it shifted the progress fraction and then
inverted, instead of shifting sigma itself. At N=25, shift=5 the correct second timestep is 991.7 and
the broken one was 827.6, so the training grid clustered its steps at the DATA end while the sampler
clusters them at the NOISE end. Nothing about that is visible in a loss curve: the student trains
happily on intervals it will never be asked to jump.

    $IWM_SERVER_PY tests/test_pdd_lingbot_parity.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.path.join(os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va"), "wan_va"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from utils.scheduler import FlowMatchScheduler  # noqa: E402

from instinctwm.train.pdd import Grid  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def served(shift: float, n: int):
    """The exact padded timestep list the server walks, from its own construction.

    wan_va_server.py:51-57 builds the scheduler with sigma_min=0.0 and extra_one_step=True, and
    line 473 pads a terminal zero.
    """
    sch = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    sch.set_timesteps(n)
    return sch, F.pad(sch.timesteps, (0, 1), mode="constant", value=0)


def main() -> int:
    # Both served streams, at the paper's N and at the config's own step counts.
    for name, shift, n in (("video", 5.0, 25), ("action", 1.0, 50),
                           ("video", 5.0, 256), ("action", 1.0, 256)):
        sch, real = served(shift, n)

        g = Grid.from_shift(n_intervals=n, block=n, shift=shift, scale=1000.0)
        mine = torch.tensor([g.cond(i) for i in range(len(g.times))])
        ok_len = len(mine) == len(real)
        d = float((mine - real).abs().max()) if ok_len else float("nan")
        check(ok_len and d < 1e-2, f"from_shift matches served {name} at N={n}",
              f"len {len(mine)}/{len(real)}  max|Δ|={d:.2e}")

        g2 = Grid.from_sigmas(sch.sigmas.tolist(), block=n, scale=1000.0)
        m2 = torch.tensor([g2.cond(i) for i in range(len(g2.times))])
        ok2 = len(m2) == len(real)
        d2 = float((m2 - real).abs().max()) if ok2 else float("nan")
        check(ok2 and d2 == 0.0, f"from_sigmas is EXACT for {name} at N={n}",
              f"max|Δ|={d2:.2e}")

    # The warp direction, stated as a property rather than a number: shift>1 must hold sigma above
    # the linear grid. This is the assertion that fails if the axis is ever inverted again.
    lin = Grid.from_shift(n_intervals=256, block=256, shift=1.0)
    vid = Grid.from_shift(n_intervals=256, block=256, shift=5.0)
    check(vid.times[1] > lin.times[1], "shift=5 holds sigma ABOVE linear (steps at the noise end)",
          f"{vid.times[1]:.5f} > {lin.times[1]:.5f}")
    check(abs(lin.times[1] - (1.0 - 1.0 / 256)) < 1e-9, "shift=1 is the identity warp")

    print("\n" + "=" * 66)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the PDD grid reproduces LingBot-VA's served schedule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

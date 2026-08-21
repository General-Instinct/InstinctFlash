#!/usr/bin/env python3
"""The PDD grid must equal LingBot-VA's real serving schedule, to the float.

Lives in InstinctFlash rather than in the submodule on purpose: it imports LingBot's scheduler, and
`instinct-pdd` must stay free of any LingBot dependency. Keeping this check on this side is what lets
both statements be true at once.

WHY IT EXISTS. `Grid.from_shift` warped the wrong axis -- it shifted the progress fraction and then
inverted, instead of shifting sigma itself. At N=25, shift=5 the correct second timestep is 991.7 and
the broken one was 827.6, so the training grid clustered its steps at the DATA end while the sampler
clusters them at the NOISE end. Nothing about that is visible in a loss curve: the student trains
happily on intervals it will never be asked to jump.

    $IFL_SERVER_PY tests/test_pdd_lingbot_parity.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The submodule is not installed; add it explicitly rather than relying on import order
# through instinctflash/__init__.py, since these tests import instinct_pdd directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "instinct-pdd" / "src"))
sys.path.insert(0, os.path.join(os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va"), "wan_va"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from utils.scheduler import FlowMatchScheduler  # noqa: E402

from instinct_pdd import Grid  # noqa: E402

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
    # Both served streams, at the config's own step counts and at the paper's N.
    for name, shift, n in (("video", 5.0, 25), ("action", 1.0, 50),
                           ("video", 5.0, 256), ("action", 1.0, 256)):
        sch, real = served(shift, n)
        sigmas = [float(s) for s in sch.sigmas]
        if abs(sigmas[-1]) > 1e-12:
            sigmas.append(0.0)

        # Exactly what LingBotChunk0VideoOracle.grid() builds.
        scale = float(sch.num_train_timesteps)
        g = Grid.from_times([1.0 - s for s in sigmas], block=n, scale=-scale, offset=scale)

        check(len(g.times) == len(real), f"{name} N={n}: grid point count matches the served list",
              f"{len(g.times)} vs {len(real)}")
        cond = torch.tensor([g.cond(i) for i in range(len(g.times))])
        d = float((cond - real).abs().max()) if len(cond) == len(real) else float("nan")
        check(d < 1e-3, f"{name} N={n}: cond(i) reproduces the served timesteps EXACTLY",
              f"max|Δ| = {d:.2e}")
        check(all(g.h(k) > 0 for k in range(g.n_intervals)),
              f"{name} N={n}: the ODE axis ascends (t = 1 - sigma)")
        check(abs(g.times[0]) < 1e-9 and abs(g.times[-1] - 1.0) < 1e-9,
              f"{name} N={n}: t runs 0 -> 1", f"{g.times[0]:.3f} -> {g.times[-1]:.3f}")

    # And the property that makes the served schedule what it is: shift > 1 keeps sigma high for
    # longer, so in t-space the early intervals are SHORT. Checked against the real scheduler rather
    # than re-derived, since re-deriving a schedule is how a training grid stops matching a sampler.
    sch5, _ = served(5.0, 256)
    sch1, _ = served(1.0, 256)
    g5 = Grid.from_times([1.0 - float(s) for s in sch5.sigmas] + [1.0], block=256)
    g1 = Grid.from_times([1.0 - float(s) for s in sch1.sigmas] + [1.0], block=256)
    check(g5.h(0) < g1.h(0),
          "shift=5 makes the first interval shorter in t (steps concentrated at the noise end)",
          f"h0: shift5 {g5.h(0):.5f} < shift1 {g1.h(0):.5f}")

    print("\n" + "=" * 66)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the adapter's grid mapping reproduces LingBot-VA's served schedule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared measurement hygiene for performance gates.

WHY THIS IS ONE MODULE AND NOT COPIED PER TEST

A perf gate has two independent verdicts, and conflating them has cost us real time:

    CORRECTNESS   unconditional. Numerics do not care who else is on the box, so this always runs
                  and always reports.
    SPEED         conditional. A latency comparison on a contended device measures the neighbour,
                  not the change. When the device is occupied the honest answer is NOT EVALUATED.

`test_triton_residual.py` learned this the hard way: a 48%-busy neighbour turned a genuine 1.20x
into a reported kernel regression, and the regression was investigated before the contention was
noticed. `test_engine_graph.py` then failed the same way during a repository reorganisation, because
the guard lived in the other file. So it lives here now, and both import it.

    from tests.perf_gate import device_busy, spread, abba, BUSY_UTIL, MAX_SPREAD

ORDER CONTROL. `abba()` runs base, treat, treat, base rather than base-then-treat, because within a
session latency drifts monotonically (clocks, thermals, cache state) and a sequential A/B charges
the whole drift to the treatment. ABBA cancels the linear component: the two base samples bracket
the two treatment samples in time, so their mean sits at the same point on the drift curve.
"""

from __future__ import annotations

import os
import subprocess

#: Utilisation on ANY device above which the box counts as occupied. Deliberately low: a 48%
#: neighbour was enough to turn a 1.20x measurement into a reported regression.
BUSY_UTIL = 15

#: ABBA repeats, and the within-arm spread above which no speed verdict is offered.
REPEATS = 3
MAX_SPREAD = 0.15


def spread(xs) -> float:
    """Relative spread of a sample: (max-min)/mean.

    Cheap, and sensitive to the single slow outlier that contention actually produces -- which a
    standard deviation would dilute across the sample.
    """
    xs = list(xs)
    m = sum(xs) / len(xs) if xs else 0.0
    return (max(xs) - min(xs)) / m if m > 0 else float("inf")


def device_busy() -> tuple[bool, str]:
    """Is anything else using the GPUs? Returns (busy, why).

    Checked by utilisation AND by foreign compute processes. Utilisation alone misses a neighbour
    that is between kernels; the process list alone misses a busy device whose owner is not visible
    to us. Either signal is enough to withhold a verdict, and so is failing to determine the state
    at all -- an unknown device is not an idle device.
    """
    try:
        u = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15).stdout
        hot = [ln for ln in u.strip().split("\n")
               if ln.strip() and int(ln.split(",")[1]) >= BUSY_UTIL]
        if hot:
            return True, f"GPU utilisation {'; '.join(x.strip() for x in hot)}%"
        pids = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=15).stdout.split()
        foreign = [x for x in pids if x.strip() and x.strip() != str(os.getpid())]
        if foreign:
            return True, f"{len(foreign)} other compute process(es) on the GPUs"
    except Exception as e:
        return True, f"could not determine device state ({type(e).__name__}); withholding a verdict"
    return False, ""


def abba(measure_base, measure_treat, repeats: int = REPEATS):
    """Measure two arms under ABBA ordering. Returns (base_samples, treat_samples).

    Each callable must return a scalar latency. Called `repeats` times each, interleaved
    base, treat, treat, base so that monotonic within-session drift cancels rather than
    accruing entirely to the treatment arm.
    """
    base, treat = [], []
    for _ in range(repeats):
        base.append(measure_base())
        treat.append(measure_treat())
        treat.append(measure_treat())
        base.append(measure_base())
    return base, treat


def speed_verdict(base, treat, label: str = "treatment") -> tuple[bool | None, str]:
    """(verdict, message) for a speed comparison. `None` means NOT EVALUATED, not failure.

    Withholds when the device is occupied, and when the within-arm spread is wide enough that the
    arms are not separable -- reporting a ratio built from noise is how a contended box gets
    recorded as a regression.
    """
    busy, why = device_busy()
    if busy:
        return None, f"NOT EVALUATED -- the device is materially occupied ({why})."
    sb, st = spread(base), spread(treat)
    if max(sb, st) > MAX_SPREAD:
        return None, (f"NOT EVALUATED -- within-arm spread {max(sb, st):.1%} exceeds "
                      f"{MAX_SPREAD:.0%}; the arms are not separable.")
    mb = sum(base) / len(base)
    mt = sum(treat) / len(treat)
    ratio = mb / mt if mt > 0 else float("inf")
    return ratio > 1.0, (f"{label}: base {mb:.3f} vs treat {mt:.3f} -> {ratio:.3f}x "
                         f"(spread base {sb:.1%}, treat {st:.1%})")

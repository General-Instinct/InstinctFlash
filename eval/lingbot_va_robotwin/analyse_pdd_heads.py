#!/usr/bin/env python3
"""Read the run's reports and rank heads by what they actually cost the endpoint.

WHY THIS EXISTS SEPARATELY FROM THE TRAINER. The trainer reports raw per-head MSE, and raw MSE is not
comparable across sigma. With snr_shift=5 the noise end is compressed hard: heads 0-31 span sigma
1.000-0.973 and carry 2.8% of the two-step jump, while heads 224-255 span 0.417-0.019 and carry 41.7%.
Target magnitude also grows toward sigma=1, since the velocity in near-pure noise is large. Both
effects inflate raw MSE exactly where influence is lowest, so ranking heads by it surfaces the heads
that matter least -- at step 2500 the reported "worst 10" were all k=34-42, contributing 7.6% of the
endpoint error between them, while k=248-255 quietly owned 44%.

The honest per-head quantity is the state error a head injects into the jump: h_k * RMSE_k. That is
what this ranks by.

    $IWM_SERVER_PY analyse_pdd_heads.py --run /home/ubuntu/iwm_results/pdd_heads_run1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
for p in (IWM_ROOT, os.path.join(IWM_ROOT, "instinct-pdd", "src"),
          os.path.join(os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va"), "wan_va")):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.scheduler import FlowMatchScheduler  # noqa: E402

from instinct_pdd import Grid  # noqa: E402


def build_grid(n_intervals: int, block: int, shift: float = 5.0):
    """The served video grid, mapped exactly as the adapter maps it."""
    sch = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    sch.set_timesteps(n_intervals)
    sig = [float(s) for s in sch.sigmas]
    if abs(sig[-1]) > 1e-12:
        sig.append(0.0)
    return Grid.from_times([1.0 - s for s in sig], block=block, scale=-1000.0, offset=1000.0)


def contributions(grid, err):
    return [grid.h(k) * math.sqrt(max(err[k], 0.0)) for k in range(grid.n_intervals)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--buckets", type=int, default=8)
    a = ap.parse_args()

    reports = sorted(Path(a.run).glob("report_step*.json"),
                     key=lambda p: int(p.stem.replace("report_step", "")))
    if not reports:
        raise SystemExit(f"no report_step*.json in {a.run}")

    loaded = [(int(p.stem.replace("report_step", "")), json.loads(p.read_text())) for p in reports]
    n = len(loaded[0][1]["per_head_error"])
    grid = build_grid(n, n // 2)

    print(f"{'step':>7} {'endpoint':>10} {'% scale':>9} {'min upd':>8} {'gate':>6} "
          f"{'raw mean':>10} {'wtd total':>10}")
    for step, d in loaded:
        err = d["per_head_error"]
        c = contributions(grid, err)
        u = d["head_updates"]
        print(f"{step:>7} {d['endpoint_rmse']:>10.4f} "
              f"{100 * d['endpoint_rmse'] / d['endpoint_scale']:>8.1f}% {min(u):>8} "
              f"{'PASS' if d['coverage_gate_pass'] else 'FAIL':>6} "
              f"{sum(err) / n:>10.4f} {sum(c):>10.4f}")

    # Per-bucket trajectory: this is the capacity question. If the high-influence band (the last
    # bucket) stops moving while the middle keeps improving, the limit is localised and unfreezing
    # the final blocks is the targeted intervention -- not full fine-tuning.
    print(f"\nweighted contribution by sigma bucket (share of endpoint error), per validation:")
    hdr = "  ".join(f"{s:>9}" for s, _ in loaded)
    print(f"  {'heads':>10} {'sigma':>16}  {hdr}")
    per_b = a.buckets
    width = n // per_b
    for b in range(per_b):
        lo, hi = b * width, (b + 1) * width
        cells = []
        for _, d in loaded:
            c = contributions(grid, d["per_head_error"])
            cells.append(f"{100 * sum(c[lo:hi]) / sum(c):>8.1f}%")
        srange = f"{grid.cond(lo) / 1000:.3f}-{grid.cond(hi - 1) / 1000:.3f}"
        print(f"  {f'{lo}-{hi - 1}':>10} {srange:>16}  " + "  ".join(cells))

    print(f"\nABSOLUTE weighted contribution (is the band actually improving, or just its share?):")
    print(f"  {'heads':>10} {'sigma':>16}  {hdr}")
    for b in range(per_b):
        lo, hi = b * width, (b + 1) * width
        cells = []
        for _, d in loaded:
            c = contributions(grid, d["per_head_error"])
            cells.append(f"{sum(c[lo:hi]):>9.4f}")
        srange = f"{grid.cond(lo) / 1000:.3f}-{grid.cond(hi - 1) / 1000:.3f}"
        print(f"  {f'{lo}-{hi - 1}':>10} {srange:>16}  " + "  ".join(cells))

    step, d = loaded[-1]
    c = contributions(grid, d["per_head_error"])
    order = sorted(range(n), key=lambda k: -c[k])[:12]
    print(f"\nworst 12 heads by contribution at step {step}:")
    print(f"  {'k':>5} {'sigma':>8} {'raw MSE':>10} {'h':>9} {'contrib':>9} {'updates':>8}")
    for k in order:
        print(f"  {k:>5} {grid.cond(k) / 1000:>8.3f} {d['per_head_error'][k]:>10.4f} "
              f"{grid.h(k):>9.5f} {c[k]:>9.5f} {d['head_updates'][k]:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

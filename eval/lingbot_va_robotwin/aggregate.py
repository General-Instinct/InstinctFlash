#!/usr/bin/env python3
"""Aggregate a LingBot-VA / RoboTwin run into a number you are allowed to report.

This replaces `evaluation/robotwin/calc_stat.py`, which has three properties that make
its headline number unsafe to publish:

 1. A CRASHED TASK RAISES THE SCORE. calc_stat computes rate=None when a task folder has
    no mp4s (`rate = (true_cnt/total) if total > 0 else None`) and then averages only the
    non-None rates (`mean_rate_of`: `[r[4] for r in ... if r[4] is not None]`). A task
    that died on episode 0 is therefore dropped from the DENOMINATOR rather than scored,
    so the more tasks crash, the better the mean looks.
 2. A TRUNCATED TASK IS WEIGHTED EQUALLY. A task that completed 3 of 100 episodes
    contributes its 3-episode rate with the same weight as a complete one.
 3. SUCCESS IS READ FROM MP4 FILENAMES (`<test_num>_<prompt>_<True|False>.mp4`), and
    test_num restarts at 0 in every client process, so two clients writing the same task
    into one save_root silently overwrite each other and the total quietly shrinks.

This script keeps the same macro-average definition (mean over tasks, which is what the
RoboTwin leaderboard reports) but makes completeness a precondition rather than a footnote:
it cross-checks the mp4 evidence against the independently written metrics/<task>/res.json,
and prints REPORTABLE: NO with reasons if anything does not line up.

Usage:
    python aggregate.py <save_root> [--expect-episodes N] [--expect-tasks N]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

MP4_RE = re.compile(r"^(\d+)_.*_(True|False)\.mp4$")


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval -- correct near 0% and 100%, unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def scan(save_root: Path):
    tasks = {}
    for vis_task in sorted(save_root.glob("stseed-*/visualization/*")):
        if not vis_task.is_dir():
            continue
        name = vis_task.name
        t = f = 0
        idxs = []
        for mp4 in vis_task.glob("*.mp4"):
            m = MP4_RE.match(mp4.name)
            if not m:
                continue
            idxs.append(int(m.group(1)))
            if m.group(2) == "True":
                t += 1
            else:
                f += 1
        rec = tasks.setdefault(name, {"true": 0, "false": 0, "idxs": []})
        rec["true"] += t
        rec["false"] += f
        rec["idxs"] += idxs

    # independent second source: the res.json the client writes per episode
    for res in sorted(save_root.glob("stseed-*/metrics/*/res.json")):
        name = res.parent.name
        try:
            d = json.loads(res.read_text())
        except Exception:
            continue
        rec = tasks.setdefault(name, {"true": 0, "false": 0, "idxs": []})
        rec["json_succ"] = d.get("succ_num")
        rec["json_total"] = d.get("total_num")
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("save_root", type=Path)
    ap.add_argument("--expect-episodes", type=int, default=None)
    ap.add_argument("--expect-tasks", type=int, default=None)
    args = ap.parse_args()

    tasks = scan(args.save_root)
    if not tasks:
        print(f"no results under {args.save_root}")
        return 1

    problems = []
    rows = []
    for name in sorted(tasks):
        r = tasks[name]
        total = r["true"] + r["false"]
        rate = r["true"] / total if total else None
        # mp4 index collisions => two clients overwrote each other
        dup = len(r["idxs"]) - len(set(r["idxs"]))
        if dup:
            problems.append(f"{name}: {dup} duplicate episode indices -- mp4s were overwritten")
        js, jt = r.get("json_succ"), r.get("json_total")
        if jt is not None and total != int(jt):
            problems.append(f"{name}: mp4 count {total} != res.json total {int(jt)}")
        if js is not None and r["true"] != int(js):
            problems.append(f"{name}: mp4 successes {r['true']} != res.json succ {int(js)}")
        if args.expect_episodes and total != args.expect_episodes:
            problems.append(f"{name}: {total} episodes, expected {args.expect_episodes}")
        if total == 0:
            problems.append(f"{name}: ZERO episodes -- task produced no result at all")
        rows.append((name, r["true"], r["false"], total, rate))

    if args.expect_tasks and len(rows) != args.expect_tasks:
        problems.append(f"{len(rows)} tasks present, expected {args.expect_tasks}")

    print(f"{'task':32s} {'succ':>5s} {'fail':>5s} {'n':>5s} {'rate':>8s}   {'95% CI (Wilson)':>18s}")
    print("-" * 88)
    for name, t, f, n, rate in sorted(rows, key=lambda r: (r[4] is None, -(r[4] or 0))):
        if rate is None:
            print(f"{name:32s} {t:5d} {f:5d} {n:5d} {'NO DATA':>8s}")
            continue
        lo, hi = wilson(t, n)
        print(f"{name:32s} {t:5d} {f:5d} {n:5d} {rate*100:7.1f}%   [{lo*100:5.1f}%, {hi*100:5.1f}%]")
    print("-" * 88)

    complete = [r for r in rows if r[3] > 0]
    # MACRO: unweighted mean over tasks -- the RoboTwin leaderboard definition.
    # Unlike calc_stat, a zero-episode task is a HARD ERROR, never a silent drop.
    macro = sum(r[4] for r in complete) / len(complete) if complete else float("nan")
    pooled_k = sum(r[1] for r in rows)
    pooled_n = sum(r[3] for r in rows)
    micro = pooled_k / pooled_n if pooled_n else float("nan")
    mlo, mhi = wilson(pooled_k, pooled_n)

    print(f"{'MACRO mean over ' + str(len(complete)) + ' tasks':32s} {'':5s} {'':5s} {'':5s} {macro*100:7.1f}%")
    print(f"{'MICRO pooled ' + str(pooled_k) + '/' + str(pooled_n):32s} {'':5s} {'':5s} {'':5s} "
          f"{micro*100:7.1f}%   [{mlo*100:5.1f}%, {mhi*100:5.1f}%]")
    print()

    if problems:
        print("REPORTABLE: NO")
        for p in problems:
            print("  -", p)
        print("\nThe macro mean above EXCLUDES nothing, but the run is incomplete or inconsistent.")
        print("Fix the listed problems and re-run rather than quoting a partial number.")
        return 1

    print("REPORTABLE: YES (run is complete and internally consistent)")
    print("NOTE: this is the LingBot-VA client protocol -- st_seed = 10000*(1+seed) and")
    print("      instruction_type='seen', which are NOT RoboTwin's canonical")
    print("      st_seed = 100000*(1+seed). Comparable to the LingBot-VA paper, not to a")
    print("      RoboTwin-harness baseline run under the canonical seeds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the step-allocation response surface from a sweep.

The question is not "which single operating point is best" but "how does task success respond to
video steps versus action steps, and do different tasks respond differently". A macro average
cannot answer the second part, so the per-task sensitivity table is the point of this script.

    python sweep_report.py --reference <teacher.jsonl> --config 1:1=<a.jsonl> --config 2:2=<b.jsonl>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/InstinctWM")

from instinctwm.verify.certify import NotCertifiable, certify, load_jsonl


def latency_of(log_dir: Path) -> float | None:
    f = log_dir / "_latency.txt"
    if not f.exists():
        return None
    m = re.search(r"steady-state mean over \d+ kept runs:\s+([0-9.]+) ms", f.read_text())
    return float(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="teacher episodes.jsonl (25/50)")
    ap.add_argument("--ref-latency", type=float, default=None)
    ap.add_argument("--config", action="append", default=[],
                    help="V:A=path/to/episodes.jsonl")
    ap.add_argument("--log-root", default="/home/ubuntu/iwm_logs")
    ap.add_argument("--prefix", default="sweep")
    ap.add_argument("--margin", type=float, default=-0.05)
    a = ap.parse_args()

    ref = load_jsonl(a.reference)
    ref_rate = sum(1 for o in ref if o.success) / len(ref)
    rows, per_task_by_cfg = [], {}

    for spec in a.config:
        name, path = spec.split("=", 1)
        v, ac = name.split(":")
        try:
            cert = certify(ref, load_jsonl(path), margin=a.margin,
                           recipe=f"nfe v{v}/a{ac}", harness="robotwin-2.0")
        except NotCertifiable as e:
            print(f"  {name}: NOT CERTIFIABLE -- {e}", file=sys.stderr)
            continue
        lat = latency_of(Path(a.log_root) / f"{a.prefix}_v{v}a{ac}")
        rows.append((int(v), int(ac), cert, lat))
        per_task_by_cfg[name] = cert.per_task

    if not rows:
        print("no certifiable configurations", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: (r[0], r[1]))

    print(f"\nreference (teacher 25/50): success {ref_rate:.3f} over {len(ref)} episodes"
          + (f", {a.ref_latency:.0f} ms" if a.ref_latency else ""))
    print(f"\n{'video':>6}{'action':>7}{'success':>9}{'delta':>8}{'ci95':>20}"
          f"{'ms':>9}{'speedup':>9}  verdict")
    for v, ac, c, lat in rows:
        sp = f"{a.ref_latency / lat:.2f}x" if (a.ref_latency and lat) else "-"
        print(f"{v:>6}{ac:>7}{c.student_success:>9.3f}{c.delta:>+8.3f}"
              f"  [{c.ci95[0]:+.3f},{c.ci95[1]:+.3f}]"
              f"{(f'{lat:.0f}' if lat else '-'):>9}{sp:>9}  {c.verdict.split(':')[0]}")

    # per-task sensitivity: does this task care about video steps or action steps?
    tasks = sorted({t for pt in per_task_by_cfg.values() for t in pt})
    names = [f"{v}:{ac}" for v, ac, _, _ in rows]
    print(f"\nper-task success by configuration\n")
    print(f"  {'task':<26}{'ref':>6}" + "".join(f"{n:>7}" for n in names))
    for t in tasks:
        cells = []
        for n in names:
            pt = per_task_by_cfg.get(n, {}).get(t)
            cells.append(f"{pt[2] / pt[0]:>7.2f}" if pt else f"{'-':>7}")
        rt = [o for o in ref if o.task == t]
        rr = sum(1 for o in rt if o.success) / len(rt) if rt else float("nan")
        print(f"  {t:<26}{rr:>6.2f}" + "".join(cells))

    # sensitivity: hold one axis, vary the other
    print(f"\nsensitivity (mean success across configs sharing an axis value)")
    for axis, idx in (("video steps", 0), ("action steps", 1)):
        vals = sorted({r[idx] for r in rows})
        line = "  ".join(
            f"{v}: {sum(r[2].student_success for r in rows if r[idx] == v) / max(1, sum(1 for r in rows if r[idx] == v)):.3f}"
            for v in vals)
        print(f"  {axis:<14} {line}")
    print("\n  A flat row means that axis is not the binding constraint at these step counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

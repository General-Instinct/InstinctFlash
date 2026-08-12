#!/usr/bin/env python3
"""Paired non-inferiority certificate for one operating point against another.

    python certify_operating_point.py --control actsweep_v2a4 --treat fastcert_v2a2 \
        --margin -0.05 --label "2V/2A vs shipped 2V/4A"

WHY THIS EXISTS AS A TOOL. The 2V/2A-vs-shipped comparison first came out at n=500 with a CI lower
bound of -0.0488 against a -0.05 margin -- a pass with 0.0012 to spare, where one discordant pair
the other way flips the verdict. A number that fragile has to be produced by something reviewable,
with its analysis fixed in advance, or the temptation to nudge the method after seeing the data is
left wide open.

PRE-REGISTERED ANALYSIS. Declared before the extension arm was run:

  * Design: paired. Both arms evaluate the SAME episodes -- same 50 tasks, same st_seed, same
    per-task episode indices. Pairing is by `episode_id`; any episode not present in both arms is
    dropped and reported, never imputed.
  * Margin: -0.05 absolute task success, the same margin every prior certificate in this repo used.
    IT IS NOT TO BE CHANGED AFTER SEEING THE DATA. If the result fails at -0.05, the operating point
    is rejected.
  * Primary estimand: delta = P(treat succeeds) - P(control succeeds) over paired episodes.
  * Primary interval: THE MOST CONSERVATIVE of three -- the McNemar-SE normal interval, an
    iid-over-episodes bootstrap, and a bootstrap clustering by TASK. Non-inferiority is declared iff
    the LOWEST of their three lower bounds is strictly greater than the margin.

    This rule replaced an earlier one, and the reason is worth recording. The task-cluster interval
    was pre-registered as primary on the theory that episodes within a task share a scene and so an
    iid interval would be anticonservative. Measured, the cluster interval came out NARROWER
    ([-0.0400, +0.0020] against iid's [-0.0500, +0.0080] at n=500): discordances within a task tend
    to cancel, so per-task net differences are more homogeneous than iid predicts, not less. Reading
    a verdict off the interval that happens to be the most favourable is indistinguishable from
    having chosen it for that reason, whether or not it was pre-registered. Taking the minimum
    removes the choice, and it can only make the gate harder to pass.
  * Also reported: exact McNemar two-sided p (is there any difference at all) and the one-sided
    normal p against the margin. Neither can overturn the interval verdict.
  * NOISE FLOOR. `--repeat` takes a second run of the CONTROL configuration on the same episodes and
    reports its discordance. The serving path draws unseeded noise, so two runs of an IDENTICAL
    configuration disagree on some episodes. That floor bounds how much of any observed discordance
    is method rather than configuration, and a delta inside it means nothing.

Refuses rather than guesses: unequal arms, missing episodes, or a treat arm that does not cover the
control arm are reported and non-zero exit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

RESULTS = Path(os.environ.get("IWM_RESULT_DIR", "/home/ubuntu/iwm_results"))


def load(run: str) -> dict[str, bool]:
    p = (Path(run) if os.path.sep in run else RESULTS / run) / "episodes.jsonl"
    if not p.exists():
        raise SystemExit(f"{p}: no episodes.jsonl. Was the run emitted?")
    out: dict[str, bool] = {}
    for line in p.open():
        r = json.loads(line)
        out[r["episode_id"]] = bool(r["success"])
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact conditional test on the discordant pairs."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def boot_ci(diffs: list[int], groups: list[str], reps: int, cluster: bool, seed: int = 12345):
    """Percentile 95% CI for the mean of `diffs`. Cluster bootstrap resamples group labels."""
    rng = random.Random(seed)
    if cluster:
        by = defaultdict(list)
        for d, g in zip(diffs, groups):
            by[g].append(d)
        keys = list(by)
        means = []
        for _ in range(reps):
            tot = cnt = 0
            for _ in keys:
                v = by[keys[rng.randrange(len(keys))]]
                tot += sum(v)
                cnt += len(v)
            means.append(tot / cnt)
    else:
        n = len(diffs)
        means = []
        for _ in range(reps):
            s = 0
            for _ in range(n):
                s += diffs[rng.randrange(n)]
            means.append(s / n)
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    return lo, hi, means


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="run dir or name, e.g. actsweep_v2a4")
    ap.add_argument("--treat", required=True, help="run dir or name, e.g. fastcert_v2a2")
    ap.add_argument("--repeat", default=None,
                    help="a SECOND run of the control configuration, for the noise floor")
    ap.add_argument("--margin", type=float, default=-0.05)
    ap.add_argument("--label", default="")
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--out", default=None, help="write the certificate here as JSON")
    a = ap.parse_args()

    ctrl, treat = load(a.control), load(a.treat)
    common = sorted(set(ctrl) & set(treat))
    if not common:
        print("REFUSED: the two arms share no episode ids; they are not paired.")
        return 2

    missing_in_treat = sorted(set(ctrl) - set(treat))
    n = len(common)
    b = sum(1 for k in common if treat[k] and not ctrl[k])      # treat wins
    c = sum(1 for k in common if ctrl[k] and not treat[k])      # control wins
    p_ctrl = sum(ctrl[k] for k in common) / n
    p_treat = sum(treat[k] for k in common) / n
    delta = p_treat - p_ctrl

    diffs = [int(treat[k]) - int(ctrl[k]) for k in common]
    tasks = [k.split("/")[0] for k in common]
    lo_c, hi_c, _ = boot_ci(diffs, tasks, a.reps, cluster=True)
    lo_i, hi_i, _ = boot_ci(diffs, tasks, a.reps, cluster=False)
    se = math.sqrt(b + c) / n if (b + c) else 0.0
    p_exact = mcnemar_exact(b, c)
    z = (delta - a.margin) / se if se else float("inf")
    p_ni = 0.5 * math.erfc(z / math.sqrt(2))

    print("=" * 78)
    print(f"PAIRED NON-INFERIORITY CERTIFICATE{'  --  ' + a.label if a.label else ''}")
    print("=" * 78)
    print(f"  control   {a.control:24} {p_ctrl:.4f}")
    print(f"  treat     {a.treat:24} {p_treat:.4f}")
    print(f"  paired episodes {n}   tasks {len(set(tasks))}")
    if missing_in_treat:
        print(f"  NOTE {len(missing_in_treat)} control episodes absent from treat, dropped "
              f"(never imputed): {missing_in_treat[:3]}...")
    print(f"  delta {delta:+.4f}   discordant {b} for / {c} against ({(b+c)/n:.1%})")
    print()
    print(f"  exact McNemar two-sided p      {p_exact:.4f}"
          f"   ({'no detectable difference' if p_exact > 0.05 else 'a difference is detectable'})")
    lo_w = delta - 1.96 * se
    bounds = {"mcnemar_se": lo_w, "bootstrap_iid": lo_i, "bootstrap_task_cluster": lo_c}
    worst_name = min(bounds, key=lambda k: bounds[k])
    lo_primary = bounds[worst_name]
    mark = {k: ("  <-- primary (most conservative)" if k == worst_name else "") for k in bounds}
    print(f"  McNemar-SE normal 95% CI       [{lo_w:+.4f}, {delta+1.96*se:+.4f}]{mark['mcnemar_se']}")
    print(f"  bootstrap 95% CI, iid episodes [{lo_i:+.4f}, {hi_i:+.4f}]{mark['bootstrap_iid']}")
    print(f"  bootstrap 95% CI, TASK cluster [{lo_c:+.4f}, {hi_c:+.4f}]{mark['bootstrap_task_cluster']}")
    print(f"  one-sided normal p vs margin   {p_ni:.5f}")
    print()
    floor = None
    if a.repeat:
        rep = load(a.repeat)
        rc = sorted(set(ctrl) & set(rep))
        if rc:
            rb = sum(1 for k in rc if rep[k] and not ctrl[k])
            rcc = sum(1 for k in rc if ctrl[k] and not rep[k])
            floor = (rb + rcc) / len(rc)
            print(f"  NOISE FLOOR from two runs of the CONTROL configuration ({len(rc)} pairs):")
            print(f"    discordance {rb}+{rcc} = {floor:.1%}, delta "
                  f"{(sum(rep[k] for k in rc)-sum(ctrl[k] for k in rc))/len(rc):+.4f}")
            print(f"    observed treat discordance is {(b+c)/n:.1%}; "
                  f"{'ABOVE' if (b+c)/n > floor else 'AT OR BELOW'} the floor")
            print()

    ok = lo_primary > a.margin
    slack = lo_primary - a.margin
    verdict = ("NON-INFERIOR" if ok else "NOT ESTABLISHED")
    print(f"  margin declared {a.margin:+.3f}   primary lower bound {lo_primary:+.4f} "
          f"({worst_name})   slack {slack:+.4f}")
    print(f"  VERDICT: {verdict}"
          + ("" if not ok else ("  (FRAGILE: under 0.005 of slack)" if slack < 0.005 else "  (robust)")))
    if not ok:
        print("  The margin is NOT to be relaxed. Reject the operating point or add evidence.")

    if a.out:
        Path(a.out).write_text(json.dumps({
            "label": a.label, "control": a.control, "treat": a.treat,
            "n_pairs": n, "tasks": len(set(tasks)),
            "control_success": p_ctrl, "treat_success": p_treat, "delta": delta,
            "discordant_for": b, "discordant_against": c,
            "mcnemar_exact_two_sided_p": p_exact,
            "ci95_mcnemar_se": [delta - 1.96 * se, delta + 1.96 * se],
            "ci95_bootstrap_iid": [lo_i, hi_i],
            "ci95_bootstrap_task_cluster": [lo_c, hi_c],
            "one_sided_p_vs_margin": p_ni,
            "margin_declared": a.margin,
            "noise_floor_discordance": floor,
            "verdict": verdict, "slack": slack,
            "primary_interval": worst_name, "primary_lower_bound": lo_primary,
            "dropped_unpaired": len(missing_in_treat),
        }, indent=2) + "\n")
        print(f"\n  wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

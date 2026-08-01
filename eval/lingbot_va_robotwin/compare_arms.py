#!/usr/bin/env python3
"""Compare two evaluation arms that ran the SAME pinned scene set.

This answers exactly one question: **did the engine change the policy's behaviour?**

It answers it two ways, and the order matters:

  1. ACTION-STREAM IDENTITY (deductive). For every seed present in both arms, compare the
     raw per-chunk action tensors the server returned, before any client-side
     unnormalization or pose composition. If every stream is bitwise identical then the
     two arms issued the identical `take_action` sequence, the simulator saw identical
     input, and the success outcomes are identical *by construction*. This is a proof, not
     an estimate, and it needs no sample size.

  2. OUTCOME AGREEMENT (empirical). Per-seed success/failure agreement plus McNemar's
     exact test on the discordant pairs.

Read (2) in light of (1). If the action streams are bitwise identical, any outcome
disagreement is RoboTwin nondeterminism, not the engine -- the harness is known to flip
some outcomes run-to-run. In that situation the empirical test is strictly weaker evidence
than the action diff, and reporting only the empirical delta would understate the result.
Conversely, if the action streams differ at all, (2) is the only evidence that counts and
it needs a measured noise floor to be interpretable.

Usage:
    python compare_arms.py --ref  /path/to/ref_actions  --opt /path/to/opt_actions \
                           [--ref-results ... --opt-results ...]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def load_arm(root: Path):
    """task -> seed -> {chunks, success}"""
    out = {}
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        per = {}
        for f in sorted(task_dir.glob("seed_*.npz")):
            seed = int(f.stem.split("_")[1])
            d = np.load(f, allow_pickle=True)
            per[seed] = {"chunks": d["chunks"], "success": bool(d["success"])}
        if per:
            out[task_dir.name] = per
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs (b wins vs c wins)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True, help="reference (stock) action-log dir")
    ap.add_argument("--opt", type=Path, required=True, help="optimized action-log dir")
    args = ap.parse_args()

    ref, opt = load_arm(args.ref), load_arm(args.opt)
    tasks = sorted(set(ref) & set(opt))
    only_ref, only_opt = sorted(set(ref) - set(opt)), sorted(set(opt) - set(ref))

    print(f"tasks in both arms: {len(tasks)}")
    if only_ref:
        print(f"  WARNING tasks only in ref: {only_ref}")
    if only_opt:
        print(f"  WARNING tasks only in opt: {only_opt}")
    print()

    worst = 0.0
    n_pairs = n_ident = n_shape_mismatch = 0
    b = c = 0            # b: ref succeeded & opt failed;  c: opt succeeded & ref failed
    agree = 0
    unpaired_seeds = 0
    per_task = []

    for t in tasks:
        seeds = sorted(set(ref[t]) & set(opt[t]))
        unpaired_seeds += len(set(ref[t]) ^ set(opt[t]))
        t_worst, t_ident = 0.0, 0
        for s in seeds:
            r, o = ref[t][s], opt[t][s]
            n_pairs += 1
            if r["chunks"].shape != o["chunks"].shape:
                # Different chunk COUNT means the episodes diverged in length -- itself a
                # behaviour change, not just a numeric one.
                n_shape_mismatch += 1
                t_worst = float("inf")
                worst = float("inf")
            else:
                d = float(np.abs(r["chunks"].astype(np.float64)
                                 - o["chunks"].astype(np.float64)).max())
                t_worst = max(t_worst, d)
                worst = max(worst, d)
                if d == 0.0:
                    n_ident += 1
                    t_ident += 1
            if r["success"] == o["success"]:
                agree += 1
            elif r["success"] and not o["success"]:
                b += 1
            else:
                c += 1
        per_task.append((t, len(seeds), t_ident, t_worst))

    print(f"{'task':32s} {'paired':>7s} {'bit-ident':>10s} {'max|d|':>12s}")
    print("-" * 66)
    for t, n, ident, w in per_task:
        ws = "SHAPE" if w == float("inf") else f"{w:.3e}"
        print(f"{t:32s} {n:7d} {ident:10d} {ws:>12s}")
    print("-" * 66)
    print()

    print("=" * 66)
    print("1) ACTION-STREAM IDENTITY")
    print(f"   paired episodes            : {n_pairs}")
    print(f"   bitwise-identical streams  : {n_ident}")
    print(f"   chunk-count mismatches     : {n_shape_mismatch}")
    print(f"   worst max|delta action|    : "
          f"{'SHAPE MISMATCH' if worst == float('inf') else f'{worst:.6e}'}")
    if unpaired_seeds:
        print(f"   WARNING unpaired seeds     : {unpaired_seeds} (scene sets not identical)")
    bitexact = (n_pairs > 0 and n_ident == n_pairs and n_shape_mismatch == 0)
    print()
    print("2) OUTCOME AGREEMENT")
    print(f"   agree                      : {agree}/{n_pairs}")
    print(f"   ref succ & opt fail (b)    : {b}")
    print(f"   opt succ & ref fail (c)    : {c}")
    print(f"   McNemar exact p            : {mcnemar_exact(b, c):.4f}")
    print()
    print("=" * 66)
    if bitexact:
        print("VERDICT: the engine did NOT change the policy. Every paired episode produced")
        print("a bitwise-identical action stream, so the trajectories are identical by")
        print("construction and the success rates are equal necessarily, not statistically.")
        if b or c:
            print()
            print(f"NOTE: {b + c} outcome(s) still disagree despite identical actions. That is")
            print("RoboTwin nondeterminism (the harness flips some outcomes run-to-run), and")
            print("it bounds the noise floor for any FUTURE lossy optimization. It is not an")
            print("effect of the engine -- identical actions cannot cause a real difference.")
        return 0
    print("VERDICT: action streams DIFFER. The engine changed the policy's behaviour.")
    print("The success comparison above is now the only evidence, and it must be read")
    print("against a measured noise floor (same server, same pinned seeds, run twice).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

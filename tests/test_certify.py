#!/usr/bin/env python3
"""Certification must be able to FAIL, and must refuse inputs it cannot honestly analyse.

A gate that only says yes is decoration, so most of this file is about the ways `certify` should
say no.

    python tests/test_certify.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from instinctflash.verify.certify import NotCertifiable, Outcome, certify

results = []


def arm(successes, tag="ep"):
    return [Outcome(f"{tag}{i}", 10000 * (1 + i), "adjust_bottle", bool(s))
            for i, s in enumerate(successes)]


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{('  -- ' + extra) if extra else ''}")
    results.append(cond)


def main() -> int:
    print("=== 1. a real regression must FAIL ===")
    # teacher 92/100, student 85/100, and the student loses the discordant pairs
    t = arm([1] * 92 + [0] * 8)
    s = arm([1] * 85 + [0] * 15)
    c = certify(t, s, margin=-0.05, harness="robotwin-2.0", recipe="test")
    print("   ", str(c).replace("\n", "\n    "))
    check("7-point drop fails a 5-point margin", not c.passed)
    check("delta is negative", c.delta < 0, f"delta={c.delta:+.3f}")

    print("\n=== 2. a genuinely equivalent student must PASS ===")
    t2 = arm([1] * 92 + [0] * 8)
    s2 = arm([1] * 92 + [0] * 8)
    c2 = certify(t2, s2, margin=-0.05)
    check("identical outcomes pass", c2.passed, c2.verdict[:46])
    check("zero discordant pairs is noted, not hidden",
          any("discordant" in n for n in c2.notes))

    print("\n=== 3. a small drop inside the margin passes ===")
    t3 = arm([1] * 92 + [0] * 8)
    s3 = arm([1] * 90 + [0] * 10)
    c3 = certify(t3, s3, margin=-0.10)
    check("2-point drop passes a 10-point margin", c3.passed, f"delta={c3.delta:+.3f}")

    print("\n=== 4. refusals: inputs that cannot support a certificate ===")
    for name, fn in [
        ("unpaired arms (different episodes)",
         lambda: certify(arm([1, 1, 0], "a"), arm([1, 0, 1], "b"), margin=-0.05)),
        ("partial overlap",
         lambda: certify(arm([1] * 5), arm([1] * 3), margin=-0.05)),
        ("duplicate episode in an arm",
         lambda: certify(arm([1, 1]) + arm([1, 1]), arm([1, 1]), margin=-0.05)),
        ("positive margin (would certify a worse model)",
         lambda: certify(arm([1, 0]), arm([1, 0]), margin=+0.05)),
        ("task mismatch on the same episode id",
         lambda: certify([Outcome("e0", 1, "taskA", True)],
                         [Outcome("e0", 1, "taskB", True)], margin=-0.05)),
    ]:
        try:
            fn()
            check(name, False, "did NOT refuse")
        except NotCertifiable as e:
            check(name, True, str(e).split(".")[0][:58])

    print("\n=== 5. the margin is recorded, so it cannot be chosen afterwards ===")
    c5 = certify(arm([1] * 90 + [0] * 10), arm([1] * 88 + [0] * 12), margin=-0.03)
    check("margin appears in the certificate", c5.margin_declared == -0.03)
    check("margin appears in the JSON", '"margin_declared": -0.03' in c5.to_json())

    print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} checks")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

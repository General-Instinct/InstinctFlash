#!/usr/bin/env python3
"""Two RoboTwin runs -> one certificate.json. The end of the Layer 1 workflow.

    checkpoint -> paired evaluation on pinned seeds -> episode JSONL -> certificate.json -> PASS/FAIL

The margin is a REQUIRED argument and is recorded in the certificate. A non-inferiority threshold
chosen after seeing the delta is a narrative, not a gate.

    python certify_run.py --teacher t.jsonl --student s.jsonl --margin -0.05 -o certificate.json
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/home/ubuntu/InstinctWM")

from instinctwm.certify import NotCertifiable, certify, load_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--margin", type=float, required=True,
                    help="acceptable success-rate LOSS, negative. -0.05 = 'may be 5 points worse'")
    ap.add_argument("-o", "--out", default="certificate.json")
    ap.add_argument("--teacher-hash", default="?")
    ap.add_argument("--student-hash", default="?")
    ap.add_argument("--recipe", default="?")
    ap.add_argument("--harness", default="robotwin-2.0")
    a = ap.parse_args()

    try:
        cert = certify(
            load_jsonl(a.teacher), load_jsonl(a.student), margin=a.margin,
            teacher_hash=a.teacher_hash, student_hash=a.student_hash,
            recipe=a.recipe, harness=a.harness,
            seeds="official LingBot-VA protocol, st_seed = 10000*(1+seed)")
    except NotCertifiable as e:
        print(f"NOT CERTIFIABLE: {e}", file=sys.stderr)
        return 2

    print(cert)
    with open(a.out, "w") as f:
        f.write(cert.to_json())
    print(f"\nwrote {a.out}")
    return 0 if cert.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

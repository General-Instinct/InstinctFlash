#!/usr/bin/env python3
"""`validate` is the trust verb: structure, the certificate, and tamper detection.

The certificate path reuses `verify.certify` (tests/test_certify.py owns the statistics); what is
tested HERE is the verb's plumbing, because each piece is a way a trust surface can lie:

  * stamping writes the certificate into provenance — the one namespace the runtime never reads —
    with sha256 of both outcome files and a self-hash;
  * a later plain `validate <dir>` verifies the self-hash, so a hand-edited verdict is a PROBLEM
    and a non-zero exit, not a nicer-looking README table;
  * a FAILED certificate is still stamped (the record of the failure is the point) and the verb
    exits 1;
  * partial certificate flags are a hard config error, not a silently-skipped analysis.

No GPU, no torch, no weights.

    python tests/test_validate_certificate.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def run(argv):
    from instinctflash.cli import main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = main(argv)
        except SystemExit as e:
            rc = int(e.code or 0)
    return rc, buf.getvalue()


def _pkg(td: Path) -> Path:
    pkg = td / "pkg"
    pkg.mkdir()
    (pkg / "config.json").write_text("{}")
    (pkg / "instinctflash.json").write_text(json.dumps({
        "instinctflash_schema": 1,
        "execution": {"model_id": "example-org/x", "backbone": "tiny-wam", "servable": True,
                      "nfe": {"action": 2}, "base_weights": "upstream/none"},
        "provenance": {"training_method": "secret"},
    }))
    (pkg / "model.safetensors").write_bytes(b"\x00")
    return pkg


def _outcomes(path: Path, successes) -> Path:
    with open(path, "w") as f:
        for i, s in enumerate(successes):
            f.write(json.dumps({"episode_id": f"ep{i}", "seed": 1000 + i,
                                "task": "adjust_bottle", "success": bool(s)}) + "\n")
    return path


def main_() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pkg = _pkg(td)
        t = _outcomes(td / "t.jsonl", [1] * 90 + [0] * 10)
        s = _outcomes(td / "s.jsonl", [1] * 89 + [0] * 11)

        print("\n=== 1. stamping: certify + write into provenance, atomically ===")
        rc, out = run(["validate", str(pkg),
                       f"--validate.teacher_outcomes={t}", f"--validate.student_outcomes={s}",
                       "--validate.margin=-0.05", "--validate.harness=robotwin-2.0",
                       "--validate.recipe=unit"])
        check(rc == 0, "structure ok + PASS certificate exits 0", str(rc))
        check("VERDICT: PASS" in out, "prints the same certificate text the harnesses produce")
        check("stamped into" in out, "and says where it wrote")

        doc = json.loads((pkg / "instinctflash.json").read_text())
        block = doc.get("provenance", {}).get("certificate")
        check(isinstance(block, dict), "provenance.certificate exists")
        check(doc["provenance"].get("training_method") == "secret",
              "stamping does not disturb the rest of provenance")
        import hashlib
        t_sha = hashlib.sha256(t.read_bytes()).hexdigest()
        check(block.get("teacher_outcomes_sha256") == t_sha == block.get("teacher_hash"),
              "the block carries the real sha256 of the teacher outcomes")
        check(block.get("student_outcomes_sha256")
              == hashlib.sha256(s.read_bytes()).hexdigest(), "and of the student outcomes")
        for field in ("verdict", "n_pairs", "margin_declared", "ci95", "stamped_at",
                      "content_sha256"):
            check(field in block, f"block records {field}")

        print("\n=== 2. a plain validate verifies the embedded certificate ===")
        rc, out = run(["validate", str(pkg)])
        check(rc == 0, "intact certificate keeps exit 0", str(rc))
        check("certificate: intact" in out, "and is reported as intact")

        print("\n=== 3. an edited verdict is detected, and fails the validation ===")
        doc["provenance"]["certificate"]["verdict"] = "PASS (hand-edited)"
        (pkg / "instinctflash.json").write_text(json.dumps(doc))
        rc, out = run(["validate", str(pkg)])
        check(rc == 1, "tampered certificate exits 1", str(rc))
        check("integrity hash" in out, "and names the problem")

        print("\n=== 4. a FAILED certificate is stamped anyway, and the verb exits 1 ===")
        s_bad = _outcomes(td / "s_bad.jsonl", [1] * 70 + [0] * 30)
        rc, out = run(["validate", str(pkg),
                       f"--validate.teacher_outcomes={t}",
                       f"--validate.student_outcomes={s_bad}", "--validate.margin=-0.05"])
        check(rc == 1, "a real regression fails the verb", str(rc))
        check("VERDICT: FAIL" in out, "with the failing verdict printed")
        doc = json.loads((pkg / "instinctflash.json").read_text())
        check(doc["provenance"]["certificate"]["verdict"].startswith("FAIL"),
              "and the failure is what got stamped — the record of a failure is the point")
        rc, out = run(["validate", str(pkg)])
        check("certificate: intact" in out and "FAIL" in out and rc == 0,
              "re-validate: the FAIL stamp is intact and REPORTED; like publishability, the "
              "verdict itself is informational on the plain path — only integrity gates",
              f"rc={rc}")

        print("\n=== 5. partial certificate flags are a hard error ===")
        rc, out = run(["validate", str(pkg), "--validate.margin=-0.05"])
        check(rc == 2, "margin without outcome files exits 2", str(rc))
        check("all three" in out, "and says which fields are needed together")

    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: validate certifies, stamps, and detects tampering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())

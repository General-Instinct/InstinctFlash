#!/usr/bin/env python3
"""The command line is part of the product surface, so it gets a gate.

"Install the package, give it a model id, run" was the stated standard and there was no CLI at all --
every question a new user has (what is this checkpoint, will this machine serve it, what would the
runtime do to it) required writing a program first. LeRobot ships `lerobot-train`, vLLM ships
`vllm serve`; five verbs is the equivalent surface here.

No GPU and no weights: what is tested is that the verbs exist, that the two which must work offline
do, and that failures are reported rather than raised as tracebacks at a user.
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
    """Invoke the CLI, capturing stdout. Returns (exit_code, output)."""
    from instinctwm.cli import main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = main(argv)
        except SystemExit as e:                                  # argparse
            rc = int(e.code or 0)
    return rc, buf.getvalue()


def test_the_verbs_exist():
    print("\n=== 1. five verbs, and help rather than a traceback ===")
    rc, out = run([])
    check(rc == 2, "bare invocation exits 2 and prints usage", str(rc))
    for verb in ("devices", "describe", "validate", "plan", "run"):
        check(verb in out, f"{verb} is listed")


def test_describe_and_validate_work_without_weights_or_gpu():
    print("\n=== 2. describe and validate are offline verbs ===")
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "p"
        pkg.mkdir()
        (pkg / "config.json").write_text("{}")
        (pkg / "instinctwm.json").write_text(json.dumps({
            "instinctwm_schema": 1,
            "execution": {"model_id": "example-org/x", "backbone": "tiny-wam", "servable": True,
                          "nfe": {"action": 2}, "base_weights": "upstream/none"},
            "provenance": {"training_method": "secret"},
        }))
        rc, out = run(["describe", str(pkg)])
        check(rc == 0, "describe succeeds on a declaration alone", str(rc))
        check("tiny-wam" in out, "and reports the backbone")
        check("secret" not in out, "without leaking a provenance value")
        check("never read by the runtime" in out, "while saying provenance exists")

        rc, out = run(["validate", str(pkg)])
        check(rc == 0, "validate succeeds", str(rc))
        check("publishable" in out, "and reports publishability")

        rc, out = run(["describe", str(Path(td) / "nope")])
        check(rc == 1, "a missing path exits 1", str(rc))
        check("Traceback" not in out, "and does not print a traceback at the user")


def test_devices_reports_or_explains():
    print("\n=== 3. devices reports a machine, or explains why it cannot ===")
    rc, out = run(["devices"])
    check(rc in (0, 1), "devices exits cleanly either way", str(rc))
    if rc == 0:
        check("features:" in out, "it lists what the device can do")
        check("decline here" in out, "and says what an absent feature means for a plan")
    else:
        check("Planning still works" in out, "it explains that planning does not need a device")


def main_() -> int:
    test_the_verbs_exist()
    test_describe_and_validate_work_without_weights_or_gpu()
    test_devices_reports_or_explains()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the CLI exists, and its offline verbs work offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())

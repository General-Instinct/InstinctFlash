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
    from instinctflash.cli import main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = main(argv)
        except SystemExit as e:                                  # argparse
            rc = int(e.code or 0)
    return rc, buf.getvalue()


def test_the_verbs_exist():
    print("\n=== 1. two documented verbs; the old five stay as hidden aliases ===")
    rc, out = run([])
    check(rc == 2, "bare invocation exits 2 and prints usage", str(rc))
    for verb in ("serve", "validate"):
        check(verb in out, f"{verb} is listed")
    for legacy in ("devices", "describe", "plan", "certify"):
        check(f"\n    {legacy}" not in out, f"{legacy} is hidden from the help (still parses)")
    # the aliases keep working — that is the compatibility contract
    rc, _ = run(["devices"])
    check(rc == 0, "the devices alias still answers", str(rc))


def test_describe_and_validate_work_without_weights_or_gpu():
    print("\n=== 2. describe and validate are offline verbs ===")
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "p"
        pkg.mkdir()
        (pkg / "config.json").write_text("{}")
        (pkg / "instinctflash.json").write_text(json.dumps({
            "instinctflash_schema": 1,
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
    print("\n=== 3. devices reports a machine, or explains why there is none ===")
    rc, out = run(["devices"])
    # BRANCH ON THE ANSWER, NOT THE EXIT CODE. This branched on rc, which worked only while
    # "no accelerator" exited 1. Once that became a successful answer -- it is the expected state on
    # the torch-free core install -- rc==0 covered both cases and the test asserted a device profile
    # in an environment that has no device. It passed under an interpreter with torch and failed
    # under the runner, which is the worst way for a test to be wrong.
    check(rc == 0, "devices always exits 0: it answers a question rather than gating on hardware",
          str(rc))
    if "no accelerator visible" in out:
        check("expected without the `runtime` extra" in out,
              "with no torch it says so, and says that is expected")
        check("APPLICABILITY UNCHECKED" in out, "and what that means for a plan")
    else:
        check("features:" in out, "with a device it lists what that device can do")
        check("decline here" in out, "and what an absent feature means for a plan")


def test_serve_verb_surface():
    print("\n=== 4. serve exists, and refuses bad input before touching a socket ===")
    # Everything here happens before the serving extra is imported, so it must hold on the
    # torch-free core install too. The wire itself is tested in tests/test_ws_server.py.
    rc, out = run(["serve", "--help"])
    check(rc == 0 and "--serve.port" in out and "--runtime.placement" in out,
          "serve -h shows the typed dotted fields, including the shared runtime section")
    check("--serve.dry_run" in out and "--serve.smoke" in out and "--serve.viz" in out,
          "and the three stop-early/observability flags")
    rc, out = run(["serve"])
    check(rc == 2, "serve without a model exits 2", str(rc))
    check("serve.model is required" in out, "and says exactly what is missing")
    rc, _ = run(["serve", "some/model", "--serve.bogus=1"])
    check(rc == 2, "an unknown --serve field is a hard error, not a silent ignore", str(rc))


def test_serve_dry_run_is_the_offline_preflight():
    print("\n=== 5. serve --serve.dry_run: device + declaration + plan, offline ===")
    import instinctflash
    sys.path.insert(0, str(ROOT))
    from examples.tiny_wam.adapter import TinyWAMAdapter
    try:
        instinctflash.register("tiny-wam", TinyWAMAdapter)
    except Exception:                                            # noqa: BLE001 - already registered
        pass
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "p"
        pkg.mkdir()
        (pkg / "config.json").write_text("{}")
        (pkg / "instinctflash.json").write_text(json.dumps({
            "instinctflash_schema": 1,
            "execution": {"model_id": "example-org/x", "backbone": "tiny-wam", "servable": True,
                          "guidance": {"action": "positive_only"}, "nfe": {"action": 2},
                          "base_weights": "upstream/none"},
        }))
        (pkg / "model.safetensors").write_bytes(b"\x00")
        rc, out = run(["serve", str(pkg), "--serve.dry_run=true"])
        check(rc == 0, "dry_run exits 0 with no torch and no weights", str(rc))
        check("preflight" in out and "declaration-only" in out, "and says what it is")
        check("device" in out, "reports the device (or its absence)")
        check("plan tier" in out or "InstinctFlash plan" in out, "and prints the plan")

        rc, out = run(["serve", str(pkg) + "-nope", "--serve.dry_run=true"])
        check(rc == 1, "a bad model path exits 1", str(rc))
        check("Traceback" not in out, "without a traceback at the user")


def main_() -> int:
    test_the_verbs_exist()
    test_describe_and_validate_work_without_weights_or_gpu()
    test_devices_reports_or_explains()
    test_serve_verb_surface()
    test_serve_dry_run_is_the_offline_preflight()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the CLI exists, and its offline verbs work offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())

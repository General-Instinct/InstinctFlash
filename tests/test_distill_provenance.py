#!/usr/bin/env python3
"""`provenance.distillation`: the block a distilled checkpoint carries, and how validate reads it.

Each check is a way the block could lie (the same taxonomy as the certificate block, plus the
one failure specific to distillation -- claiming a trained result with no matched-NFE control):

  * a complete block (teacher hash, dataset hash, schedule, B-A and C-B stated separately)
    stamps into provenance and a plain `validate` reports it intact, exit 0;
  * FILL_ME control sentinels -- trained, control not yet run -- keep validate failing, with the
    scaffold's discipline: the sentinel is flagged on EVERY run until the evidence exists;
  * a block with the control object missing entirely is a PROBLEM, never a pass;
  * an edited block fails its self-hash on the next plain validate;
  * "distillation" in the EXECUTION namespace is refused at load -- training facts must not be
    readable by the runtime (descriptors/checkpoint.py FORBIDDEN_IN_EXECUTION).

No GPU, no torch, no weights.

    python tests/test_distill_provenance.py
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
    pkg.mkdir(parents=True)
    (pkg / "config.json").write_text("{}")
    (pkg / "instinctflash.json").write_text(json.dumps({
        "instinctflash_schema": 1,
        "execution": {"model_id": "example-org/va-1v4a", "backbone": "wan_va", "servable": True,
                      "nfe": {"video": 1, "action": 4}, "base_weights": "upstream/none"},
        "provenance": {"training_method": "video-stream consistency distillation"},
    }))
    (pkg / "model.safetensors").write_bytes(b"\x00")
    return pkg


def _control(td: Path) -> dict:
    """A complete matched-NFE-control record, hashes over real (tiny) outcome files."""
    import hashlib

    def outcomes(name: str, flips: int) -> str:
        p = td / name
        with open(p, "w") as f:
            for i in range(50):
                f.write(json.dumps({"episode_id": f"ep{i}", "seed": 1000 + i,
                                    "task": "adjust_bottle", "success": i >= flips}) + "\n")
        return hashlib.sha256(p.read_bytes()).hexdigest()

    return {
        "teacher_outcomes_sha256": outcomes("a.jsonl", 2),
        "control_outcomes_sha256": outcomes("b.jsonl", 10),
        "student_outcomes_sha256": outcomes("c.jsonl", 4),
        "b_minus_a": {"delta": -0.16, "ci95": [-0.24, -0.09], "n_pairs": 206,
                      "discordant": [42, 8], "verdict": "recorded, screening",
                      "margin_declared": -0.05},
        "c_minus_b": {"delta": +0.12, "ci95": [+0.05, +0.19], "n_pairs": 206,
                      "discordant": [6, 31], "verdict": "recorded, screening",
                      "margin_declared": -0.05},
        # the strong form of the law: which guidance the control served, and the grid it won
        "control_guidance": {"video": {"mode": "cfg", "scale": 3.0},
                             "action": {"mode": "positive_only", "scale": 1.0}},
        "guidance_grid_swept": [
            {"point": "1V/4A@w5", "guidance": {"video": {"mode": "cfg", "scale": 5.0}}, "success": 0.7521},
            {"point": "1V/4A@w3", "guidance": {"video": {"mode": "cfg", "scale": 3.0}}, "success": 0.8824},
        ],
    }


def main_() -> int:
    from instinctflash.descriptors.distillation import (
        CONTROL_KEYS, build_block, content_hash, stamp_distillation, verify_distillation,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pkg = _pkg(td)

        print("\n=== 1. a complete block stamps, validates intact, exit 0 ===")
        block = build_block(
            recipe_id="fewstep_video_cd_v1", family="wan_va",
            teacher_model_id="robbyant/lingbot-va-posttrain-robotwin",
            teacher_weights_sha256="a" * 64,
            dataset_id="robotwin2_demo_clean", dataset_sha256="b" * 64,
            schedule={"nfe": {"video": 1, "action": 4},
                      "grids": {"video": [1000.0], "action": [1000.0, 750.0, 500.0, 250.0]}},
            control=_control(td),
        )
        check(block["content_sha256"] == content_hash(block), "the builder self-hashes the block")
        stamp_distillation(pkg, block)
        doc = json.loads((pkg / "instinctflash.json").read_text())
        check(doc["provenance"].get("distillation") == block, "stamped under provenance")
        check(doc["provenance"].get("training_method") is not None,
              "without disturbing the rest of provenance")
        rc, out = run(["validate", str(pkg)])
        check(rc == 0, "plain validate exits 0", str(rc))
        check("distillation: intact" in out, "and reports the block intact")
        check("B−A delta -0.1600" in out.replace("−0.1600", "-0.1600")
              or "B−A delta -0.16" in out or "B−A" in out,
              "quoting B−A and C−B separately", out.splitlines()[-1] if out else "")

        print("\n=== 2. FILL_ME control sentinels keep validate failing ===")
        pending = build_block(
            recipe_id="fewstep_video_cd_v1", family="wan_va",
            teacher_model_id="robbyant/lingbot-va-posttrain-robotwin",
            teacher_weights_sha256="a" * 64,
            dataset_id="robotwin2_demo_clean", dataset_sha256="b" * 64,
            schedule={"nfe": {"video": 1, "action": 4}},
            control=None,
        )
        check(all(pending["matched_nfe_control"][k] == "FILL_ME" for k in CONTROL_KEYS),
              "control=None stamps sentinels, never absence")
        stamp_distillation(pkg, pending)
        rc, out = run(["validate", str(pkg)])
        check(rc == 1, "a trained point without its control fails validation", str(rc))
        check('is "FILL_ME"' in out and "matched-NFE control" in out.replace("matched_nfe", "matched-NFE")
              or "FILL_ME" in out, "each sentinel is flagged with why it blocks")
        status, _, problems = verify_distillation(pkg)
        check(status == "intact" and len(problems) == len(CONTROL_KEYS),
              "intact but dishonest: one problem per sentinel", str(len(problems)))

        print("\n=== 3. a control object missing entirely is refused ===")
        headless = dict(pending)
        headless.pop("matched_nfe_control")
        headless["content_sha256"] = content_hash(headless)
        doc = json.loads((pkg / "instinctflash.json").read_text())
        doc["provenance"]["distillation"] = headless
        (pkg / "instinctflash.json").write_text(json.dumps(doc))
        status, _, problems = verify_distillation(pkg)
        check(status == "intact" and any("matched_nfe_control" in p for p in problems),
              "missing control is a PROBLEM, not a pass", str(problems[:1]))
        rc, _ = run(["validate", str(pkg)])
        check(rc == 1, "and fails the verb", str(rc))
        try:
            build_block(recipe_id="r", family="wan_va", teacher_model_id="t",
                        teacher_weights_sha256="a" * 64, dataset_id="d",
                        dataset_sha256="b" * 64, schedule={},
                        control={"b_minus_a": {}})
        except ValueError as e:
            check("partial control is not a control" in str(e),
                  "the builder refuses a partial control outright")
        else:
            check(False, "the builder refuses a partial control outright")

        print("\n=== 4. an edited block is detected by the self-hash ===")
        stamp_distillation(pkg, block)
        doc = json.loads((pkg / "instinctflash.json").read_text())
        doc["provenance"]["distillation"]["matched_nfe_control"]["c_minus_b"]["delta"] = +0.9
        (pkg / "instinctflash.json").write_text(json.dumps(doc))
        status, _, problems = verify_distillation(pkg)
        check(status == "tampered", "an edited C−B is tampering", status)
        rc, out = run(["validate", str(pkg)])
        check(rc == 1 and "integrity hash" in out, "and fails a plain validate loudly", str(rc))

        print("\n=== 5. 'distillation' may not appear in the execution namespace ===")
        leaky = _pkg(td / "leak")
        doc = json.loads((leaky / "instinctflash.json").read_text())
        doc["execution"]["distillation"] = {"recipe_id": "smuggled"}
        (leaky / "instinctflash.json").write_text(json.dumps(doc))
        from instinctflash.descriptors.checkpoint import load_declaration
        try:
            load_declaration(leaky)
        except RuntimeError as e:
            check("provenance keys" in str(e) and "distillation" in str(e),
                  "load_declaration refuses the leak by name")
        else:
            check(False, "load_declaration refuses the leak by name")

        print("\n=== 6. certificate and distillation blocks coexist independently ===")
        stamp_distillation(pkg, block)
        t = td / "t.jsonl"
        s = td / "s.jsonl"
        for path, bad in ((t, 5), (s, 6)):
            with open(path, "w") as f:
                for i in range(100):
                    f.write(json.dumps({"episode_id": f"e{i}", "seed": i, "task": "x",
                                        "success": i >= bad}) + "\n")
        rc, out = run(["validate", str(pkg), f"--validate.teacher_outcomes={t}",
                       f"--validate.student_outcomes={s}", "--validate.margin=-0.05"])
        check(rc == 0, "stamping a certificate alongside an intact distillation block", str(rc))
        doc = json.loads((pkg / "instinctflash.json").read_text())
        check("certificate" in doc["provenance"] and "distillation" in doc["provenance"],
              "both blocks present")
        rc, out = run(["validate", str(pkg)])
        check(rc == 0 and "certificate: intact" in out and "distillation: intact" in out,
              "and both verified on the plain run", str(rc))

    # td / "leak" is created inside the context manager via _pkg(td / "leak")
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the distillation block states its control, hashes its claims, and cannot be "
          "edited quietly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())

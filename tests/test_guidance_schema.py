#!/usr/bin/env python3
"""The guidance leg of the operating-point tuple: one schema, every consumer reads the same thing.

Pinned here (CPU, no torch, no weights):

  * the three declaration forms -- a mode string (the pre-campaign schema, inherited scale
    recorded as such), a numeric scale, {mode, scale} -- resolve to one canonical (mode, scale)
    and identical serving canonicalises identically;
  * contradictions and garbage are refused BY NAME at declaration load (checkpoint.py), never
    ignored at serve time;
  * capability tokens keep the historical spelling for the string form and carry the scale for
    the others, so a plan can tell w=3 from w=5;
  * the scaffold inherits guidance with its classification printed -- a SERVING choice, not a
    training fact -- as the resolved (mode@scale, batching) tuple;
  * the plan preflight prints the operating point as the tuple.

Why this exists: the few-step campaign measured the SAME schedule (LingBot-VA 1V/4A, pinned
scenes) at 0.752 (w=5), 0.882 (w=3), 0.885 (w=1, batch-1) and 0.276 (w=9). `nfe` alone does
not name an operating point; (schedule grid, per-stream guidance scale, CFG batching) does.
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


FAMILY = {"video": ("cfg", 5.0), "action": ("positive_only", 1.0)}


def test_three_forms_one_meaning():
    print("\n=== 1. three declaration forms, one canonical (mode, scale) ===")
    from instinctflash.descriptors.guidance import (
        GuidanceDeclarationError, canonical_guidance, parse_declared, resolve,
    )
    r = resolve({"video": "cfg", "action": "positive_only"}, FAMILY)
    check(r["video"].scale == 5.0 and r["video"].scale_inherited
          and r["video"].scale_source == "inherited from the family default",
          "a mode string takes the family scale and RECORDS the inheritance")
    check(r["action"].scale == 1.0 and r["action"].scale_source == "implied by mode"
          and not r["action"].negative_branch,
          "positive_only implies scale 1 and no negative branch")
    check(r["video"].negative_branch, "cfg@5 requests the negative branch")
    check(resolve({"video": 3}, FAMILY)["video"].canonical() == {"mode": "cfg", "scale": 3.0},
          "a bare number is a CFG scale")
    check(resolve({"video": "3"}, FAMILY)["video"].canonical() == {"mode": "cfg", "scale": 3.0},
          "a numeric CLI string is the same scale (the worker path always sends strings)")
    off = resolve({"video": {"mode": "cfg", "scale": 1.0}}, FAMILY)["video"]
    check(off.mode == "cfg" and off.scale == 1.0 and not off.negative_branch,
          "cfg at w=1 keeps its mode and requests no negative branch (batch-1)")
    check(canonical_guidance({"action": "positive_only"}, FAMILY)["action"]
          == canonical_guidance({"action": {"mode": "positive_only", "scale": 1.0}}, FAMILY)["action"],
          "identical serving canonicalises identically across forms")
    check(resolve({}, FAMILY)["video"].scale_source == "inherited from the family default",
          "an undeclared stream is the family default, recorded as inherited")
    check(parse_declared("cfg").form == "mode" and parse_declared(2).form == "scale"
          and parse_declared({"mode": "cfg", "scale": 2}).form == "mode+scale",
          "each form is named")
    for bad, needle in (("bogus", "neither a guidance mode nor a numeric scale"),
                        ({"mode": "positive_only", "scale": 3}, "discards the negative branch"),
                        ({"mode": "none", "scale": 2}, "discards the negative branch"),
                        (True, "boolean"), (-1, "non-negative"), ({}, "declares nothing"),
                        ({"mode": "cfg", "extra": 1}, "unknown keys"), ([5], "not a guidance declaration")):
        try:
            parse_declared(bad, where="t")
        except GuidanceDeclarationError as e:
            check(needle in str(e), f"refused: {bad!r}", str(e)[:70])
        else:
            check(False, f"refused: {bad!r}")


def test_load_declaration_refuses_a_bad_guidance_block():
    print("\n=== 2. the declaration boundary refuses what the runtime could not serve as written ===")
    from instinctflash.descriptors.checkpoint import load_declaration

    def write(td: Path, guidance) -> Path:
        d = td / "pkg"
        d.mkdir(exist_ok=True)
        (d / "instinctflash.json").write_text(json.dumps({
            "instinctflash_schema": 1,
            "execution": {"model_id": "org/x", "backbone": "wan_va", "servable": True,
                          "nfe": {"video": 1, "action": 4}, "guidance": guidance}}))
        return d

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for ok in ({"video": "cfg", "action": "positive_only"}, {"video": 3.0},
                   {"video": {"mode": "cfg", "scale": 1.0}, "action": "positive_only"}):
            decl = load_declaration(write(td, ok))
            check(dict(decl.guidance) == ok, f"accepted verbatim: {ok}")
        for bad in ({"video": "bogus"}, {"action": {"mode": "positive_only", "scale": 3}},
                    {"video": True}):
            try:
                load_declaration(write(td, bad))
            except RuntimeError as e:
                check("execution.guidance" in str(e), f"refused at load: {bad}", str(e)[:80])
            else:
                check(False, f"refused at load: {bad}")
        # legacy delta.json takes the same check
        legacy = td / "legacy"
        legacy.mkdir()
        (legacy / "delta.json").write_text(json.dumps({"model_id": "m", "backbone": "wan_va",
                                                       "coverage_gate_pass": True,
                                                       "guidance": {"video": "nonsense"}}))
        try:
            load_declaration(legacy)
        except ValueError as e:
            check("nonsense" in str(e), "the legacy reader refuses too")
        else:
            check(False, "the legacy reader refuses too")


def test_capability_tokens():
    print("\n=== 3. capability tokens: historical spelling kept, scales carried ===")
    from instinctflash.descriptors.checkpoint import ExecutionDeclaration
    from instinctflash.descriptors.package import Checkpoint
    ex = ExecutionDeclaration(model_id="m", backbone="wan_va", servable=True,
                              guidance={"video": "cfg", "action": "positive_only"})
    caps = Checkpoint("x", ex).capabilities()
    check({"guidance:video=cfg", "guidance:action=positive_only"} <= caps,
          "the string form's tokens are unchanged", str(sorted(c for c in caps if c.startswith("guidance"))))
    ex2 = ExecutionDeclaration(model_id="m", backbone="wan_va", servable=True,
                               guidance={"video": {"mode": "cfg", "scale": 3.0}, "action": 1.0})
    caps2 = Checkpoint("x", ex2).capabilities()
    check({"guidance:video=cfg@3", "guidance:action=cfg@1"} <= caps2,
          "a declared scale is part of the token (w=3 is not w=5)",
          str(sorted(c for c in caps2 if c.startswith("guidance"))))


def test_scaffold_inherits_guidance_with_its_classification_printed():
    print("\n=== 4. the scaffold inherits guidance as a SERVING choice, and prints the tuple ===")
    from instinctflash.descriptors.scaffold import scaffold_declaration
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "wanva-ft"
        d.mkdir()
        (d / "config.json").write_text(json.dumps(
            {"_class_name": "WanTransformer3DModel", "action_dim": 30, "num_layers": 30}))
        (d / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00" * 4096)
        plan = scaffold_declaration(d, "robbyant/lingbot-va-posttrain-robotwin")
        f = {x.key: x for x in plan.fields}["guidance"]
        check(f.status == "inherited" and f.value == {"video": "cfg", "action": "positive_only"},
              "inherited, value untouched (the string form stays valid)")
        check("SERVING choice, not a training fact" in f.note,
              "the classification is stated: a serving choice, inheritable, not a training fact")
        check("video=cfg@5 (scale inherited)" in f.note and "action=positive_only@1" in f.note,
              "the resolved per-stream tuple is printed", f.note)
        check("batch-2 on 8 of 8 declared forwards" in f.note,
              "with the CFG batching it implies at the inherited 2V/4A schedule", f.note)
        check("declares its own {mode, scale}" in f.note,
              "and says what a re-tuned or CFG-folded checkpoint must do instead")
        check("guidance" in plan.explain() and "SERVING choice" in plan.explain(),
              "explain() carries it")


def test_plan_preflight_prints_the_operating_point_tuple():
    print("\n=== 5. `instinctflash plan` prints the operating point as the tuple ===")
    from instinctflash.cli import main
    with tempfile.TemporaryDirectory() as td:
        Path(td, "instinctflash.json").write_text(json.dumps({
            "instinctflash_schema": 1,
            "execution": {"model_id": "org/va-2v2a-w1", "backbone": "wan_va", "servable": True,
                          "nfe": {"video": 2, "action": 2},
                          "guidance": {"video": {"mode": "cfg", "scale": 1.0},
                                       "action": "positive_only"}}}))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["plan", td])
        out = buf.getvalue()
    check(rc == 0, "plan exits 0", out[-200:])
    line = next((ln for ln in out.splitlines() if "operating point:" in ln), "")
    check("schedule {kv_refresh=2 + video=2 + action=2}" in line, "schedule leg", line)
    check("video=cfg@1" in line and "action=positive_only@1" in line, "guidance leg", line)
    check("batch-1 on all 6 declared forwards" in line, "CFG-batching leg, derived", line)
    check("guidance:video=cfg@1" in out, "and the capability token carries the scale")


def main_() -> int:
    test_three_forms_one_meaning()
    test_load_declaration_refuses_a_bad_guidance_block()
    test_capability_tokens()
    test_scaffold_inherits_guidance_with_its_classification_printed()
    test_plan_preflight_prints_the_operating_point_tuple()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: one guidance schema, and every consumer reads the same operating-point tuple.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())

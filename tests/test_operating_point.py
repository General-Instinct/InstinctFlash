#!/usr/bin/env python3
"""An operating point is a declared fact, and every placement path must obey it.

Four defects lived here at once, all of them silent, all found while costing a step-reduction
decision. Each one served something other than what the checkpoint declared:

  1. `worker_command` read only the `nfe=` override, so a checkpoint declaring 2V/4A ran at the
     upstream 25/50 default in a worker and at 2/4 in-process. One declaration, two behaviours.
  2. `worker_command` pointed LINGBOT_CKPT at `base_weights` -- a Hub repo id -- so the worker
     loaded the BASE transformer and ignored the published package.
  3. `execution.guidance` was parsed, echoed by describe(), and applied nowhere.
  4. The planner priced `adapter.spec()`'s 79-forward cycle while a 10-forward cycle ran.

No GPU, no weights: these are all about whether the declaration reaches the thing that acts on it.
"""
from __future__ import annotations

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


def test_with_nfe_rewrites_the_schedule():
    print("\n=== 1. AdapterSpec.with_nfe reprices the declared schedule ===")
    from instinctflash import load
    spec = load("wan_va").spec()
    check(spec.total_forwards() == 79, "the adapter states the model's own schedule", "79")
    at24 = spec.with_nfe({"video": 2, "action": 4})
    check(at24.forwards_breakdown() == "kv_refresh=2 + video=2 + action=4",
          "phases named in the declaration change; others do not", at24.forwards_breakdown())
    # TWO DIFFERENT NUMBERS, both correct, and conflating them is how a latency estimate goes
    # wrong: total_forwards() sums DECLARED DENOISE forwards (8 here), while the cycle EXECUTES 10,
    # because each denoise loop pads one terminal cache-only forward. Frontier arithmetic wants the
    # executed count; profitability arguments about denoise work want this one.
    check(at24.total_forwards() == 8, "declared denoise forwards at 2V/4A", str(at24.total_forwards()))
    executed = at24.total_forwards() + sum(1 for p in at24.phases if p.truncatable)
    check(executed == 10, "executed forwards = declared + one cache-only per truncatable phase",
          str(executed))
    check(spec.total_forwards() == 79, "the original spec is untouched (frozen dataclass)")


def test_commit_steps_are_remapped():
    print("\n=== 2. a terminal commit stays terminal ===")
    # THE SUBTLE ONE. video declares commit_steps={25} of 26 forwards: "the last one writes K/V".
    # Rewrite nfe to 2 and leave that index alone and it points two forwards past the end, so
    # cfg_elision -- which reads commit_steps to decide which forwards must keep their guidance
    # branch -- would elide the committing forward. The episode then goes wrong several chunks later.
    from instinctflash import load
    spec = load("wan_va").spec()
    before = spec.phase("video")
    check(before.commit_steps == frozenset({before.nfe - 1}), "video commits on its last forward",
          str(sorted(before.commit_steps)))
    after = spec.with_nfe({"video": 2, "action": 4}).phase("video")
    check(after.commit_steps == frozenset({1}), "and still does after the rewrite",
          str(sorted(after.commit_steps)))
    check(max(after.commit_steps) < after.nfe, "no commit index is out of range")
    a = spec.with_nfe({"action": 4}).phase("action")
    check(max(a.commit_steps) < a.nfe, "same for the action phase", str(sorted(a.commit_steps)))
    kv = spec.with_nfe({"video": 1}).phase("kv_refresh")
    check(kv.commit_steps == frozenset({0, 1}), "a phase not named keeps both its commits")
    lo = spec.with_nfe({"video": 0}).phase("video")
    check(lo.nfe == lo.min_nfe, "below min_nfe is raised to it", f"nfe={lo.nfe}")


def test_planner_plans_the_declared_schedule():
    print("\n=== 3. the plan is priced at the schedule that will run ===")
    from instinctflash import load, Optimizer
    spec = load("wan_va").spec()
    teacher = Optimizer().compile(spec).explain()
    shipped = Optimizer().compile(spec.with_nfe({"video": 2, "action": 4})).explain()
    check("79" in teacher, "the teacher schedule reports 79 forwards somewhere")
    check("79" not in shipped, "the shipped plan does NOT quote a 79-forward cycle")
    src = (ROOT / "instinctflash" / "runtime" / "facade.py").read_text()
    check("with_nfe(schedule)" in src, "from_pretrained applies the declaration before compiling")
    check("ckpt.execution.nfe" in src, "and the declaration is the base, not just the override")


def test_declared_guidance_is_applied():
    print("\n=== 4. execution.guidance reaches the server ===")
    from instinctflash.adapters.lingbot_va import apply_declared_guidance

    class Cfg:
        guidance_scale = 5.0
        action_guidance_scale = 1.0

    c = Cfg()
    got = apply_declared_guidance(c, {"video": "cfg", "action": "positive_only"})
    check(c.guidance_scale == 5.0, "'cfg' leaves the model's own scale alone", str(c.guidance_scale))
    check(c.action_guidance_scale == 1.0, "'positive_only' pins the scale to 1.0")

    c2 = Cfg()
    apply_declared_guidance(c2, {"video": "positive_only"})
    check(c2.guidance_scale == 1.0,
          "a guidance-distilled student turns video CFG OFF", str(c2.guidance_scale))
    c3 = Cfg()
    apply_declared_guidance(c3, {"video": 3.5})
    check(c3.guidance_scale == 3.5, "an explicit numeric scale wins", str(c3.guidance_scale))
    c4 = Cfg()
    apply_declared_guidance(c4, {"video": {"mode": "cfg", "scale": 7.0}})
    check(c4.guidance_scale == 7.0, "and so does a {mode, scale} form")
    apply_declared_guidance(c4, {"nonexistent_stream": "cfg"})
    check(True, "an unknown stream is ignored rather than raising")


def test_both_placements_serve_the_same_declaration():
    print("\n=== 5. worker and in-process obey the same declaration ===")
    from instinctflash.adapters.lingbot_va import LingBotVA
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "pkg"
        pkg.mkdir()
        (pkg / "config.json").write_text("{}")
        (pkg / "model.safetensors").write_bytes(b"\x00")
        base = Path(td) / "base"
        for comp in LingBotVA.FROZEN_COMPONENTS:
            (base / comp).mkdir(parents=True, exist_ok=True)

        class Ex:
            nfe = {"video": 2, "action": 4}
            guidance = {"video": "cfg", "action": "positive_only"}
            extra = {"base_weights": str(base)}

        class Ck:
            path = str(pkg)
            model_id = "example-org/x"
            execution = Ex()

        import os
        os.environ["IFL_CACHE"] = str(Path(td) / "cache")
        try:
            argv, env = LingBotVA().worker_command(
                Ck(), None, port=1234, python="python3", device=None, nfe=None)
        finally:
            os.environ.pop("IFL_CACHE", None)

        joined = " ".join(argv)
        check("--degrade-nfe 2,4" in joined,
              "the worker gets the DECLARED nfe with no override passed", joined.split()[-4:])
        check("--guidance" in joined and "action=positive_only" in joined,
              "and the declared guidance")
        ck = env.get("LINGBOT_CKPT", "")
        check(str(base) != ck, "LINGBOT_CKPT is NOT the base pointer")
        check("composed" in ck, "it is the composed tree, as in-process uses", ck[-46:])
        argv2, _ = LingBotVA().worker_command(
            Ck(), None, port=1234, python="python3", device=None, nfe={"action": 2})
        check("--degrade-nfe 2,2" in " ".join(argv2), "an explicit override still wins")


def main() -> int:
    test_with_nfe_rewrites_the_schedule()
    test_commit_steps_are_remapped()
    test_planner_plans_the_declared_schedule()
    test_declared_guidance_is_applied()
    test_both_placements_serve_the_same_declaration()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the operating point is declaration-driven in every placement path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

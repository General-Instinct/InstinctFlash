#!/usr/bin/env python3
"""An operating point is a declared fact, and every placement path must obey it.

Five defects lived here, all of them silent, all found while costing a step-reduction decision.
Each one served something other than what the checkpoint declared:

  1. `worker_command` read only the `nfe=` override, so a checkpoint declaring 2V/4A ran at the
     upstream 25/50 default in a worker and at 2/4 in-process. One declaration, two behaviours.
  2. `worker_command` pointed LINGBOT_CKPT at `base_weights` -- a Hub repo id -- so the worker
     loaded the BASE transformer and ignored the published package.
  3. `execution.guidance` was parsed, echoed by describe(), and applied nowhere.
  4. The planner priced `adapter.spec()`'s 79-forward cycle while a 10-forward cycle ran.
  5. A numeric guidance scale on the worker path (`--guidance video=3`, a CLI STRING) fell
     through every case and applied NOTHING (found 2026-08-31 preparing the guidance x NFE
     sweep); and the planner priced batch-2 forwards for a checkpoint whose declared guidance
     the server ran at batch-1.

The few-step campaign's consequence (RFC §11) is what section 6 pins: an operating point is the
tuple (schedule grid, per-stream guidance scale, CFG batching) -- the same 1V/4A schedule scored
0.752 at w=5 and 0.885 at w=1 -- so the declaration, the served scale, the worker flag and the
plan's pricing must all carry the same tuple, and the plan must print it.

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
    c5 = Cfg()
    got5 = apply_declared_guidance(c5, {"video": "3"})
    check(c5.guidance_scale == 3.0 and got5 == {"video": 3.0},
          "a numeric scale as a CLI STRING applies (the worker path always sends strings; "
          "before 2026-08-31 '--guidance video=3' silently applied nothing)",
          str(c5.guidance_scale))
    c6 = Cfg()
    try:
        apply_declared_guidance(c6, {"video": "bogus"})
    except ValueError as e:
        check("bogus" in str(e) and c6.guidance_scale == 5.0,
              "an unreadable value is REFUSED by name, not silently ignored (the failure class "
              "this module exists to prevent is serving something other than the declaration)")
    else:
        check(False, "an unreadable value is REFUSED by name, not silently ignored")
    c7 = Cfg()
    got7 = apply_declared_guidance(c7, {"video": {"mode": "cfg", "scale": 1.0}})
    check(c7.guidance_scale == 1.0 and got7 == {"video": 1.0},
          "cfg at a declared scale of 1.0 writes 1.0 -- the guidance-off, batch-1 point the "
          "campaign found free (h1_report §4b: 1V/4A@w1 0.885 vs 0.752 at the shipped w=5)")
    c8 = Cfg()
    try:
        apply_declared_guidance(c8, {"action": {"mode": "positive_only", "scale": 3.0}})
    except ValueError as e:
        check("negative branch" in str(e) and c8.action_guidance_scale == 1.0,
              "positive_only WITH a scale > 1 is a contradiction and is refused before it turns "
              "the action combine on")
    else:
        check(False, "positive_only WITH a scale > 1 is refused")
    c9 = Cfg()
    got9 = apply_declared_guidance(c9, {"video": "cfg"})
    check(c9.guidance_scale == 5.0 and got9 == {},
          "'cfg' by name is the family default, recorded as inherited: nothing is written")


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
            # geometry declared like any real wan_va checkpoint must: the adapter now refuses
            # to guess it (see tests/test_va_geometry.py), and the worker carries it as
            # --geometry overrides
            extra = {"base_weights": str(base),
                     "obs_cam_keys": ["observation.images.cam_high",
                                      "observation.images.cam_left_wrist",
                                      "observation.images.cam_right_wrist"],
                     "height": 256, "width": 320, "env_type": "robotwin_tshape"}

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

        class ExScale(Ex):
            guidance = {"video": {"mode": "cfg", "scale": 3.0}, "action": "positive_only"}

        class CkScale(Ck):
            execution = ExScale()

        os.environ["IFL_CACHE"] = str(Path(td) / "cache")
        try:
            argv_w, _ = LingBotVA().worker_command(
                CkScale(), None, port=1234, python="python3", device=None, nfe=None)
        finally:
            os.environ.pop("IFL_CACHE", None)
        flag = argv_w[argv_w.index("--guidance") + 1]
        check(flag == "action=positive_only,video=3",
              "a {mode, scale} declaration travels to the worker as the SERVED scale (a number "
              "the worker re-parses through the same resolver), positive_only as its mode name",
              flag)
        from instinctflash.adapters.lingbot_va import apply_declared_guidance

        class Cfg2:
            guidance_scale = 5.0
            action_guidance_scale = 1.0

        c = Cfg2()
        apply_declared_guidance(c, dict(p.split("=", 1) for p in flag.split(",")))
        check(c.guidance_scale == 3.0 and c.action_guidance_scale == 1.0,
              "and the worker-side parse of that flag serves the identical scales")


def test_plan_is_priced_at_the_declared_guidance_and_prints_the_tuple():
    print("\n=== 6. the operating point is (schedule, guidance, CFG batching), and the plan says so ===")
    from instinctflash import Optimizer, load
    from instinctflash.adapters.base import GuidanceMode
    spec = load("wan_va").spec().with_nfe({"video": 1, "action": 4})

    shipped = spec.with_guidance({"video": "cfg", "action": "positive_only"})
    check(shipped.guidance["video"].scale == 5.0
          and shipped.guidance_resolution["video"] == "inherited from the family default",
          "the string form inherits the family scale and RECORDS that it was inherited",
          str(shipped.guidance_resolution))
    b = shipped.cfg_batching()
    check(b["batch2_forwards"] == 7 and b["batch1_forwards"] == 0,
          "video cfg@5 makes every declared forward batch-2 (the action branch is computed then "
          "discarded -- the fact cfg_branch_elision exploits)", str(b))

    off = spec.with_guidance({"video": "positive_only"})
    check(off.guidance["video"].mode is GuidanceMode.POSITIVE_ONLY and off.guidance["video"].scale == 1.0,
          "positive_only rewrites the video rule to scale 1")
    b = off.cfg_batching()
    check(b["batch2_forwards"] == 0 and b["batch1_forwards"] == 7 and not b["negative_branch_requested_by"],
          "and with no stream requesting a negative branch every forward is batch-1", str(b))
    w1 = spec.with_guidance({"video": {"mode": "cfg", "scale": 1.0}})
    check(not w1.guidance["video"].requests_negative_branch and w1.cfg_batching()["batch2_forwards"] == 0,
          "cfg at w=1 requests no negative branch either: CFG batching is DERIVED from the served "
          "scale, never declared")
    w3 = spec.with_guidance({"video": 3})
    check(w3.guidance["video"].scale == 3.0 and w3.guidance_resolution["video"] == "declared"
          and w3.cfg_batching()["batch2_forwards"] == 7,
          "a numeric declaration is a served CFG scale: w=3, batch-2")

    from instinctflash.planners.planner import Tier
    plan_on = Optimizer(tier_ceiling=Tier.NUMERIC).compile(shipped)
    plan_off = Optimizer(tier_ceiling=Tier.NUMERIC).compile(off)
    by_name = lambda plan: {r.name: r for r in plan.results}  # noqa: E731
    check(by_name(plan_on)["cfg_branch_elision"].applies,
          "at the shipped guidance cfg_branch_elision has a discarded branch to elide (NUMERIC ceiling)",
          by_name(plan_on)["cfg_branch_elision"].reason)
    check(not by_name(plan_off)["cfg_branch_elision"].applies
          and "batch-1" in by_name(plan_off)["cfg_branch_elision"].reason,
          "at guidance-off it declines, because the batch it would trim is not there",
          by_name(plan_off)["cfg_branch_elision"].reason)
    check("operating point:" in plan_off.explain() and "batch-1 on all 7 declared forwards" in plan_off.explain(),
          "the plan PRINTS the tuple", plan_off.explain().splitlines()[2])
    check("video=cfg@5 [inherited from the family default]" in plan_on.operating_point()
          if callable(getattr(plan_on, "operating_point", None)) else
          "video=cfg@5 [inherited from the family default]" in plan_on.operating_point,
          "...including where an inherited scale came from")
    some_pass = plan_off.results[0].name
    check(plan_off.without(some_pass).operating_point == plan_off.operating_point
          and plan_off.bitexact_subset().operating_point == plan_off.operating_point,
          "derived plans keep the operating point")

    # the facade applies the DECLARATION's guidance, not just its nfe
    from instinctflash.runtime.facade import plan_declaration
    with tempfile.TemporaryDirectory() as td:
        Path(td, "instinctflash.json").write_text(json.dumps({
            "instinctflash_schema": 1,
            "execution": {"model_id": "org/va-1v4a-w1", "backbone": "wan_va", "servable": True,
                          "nfe": {"video": 1, "action": 4},
                          "guidance": {"video": {"mode": "cfg", "scale": 1.0},
                                       "action": "positive_only"}}}))
        _, _, plan, _ = plan_declaration(td, probe_device=False)
    check("video=cfg@1" in plan.operating_point and "batch-1 on all 7" in plan.operating_point,
          "plan_declaration prices the checkpoint's declared guidance", plan.operating_point)
    check(not {r.name: r for r in plan.results}["cfg_branch_elision"].applies,
          "so the guidance-off checkpoint's plan does not claim an elision that cannot happen")


def main() -> int:
    test_with_nfe_rewrites_the_schedule()
    test_commit_steps_are_remapped()
    test_planner_plans_the_declared_schedule()
    test_declared_guidance_is_applied()
    test_both_placements_serve_the_same_declaration()
    test_plan_is_priced_at_the_declared_guidance_and_prints_the_tuple()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the operating point is declaration-driven in every placement path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

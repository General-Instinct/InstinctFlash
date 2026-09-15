#!/usr/bin/env python3
"""The distillation skeleton: adapter declarations, the E8 grids, and the control gate.

Three things are load-bearing enough to pin on CPU before any trainer exists:

  * `flow_match_times` reproduces the DEPLOYED grids bit-for-bit at the points we have
    measured (mapping memo E8) — training on a grid the sampler never visits is the class of
    silent failure the whole schedule declaration exists to prevent;
  * the family adapters declare what the campaigns measured (streams, shifts, guidance,
    certified points, the H1 trainable split), their capability gate is JVP-free and
    non-adversarial (C1/C2), and their training halves are LOUD seams, not silent stubs;
  * `verify_point` REFUSES a trained result without its matched-NFE control, refuses a
    control at a different schedule, and — with all three arms — states B−A and C−B
    separately with exactly certify()'s numbers.
  * THE STRONG FORM (RFC §11): the control is matched on the whole operating point (schedule
    grid, per-stream guidance, CFG batching) and must be the BEST untrained configuration over
    the guidance grid swept at that schedule — a control whose grid was not swept, or that is
    not its best, is refused. The H1 screen is the precedent: +14.3 pp vs the shipped w=5
    control was +0.0 vs the best untrained knob (w=3).
  * "NO DISTILLATION NEEDED" is a first-class verdict: `screen_verdict` over a frontier report
    decides before any trainer runs, `train()` requires the verdict, and the verdict owes the
    frontier + intervals + a certification prereg stub for the winning untrained point.

No GPU, no torch, no weights.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from instinctflash.distill import (  # noqa: E402
    ControlGateViolation,
    DistillPipeline,
    ExperimentalSeam,
    NoDistillationNeeded,
    ScreenVerdict,
    canonical_schedule,
    flow_match_times,
    get_family,
    registered_families,
    screen_verdict,
    verify_point,
)
from instinctflash.distill.pipeline import (  # noqa: E402
    VERDICT_EXTEND_SCREEN, VERDICT_NO_DISTILLATION_NEEDED, VERDICT_TRAIN,
)
from instinctflash.train.recipe import Environment  # noqa: E402
from instinctflash.verify.certify import Outcome  # noqa: E402
from tests.run_tests import run_module_tests  # noqa: E402


def test_flow_match_grids_reproduce_the_deployed_schedules() -> None:
    # E8: video snr_shift 5 -> 2V grid t = {1000, 833.3}; action shift 1 -> 4A linear grid.
    video = flow_match_times(2, 5.0)
    assert video[0] == 1000.0
    assert abs(video[1] - 833.3333333333334) < 1e-9, video
    action = flow_match_times(4, 1.0)
    assert action == (1000.0, 750.0, 500.0, 250.0)
    # 1-step of ANY shift starts at the trained anchor t=1000 — the mechanism behind the
    # pi05 nfe1 == nfe10 result (E5): a 1-step sampler visits only the model's best point.
    assert flow_match_times(1, 5.0) == (1000.0,)
    assert flow_match_times(1, 1.0) == (1000.0,)


def test_family_adapters_declare_the_measured_facts() -> None:
    registered = registered_families()
    assert {"wan_va", "pi05"} <= set(registered)

    va = get_family("wan_va")
    names = [s.name for s in va.streams()]
    assert names == ["video", "action"]
    video, action = va.stream("video"), va.stream("action")
    assert (video.shift, video.guidance_scale, video.guidance_mode) == (5.0, 5.0, "cfg")
    assert (action.shift, action.guidance_scale, action.guidance_mode) == (1.0, 1.0, "positive_only")
    assert (video.certified_nfe, action.certified_nfe) == (2, 4)  # the 2V/4A certificate
    grids = va.schedule({"video": 1, "action": 4})
    assert grids["video"].times == (1000.0,)
    assert grids["action"].times == (1000.0, 750.0, 500.0, 250.0)
    # H1's asymmetry is a declaration, not a comment: video trains, action is untouched
    assert not va.trainable_set("video").untouched
    assert va.trainable_set("action").untouched
    # H3: the grid is a declared search axis
    shifted = va.schedule_grid("action", 2, shift=3.0)
    assert shifted.shift == 3.0 and shifted.times[0] == 1000.0 and shifted.times[1] > 500.0

    pi = get_family("pi05")
    assert [s.name for s in pi.streams()] == ["action"]
    assert pi.stream("action").teacher_nfe == 10
    assert pi.schedule_grid("action", 10).times[0] == 1000.0

    # the capability gate: JVP-free and non-adversarial by default (C1/C2), checked closed
    for adapter in registered.values():
        ok, why = adapter.requires().satisfied_by(Environment(has_jvp_attention=False))
        assert ok, (adapter.family, why)
        caps = adapter.requires()
        assert not caps.jvp_through_attention and not caps.adversarial


def test_training_halves_are_loud_seams() -> None:
    va = get_family("wan_va")
    for call in (
        lambda: va.build_student(None, {}),
        lambda: va.consistency_target("video", None, None, va.schedule_grid("video", 1)),
    ):
        try:
            call()
        except ExperimentalSeam as seam:
            assert "H1" in str(seam) and "h1_prereg" in str(seam)
        else:
            raise AssertionError("the wan_va training seam did not raise")
    pipeline = DistillPipeline("wan_va", {"video": 1, "action": 4})
    # Pillar 1 before Pillar 2: no screen verdict, no training
    try:
        pipeline.train(teacher=None)
    except ControlGateViolation as refusal:
        assert "no screen verdict" in str(refusal)
    else:
        raise AssertionError("train() ran without a screen verdict")
    go = ScreenVerdict("wan_va", {"video": 1, "action": 4}, -0.05, {}, (), {"point": "x"},
                       VERDICT_TRAIN, "unit")
    try:
        pipeline.train(teacher=None, screen=go)
    except ExperimentalSeam:
        pass
    else:
        raise AssertionError("the pipeline routed around the training seam")


def _outcomes(rates: list[bool], arm: str) -> list[Outcome]:
    return [
        Outcome(f"task{i % 5}/s{i}", 10_000 + i, f"task{i % 5}", success)
        for i, success in enumerate(rates)
    ]


VA_GUIDANCE = {"video": ("cfg", 5.0), "action": ("positive_only", 1.0)}
#: the guidance grid swept at 1V/4A on the campaign's pinned scenes (gw_autotable.md, h1_report §4b)
VA_1V4A_GRID = (
    {"point": "1V/4A@w5", "guidance": {"video": "cfg"}, "success": 0.7521},
    {"point": "1V/4A@w3", "guidance": {"video": 3}, "success": 0.8824},
    {"point": "1V/4A@w1", "guidance": {"video": {"mode": "cfg", "scale": 1.0}}, "success": 0.8833},
    {"point": "1V/4A@w7", "guidance": {"video": 7}, "success": 0.5109},
    {"point": "1V/4A@w9", "guidance": {"video": 9}, "success": 0.2759},
)


def test_verify_point_refuses_without_the_matched_control() -> None:
    teacher = _outcomes([True] * 90 + [False] * 10, "teacher")
    student = _outcomes([True] * 85 + [False] * 15, "student")
    schedule = {"nfe": {"video": 1, "action": 4}, "guidance": {"video": {"mode": "cfg", "scale": 1.0}}}
    for control, control_schedule, grid, expected in (
        (None, schedule, VA_1V4A_GRID, "no untrained matched-NFE control"),
        ([], schedule, VA_1V4A_GRID, "no untrained matched-NFE control"),
        (teacher, None, VA_1V4A_GRID, "no declared schedule"),
        (teacher, {"nfe": {"video": 2, "action": 4}, "guidance": schedule["guidance"]}, VA_1V4A_GRID,
         "different experiment"),
        # THE OPERATING POINT, not just nfe: a control at the shipped guidance (no guidance key =
        # the family's cfg@5) is not execution-matched to a guidance-off student
        (teacher, {"nfe": {"video": 1, "action": 4}}, VA_1V4A_GRID, "different experiment"),
        # the strong form: the grid must have been swept...
        (teacher, schedule, None, "guidance grid was not swept"),
        (teacher, schedule, [], "guidance grid was not swept"),
        # ...the control must be in it...
        (teacher, schedule, [g for g in VA_1V4A_GRID if g["point"] != "1V/4A@w1"], "not in the swept grid"),
        # ...and it must be the grid's best: a student verified against 1V/4A@w5 (0.752) while
        # w3 scored 0.882 is the H1 confound (+14.3 pp vs w5, +0.0 vs w3)
        (teacher, {"nfe": {"video": 1, "action": 4}, "guidance": {"video": "cfg"}}, VA_1V4A_GRID,
         "not the best untrained configuration"),
        # ...measured: a grid row without a success cannot be compared
        (teacher, schedule, [*VA_1V4A_GRID, {"point": "1V/4A@w2", "guidance": {"video": 2}}],
         "carries no success"),
    ):
        student_schedule = schedule
        if expected == "not the best untrained configuration":
            student_schedule = {"nfe": {"video": 1, "action": 4}, "guidance": {"video": "cfg"}}
        try:
            verify_point(
                point="1V/4A", teacher=teacher, control=control, student=student,
                student_schedule=student_schedule, control_schedule=control_schedule, margin=-0.05,
                control_grid=grid, guidance_axis_required=True, family_guidance=VA_GUIDANCE,
            )
        except ControlGateViolation as violation:
            assert expected in str(violation), (expected, str(violation))
            assert "matched-NFE-control law" in str(violation)
        else:
            raise AssertionError(f"the gate did not fire for: {expected}")
    # a family with no negative branch anywhere (pi05) has no guidance axis to sweep: no grid needed
    control = _outcomes([True] * 70 + [False] * 30, "control")
    report = verify_point(
        point="nfe1", teacher=teacher, control=control, student=student,
        student_schedule={"nfe": {"action": 1}}, control_schedule={"nfe": {"action": 1}},
        margin=-0.05, guidance_axis_required=False, family_guidance={"action": ("none", 1.0)},
    )
    assert report.control_guidance == {"action": {"mode": "none", "scale": 1.0}}
    # canonicalisation: identical serving, different spellings, one operating point
    a = canonical_schedule({"nfe": {"video": 1, "action": 4}, "guidance": {"video": "positive_only"}}, VA_GUIDANCE)
    b = canonical_schedule({"nfe": {"action": 4, "video": 1},
                            "guidance": {"video": {"mode": "positive_only", "scale": 1.0},
                                         "action": "positive_only"}}, VA_GUIDANCE)
    assert a == b, (a, b)


def test_verify_point_states_b_minus_a_and_c_minus_b_separately() -> None:
    # A 90%, B 70% (step reduction costs 20 points), C 85% (training bought 15 back).
    teacher = _outcomes([True] * 90 + [False] * 10, "teacher")
    control = _outcomes([True] * 70 + [False] * 30, "control")
    student = _outcomes([True] * 85 + [False] * 15, "student")
    schedule = {"nfe": {"video": 1, "action": 4}, "guidance": {"video": {"mode": "cfg", "scale": 1.0}}}
    report = verify_point(
        point="1V/4A", teacher=teacher, control=control, student=student,
        student_schedule=schedule, control_schedule=dict(schedule), margin=-0.05,
        control_grid=VA_1V4A_GRID, family_guidance=VA_GUIDANCE,
    )
    assert report.control_guidance["video"] == {"mode": "cfg", "scale": 1.0}
    assert [g["point"] for g in report.guidance_grid_swept] == [g["point"] for g in VA_1V4A_GRID]
    assert "best untrained configuration" in report.summary()
    assert abs(report.b_minus_a.delta - (-0.20)) < 1e-12
    assert abs(report.c_minus_b.delta - (+0.15)) < 1e-12
    assert abs(report.c_minus_a.delta - (-0.05)) < 1e-12
    summary = report.summary()
    assert "cost of step reduction" in summary and "what training bought" in summary
    # the control block for provenance carries both deltas and all three outcome hashes
    block = report.control_block(
        teacher_outcomes_sha256="a" * 64,
        control_outcomes_sha256="b" * 64,
        student_outcomes_sha256="c" * 64,
    )
    assert block["b_minus_a"]["delta"] == report.b_minus_a.delta
    assert block["c_minus_b"]["delta"] == report.c_minus_b.delta
    assert block["control_guidance"]["video"]["scale"] == 1.0
    assert len(block["guidance_grid_swept"]) == len(VA_1V4A_GRID)
    from instinctflash.descriptors.distillation import CONTROL_KEYS

    assert set(CONTROL_KEYS) <= set(block)
    # the reading the law exists for: C−B indistinguishable from zero is said out loud
    flat = verify_point(
        point="1V/4A", teacher=teacher, control=control, student=control,
        student_schedule=schedule, control_schedule=dict(schedule), margin=-0.05,
        control_grid=VA_1V4A_GRID, family_guidance=VA_GUIDANCE,
    )
    assert flat.training_bought_nothing and "training bought nothing" in flat.summary()


def test_pipeline_verify_and_stamp_enforces_the_gate_end_to_end() -> None:
    import json

    with tempfile.TemporaryDirectory() as temporary:
        td = Path(temporary)
        pkg = td / "pkg"
        pkg.mkdir()
        (pkg / "config.json").write_text("{}")
        (pkg / "instinctflash.json").write_text(json.dumps({
            "instinctflash_schema": 1,
            "execution": {"model_id": "x", "backbone": "wan_va", "servable": True,
                          "nfe": {"video": 1, "action": 4}},
        }))

        def jsonl(name: str, flips: int) -> Path:
            p = td / name
            with open(p, "w") as f:
                for i in range(60):
                    f.write(json.dumps({"episode_id": f"e{i}", "seed": i, "task": f"t{i % 3}",
                                        "success": i >= flips}) + "\n")
            return p

        teacher, control, student = jsonl("a.jsonl", 3), jsonl("b.jsonl", 12), jsonl("c.jsonl", 6)
        pipeline = DistillPipeline("wan_va", {"video": 1, "action": 4}, out_dir=pkg,
                                   dataset="robotwin2", recipe_id="unit")
        try:
            pipeline.verify_and_stamp(
                point="1V/4A", teacher_outcomes=teacher, control_outcomes=None,
                student_outcomes=student, control_schedule=None, margin=-0.05,
            )
        except ControlGateViolation:
            pass
        else:
            raise AssertionError("verify_and_stamp accepted a missing control file")

        control_schedule = {
            "nfe": {"video": 1, "action": 4},
            "grids": {"action": [1000.0, 750.0, 500.0, 250.0], "video": [1000.0]},
        }
        # wan_va serves CFG on video, so the guidance grid is mandatory at this schedule
        try:
            pipeline.verify_and_stamp(
                point="1V/4A", teacher_outcomes=teacher, control_outcomes=control,
                student_outcomes=student, control_schedule=control_schedule, margin=-0.05,
            )
        except ControlGateViolation as refusal:
            assert "guidance grid was not swept" in str(refusal)
        else:
            raise AssertionError("verify_and_stamp accepted a control whose guidance grid was not swept")
        # the shipped-guidance control IS the family default here, and the grid says w=1 beats it:
        # this student (trained at the shipped point) is verified against the wrong control
        try:
            pipeline.verify_and_stamp(
                point="1V/4A", teacher_outcomes=teacher, control_outcomes=control,
                student_outcomes=student, control_schedule=control_schedule, margin=-0.05,
                control_grid=VA_1V4A_GRID,
            )
        except ControlGateViolation as refusal:
            assert "not the best untrained configuration" in str(refusal)
        else:
            raise AssertionError("a control that is not the best of its grid was accepted")
        # a student and control both at the guidance-off point, with the grid: the stamp lands
        pipeline = DistillPipeline("wan_va", {"video": 1, "action": 4},
                                   guidance={"video": {"mode": "cfg", "scale": 1.0}},
                                   out_dir=pkg, dataset="robotwin2", recipe_id="unit")
        control_schedule["guidance"] = {"video": 1.0}
        report = pipeline.verify_and_stamp(
            point="1V/4A@w1", teacher_outcomes=teacher, control_outcomes=control,
            student_outcomes=student, control_schedule=control_schedule, margin=-0.05,
            control_grid=VA_1V4A_GRID,
            teacher_model_id="robbyant/lingbot-va-posttrain-robotwin",
            teacher_weights_sha256="a" * 64, dataset_sha256="b" * 64,
        )
        assert abs(report.b_minus_a.delta - (-0.15)) < 1e-12
        assert abs(report.c_minus_b.delta - (+0.10)) < 1e-12
        # the stamp is the block validate verifies; it must come out intact and complete
        from instinctflash.descriptors.distillation import verify_distillation

        status, block, problems = verify_distillation(pkg)
        assert status == "intact" and not problems, (status, problems)
        assert block["schedule"]["nfe"] == {"video": 1, "action": 4}
        assert block["schedule"]["guidance"]["video"] == {"mode": "cfg", "scale": 1.0}
        assert block["schedule"]["guidance"]["action"] == {"mode": "positive_only", "scale": 1.0}
        assert block["matched_nfe_control"]["b_minus_a"]["delta"] == report.b_minus_a.delta
        assert block["matched_nfe_control"]["control_guidance"]["video"]["scale"] == 1.0
        assert [g["point"] for g in block["matched_nfe_control"]["guidance_grid_swept"]][:2] == ["1V/4A@w5", "1V/4A@w3"]


def _va_report() -> dict:
    """A frontier report in the sweep-report row shape, carrying the campaign's measured numbers
    at three schedules (h1_report.md §4b, gw_autotable.md, untrained_frontier.md)."""
    def row(point, nfe, guidance, success, delta, lo, hi, n, b1, b2):
        return {"point": point, "nfe": nfe, "guidance": guidance, "status": "SCREENED",
                "point_success": success, "delta": delta, "interval": [lo, hi],
                "interval_method": "most_conservative_of_three", "deciding_lower_bound": lo,
                "n_pairs": n, "forwards_batch1": b1, "forwards_batch2": b2}
    w = lambda s: {"video": {"mode": "cfg", "scale": s}}  # noqa: E731
    return {
        "sweep": {"baseline": {"id": "2V/4A@w5", "nfe": {"video": 2, "action": 4}},
                  "screening": {"margin": -0.05}},
        "baseline": {"success": 0.9138, "n": 232, "guidance": w(5.0), "forwards_batch1": 0, "forwards_batch2": 10},
        "rows": [
            row("2V/2A@w5", {"video": 2, "action": 2}, w(5.0), 0.8821, -0.0306, -0.0716, 0.0105, 229, 0, 8),
            row("2V/2A@w3", {"video": 2, "action": 2}, w(3.0), 0.8833, -0.0262, -0.0699, 0.0175, 233, 0, 8),
            row("2V/2A@w1", {"video": 2, "action": 2}, w(1.0), 0.9208, 0.0090, -0.0350, 0.0580, 227, 8, 0),
            row("1V/4A@w5", {"video": 1, "action": 4}, w(5.0), 0.7478, -0.1681, -0.2443, -0.0961, 226, 0, 9),
            row("1V/4A@w3", {"video": 1, "action": 4}, w(3.0), 0.8824, -0.0219, -0.0666, 0.0227, 228, 0, 9),
            row("1V/4A@w1", {"video": 1, "action": 4}, w(1.0), 0.8833, -0.0350, -0.0910, 0.0180, 227, 9, 0),
            row("1V/1A@w5", {"video": 1, "action": 1}, w(5.0), 0.5625, -0.3571, -0.4484, -0.2696, 224, 0, 6),
            row("1V/1A@w1", {"video": 1, "action": 1}, w(1.0), 0.6777, -0.2200, -0.2941, -0.1500, 229, 6, 0),
        ],
        "table_markdown": "| unit frontier |",
    }


def test_no_distillation_needed_is_a_first_class_verdict() -> None:
    report = _va_report()
    # 2V/2A: the best untrained configuration (w=1, batch-1) clears the margin -> do not train
    v = screen_verdict(report, family="wan_va", nfe={"video": 2, "action": 2})
    assert v.verdict == VERDICT_NO_DISTILLATION_NEEDED and v.no_distillation_needed
    assert v.best["point"] == "2V/2A@w1" and v.margin == -0.05
    assert [c["point"] for c in v.candidates] == ["2V/2A@w5", "2V/2A@w3", "2V/2A@w1"]
    artifact = v.artifact()
    assert artifact["kind"] == "fewstep_screen_verdict" and artifact["frontier_table_markdown"] == "| unit frontier |"
    assert "DRAFT pre-registration (NOT RUN)" in artifact["certification_prereg_stub_markdown"]
    assert "2V/2A@w1" in artifact["certification_prereg_stub_markdown"]
    assert "8 batch-1 + 0 batch-2 forwards/cycle" in artifact["certification_prereg_stub_markdown"]
    assert "NO DISTILLATION NEEDED" in v.headline()
    # ...and the trainer does not run: the pipeline raises the verdict, carrying the artifact
    pipeline = DistillPipeline("wan_va", {"video": 2, "action": 2}, guidance={"video": 1.0})
    try:
        pipeline.train(teacher=None, screen=pipeline.screen_verdict(report))
    except NoDistillationNeeded as verdict:
        assert verdict.verdict.best["point"] == "2V/2A@w1"
        assert verdict.verdict.control_grid()[2]["success"] == 0.9208
    else:
        raise AssertionError("train() ran where the screen said no distillation is needed")
    # 1V/4A: the best untrained point straddles the margin at screening n -> extend/certify first,
    # not train (the H1 case; training later measured +1.3 pp n.s. at this exact point)
    v = screen_verdict(report, family="wan_va", nfe={"video": 1, "action": 4})
    assert v.verdict == VERDICT_EXTEND_SCREEN and v.best["point"] == "1V/4A@w1"
    try:
        DistillPipeline("wan_va", {"video": 1, "action": 4}, guidance={"video": 1.0}).train(teacher=None, screen=v)
    except ControlGateViolation as refusal:
        assert "extend_screen" in str(refusal)
    else:
        raise AssertionError("train() ran on an unresolved screen")
    # 1V/1A: a real cliff even with guidance off -> training has something to buy; the control
    # is the best untrained configuration (w=1), never the shipped w=5
    v = screen_verdict(report, family="wan_va", nfe={"video": 1, "action": 1})
    assert v.verdict == VERDICT_TRAIN and v.best["point"] == "1V/1A@w1"
    assert "TRAIN" in v.headline() and "standing control" in v.headline()
    # a schedule the report never screened cannot be decided
    v = screen_verdict(report, family="wan_va", nfe={"video": 3, "action": 4})
    assert v.verdict == VERDICT_EXTEND_SCREEN and v.best is None


def test_screening_spec_always_carries_the_guidance_axis_on_a_cfg_family() -> None:
    driver = {"command": ["x"], "environment": {}, "timeout_seconds": 1, "revision": "r"}
    spec = DistillPipeline("wan_va", {"video": 1, "action": 4}).screening_spec(name="unit", driver=driver)
    # default grid on a CFG family: the family's own scale and the negative branch off
    assert spec["guidance_axis"] == {"video": [5.0, 1.0]}
    ids = [p["id"] for p in spec["points"]]
    assert ids == ["1V/4A@vw5", "1V/4A@vw1"], ids
    assert spec["points"][1]["guidance"] == {"video": 1.0}
    assert spec["baseline"] == {"id": "2V/4A", "nfe": {"video": 2, "action": 4}}
    from benchmarks.vla.schedule_sweep import validate_sweep

    validate_sweep(spec)  # what the trainer emits, the sweep accepts
    spec = DistillPipeline("wan_va", {"video": 1, "action": 4}).screening_spec(
        name="unit", driver=driver, guidance_grid={"video": [1, 3, 5, 7, 9]})
    assert [p["guidance"]["video"] for p in spec["points"]] == [1, 3, 5, 7, 9]
    validate_sweep(spec)
    # pi05 has no negative branch anywhere: no axis, one point
    spec = DistillPipeline("pi05", {"action": 1}).screening_spec(
        name="unit", driver=driver, baseline_nfe={"action": 10})
    assert "guidance_axis" not in spec and [p["id"] for p in spec["points"]] == ["1A"]
    assert not get_family("pi05").guidance_axis_required() and get_family("wan_va").guidance_axis_required()


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))

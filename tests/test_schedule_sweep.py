#!/usr/bin/env python3
"""The schedule-sweep plan/suite type: the frontier campaign, productized.

What is pinned here, and why:

  * SPEC -> ARMS -> PLAN is deterministic and counterbalanced, and the sweep declaration is
    covered by the plan digest -- a sweep that cannot be reproduced from its spec is a
    hand-run campaign with extra steps.
  * The frontier report's TABLE SHAPE is checked against the real campaign artifact
    (`tests/fixtures/fewstep_frontier/frontier_table.md`, the untrained low-step frontier of
    2026-08-31): same nine columns, same per-point row form, same baseline/noise-floor rows.
    The campaign is the reference implementation; the product must emit what it emitted.
  * SCREENING FRAMING: certify() verdict lines are recorded in the JSON, and the table never
    quotes a SHIP/verdict line (frontier prereg §0).
  * Paired deltas from a synthetic end-to-end run are checked against hand-computed values --
    the driver's outcomes are a pure function of (task, seed, schedule), so every delta is
    exact and a regression in pairing or in the delta arithmetic cannot hide.
  * THE GUIDANCE AXIS: a point is (schedule, guidance, CFG batching). Points may vary nfe and
    guidance jointly, the arm carries the guidance leg, the table reports batch-1 and batch-2
    forwards separately, and the report names the BEST untrained configuration per schedule --
    the strengthened matched-NFE control (RFC §11; the H1 screen's +14.3 pp vs the shipped
    guidance was +0.0 vs the best untrained knob).

No GPU, no torch, no model weights.
"""
from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.vla.plan import validate_plan  # noqa: E402
from benchmarks.vla.registry import Registry, load_registry  # noqa: E402
from benchmarks.vla.runner import execute_plan  # noqa: E402
from benchmarks.vla.schedule_sweep import (  # noqa: E402
    arms_from_sweep,
    build_sweep_plan,
    build_sweep_report,
    frontier_table,
    repeat_arm_id,
    validate_sweep,
)
from benchmarks.vla.util import ConfigurationError, sha256_json  # noqa: E402
from tests.run_tests import run_module_tests  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "fewstep_frontier" / "frontier_table.md"
MODEL = "robbyant/lingbot-va-posttrain-robotwin"


#: Success is a threshold on a pure function of (task, seed); the threshold is a pure function
#: of the schedule. Identical schedules therefore produce identical outcomes (a zero-discordance
#: noise floor), and every paired delta is exactly the measure of scenes between two thresholds.
SWEEP_DRIVER = '''
import argparse, hashlib, json, random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
job = json.loads(args.request.read_text())
request = job["request"]
required = request["suite"]["required_metrics"]
schedule = request["arm"]["operating_point"]["schedule"]
nfe = schedule["nfe"]
rate = min(0.95, 0.40 + 0.10 * nfe["video"] + 0.08 * nfe["action"])
# the campaign's shape: at one video step the shipped w=5 is a cliff, w=3 and w=1 are not
g = (schedule.get("guidance") or {}).get("video")
w = g.get("scale", 5.0) if isinstance(g, dict) else (g if isinstance(g, (int, float)) else (1.0 if g == "positive_only" else 5.0))
if nfe["video"] == 1:
    rate -= 0.20 if w >= 5 else (0.05 if w > 1 else 0.0)
draw = random.Random(f"{request['task']}|{request['requested_seed']}").random()
metrics = {}
if "finite" in required:
    metrics["finite"] = True
if "success" in required:
    metrics["success"] = bool(draw < rate)
if "action_digest" in required:
    payload = f"{request['task']}|{request['requested_seed']}|{sorted(nfe.items())}"
    metrics["action_digest"] = hashlib.sha256(payload.encode()).hexdigest()
if "latency_ms" in required:
    base = 30.0 + 10.0 * (nfe["video"] + nfe["action"])
    metrics["latency_ms"] = [base] * request["measurement"]["iterations"]
args.output.write_text(json.dumps({
    "schema_version": 1,
    "job_id": job["job_id"],
    "request_sha256": job["request_sha256"],
    "status": "completed",
    "resolved_seed": request["requested_seed"],
    "metrics": metrics,
    "provenance": {
        "model_revision": request["model"]["checkpoint"]["revision"],
        "driver_revision": job["driver"]["revision"],
        "environment_fingerprint": hashlib.sha256(b"sweep-driver").hexdigest(),
        "synthetic": True,
    },
}))
'''


def _sweep(driver_command: list[str]) -> dict:
    return {
        "schema_version": 1,
        "name": "unit untrained low-step frontier",
        "baseline": {"id": "2V/4A", "nfe": {"video": 2, "action": 4}, "forwards_per_cycle": 10},
        "points": [
            {"id": "3V/4A", "nfe": {"video": 3, "action": 4}, "forwards_per_cycle": 11},
            {"id": "1V/4A", "nfe": {"video": 1, "action": 4}, "forwards_per_cycle": 9},
            # the guidance axis, varied jointly with nfe: the same schedule at w=3 and at w=1
            {"id": "1V/4A@w3", "nfe": {"video": 1, "action": 4}, "forwards_per_cycle": 9,
             "guidance": {"video": 3}, "batch2_forwards_per_cycle": 9},
            {"id": "1V/4A@w1", "nfe": {"video": 1, "action": 4}, "forwards_per_cycle": 9,
             "guidance": {"video": {"mode": "cfg", "scale": 1.0}, "action": "positive_only"}},
            {"id": "1V/1A", "nfe": {"video": 1, "action": 1}, "forwards_per_cycle": 5},
        ],
        "repeat_baseline": True,
        "driver": {
            "command": driver_command,
            "environment": {},
            "timeout_seconds": 120,
            "revision": "sweep-unit-v1",
        },
        "screening": {"margin": -0.05, "interval": "wald_central95", "min_pairs": 1},
        "control_steps_per_cycle": 32,
    }


def _sweep_registry() -> Registry:
    registry = load_registry()
    raw = copy.deepcopy(registry.raw)
    raw["profiles"]["sweep_unit"] = {
        "suites": ["robotwin50_easy", "single_gpu_latency"],
        "limits": {"tasks": 5, "seeds_per_task": {"closed_loop": 4, "latency": 1}},
        "latency": {"warmup": 1, "iterations": 3},
        "arm_repeats": {"closed_loop": 1, "latency": 1},
    }
    return Registry(raw=raw, digest=sha256_json(raw), path=registry.path)


def test_sweep_spec_is_validated() -> None:
    spec = _sweep(["python3", "driver.py"])
    validate_sweep(spec)
    bad = copy.deepcopy(spec)
    bad["points"][0]["id"] = "2V/4A"  # collides with the baseline
    try:
        validate_sweep(bad)
    except ConfigurationError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("a point colliding with the baseline id was accepted")
    bad = copy.deepcopy(spec)
    bad["points"][0]["nfe"] = 2  # a bare count silently applies to both streams
    try:
        validate_sweep(bad)
    except ConfigurationError as error:
        assert "phase name" in str(error)
    else:
        raise AssertionError("a bare nfe count was accepted")
    bad = copy.deepcopy(spec)
    bad["screening"]["interval"] = "eyeballed"
    try:
        validate_sweep(bad)
    except ConfigurationError as error:
        assert "decision rule" in str(error)
    else:
        raise AssertionError("an undeclared interval method was accepted")
    # the guidance leg is validated by the declaration schema, per stream, never guessed
    bad = copy.deepcopy(spec)
    bad["points"][2]["guidance"] = {"video": "bogus"}
    try:
        validate_sweep(bad)
    except ConfigurationError as error:
        assert "neither a guidance mode nor a numeric scale" in str(error)
    else:
        raise AssertionError("an unreadable guidance value was accepted")
    bad = copy.deepcopy(spec)
    bad["points"][2]["guidance"] = {"action": {"mode": "positive_only", "scale": 3}}
    try:
        validate_sweep(bad)
    except ConfigurationError as error:
        assert "discards the negative branch" in str(error)
    else:
        raise AssertionError("a contradictory guidance value was accepted")
    bad = copy.deepcopy(spec)
    bad["points"][2]["batch2_forwards_per_cycle"] = 12  # more than forwards_per_cycle
    try:
        validate_sweep(bad)
    except ConfigurationError as error:
        assert "batch2_forwards_per_cycle" in str(error)
    else:
        raise AssertionError("an impossible batch split was accepted")


def test_sweep_compiles_to_arms_with_the_baseline_as_control() -> None:
    spec = _sweep(["python3", "driver.py"])
    arms = arms_from_sweep(spec)
    assert arms["control_arm"] == "2V/4A"
    identifiers = [arm["id"] for arm in arms["arms"]]
    assert identifiers == ["2V/4A", "3V/4A", "1V/4A", "1V/4A@w3", "1V/4A@w1", "1V/1A",
                           repeat_arm_id(spec)]
    # the guidance leg rides in the arm's schedule, so the driver serves the declared scale
    by_id = {arm["id"]: arm for arm in arms["arms"]}
    assert by_id["1V/4A@w3"]["operating_point"]["schedule"]["guidance"] == {"video": 3}
    assert "guidance" not in by_id["1V/4A"]["operating_point"]["schedule"]
    assert by_id["1V/4A@w1"]["operating_point"]["schedule"]["nfe"] == by_id["1V/4A"]["operating_point"]["schedule"]["nfe"]
    roles = {arm["id"]: arm["role"] for arm in arms["arms"]}
    assert roles["2V/4A"] == "control"
    assert all(roles[i] == "treatment" for i in identifiers[1:])
    # the repeat arm serves the IDENTICAL configuration: that is what makes it a noise floor
    by_id = {arm["id"]: arm for arm in arms["arms"]}
    assert (
        by_id[repeat_arm_id(spec)]["operating_point"]["schedule"]
        == by_id["2V/4A"]["operating_point"]["schedule"]
    )
    # every treatment arm carries the declared screening analysis, and says its gates are not
    # the decision surface
    for arm in arms["arms"]:
        if arm["role"] != "treatment":
            continue
        assert arm["gates"]["success"] == {
            "margin": -0.05, "interval": "wald_central95", "min_pairs": 1
        }
        assert "decision surface" in arm["gates"]["note"]


def test_sweep_plan_is_deterministic_counterbalanced_and_self_describing() -> None:
    registry = _sweep_registry()
    spec = _sweep(["python3", "driver.py"])
    first = build_sweep_plan(registry, spec, "sweep_unit", [MODEL])
    second = build_sweep_plan(registry, spec, "sweep_unit", [MODEL])
    assert first == second
    validate_plan(first)
    assert first["sweep"] == spec
    # the sweep block is covered by the plan digest: editing it breaks plan_id
    tampered = json.loads(json.dumps(first))
    tampered["sweep"]["points"][0]["nfe"]["video"] = 99
    try:
        validate_plan(tampered)
    except ConfigurationError as error:
        assert "digest" in str(error)
    else:
        raise AssertionError("an edited sweep declaration passed plan validation")
    # every pair contains every arm, and arm order is counterbalanced across pairs
    by_pair: dict[str, list[str]] = {}
    for job in first["jobs"]:
        by_pair.setdefault(job["request"]["pair_id"], []).append(job["request"]["arm"]["id"])
    arm_ids = {arm["id"] for arm in first["arms"]}
    assert all(set(order) == arm_ids for order in by_pair.values())
    assert any(order[0] == "2V/4A" for order in by_pair.values())
    assert any(order[-1] == "2V/4A" for order in by_pair.values())


def _run_sweep(root: Path) -> tuple[Registry, Path, dict]:
    driver = root / "sweep_driver.py"
    driver.write_text(SWEEP_DRIVER)
    registry = _sweep_registry()
    spec = _sweep([sys.executable, str(driver)])
    plan = build_sweep_plan(registry, spec, "sweep_unit", [MODEL])
    run_root = root / "run"
    progress = execute_plan(plan, run_root, repo_root=ROOT)
    assert progress["completed"] == progress["expected"], progress["failed"]
    return registry, run_root, spec


def _expected_delta(registry: Registry, nfe_a: dict, nfe_b: dict) -> float:
    """The driver's exact paired delta over the planned closed-loop scenes."""
    import random

    suite = registry.suites["robotwin50_easy"]
    tasks = suite["tasks"][:5]
    seeds = [int(suite["seed_base"]) + t * 10_000 + s for t in range(len(tasks)) for s in range(4)]
    scenes = [(task, int(suite["seed_base"]) + index * 10_000 + s)
              for index, task in enumerate(tasks) for s in range(4)]

    def rate(point):
        nfe = point["nfe"] if "nfe" in point else point
        r = min(0.95, 0.40 + 0.10 * nfe["video"] + 0.08 * nfe["action"])
        g = (point.get("guidance") or {}).get("video") if isinstance(point, dict) else None
        w = (g.get("scale", 5.0) if isinstance(g, dict)
             else (g if isinstance(g, (int, float)) else (1.0 if g == "positive_only" else 5.0)))
        if nfe["video"] == 1:
            r -= 0.20 if w >= 5 else (0.05 if w > 1 else 0.0)
        return r

    delta = 0
    for task, seed in scenes:
        draw = random.Random(f"{task}|{seed}").random()
        delta += (draw < rate(nfe_b)) - (draw < rate(nfe_a))
    assert seeds  # the seed arithmetic above must match plan.py's
    return delta / len(scenes)


def test_frontier_report_matches_the_campaign_artifact_shape() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        registry, run_root, spec = _run_sweep(Path(temporary))
        report = build_sweep_report(run_root, registry, allow_synthetic=True)

        # -- table shape vs the real campaign artifact (the reference fixture) ------------
        fixture_lines = FIXTURE.read_text().splitlines()
        fixture_header = next(line for line in fixture_lines if line.startswith("| point"))
        fixture_columns = [c.strip() for c in fixture_header.strip("|").split("|")]
        table = report["table_markdown"].splitlines()
        header = next(line for line in table if line.startswith("| point"))
        columns = [c.strip() for c in header.strip("|").split("|")]
        # the campaign artifact (2026-08-31) had one forwards column; the product reports the
        # CFG-batching leg of the operating point -- batch-1 and batch-2 forwards -- SEPARATELY
        # (RFC §11), so its single `forwards/cycle` column is the only one that splits in two
        assert len(fixture_columns) == 9 and len(columns) == 10
        assert fixture_columns[1] == "forwards/cycle"
        assert columns[1:3] == ["batch-1 fwd/cycle", "batch-2 fwd/cycle"]
        # identical column meanings otherwise; the two campaign-specific words (the baseline
        # arm's name inside the delta column, the interval rule inside the interval column) are
        # declared per sweep rather than hardcoded
        assert [columns[0], *columns[3:6]] == [fixture_columns[0], *fixture_columns[2:5]]
        assert columns[8:] == fixture_columns[7:]
        assert columns[6].startswith("Δ vs 2V/4A") and fixture_columns[5].startswith("Δ vs 2V/4A")
        assert columns[7].startswith("95% interval") and fixture_columns[6].startswith("95% interval")
        # one row per schedule point, baseline row first and marked as the control
        baseline_row = next(line for line in table if "baseline arm" in line)
        assert "(control)" in baseline_row
        for point in spec["points"]:
            assert any(line.startswith(f"| {point['id']} |") for line in table), point["id"]
        # the campaign artifact ends with the same screening disclaimer this table must carry
        assert any("not certification" in line for line in table)
        assert any("not certification" in line for line in fixture_lines)
        # noise-floor line, from the repeat arm
        assert any(line.startswith("Noise floor") for line in table)

        # -- screening framing: verdicts recorded, never quoted --------------------------
        assert report["screening"] is True
        assert "SHIP" not in report["table_markdown"]
        assert "VERDICT" not in report["table_markdown"]
        for row in report["rows"]:
            assert row["certify_verdict_recorded_not_quoted"]

        # -- the paired numbers are the driver's exact numbers ---------------------------
        rows = {row["point"]: row for row in report["rows"]}
        for point in spec["points"]:
            expected = _expected_delta(registry, spec["baseline"], point)
            assert abs(rows[point["id"]]["delta"] - expected) < 1e-12, point["id"]
            assert rows[point["id"]]["n_pairs"] == 20
        # the guidance axis: same schedule, different served scale, different outcome
        assert rows["1V/4A@w1"]["point_success"] >= rows["1V/4A@w3"]["point_success"] > rows["1V/4A"]["point_success"]
        assert rows["1V/4A@w3"]["guidance"] == {"video": {"mode": "cfg", "scale": 3.0}}
        assert rows["1V/4A@w1"]["guidance"]["video"] == {"mode": "cfg", "scale": 1.0}
        assert rows["1V/4A"]["guidance"] is None
        # CFG batching reported separately: declared split, derived split (wan_va adapter,
        # guidance off => all batch-1), and the baseline's family default (cfg@5 => all batch-2)
        assert (rows["1V/4A@w3"]["forwards_batch1"], rows["1V/4A@w3"]["forwards_batch2"]) == (0, 9)
        assert (rows["1V/4A@w1"]["forwards_batch1"], rows["1V/4A@w1"]["forwards_batch2"]) == (9, 0)
        assert (report["baseline"]["forwards_batch1"], report["baseline"]["forwards_batch2"]) == (0, 10)
        w1_row = next(line for line in table if line.startswith("| 1V/4A@w1 |"))
        assert w1_row.split("|")[2].strip() == "9" and w1_row.split("|")[3].strip() == "0"
        # the strengthened control: per schedule, the best over the guidance grid swept
        best = report["best_untrained_per_schedule"]
        group = best[json.dumps({"action": 4, "video": 1}, sort_keys=True, separators=(",", ":"))]
        assert [c["point"] for c in group["candidates"]] == ["1V/4A", "1V/4A@w3", "1V/4A@w1"]
        assert group["best"]["point"] == "1V/4A@w1"
        assert len(group["guidance_grid_swept"]) == 3 and group["guidance_grid_swept"][0] is None
        assert "Best untrained configuration per schedule" in report["table_markdown"]
        assert "| 1V/4A@w1 |" in next(l for l in table if l.startswith("| action=4, video=1 |"))
        floor = report["noise_floor"]
        assert floor["delta"] == 0.0 and floor["discordance"] == 0.0
        # latency: the driver's schedule-derived constants -> exact p50s and effective Hz
        assert rows["1V/1A"]["cycle_p50_ms"] == 50.0
        assert abs(rows["1V/1A"]["effective_hz"] - 32000.0 / 50.0) < 1e-9
        assert report["baseline"]["cycle_p50_ms"] == 90.0
        assert report["baseline"]["n"] == 20

        # -- synthetic evidence can never be reported as screening without saying so -----
        refused = build_sweep_report(run_root, registry)
        assert refused["synthetic"] is True
        assert refused["reportable_as_screening"] is False

        # -- artifacts land next to the evidence -----------------------------------------
        assert (run_root / "frontier.json").exists()
        assert (run_root / "frontier.md").read_text() == report["table_markdown"]


def test_sweep_report_refuses_a_plain_run() -> None:
    # a run whose plan never declared a sweep has no baseline semantics to pair against
    registry = _sweep_registry()
    spec = _sweep([sys.executable, "driver.py"])
    plan = build_sweep_plan(registry, spec, "sweep_unit", [MODEL])
    plan.pop("sweep")
    plan.pop("plan_id")
    plan["plan_id"] = sha256_json(plan)
    with tempfile.TemporaryDirectory() as temporary:
        run_root = Path(temporary) / "run"
        (run_root / "results").mkdir(parents=True)
        from benchmarks.vla.util import write_json_atomic

        write_json_atomic(run_root / "plan.json", plan)
        try:
            build_sweep_report(run_root, registry)
        except ConfigurationError as error:
            assert "no sweep" in str(error)
        else:
            raise AssertionError("sweep-report accepted a plan with no sweep declaration")


def test_frontier_table_renders_not_evaluated_rows() -> None:
    # a point with no paired closed-loop evidence is reported NOT EVALUATED, never invented
    report = {
        "sweep": _sweep(["python3", "driver.py"]),
        "max_pairs": 0,
        "interval_method": "wald_central95",
        "baseline": {"success": None, "n": None, "cycle_p50_ms": None, "effective_hz": None},
        "rows": [
            {
                "point": "1V/4A",
                "cells": ["1V/4A", "n/d", "n/d", "n/e", "n/e", "n/e",
                          "NOT EVALUATED — no paired closed-loop evidence", "—", "—", "—"],
            }
        ],
        "noise_floor": None,
    }
    table = frontier_table(report)
    assert "NOT EVALUATED" in table
    row = next(line for line in table.splitlines() if line.startswith("| 1V/4A"))
    assert len([c for c in row.strip("|").split("|")]) == 10
    assert not re.search(r"\bSHIP\b", table)


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))

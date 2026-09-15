"""A descriptive summary must preserve negative controls and gate updated routes."""

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "eval/user_e2e_2026-09-14/render_summary.py"
SPEC = importlib.util.spec_from_file_location("render_user_summary", SCRIPT)
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


def fixture_report():
    cells, comparisons = [], []
    def row(family, arm, ms, identifier=None, schedule=None, group=None, precision="native", tier="bitexact"):
        identifier = identifier or f"{family}-{arm}"
        cell = {"id": identifier, "family": family, "arm": arm, "status": "passed", "precision": precision,
                "model_id": f"owner/{family}", "revision": "a" * 40, "comparison_group": group or family,
                "effective_schedule": schedule or {"nfe": {"action": 10}}, "guidance": "checkpoint",
                "experimental": arm == "operating_point", "runtime_kwargs": {"tier_ceiling": tier},
                "execution_policy": {"tier_ceiling": tier}, "device": "NVIDIA Thor",
                "primary_metric": "continuation_p50_ms" if family in ("va", "dreamzero") else "generation_p50_ms",
                "primary": {"p50_ms": ms, "count": 12 if family in ("va", "dreamzero") else 20},
                "receipt": {"sha256": hashlib.sha256(identifier.encode()).hexdigest()},
                "task_quality_validated": False, "recommended": False}
        cells.append(cell)
        return cell
    def compare(baseline, candidate, *, changed=False):
        comparisons.append({"baseline": baseline["id"], "candidate": candidate["id"], "status": "passed",
                            "kind": "changed_policy_latency_ratio" if changed else "matched_policy_latency_ratio",
                            "same_sampling_policy": not changed,
                            "ratio_of_p50": baseline["primary"]["p50_ms"] / candidate["primary"]["p50_ms"]})
    eager, selected = {}, {}
    for family in RENDERER.FAMILIES:
        eager[family] = row(family, "eager_native", 100)
        default = row(family, "runtime_default", 80)
        selected[family] = row(family, "runtime_selected", 120 if family in ("vla4", "vla2") else 50,
                               precision="fp8", tier="numeric")
        compare(eager[family], default)
        compare(eager[family], selected[family])
    for family in ("va", "dreamzero"):
        schedule = {"nfe": {"video": 2, "action": 4}} if family == "va" else {"nfe": {"video_action": 16}, "step_cache": "dynamic"}
        native = row(family, "operating_point", 1, identifier=f"{family}-extra-native", schedule=schedule,
                     group=f"{family}-extra", tier="behavioral")
        fp8 = row(family, "operating_point", .5, identifier=f"{family}-extra-fp8", schedule=schedule,
                  group=f"{family}-extra", precision="fp8", tier="behavioral")
        compare(native, fp8)
        compare(eager[family], native, changed=True)
        compare(eager[family], fp8, changed=True)
    for family, identifier, ms, numeric in (("vla4", "vla4-runtime_update", 30, False),
                                            ("vla2", "vla2-runtime_update", 90, False),
                                            ("vla2", "vla2-runtime_numeric", 40, True)):
        update = row(family, "runtime_update", ms, identifier=identifier,
                     precision="native" if numeric else "fp8", tier="numeric")
        baseline = eager[family] if numeric else selected[family]
        update["baseline_cell"] = baseline["id"]
        compare(baseline, update)
    report = {"schema": "instinctflash.user_report.v1", "status": "passed", "errors": [], "pending": [],
              "task_quality_validated": False, "recommended": False, "cells": cells, "comparisons": comparisons,
              "counts": {"expected_cells": 31, "main_cells": 24, "operating_point_cells": 4, "runtime_update_cells": 3,
                         "passed_cells": 31, "failed_cells": 0, "incomplete_cells": 0, "failed_comparisons": 0}}
    gate = {"schema": 1, "gates": {cell["id"]: {"status": "passed", "receipt_sha256": cell["receipt"]["sha256"]}
                                     for cell in cells if cell["arm"] == "runtime_update"}}
    return report, gate


def test_fastest_runtime_uses_passed_updates_and_keeps_slow_original_fp8_controls():
    report, gate = fixture_report()
    before = copy.deepcopy(report)
    rows, excluded = RENDERER.build_summary(report, gate)
    vla4, vla2 = rows[1], rows[2]
    assert vla4["fastest_cell"] == "vla4-runtime_update"
    assert vla4["fastest_route"] == "FP8; ceiling NUMERIC" and vla4["fastest_p50_ms"] == 30
    assert vla4["fastest_ratio_vs_eager"] == 100 / 30
    assert vla2["fastest_cell"] == "vla2-runtime_numeric" and vla2["fastest_route"] == "native; ceiling NUMERIC"
    assert vla2["default_cell"] == "vla2-runtime_default" and vla2["default_route"] == "native; ceiling BITEXACT"
    assert not excluded and report == before
    assert next(row for row in report["cells"] if row["id"] == "vla4-runtime_selected")["primary"]["p50_ms"] == 120


@pytest.mark.parametrize("status", ["failed", "inactive", "incomplete", "missing"])
def test_unvalidated_update_is_excluded_without_relabelling_original_selected_cell(status):
    report, gate = fixture_report()
    if status == "missing":
        gate["gates"].pop("vla4-runtime_update")
    else:
        gate["gates"]["vla4-runtime_update"]["status"] = status
    rows, excluded = RENDERER.build_summary(report, gate)
    assert rows[1]["fastest_cell"] == "vla4-runtime_default"
    assert rows[1]["fastest_p50_ms"] == 80
    assert excluded == ["vla4-runtime_update"]


def test_all_runtime_routes_slower_than_eager_keep_a_negative_ratio():
    report, gate = fixture_report()
    default = next(row for row in report["cells"] if row["id"] == "vla4-runtime_default")
    default["primary"]["p50_ms"] = 110
    gate["gates"]["vla4-runtime_update"]["status"] = "failed"
    rows, _ = RENDERER.build_summary(report, gate)
    assert rows[1]["fastest_cell"] == "vla4-runtime_default"
    assert rows[1]["eager_p50_ms"] == 100 and rows[1]["fastest_p50_ms"] == 110
    assert rows[1]["fastest_ratio_vs_eager"] == 100 / 110 < 1


def test_operating_points_never_enter_main_minimum_and_keep_actual_schedules():
    report, gate = fixture_report()
    rows, _ = RENDERER.build_summary(report, gate)
    assert len(rows) == 12
    assert all(row["section"] == "same_sampling_policy" and row["fastest_p50_ms"] >= 30 for row in rows[:8])
    assert all(row["section"] == "changed_operating_point" for row in rows[8:])
    assert rows[8]["schedule_label"] == "2 video / 4 action steps"
    assert rows[10]["schedule_label"] == "16 solver updates; dynamic reuse"
    assert rows[8]["operating_ratio_vs_eager"] == 100


@pytest.mark.parametrize("problem", ["incomplete", "counts", "cell_status", "quality", "zero", "nan", "samples",
                                     "different_policy", "missing_comparison", "gate_hash", "missing_gate_hash", "cross_policy_ratio"])
def test_refuses_incomplete_or_inconsistent_evidence(problem):
    report, gate = fixture_report()
    if problem == "incomplete":
        report["status"] = "incomplete"
    elif problem == "counts":
        report["counts"]["passed_cells"] = 30
    elif problem == "cell_status":
        report["cells"][0]["status"] = "failed"
    elif problem == "quality":
        report["task_quality_validated"] = True
    elif problem in ("zero", "nan"):
        report["cells"][0]["primary"]["p50_ms"] = 0 if problem == "zero" else float("nan")
    elif problem == "samples":
        report["cells"][0]["primary"]["count"] = 19
    elif problem == "different_policy":
        report["cells"][1]["effective_schedule"] = {"nfe": {"action": 1}}
    elif problem == "missing_comparison":
        report["comparisons"].pop(0)
    elif problem == "gate_hash":
        gate["gates"]["vla4-runtime_update"]["receipt_sha256"] = "0" * 64
    elif problem == "missing_gate_hash":
        gate["gates"]["vla4-runtime_update"].pop("receipt_sha256")
    else:
        next(row for row in report["comparisons"] if row["kind"] == "changed_policy_latency_ratio")["ratio_of_p50"] = 999
    with pytest.raises(ValueError):
        RENDERER.build_summary(report, gate)


def test_renderer_writes_bound_csv_and_under_80_line_rst_without_changing_inputs(tmp_path):
    report, gate = fixture_report()
    report_path, gate_path = tmp_path / "report.json", tmp_path / "gate.json"
    report_path.write_text(json.dumps(report))
    gate["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    gate_path.write_text(json.dumps(gate))
    before = report_path.read_bytes(), gate_path.read_bytes()
    csv_path, rst_path = tmp_path / "results.csv", tmp_path / "results.rst"
    result = subprocess.run([sys.executable, "-I", str(SCRIPT), "--report", str(report_path), "--prompt-gate", str(gate_path),
                             "--csv-output", str(csv_path), "--rst-output", str(rst_path)], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12 and rows[1]["fastest_cell"] == "vla4-runtime_update"
    assert all(row["report_sha256"] == hashlib.sha256(before[0]).hexdigest() for row in rows)
    assert all(row["prompt_gate_sha256"] == hashlib.sha256(before[1]).hexdigest() for row in rows)
    assert all(row["task_quality_validated"] == row["recommended"] == "False" for row in rows)
    rst = rst_path.read_text()
    assert len(rst.splitlines()) <= 80
    assert "2,146 loaded tensors" in rst and "12 continuation calls" in rst and "51-call queue" in rst
    assert "Fastest Runtime" in rst and "Changed operating points (separate)" in rst
    assert before == (report_path.read_bytes(), gate_path.read_bytes())


def test_incomplete_report_does_not_write_placeholder_results(tmp_path):
    report, gate = fixture_report()
    report["status"] = "incomplete"
    report_path, gate_path = tmp_path / "report.json", tmp_path / "gate.json"
    report_path.write_text(json.dumps(report))
    gate["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    gate_path.write_text(json.dumps(gate))
    csv_path, rst_path = tmp_path / "results.csv", tmp_path / "results.rst"
    with pytest.raises(ValueError, match="fully passed"):
        RENDERER.render(report_path, gate_path, csv_path, rst_path)
    assert not csv_path.exists() and not rst_path.exists()


@pytest.mark.parametrize("missing", [False, True])
def test_prompt_gate_must_bind_the_exact_report_bytes(tmp_path, missing):
    report, gate = fixture_report()
    report_path, gate_path = tmp_path / "report.json", tmp_path / "gate.json"
    report_path.write_text(json.dumps(report))
    if not missing:
        gate["report_sha256"] = "0" * 64
    gate_path.write_text(json.dumps(gate))
    csv_path, rst_path = tmp_path / "results.csv", tmp_path / "results.rst"
    with pytest.raises(ValueError, match="report SHA-256 mismatch"):
        RENDERER.render(report_path, gate_path, csv_path, rst_path)
    assert not csv_path.exists() and not rst_path.exists()

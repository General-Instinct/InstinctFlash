"""Reject invalid or unrelated action/capture evidence for the prompt repair."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("prompt_update_gate",
    Path(__file__).resolve().parents[1] / "eval/user_e2e_2026-09-14/prompt_update_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def pair():
    old = dict(family="vla2", precision="fp8", device="Thor", torch="2.8", model_id="model",
        revision="revision", cases=[dict(i=i) for i in range(25)], runtime_kwargs={"precision": "fp8"},
        optimizer_environment={}, effective_schedule={"nfe": 10}, guidance="none", ok=True)
    new = deepcopy(old)
    new.update(sources={"/env/flash_rt/frontend.py": "a"*64},
        graph_stats=dict(captured=True, captures=1, reuses=24, calibrations=25))
    actions = np.arange(25*50*14, dtype=np.float32).reshape(25, 50, 14)
    return old, new, actions, actions.copy()


def check(values):
    return gate.check_update(*values, source_suffix="/flash_rt/frontend.py", source_sha256="a"*64,
                             counters=dict(captures=1, reuses=24, calibrations=25))


def test_exact_full_horizon_with_observation_recalibration_passes():
    result = check(pair())
    assert result["status"] == "passed" and result["exact_action_bytes"]
    assert result["task_quality_validated"] is False


@pytest.mark.parametrize("mutation", ["last_action", "dtype", "shape", "nonfinite", "signed_zero"])
def test_full_public_actions_are_required(mutation):
    values = list(pair())
    if mutation == "last_action":
        values[3][-1, -1, -1] += 1
    elif mutation == "dtype":
        values[3] = values[3].astype(np.float64)
    elif mutation == "shape":
        values[3] = values[3][:, :1]
    elif mutation == "nonfinite":
        values[2][-1, -1, -1] = values[3][-1, -1, -1] = np.nan
    else:
        values[3][0, 0, 0] = -0.0
    assert check(values)["status"] == "failed"


@pytest.mark.parametrize("field", ["cases", "model_id", "revision", "device", "precision",
                                    "effective_schedule", "runtime_kwargs"])
def test_other_inputs_policies_and_checkpoints_cannot_certify_a_repair(field):
    values = list(pair())
    values[1][field] = "different"
    assert check(values)["status"] == "failed"


@pytest.mark.parametrize("counter,value", [("calibrations", 1), ("captures", 25), ("reuses", 0),
                                             ("captured", False), ("captures", True)])
def test_redundant_capture_or_removed_calibration_fails(counter, value):
    values = list(pair())
    values[1]["graph_stats"][counter] = value
    assert check(values)["status"] == "failed"


def test_declared_but_not_loaded_frontend_does_not_pass():
    values = list(pair())
    values[1]["sources"]["/env/flash_rt/frontend.py"] = "b"*64
    assert check(values)["status"] == "failed"


@pytest.mark.parametrize("missing_binding", ["report", "provenance"])
def test_audits_must_bind_the_current_matrix(tmp_path, missing_binding):
    matrix = tmp_path / "matrix_final.json"
    matrix.write_text('{"cells": []}')
    digest = gate.sha(matrix)
    report = dict(status="passed", matrix={"sha256": "0"*64 if missing_binding == "report" else digest})
    provenance = dict(status="passed", artifacts=[])
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "provenance_report.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="matrix"):
        gate.build_gate(tmp_path)


def receipt_audits():
    identifiers = [f"{family}-{arm}" for family in ("pi05", "vla4", "vla2", "groot", "edge", "nano", "va", "dreamzero")
                   for arm in ("eager_native", "runtime_default", "runtime_selected")]
    identifiers += ["va-2v4a-native", "va-2v4a-fp8", "dreamzero-dynamic-native", "dreamzero-dynamic-fp8",
                    "vla4-runtime_update", "vla2-runtime_update", "vla2-runtime_numeric"]
    report = {"status": "passed", "cells": [{"id": identifier, "status": "passed",
               "receipt": {"sha256": hashlib.sha256(identifier.encode()).hexdigest()}} for identifier in identifiers]}
    return report, deepcopy(report), set(identifiers)


def test_complete_report_and_provenance_bind_identical_receipt_bytes():
    report, provenance, identifiers = receipt_audits()
    before = deepcopy((report, provenance))
    rows = gate.bind_report_receipts(report, provenance, identifiers)
    assert set(rows) == identifiers and len(rows) == 31
    assert (report, provenance) == before


@pytest.mark.parametrize("audit_name", ["report", "provenance"])
@pytest.mark.parametrize("problem", ["missing", "duplicate", "unknown", "unpassed", "missing_hash", "invalid_hash"])
def test_receipt_join_requires_complete_unique_passed_hash_bindings(audit_name, problem):
    report, provenance, identifiers = receipt_audits()
    audit = report if audit_name == "report" else provenance
    if problem == "missing":
        audit["cells"].pop()
    elif problem == "duplicate":
        audit["cells"][-1] = deepcopy(audit["cells"][0])
    elif problem == "unknown":
        audit["cells"][-1]["id"] = "another-unmeasured-cell"
    elif problem == "unpassed":
        audit["cells"][-1]["status"] = "incomplete"
    elif problem == "missing_hash":
        audit["cells"][-1].pop("receipt")
    else:
        audit["cells"][-1]["receipt"]["sha256"] = "A" * 64
    with pytest.raises(ValueError, match=audit_name):
        gate.bind_report_receipts(report, provenance, identifiers)


@pytest.mark.parametrize("changed_cell", ["vla4-runtime_update", "vla2-runtime_default"])
def test_same_matrix_cannot_pair_new_report_receipts_with_stale_provenance(tmp_path, changed_cell):
    report, provenance, identifiers = receipt_audits()
    next(row for row in provenance["cells"] if row["id"] == changed_cell)["receipt"]["sha256"] = "0" * 64
    matrix = {"cells": [{"id": identifier, "receipt": f"cells/{identifier}/receipt.json"}
                         for identifier in sorted(identifiers)]}
    matrix_path = tmp_path / "matrix_final.json"
    matrix_path.write_text(json.dumps(matrix))
    report["matrix"] = {"sha256": gate.sha(matrix_path)}
    provenance["artifacts"] = [{"sha256": gate.sha(matrix_path)}]
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "provenance_report.json").write_text(json.dumps(provenance))
    # Both audit statuses and matrix hashes pass; the cross-audit join must fail
    # before attempting to load any model receipt, action archive or source.
    with pytest.raises(ValueError, match=f"report/provenance receipt SHA-256 mismatch: {changed_cell}"):
        gate.build_gate(tmp_path)

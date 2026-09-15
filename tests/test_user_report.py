"""CPU artifact/protocol checks, using synthetic receipts rather than models."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regression.user_report import InvalidReceipt, build_report, main, write_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def request(index, *, history=False, queue=False):
    return {
        "i": index, "episode": 0 if queue else index // 3 if history else index,
        "cycle": index if queue else index % 3 if history else 0,
        "phase": "queue_drain" if queue else "warmup" if index < (3 if history else 5) else "measured",
        "seed": 1000 + index,
        "request_sha256": hashlib.sha256(f"recorded-input-{index}".encode()).hexdigest(),
        "feedback_sha256": hashlib.sha256(f"recorded-feedback-{index}".encode()).hexdigest() if history else None,
        "call_kind": "continuous_select_action" if queue else "predict_commit" if history else "reset_predict",
    }


def make_cell(root, arm, *, family="vla4", identifier=None, scale=1.0,
              group=None, schedule=None, shape=None):
    identifier = identifier or f"{family}-{arm}"
    history = family in ("va", "dreamzero")
    count = 21 if history else 25
    shape = shape or ([16, 2, 16] if family == "va" else [7] if family == "pi05" else [2, 3])
    schedule = schedule or ({"video": 25, "action": 50} if family == "va" else {"action": 10})
    cases = [request(i, history=history) for i in range(count)]
    milliseconds = [10000 * scale if case["phase"] == "warmup" else (10 + i) * scale
                    for i, case in enumerate(cases)]
    actions = np.arange(count * np.prod(shape), dtype=np.float32).reshape(count, *shape) / 100
    np.savez(root / f"{identifier}.npz", actions=actions)
    kwargs = {"precision": "fp8", "tier": "numeric"} if arm == "runtime_selected" else {}
    receipt = {
        "schema": 1, "ok": True, "cell_id": identifier, "family": family, "arm": arm,
        "model_id": f"publisher/{family}", "revision": "a" * 40,
        "precision": "fp8" if arm == "runtime_selected" else "native",
        "device": "NVIDIA Thor", "torch": "2.11.0+cu130",
        "runtime_kwargs": kwargs, "optimizer_environment": {},
        "default_schedule": schedule, "effective_schedule": schedule, "guidance": {"video": 5},
        "execution_policy": {"tier": "numeric" if arm == "runtime_selected" else "bitexact"},
        "setup_seconds": 1.25, "setup_scope": "construction and initial reset/lazy model loading",
        "timing_scope": "synchronized public predict with prepared inputs, including commit",
        "input_contract": {"recorded_images": True, "synthetic_state": True},
        "cases": cases, "calls": [{"i": i, "ms": ms, "shape": shape} for i, ms in enumerate(milliseconds)],
        "actions_sha256": digest(root / f"{identifier}.npz"),
    }
    if family == "pi05":
        np.savez(root / f"{identifier}.queue.npz", actions=np.zeros((51, *shape), dtype=np.float32))
        receipt["queue_drain"] = {
            "archive": f"{identifier}.queue.npz", "actions_sha256": digest(root / f"{identifier}.queue.npz"),
            "reset_between_calls": False, "expected_calls": 51,
            "cases": [request(i, queue=True) for i in range(51)],
            "calls": [{"i": i, "ms": (100 if i in (0, 50) else .1) * scale, "shape": shape} for i in range(51)],
        }
    save(root / f"{identifier}.json", receipt)
    return {"id": identifier, "family": family, "arm": arm, "receipt": f"{identifier}.json",
            "model_id": receipt["model_id"], "revision": receipt["revision"], "action_shape": shape,
            "comparison_group": group or family, "experimental": arm == "operating_point",
            "expected_runtime_kwargs": kwargs, "expected_optimizer_environment": {}}, receipt


def make_matrix(root, family="vla4"):
    cells = [make_cell(root, arm, family=family, scale=scale)[0]
             for arm, scale in (("eager_native", 1), ("runtime_default", .8), ("runtime_selected", .5))]
    matrix = {"schema": 1, "expected_main_cells": 3, "expected_families": [family], "cells": cells}
    save(root / "matrix.json", matrix)
    return root / "matrix.json", matrix


def edit_receipt(root, cell, transform):
    path = root / cell["receipt"]
    receipt = json.loads(path.read_text())
    transform(receipt)
    save(path, receipt)


def test_recomputes_raw_generation_latency_and_writes_csv_without_admission(tmp_path):
    path, matrix = make_matrix(tmp_path)
    # A stale producer summary has no effect on independent statistics.
    edit_receipt(tmp_path, matrix["cells"][0], lambda r: r.update(p50_ms=.001, task_quality_validated=True))
    report = build_report(path)
    assert report["status"] == "passed"
    assert report["counts"] == {
        "expected_cells": 3, "main_cells": 3, "operating_point_cells": 0,
        "runtime_update_cells": 0,
        "passed_cells": 3, "failed_cells": 0, "incomplete_cells": 0, "failed_comparisons": 0,
        "finite_public_actions": 75, "finite_queue_actions": 0,
    }
    native = report["cells"][0]
    assert native["primary"] == {"count": 20, "p50_ms": 24.5, "mean_ms": 24.5, "min_ms": 15., "max_ms": 34.}
    assert native["first_predict_ms"] == 10000
    assert native["setup_plus_first_predict_ms"] == 11250
    assert report["comparisons"][1]["ratio_of_p50"] == 2
    assert report["comparisons"][1]["median_paired_ratio"] == 2
    assert report["comparisons"][1]["actions"]["exact_bytes"]
    assert all(not row["task_quality_validated"] and not row["recommended"] for row in report["cells"])
    assert report["task_quality_validated"] is False
    assert report["matrix"]["sha256"] == digest(path)
    csv_path = tmp_path / "report.csv"
    write_csv(report, csv_path)
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert rows[2]["matched_ratio_of_p50"] == "2.0"
    assert json.loads(rows[2]["runtime_kwargs"]) == {"precision": "fp8", "tier": "numeric"}
    assert rows[2]["task_quality_validated"] == "False"


def test_history_continuation_and_full_multi_axis_actions_are_preserved(tmp_path):
    path, _ = make_matrix(tmp_path, "va")
    report = build_report(path)
    assert report["status"] == "passed"
    row = report["cells"][0]
    assert row["total_calls"] == 21 and row["warmup_calls"] == 3
    assert row["measured"]["count"] == 18
    assert row["primary_metric"] == "continuation_p50_ms"
    assert row["primary"]["count"] == 12
    assert row["primary"]["p50_ms"] == 22
    assert row["new_episode"]["count"] == 6
    assert row["actions"]["shape"] == [21, 16, 2, 16]
    assert report["comparisons"][0]["paired_samples"] == 12


@pytest.mark.parametrize("problem,fragment", [
    ("nan_rehashed", "nonfinite"), ("bad_hash", "SHA-256 mismatch"),
    ("truncated_calls", "expected 25"), ("wrong_phase", "incorrect warmup"),
    ("wrong_call_index", "sequential"), ("shape_flattened", "full public action archive shape"),
    ("wrong_call_shape", "output shape mismatch"), ("zero_ms", "finite and positive"),
    ("missing_request_hash", "request_sha256/input_sha256"),
    ("missing_effective_schedule", "missing effective_schedule"),
    ("missing_timing_scope", "missing timing_scope/scope"),
    ("missing_input_contract", "missing input_contract"),
    ("default_env", "optimizer environment overrides"),
    ("default_options", "changes public functional options"),
])
def test_invalid_artifacts_or_claimed_protocol_fail(tmp_path, problem, fragment):
    path, matrix = make_matrix(tmp_path)
    cell = matrix["cells"][1] if problem.startswith("default_") else matrix["cells"][0]
    receipt_path = tmp_path / cell["receipt"]
    receipt = json.loads(receipt_path.read_text())
    archive = receipt_path.with_suffix(".npz")
    if problem in ("nan_rehashed", "shape_flattened"):
        with np.load(archive) as loaded:
            array = loaded["actions"].copy()
        if problem == "nan_rehashed":
            array[-1, -1, -1] = np.nan
        else:
            array = array.reshape(25, 6)
        np.savez(archive, actions=array)
        receipt["actions_sha256"] = digest(archive)
    elif problem == "bad_hash":
        receipt["actions_sha256"] = "0" * 64
    elif problem == "truncated_calls":
        receipt["calls"].pop()
    elif problem == "wrong_phase":
        receipt["cases"][5]["phase"] = "warmup"
    elif problem == "wrong_call_index":
        receipt["calls"][2]["i"] = 3
    elif problem == "wrong_call_shape":
        receipt["calls"][4]["shape"] = [6]
    elif problem == "zero_ms":
        receipt["calls"][4]["ms"] = 0
    elif problem == "missing_request_hash":
        receipt["cases"][3].pop("request_sha256")
    elif problem == "missing_effective_schedule":
        receipt.pop("effective_schedule")
    elif problem == "missing_timing_scope":
        receipt.pop("timing_scope")
    elif problem == "missing_input_contract":
        receipt.pop("input_contract")
    elif problem == "default_env":
        receipt["optimizer_environment"] = {"IFL_COSMOS3_GEN_REGIONS": "1"}
        cell["expected_optimizer_environment"] = receipt["optimizer_environment"]
    elif problem == "default_options":
        receipt["runtime_kwargs"] = {"nfe": 1}
        cell["expected_runtime_kwargs"] = receipt["runtime_kwargs"]
    save(receipt_path, receipt)
    save(path, matrix)
    report = build_report(path)
    assert report["status"] == "failed"
    failed = [row for row in report["cells"] if row["status"] == "failed"]
    assert len(failed) == 1 and fragment in failed[0]["errors"][0]


@pytest.mark.parametrize("field", ["request_sha256", "feedback_sha256", "seed", "call_kind"])
def test_request_mismatch_refuses_ratio_even_when_actions_match(tmp_path, field):
    path, matrix = make_matrix(tmp_path, "va")
    def change(receipt):
        receipt["cases"][3][field] = 99 if field == "seed" else "another_call" if field == "call_kind" else "f" * 64
    edit_receipt(tmp_path, matrix["cells"][2], change)
    report = build_report(path)
    assert report["status"] == "failed"
    comparison = report["comparisons"][1]
    assert comparison["status"] == "failed"
    assert "requests" in comparison["errors"][0]
    assert "ratio_of_p50" not in comparison


@pytest.mark.parametrize("field,value", [
    ("effective_schedule", {"video_action": 16, "fixed_mask": [0, 1, 6, 11]}),
    ("guidance", {"video": 1}),
    ("device", "other GPU"),
    ("torch", "other version"),
    ("timing_scope", "forward only, excludes decode"),
])
def test_declared_same_group_requires_policy_and_measurement_equivalence(tmp_path, field, value):
    path, matrix = make_matrix(tmp_path, "dreamzero")
    edit_receipt(tmp_path, matrix["cells"][2], lambda receipt: receipt.update({field: value}))
    report = build_report(path)
    assert report["status"] == "failed"
    assert report["comparisons"][1]["status"] == "failed"
    assert "ratio_of_p50" not in report["comparisons"][1]


def test_operating_points_have_their_own_reference_and_explicit_cross_policy_ratio(tmp_path):
    path, matrix = make_matrix(tmp_path, "va")
    for name, scale in (("va-short-native", .1), ("va-short-fp8", .05)):
        cell, receipt = make_cell(tmp_path, "operating_point", family="va", identifier=name,
                                  group="va-2v4a", scale=scale, schedule={"video": 2, "action": 4})
        receipt["default_schedule"] = {"video": 25, "action": 50}
        cell["cross_policy_baseline"] = "va-eager_native"
        if name.endswith("native"):
            cell["comparison_baseline"] = True
        else:
            receipt["precision"] = "fp8"
            receipt["runtime_kwargs"] = {"precision": "fp8", "nfe": {"video": 2, "action": 4}}
            cell["expected_runtime_kwargs"] = receipt["runtime_kwargs"]
        save(tmp_path / cell["receipt"], receipt)
        matrix["cells"].append(cell)
    save(path, matrix)
    report = build_report(path)
    assert report["status"] == "passed"
    assert report["counts"]["operating_point_cells"] == 2
    comparisons = report["comparisons"]
    assert len(comparisons) == 5
    within = next(c for c in comparisons if c["baseline"] == "va-short-native")
    assert within["same_sampling_policy"] and within["ratio_of_p50"] == 2
    cross = [c for c in comparisons if c["kind"] == "changed_policy_latency_ratio"]
    assert len(cross) == 2
    assert cross[0]["ratio_of_p50"] == pytest.approx(10)
    assert not cross[0]["same_sampling_policy"]
    assert "not an architecture-only speedup" in cross[0]["interpretation"]
    assert not cross[0]["task_quality_validated"]


def test_pi05_queue_drain_is_separate_full_51_public_calls(tmp_path):
    path, _ = make_matrix(tmp_path, "pi05")
    report = build_report(path)
    assert report["status"] == "passed"
    assert report["counts"]["finite_public_actions"] == 75
    assert report["counts"]["finite_queue_actions"] == 153
    row = report["cells"][0]
    assert row["primary"]["count"] == 20 and row["primary"]["p50_ms"] == 24.5
    assert row["actions"]["shape"] == [25, 7]
    assert row["queue_drain"]["actions"]["shape"] == [51, 7]
    assert row["queue_drain"]["middle_49_calls"]["p50_ms"] == .1
    assert row["queue_drain"]["total_ms"] == pytest.approx(204.9)
    assert report["comparisons"][1]["queue_drain_total_ratio"] == 2


@pytest.mark.parametrize("problem", ["reset", "missing", "truncated", "nonfinite", "different_requests"])
def test_pi05_queue_drain_must_be_complete_finite_and_matched(tmp_path, problem):
    path, matrix = make_matrix(tmp_path, "pi05")
    cell = matrix["cells"][2]
    def change(receipt):
        queue = receipt["queue_drain"]
        if problem == "reset":
            queue["reset_between_calls"] = True
        elif problem == "missing":
            receipt.pop("queue_drain")
        elif problem == "truncated":
            queue["calls"].pop()
        elif problem == "nonfinite":
            archive = tmp_path / queue["archive"]
            actions = np.zeros((51, 7), dtype=np.float32)
            actions[50, -1] = np.inf
            np.savez(archive, actions=actions)
            queue["actions_sha256"] = digest(archive)
        else:
            queue["cases"][50]["request_sha256"] = "c" * 64
    edit_receipt(tmp_path, cell, change)
    report = build_report(path)
    assert report["status"] == "failed"
    assert report["counts"]["failed_cells"] + report["counts"]["failed_comparisons"] == 1
    if problem == "different_requests":
        assert "ratio_of_p50" not in report["comparisons"][1]


def test_missing_cells_or_archives_are_incomplete_not_success(tmp_path):
    path, matrix = make_matrix(tmp_path)
    (tmp_path / matrix["cells"][0]["receipt"]).unlink()
    (tmp_path / matrix["cells"][1]["receipt"]).with_suffix(".npz").unlink()
    report = build_report(path)
    assert report["status"] == "incomplete"
    assert report["counts"]["passed_cells"] == 1
    assert report["counts"]["incomplete_cells"] == 2
    assert not report["comparisons"]
    assert main(["--matrix", str(path), "--json-output", str(tmp_path / "report.json"),
                 "--csv-output", str(tmp_path / "report.csv")]) == 2
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "incomplete"


def test_manifest_coverage_does_not_silently_drop_a_model(tmp_path):
    path, matrix = make_matrix(tmp_path)
    matrix["expected_main_cells"] = 24
    matrix["expected_families"].append("dreamzero")
    save(path, matrix)
    report = build_report(path)
    assert report["status"] == "failed"
    assert any("expected 24" in error for error in report["errors"])
    assert any("dreamzero" in error for error in report["errors"])


def test_frozen_protocol_binding_changes_fail_and_missing_is_pending(tmp_path):
    path, matrix = make_matrix(tmp_path)
    protocol = tmp_path / "protocol.json"
    save(protocol, {"scope": "prepared public predict"})
    matrix["bindings"] = [{"path": "protocol.json", "sha256": digest(protocol)}]
    save(path, matrix)
    assert build_report(path)["status"] == "passed"
    save(protocol, {"scope": "kernel microbenchmark"})
    assert build_report(path)["status"] == "failed"
    protocol.unlink()
    assert build_report(path)["status"] == "incomplete"


def test_accepts_model_and_input_hash_alias_but_rejects_conflicts(tmp_path):
    path, matrix = make_matrix(tmp_path)
    def change(receipt):
        receipt["model"] = receipt.pop("model_id")
        for case in receipt["cases"]:
            case["input_sha256"] = case.pop("request_sha256")
    edit_receipt(tmp_path, matrix["cells"][0], change)
    assert build_report(path)["status"] == "passed"
    edit_receipt(tmp_path, matrix["cells"][0], lambda r: r.update(model_id="conflicting-model"))
    assert build_report(path)["status"] == "failed"


def test_descriptive_action_difference_is_not_an_automatic_quality_verdict(tmp_path):
    path, matrix = make_matrix(tmp_path)
    cell = matrix["cells"][2]
    archive = (tmp_path / cell["receipt"]).with_suffix(".npz")
    with np.load(archive) as loaded:
        actions = loaded["actions"].copy()
    actions[-1, -1, -1] += 100
    np.savez(archive, actions=actions)
    edit_receipt(tmp_path, cell, lambda r: r.update(actions_sha256=digest(archive)))
    report = build_report(path)
    assert report["status"] == "passed"
    comparison = report["comparisons"][1]
    assert comparison["actions"]["max_abs"] == pytest.approx(100)
    assert not comparison["actions"]["exact_values"]
    assert not comparison["task_quality_validated"] and not comparison["recommended"]


def test_duplicate_json_keys_and_nonfinite_json_are_rejected(tmp_path):
    path, matrix = make_matrix(tmp_path)
    receipt_path = tmp_path / matrix["cells"][0]["receipt"]
    receipt_path.write_text('{"schema": 1, "schema": 1}')
    report = build_report(path)
    assert report["status"] == "failed"
    assert "duplicate JSON key" in report["cells"][0]["errors"][0]
    receipt_path.write_text('{"value": 1e999}')
    assert build_report(path)["status"] == "failed"
    matrix["cells"].append(dict(matrix["cells"][0]))
    save(path, matrix)
    with pytest.raises(InvalidReceipt, match="duplicate cell ids"):
        build_report(path)


def test_finite_float64_metric_overflow_is_reported_without_invalid_json(tmp_path):
    path, matrix = make_matrix(tmp_path)
    for cell, sign in zip(matrix["cells"], (1, 1, -1)):
        archive = (tmp_path / cell["receipt"]).with_suffix(".npz")
        np.savez(archive, actions=np.full((25, 2, 3), sign * np.finfo(np.float64).max))
        edit_receipt(tmp_path, cell, lambda r: r.update(actions_sha256=digest(archive)))
    report = build_report(path)
    assert report["status"] == "passed"
    assert report["comparisons"][1]["actions"]["metric_overflow"]
    assert report["comparisons"][1]["actions"]["max_abs"] is None
    json.dumps(report, allow_nan=False)


def test_corrupt_rehashed_archive_fails_without_crashing_the_report(tmp_path):
    path, matrix = make_matrix(tmp_path)
    cell = matrix["cells"][0]
    archive = (tmp_path / cell["receipt"]).with_suffix(".npz")
    archive.write_bytes(b"PK\x03\x04truncated zip")
    edit_receipt(tmp_path, cell, lambda r: r.update(actions_sha256=digest(archive)))
    report = build_report(path)
    assert report["status"] == "failed"
    assert report["counts"]["failed_cells"] == 1


def test_native_precision_with_numeric_options_remains_an_explicit_candidate(tmp_path):
    path, matrix = make_matrix(tmp_path, "edge")
    cell = matrix["cells"][2]
    cell["expected_runtime_kwargs"] = {"tier_ceiling": "numeric"}
    cell["expected_optimizer_environment"] = {"IFL_COSMOS3_GEN_REGIONS": "1"}
    edit_receipt(tmp_path, cell, lambda r: r.update(precision="native", runtime_kwargs=cell["expected_runtime_kwargs"],
                                                 optimizer_environment=cell["expected_optimizer_environment"]))
    save(path, matrix)
    report = build_report(path)
    assert report["status"] == "passed"
    assert report["cells"][2]["usability"] == "explicit numerical candidate on recorded requests"
    assert not report["cells"][2]["recommended"]


def test_eager_reference_cannot_declare_applied_runtime_passes(tmp_path):
    path, matrix = make_matrix(tmp_path)
    edit_receipt(tmp_path, matrix["cells"][0], lambda r: r.update(applied_passes=["graph_capture"]))
    assert build_report(path)["status"] == "failed"


def test_capture_request_hash_binds_values_shape_dtype_and_feedback():
    from benchmarks.regression.user_e2e import request_hash
    arrays = np.arange(12, dtype=np.float32).reshape(2, 6)
    base = {"observation": arrays, "prompt": "move"}
    value = request_hash(base)
    assert request_hash({"prompt": "move", "observation": arrays.copy()}) == value
    assert request_hash({"observation": arrays.astype(np.float64), "prompt": "move"}) != value
    assert request_hash({"observation": arrays.reshape(3, 4), "prompt": "move"}) != value
    changed = arrays.copy()
    changed[-1, -1] += 1
    assert request_hash({"observation": changed, "prompt": "move"}) != value
    assert request_hash({"observation": arrays, "prompt": "stop"}) != value
    assert request_hash({"executed_action": arrays}) != request_hash({"executed_action": changed})


def test_capture_prepared_inputs_preserve_each_public_family_contract():
    from benchmarks.regression.user_e2e import RecordedInputs
    fixture = RecordedInputs.__new__(RecordedInputs)
    fixture.frames = [[np.full((24, 32, 3), frame + camera, dtype=np.uint8) for camera in range(3)]
                      for frame in range(13)]
    fixture.feedback = np.zeros((3, 16, 2, 16), dtype=np.float32)
    pi05 = fixture.observation("pi05", 1, 0)
    assert pi05["observation.images.image"].shape == (3, 24, 32)
    assert pi05["observation.images.image"].dtype == np.float32
    assert 0 <= pi05["observation.images.image"].min() <= pi05["observation.images.image"].max() <= 1
    assert pi05["observation.state"].shape == (8,)
    for family in ("vla4", "vla2"):
        observation = fixture.observation(family, 1, 0)
        assert observation["observation.state"].shape == (14,)
        assert sum(key.startswith("observation.images.") for key in observation) == 3
        assert observation["observation.images.cam_high"].dtype == np.uint8
    groot = fixture.observation("groot", 1, 0)
    assert len(groot["images"]) == 2
    assert sum(value.size for value in groot["state"].values()) == 17
    assert np.array_equal(groot["state"]["eef_9d"][3:], [1, 0, 0, 0, 1, 0])
    for family in ("edge", "nano"):
        observation = fixture.observation(family, 1, 0)
        assert observation["image"].shape == (540, 640, 3)
        assert observation["image"].dtype == np.uint8
        assert observation["state"].shape == (8,)
    assert [len(fixture.observation("va", i, i)["obs"]) for i in range(3)] == [1, 4, 8]
    assert [fixture.observation("dreamzero", i, i)["observation/wrist_image_left"].shape[0]
            for i in range(3)] == [1, 4, 4]


def add_observed_schedule(root, matrix, *, family="vla4"):
    matrix["require_observed_schedule"] = True
    mask = [i in (0, 1, 2, 6, 10, 13, 14, 15) for i in range(16)]
    for cell in matrix["cells"]:
        def change(receipt):
            nfe = {"video_action": 16} if family == "dreamzero" else {"prefix": 1, "action": 10}
            receipt["default_schedule"] = nfe
            receipt["effective_schedule"] = {"nfe": nfe}
            observed = {"video_action": 16} if family == "dreamzero" else {"action": 10}
            receipt["observed_nfe_before"] = dict(observed)
            receipt["observed_nfe_after"] = dict(observed)
            if family == "dreamzero":
                receipt["effective_schedule"].update(step_cache="checkpoint", checkpoint_dit_step_mask=mask)
                observed = {"steps": {"video_action": 16, "kv_commit": 1},
                            "guidance": {"video_action": ["cfg", 5.0]},
                            "dit_step_mask": mask, "dynamic_cache_schedule": False}
                receipt["observed_schedule_before"] = observed
                receipt["observed_schedule_after"] = dict(observed)
            cell["effective_schedule"] = receipt["effective_schedule"]
        edit_receipt(root, cell, change)


@pytest.mark.parametrize("problem", [None, "missing", "changed_after", "wrong_both", "inactive_only"])
def test_requires_loaded_nfe_evidence_beyond_copied_policy_label(tmp_path, problem):
    path, matrix = make_matrix(tmp_path)
    add_observed_schedule(tmp_path, matrix)
    save(path, matrix)
    if problem:
        def change(receipt):
            if problem == "missing":
                receipt.pop("observed_nfe_before")
                receipt.pop("observed_nfe_after")
            elif problem == "changed_after":
                receipt["observed_nfe_after"] = {"action": 1}
            else:
                value = {"action": 1} if problem == "wrong_both" else {"prefix": 1}
                receipt["observed_nfe_before"] = value
                receipt["observed_nfe_after"] = value
        edit_receipt(tmp_path, matrix["cells"][2], change)
    report = build_report(path)
    assert report["status"] == ("failed" if problem else "passed")
    if problem:
        assert report["counts"]["failed_cells"] == 1
    else:
        assert report["cells"][0]["observed_schedule"]["nfe_before"] == {"action": 10}


@pytest.mark.parametrize("problem", [None, "missing", "wrong_mask", "wrong_cfg", "wrong_dynamic", "wrong_kv", "changed_after"])
def test_dreamzero_observed_schedule_binds_grid_mask_cfg_dynamic_and_kv(tmp_path, problem):
    path, matrix = make_matrix(tmp_path, "dreamzero")
    add_observed_schedule(tmp_path, matrix, family="dreamzero")
    save(path, matrix)
    if problem:
        def change(receipt):
            observed = receipt["observed_schedule_before"]
            if problem == "missing":
                receipt.pop("observed_schedule_before")
                return
            if problem == "wrong_mask":
                observed["dit_step_mask"][15] = False
            elif problem == "wrong_cfg":
                observed["guidance"]["video_action"][1] = 1.
            elif problem == "wrong_dynamic":
                observed["dynamic_cache_schedule"] = True
            elif problem == "wrong_kv":
                observed["steps"]["kv_commit"] = 0
            else:
                receipt["observed_schedule_after"]["dynamic_cache_schedule"] = True
                return
            receipt["observed_schedule_after"] = observed
        edit_receipt(tmp_path, matrix["cells"][2], change)
    report = build_report(path)
    assert report["status"] == ("failed" if problem else "passed")
    if problem:
        assert report["counts"]["failed_cells"] == 1


def add_runtime_update(root, matrix, *, family="vla4", baseline=None):
    cell, receipt = make_cell(root, "runtime_selected", family=family,
                              identifier=f"{family}-runtime_update", scale=.25)
    cell.update(arm="runtime_update", baseline_cell=baseline or f"{family}-runtime_selected")
    receipt["arm"] = "runtime_update"
    save(root / cell["receipt"], receipt)
    matrix["cells"].append(cell)
    return cell


@pytest.mark.parametrize("baseline,ratio", [("vla4-runtime_selected", 2), ("vla4-eager_native", 4)])
def test_runtime_update_uses_explicit_baseline_without_rewriting_main_comparisons(tmp_path, baseline, ratio):
    path, matrix = make_matrix(tmp_path)
    original = build_report(path)
    update = add_runtime_update(tmp_path, matrix, baseline=baseline)
    edit_receipt(tmp_path, update, lambda receipt: receipt.update(recommended=True, task_quality_validated=True))
    save(path, matrix)
    report = build_report(path)
    assert report["status"] == "passed"
    assert report["counts"]["main_cells"] == 3
    assert report["counts"]["operating_point_cells"] == 0
    assert report["counts"]["runtime_update_cells"] == 1
    assert report["comparisons"][:2] == original["comparisons"]
    comparison = report["comparisons"][2]
    assert comparison["baseline"] == baseline
    assert comparison["candidate"] == update["id"]
    assert comparison["kind"] == "matched_policy_latency_ratio"
    assert comparison["ratio_of_p50"] == ratio and comparison["paired_samples"] == 20
    assert comparison["same_sampling_policy"]
    row = report["cells"][3]
    assert row["total_calls"] == 25 and row["warmup_calls"] == 5
    assert row["experimental"] is False
    assert row["usability"] == "runtime source update candidate on recorded requests"
    assert not row["recommended"] and not row["task_quality_validated"]
    assert not comparison["recommended"] and not comparison["task_quality_validated"]
    csv_path = tmp_path / "updated.csv"
    write_csv(report, csv_path)
    with csv_path.open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert csv_rows[3]["arm"] == "runtime_update"
    assert csv_rows[3]["matched_baseline"] == baseline
    assert csv_rows[3]["cross_policy_baseline"] == ""


@pytest.mark.parametrize("problem", ["missing", "unknown", "self", "default_arm", "other_group",
                                     "other_family", "experimental", "implicit_experimental",
                                     "group_baseline", "cross_policy"])
def test_runtime_update_requires_an_explicit_nonexperimental_comparable_main_baseline(tmp_path, problem):
    path, matrix = make_matrix(tmp_path)
    cell = add_runtime_update(tmp_path, matrix)
    if problem == "missing":
        cell.pop("baseline_cell")
    elif problem == "unknown":
        cell["baseline_cell"] = "not-in-matrix"
    elif problem == "self":
        cell["baseline_cell"] = cell["id"]
    elif problem == "default_arm":
        cell["baseline_cell"] = "vla4-runtime_default"
    elif problem == "other_group":
        cell["comparison_group"] = "another-group"
    elif problem == "other_family":
        cell["family"] = "vla2"
        edit_receipt(tmp_path, cell, lambda receipt: receipt.update(family="vla2"))
    elif problem == "experimental":
        cell["experimental"] = True
    elif problem == "implicit_experimental":
        cell.pop("experimental")
    elif problem == "group_baseline":
        cell["comparison_baseline"] = True
    else:
        cell["cross_policy_baseline"] = "vla4-eager_native"
    save(path, matrix)
    report = build_report(path)
    assert report["status"] == "failed"
    assert report["errors"]
    assert len(report["comparisons"]) == 2


@pytest.mark.parametrize("baseline_arm", ["operating_point", "runtime_update"])
def test_runtime_update_rejects_operating_point_and_update_chain_baselines(tmp_path, baseline_arm):
    path, matrix = make_matrix(tmp_path)
    cell = add_runtime_update(tmp_path, matrix, baseline="another-candidate")
    baseline, receipt = make_cell(tmp_path, "runtime_selected", identifier="another-candidate")
    baseline["arm"] = receipt["arm"] = baseline_arm
    if baseline_arm == "operating_point":
        baseline.update(comparison_group="extra", comparison_baseline=True, experimental=True)
    else:
        baseline["baseline_cell"] = "vla4-runtime_selected"
    save(tmp_path / baseline["receipt"], receipt)
    matrix["cells"].append(baseline)
    save(path, matrix)
    report = build_report(path)
    assert report["status"] == "failed"
    assert f"{cell['id']}: runtime_update baseline must be eager_native or runtime_selected" in report["errors"]
    assert not any(comparison["candidate"] == cell["id"] for comparison in report["comparisons"])


@pytest.mark.parametrize("valid_structure", [False, True])
def test_runtime_update_missing_receipt_does_not_hide_invalid_baseline(tmp_path, valid_structure):
    path, matrix = make_matrix(tmp_path)
    cell = add_runtime_update(tmp_path, matrix)
    (tmp_path / cell["receipt"]).unlink()
    if not valid_structure:
        cell.pop("baseline_cell")
    save(path, matrix)
    report = build_report(path)
    assert report["status"] == ("incomplete" if valid_structure else "failed")
    assert report["counts"]["incomplete_cells"] == 1
    assert bool(report["errors"]) is not valid_structure


@pytest.mark.parametrize("problem", ["request", "dtype", "integer_dtype", "hash", "schedule", "guidance",
                                     "default_schedule", "precision", "kwargs", "environment", "truncated"])
def test_runtime_update_keeps_request_dtype_hash_and_policy_checks_strict(tmp_path, problem):
    path, matrix = make_matrix(tmp_path)
    cell = add_runtime_update(tmp_path, matrix)
    receipt_path = tmp_path / cell["receipt"]
    receipt = json.loads(receipt_path.read_text())
    if problem == "request":
        receipt["cases"][7]["request_sha256"] = "e" * 64
    elif problem in ("dtype", "integer_dtype"):
        archive = receipt_path.with_suffix(".npz")
        with np.load(archive) as loaded:
            values = loaded["actions"].astype(np.float64 if problem == "dtype" else np.int64)
        np.savez(archive, actions=values)
        receipt["actions_sha256"] = digest(archive)
    elif problem == "hash":
        receipt["actions_sha256"] = "0" * 64
    elif problem == "schedule":
        receipt["effective_schedule"] = {"action": 1}
    elif problem == "guidance":
        receipt["guidance"] = {"video": 1}
    elif problem == "default_schedule":
        receipt["default_schedule"] = {"action": 1}
    elif problem == "precision":
        receipt["precision"] = "native"
    elif problem == "kwargs":
        receipt["runtime_kwargs"] = {"precision": "fp8", "tier": "behavioral"}
        cell["expected_runtime_kwargs"] = receipt["runtime_kwargs"]
    elif problem == "environment":
        receipt["optimizer_environment"] = {"IFL_UNREQUESTED_OPTION": "1"}
        cell["expected_optimizer_environment"] = receipt["optimizer_environment"]
    else:
        receipt["calls"].pop()
    save(receipt_path, receipt)
    save(path, matrix)
    report = build_report(path)
    assert report["status"] == "failed"
    assert report["counts"]["failed_cells"] + report["counts"]["failed_comparisons"] == 1
    for comparison in report["comparisons"]:
        if comparison["candidate"] == cell["id"]:
            assert comparison["status"] == "failed" and "ratio_of_p50" not in comparison


def test_runtime_update_preserves_pi05_public_queue_dtype_contract(tmp_path):
    path, matrix = make_matrix(tmp_path, "pi05")
    cell = add_runtime_update(tmp_path, matrix, family="pi05")
    receipt_path = tmp_path / cell["receipt"]
    receipt = json.loads(receipt_path.read_text())
    archive = tmp_path / receipt["queue_drain"]["archive"]
    np.savez(archive, actions=np.zeros((51, 7), dtype=np.float64))
    receipt["queue_drain"]["actions_sha256"] = digest(archive)
    save(receipt_path, receipt)
    save(path, matrix)
    report = build_report(path)
    assert report["status"] == "failed"
    comparison = report["comparisons"][-1]
    assert comparison["errors"] == ["incomparable fields: queue_drain action dtype"]
    assert "ratio_of_p50" not in comparison


def test_runtime_update_preserves_original_24_main_and_four_operating_point_cells(tmp_path):
    families = ("pi05", "vla4", "vla2", "groot", "edge", "nano", "va", "dreamzero")
    cells = [make_cell(tmp_path, arm, family=family)[0]
             for family in families for arm in ("eager_native", "runtime_default", "runtime_selected")]
    for family in ("va", "dreamzero"):
        for index in range(2):
            cell, _ = make_cell(tmp_path, "operating_point", family=family,
                                 identifier=f"{family}-extra-{index}", group=f"{family}-extra")
            if index == 0:
                cell["comparison_baseline"] = True
            else:
                cell["baseline_cell"] = f"{family}-extra-0"
            cells.append(cell)
    matrix = {"schema": 1, "expected_main_cells": 24, "expected_operating_point_cells": 4,
              "expected_families": list(families), "cells": cells}
    path = tmp_path / "matrix.json"
    save(path, matrix)
    initial = build_report(path)
    assert initial["status"] == "passed"
    add_runtime_update(tmp_path, matrix)
    save(path, matrix)
    updated = build_report(path)
    assert updated["status"] == "passed"
    assert updated["counts"]["expected_cells"] == 29
    assert updated["counts"]["main_cells"] == 24
    assert updated["counts"]["operating_point_cells"] == 4
    assert updated["counts"]["runtime_update_cells"] == 1
    assert updated["comparisons"][:-1] == initial["comparisons"]
    # An update may not replace any required original operating-point or main cell.
    matrix["cells"].pop(27)
    save(path, matrix)
    assert build_report(path)["status"] == "failed"
    matrix["cells"].pop(0)
    save(path, matrix)
    assert any("expected 24" in error for error in build_report(path)["errors"])


def test_explicit_runtime_update_count_is_enforced(tmp_path):
    path, matrix = make_matrix(tmp_path)
    add_runtime_update(tmp_path, matrix)
    matrix["expected_runtime_update_cells"] = 2
    save(path, matrix)
    report = build_report(path)
    assert report["status"] == "failed"
    assert report["errors"] == ["matrix declares 1 runtime-update cells, expected 2"]

"""CPU-only validation of public-user benchmark receipts and full action archives.

The matrix fixes the expected cells, public output shapes, requested options and
comparison groups before results are read. This module does not import Runtime
or Torch and never evaluates a model. A passed report establishes artifact and
protocol consistency on the recorded requests, not task quality or admission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any
from zipfile import BadZipFile

import numpy as np


MAIN_ARMS = ("eager_native", "runtime_default", "runtime_selected")
HISTORY_FAMILIES = frozenset(("va", "dreamzero"))
REPORT_SCHEMA = "instinctflash.user_report.v1"


class InvalidReceipt(ValueError):
    """An existing artifact violates the declared measurement protocol."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidReceipt(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json(path: Path) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _check(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise InvalidReceipt(f"nonfinite JSON value: {value}")

    with path.open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=unique, parse_constant=invalid)
    # Overflowing JSON numbers such as 1e999 are not parse_constant tokens.
    _canonical(result)
    return result


def _path(root: Path, value: str) -> Path:
    _check(isinstance(value, str) and bool(value), "artifact path must be nonempty")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    _check(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, positive: bool = False) -> float:
    _check(type(value) in (int, float), f"{name} must be a number")
    _check(math.isfinite(value) and (value > 0 if positive else value >= 0),
           f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _hash(value: Any, name: str) -> str:
    _check(isinstance(value, str) and len(value) == 64
           and all(character in "0123456789abcdef" for character in value),
           f"{name} must be a lowercase SHA-256")
    return value


def _required(receipt: dict, name: str) -> Any:
    _check(name in receipt and receipt[name] is not None, f"receipt missing {name}")
    return receipt[name]


def _shape(value: Any) -> list[int]:
    _check(isinstance(value, list) and bool(value), "action_shape must be a nonempty list")
    for dimension in value:
        _integer(dimension, "action_shape dimension", 1)
    return value


def _request_signature(case: dict) -> dict:
    request_hash = case.get("request_sha256", case.get("input_sha256"))
    _hash(request_hash, "request_sha256/input_sha256")
    if "request_sha256" in case and "input_sha256" in case:
        _check(case["request_sha256"] == case["input_sha256"],
               "ambiguous request_sha256 and input_sha256")
    feedback = case.get("feedback_sha256")
    if feedback is not None:
        _hash(feedback, "feedback_sha256")
    _integer(case.get("seed"), "case seed")
    _integer(case.get("episode"), "case episode")
    _check(isinstance(case.get("call_kind"), str) and bool(case["call_kind"]),
           "case call_kind must describe the public call")
    return {key: case.get(key) for key in ("i", "episode", "cycle", "phase", "seed", "call_kind")} | {
        "request_sha256": request_hash, "feedback_sha256": feedback,
    }


def _samples(cases: Any, calls: Any, shape: list[int], *, history: bool,
             queue: bool = False) -> tuple[list[dict], list[float], list[int]]:
    count = 51 if queue else (21 if history else 25)
    _check(isinstance(cases, list) and isinstance(calls, list), "cases and calls must be lists")
    _check(len(cases) == len(calls) == count,
           f"expected {count} cases/calls, got {len(cases)}/{len(calls)}")
    signatures, timings, primary = [], [], []
    for index, (case, call) in enumerate(zip(cases, calls)):
        _check(isinstance(case, dict) and isinstance(call, dict), "case/call must be objects")
        _check(type(case.get("i")) is int and case["i"] == index
               and type(call.get("i")) is int and call["i"] == index,
               f"case/call index must be sequential: {index}")
        cycle = _integer(case.get("cycle"), "case cycle")
        if queue:
            _check(case.get("phase") == "queue_drain", "queue calls require phase=queue_drain")
        else:
            expected_cycle = index % 3 if history else 0
            expected_phase = "warmup" if index < (3 if history else 5) else "measured"
            _check(cycle == expected_cycle and case.get("phase") == expected_phase,
                   f"case {index} has incorrect warmup/measured phase or history cycle")
            if history:
                _check(case.get("episode") == index // 3,
                       f"case {index} has incorrect history episode")
        signature = _request_signature(case)
        for key in signature:
            if key in call:
                _check(call[key] == signature[key], f"case/call disagree on {key} at {index}")
        _check(call.get("shape") == shape, f"call {index} public output shape mismatch")
        timings.append(_number(call.get("ms"), f"call {index} ms", positive=True))
        signatures.append(signature)
        if queue or (case["phase"] == "measured" and (not history or cycle > 0)):
            primary.append(index)
    return signatures, timings, primary


def _actions(root: Path, archive: str, key: str, expected_hash: Any,
             count: int, shape: list[int]) -> tuple[np.ndarray, dict]:
    path = _path(root, archive)
    expected_hash = _hash(expected_hash, "actions_sha256")
    actual_hash = _sha256(path)
    _check(actual_hash == expected_hash, f"action archive SHA-256 mismatch: {archive}")
    with np.load(path, allow_pickle=False) as loaded:
        _check(key in loaded.files, f"missing action array {key}")
        array = loaded[key].copy()
        unverified_keys = sorted(set(loaded.files) - {key})
    _check(array.dtype.kind == "f", "public actions must be floating point")
    _check(list(array.shape) == [count, *shape],
           f"full public action archive shape {list(array.shape)} != {[count, *shape]}")
    _check(bool(np.isfinite(array).all()), "action archive contains nonfinite values")
    content = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return array, {
        "path": str(path.resolve()), "sha256": actual_hash,
        "actions_key": key, "shape": list(array.shape), "dtype": str(array.dtype),
        "finite": True, "array_bytes_sha256": content,
        "additional_unverified_arrays": unverified_keys,
    }


def _distribution(values: list[float]) -> dict:
    _check(bool(values), "no timing samples")
    return {"count": len(values), "p50_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values), "min_ms": min(values), "max_ms": max(values)}


def _observed_schedule(receipt: dict, cell: dict, required: bool) -> dict:
    before, after = receipt.get("observed_nfe_before"), receipt.get("observed_nfe_after")
    if not required and before is None and after is None:
        return {"status": "not_required_by_matrix"}
    _check(isinstance(before, dict) and bool(before) and before == after,
           "observed_nfe_before/after must be nonempty and identical")
    expected = receipt["effective_schedule"].get("nfe")
    _check(isinstance(expected, dict) and bool(expected), "effective_schedule missing nfe for observed checks")
    for key, value in before.items():
        _integer(value, f"observed nfe {key}", 1)
        _check(expected.get(key) == value, f"observed nfe {key} differs from effective_schedule")
    active_streams = {"va": {"video", "action"}, "dreamzero": {"video_action"}}
    _check(active_streams.get(cell["family"], {"action"}) <= before.keys(),
           "observed nfe omits an active sampling stream")
    result = {"status": "passed", "nfe_before": before, "nfe_after": after}
    if cell["family"] == "dreamzero":
        initial, final = receipt.get("observed_schedule_before"), receipt.get("observed_schedule_after")
        _check(isinstance(initial, dict) and bool(initial) and initial == final,
               "DreamZero observed_schedule_before/after must be nonempty and identical")
        _check(initial.get("steps") == {"video_action": 16, "kv_commit": 1},
               "DreamZero observed scheduler/KV steps differ")
        _check(initial.get("guidance", {}).get("video_action") == ["cfg", 5.0],
               "DreamZero observed CFG differs")
        mask = cell.get("effective_schedule", {}).get("checkpoint_dit_step_mask")
        _check(isinstance(mask, list) and len(mask) == 16 and all(type(x) is bool for x in mask),
               "matrix must bind DreamZero checkpoint_dit_step_mask")
        _check(initial.get("dit_step_mask") == mask, "DreamZero observed fixed mask differs")
        dynamic = receipt["effective_schedule"].get("step_cache") == "dynamic"
        _check(initial.get("dynamic_cache_schedule") is dynamic, "DreamZero observed dynamic flag differs")
        if dynamic:
            _check(receipt["effective_schedule"].get("profile") == "dreamzero_velocity_v1",
                   "unsupported declared DreamZero dynamic profile")
        result.update(schedule_before=initial, schedule_after=final)
    return result


def _validate_cell(cell: dict, root: Path, *, require_observed_schedule: bool = False) -> tuple[dict, dict]:
    receipt_path = _path(root, cell["receipt"])
    receipt = _json(receipt_path)
    _check(isinstance(receipt, dict), "receipt must be an object")
    _check(type(receipt.get("schema")) is int and receipt["schema"] == 1, "receipt schema must be 1")
    _check(receipt.get("ok") is True, "receipt does not declare successful completion")
    for key in ("family", "arm"):
        _check(receipt.get(key) == cell[key], f"receipt {key} does not match matrix")
    _check(receipt.get("cell_id") == cell["id"], "receipt cell_id does not match matrix")
    model_id = receipt.get("model_id", receipt.get("model"))
    _check(isinstance(model_id, str) and bool(model_id), "receipt missing model_id/model")
    if "model_id" in receipt and "model" in receipt:
        _check(receipt["model_id"] == receipt["model"], "conflicting model_id/model")
    _check(model_id == cell.get("model_id", cell.get("model")), "model does not match matrix")
    _check(receipt.get("revision") == cell.get("revision")
           and isinstance(receipt.get("revision"), str) and bool(receipt["revision"]),
           "revision does not match matrix")
    shape = _shape(cell.get("action_shape"))
    for key in ("runtime_kwargs", "optimizer_environment"):
        actual = _required(receipt, key)
        _check(isinstance(actual, dict), f"{key} must be an object")
        expected = cell.get(f"expected_{key}")
        _check(isinstance(expected, dict), f"matrix missing expected_{key}")
        _check(actual == expected, f"{key} does not match frozen matrix")
    precision = _required(receipt, "precision")
    _check(isinstance(precision, str), "precision must be a string")
    if "precision" in cell:
        _check(precision == cell["precision"], "precision does not match matrix")
    if cell["arm"] == "runtime_default":
        _check(precision == "native", "runtime_default must use native precision")
        allowed = {"revision", "device", "cache_dir", "local_files_only"}
        _check(not (set(receipt["runtime_kwargs"]) - allowed),
               "runtime_default changes public functional options")
        _check(not receipt["optimizer_environment"], "runtime_default has optimizer environment overrides")
    if cell["arm"] == "eager_native":
        _check(precision == "native", "eager_native must use native precision")
        _check(not receipt["runtime_kwargs"], "eager_native must not construct Runtime")
        _check(not receipt.get("applied", []) and not receipt.get("applied_passes", []),
               "eager_native declares Runtime optimization passes")
    default_schedule = _required(receipt, "default_schedule")
    effective_schedule = _required(receipt, "effective_schedule")
    _check(isinstance(default_schedule, dict) and bool(default_schedule), "default_schedule must be an object")
    _check(isinstance(effective_schedule, dict) and bool(effective_schedule), "effective_schedule must be an object")
    for key in ("effective_schedule", "guidance", "default_schedule"):
        if key in cell:
            _check(receipt.get(key) == cell[key], f"{key} does not match frozen matrix")
    observed_schedule = _observed_schedule(receipt, cell, require_observed_schedule)
    guidance = _required(receipt, "guidance")
    device = _required(receipt, "device")
    _check(isinstance(device, str) and bool(device), "device must be a string")
    torch_version = _required(receipt, "torch")
    _check(isinstance(torch_version, str) and bool(torch_version), "torch version must be a string")
    execution_policy = _required(receipt, "execution_policy")
    _check(isinstance(execution_policy, dict), "execution_policy must be an object")
    timing_scope = receipt.get("timing_scope", receipt.get("scope"))
    _check(isinstance(timing_scope, str) and bool(timing_scope), "receipt missing timing_scope/scope")
    input_contract = _required(receipt, "input_contract")
    _check(isinstance(input_contract, (str, dict)) and bool(input_contract), "input_contract must describe the prepared inputs")
    history = cell["family"] in HISTORY_FAMILIES
    signatures, timings, primary = _samples(receipt.get("cases"), receipt.get("calls"), shape, history=history)
    archive = receipt.get("actions_archive", str(receipt_path.with_suffix(".npz")))
    actions, action_info = _actions(root, archive, receipt.get("actions_key", "actions"),
                                    receipt.get("actions_sha256"), len(timings), shape)
    measured = [i for i, case in enumerate(signatures) if case["phase"] == "measured"]
    new_episode = [i for i in measured if signatures[i]["cycle"] == 0]
    setup = _number(_required(receipt, "setup_seconds"), "setup_seconds")
    row = {
        "id": cell["id"], "family": cell["family"], "arm": cell["arm"], "status": "passed",
        "model_id": model_id, "revision": receipt["revision"], "precision": precision,
        "comparison_group": cell["comparison_group"], "experimental": bool(cell.get("experimental", False)),
        "runtime_kwargs": receipt["runtime_kwargs"], "optimizer_environment": receipt["optimizer_environment"],
        "default_schedule": default_schedule, "effective_schedule": effective_schedule, "guidance": guidance,
        "observed_schedule": observed_schedule,
        "execution_policy": execution_policy, "device": device, "torch": torch_version,
        "timing_scope": timing_scope, "input_contract": input_contract,
        "setup_seconds": setup, "setup_scope": receipt.get("setup_scope", "unspecified"),
        "first_predict_ms": timings[0], "measured": _distribution([timings[i] for i in measured]),
        "primary_metric": "continuation_p50_ms" if history else "generation_p50_ms",
        "primary": _distribution([timings[i] for i in primary]),
        "new_episode": _distribution([timings[i] for i in new_episode]),
        "total_calls": len(timings), "warmup_calls": len(timings) - len(measured),
        "timed_session_ms": sum(timings), "timed_session_mean_ms": statistics.fmean(timings),
        "setup_plus_first_predict_ms": setup * 1000 + timings[0],
        "setup_plus_timed_calls_mean_ms": (setup * 1000 + sum(timings)) / len(timings),
        "amortization_scope": "setup plus recorded predict calls; excludes unreported work between calls",
        "action_shape": shape, "actions": action_info,
        "receipt": {"path": str(receipt_path.resolve()), "sha256": _sha256(receipt_path)},
        "request_sequence_sha256": hashlib.sha256(_canonical(signatures).encode()).hexdigest(),
        "task_quality_validated": False, "recommended": False,
        "usability": ("public default succeeded on recorded requests" if cell["arm"] == "runtime_default"
                      else "experimental operating point" if cell["arm"] == "operating_point"
                      else "runtime source update candidate on recorded requests" if cell["arm"] == "runtime_update"
                      else "explicit numerical candidate on recorded requests" if precision != "native"
                      or receipt["runtime_kwargs"].get("tier_ceiling") in ("numeric", "behavioral")
                      or receipt["optimizer_environment"]
                      else "succeeded on recorded requests"),
    }
    if cell["arm"] == "runtime_update":
        row["baseline_cell"] = cell.get("baseline_cell")
    evidence = {"actions": actions, "requests": signatures, "timings": timings, "primary_indices": primary,
                "schedule": effective_schedule, "guidance": guidance, "model_id": model_id,
                "revision": receipt["revision"], "device": device, "torch": torch_version,
                "action_shape": shape, "timing_scope": row["timing_scope"],
                "input_contract": input_contract}
    queue = receipt.get("queue_drain")
    if cell["family"] == "pi05":
        _check(isinstance(queue, dict), "pi05 requires a separate 51-call public queue_drain")
    if queue is not None:
        _check(cell["family"] == "pi05", "queue_drain is only defined for pi05")
        _check(queue.get("reset_between_calls") is False, "queue_drain must not reset between calls")
        _check(queue.get("expected_calls", 51) == 51, "queue_drain must contain 51 calls")
        queue_cases, queue_ms, _ = _samples(queue.get("cases"), queue.get("calls"), shape, history=False, queue=True)
        queue_actions, queue_info = _actions(root, queue.get("archive"), queue.get("actions_key", "actions"),
                                             queue.get("actions_sha256"), 51, shape)
        row["queue_drain"] = {
            "metrics": _distribution(queue_ms), "first_call_ms": queue_ms[0], "call_51_ms": queue_ms[50],
            "middle_49_calls": _distribution(queue_ms[1:50]), "total_ms": sum(queue_ms),
            "reset_between_calls": False, "actions": queue_info,
            "request_sequence_sha256": hashlib.sha256(_canonical(queue_cases).encode()).hexdigest(),
            "scope": "separate continuous public select_action calls; excluded from generation latency",
        }
        evidence.update(queue_actions=queue_actions, queue_requests=queue_cases)
    return row, evidence


def _action_difference(left: np.ndarray, right: np.ndarray) -> dict:
    # The archives may contain finite float64 values too large for a representable
    # difference. Keep the report valid JSON without mistaking metric overflow for
    # a nonfinite public action, and compute the norm with scaling when possible.
    with np.errstate(over="ignore", invalid="ignore"):
        difference = left.astype(np.float64) - right.astype(np.float64)
        maximum = float(np.max(np.abs(difference)))
        rmse = maximum * float(np.sqrt(np.mean((difference / maximum) ** 2))) if 0 < maximum < math.inf else maximum
    return {"exact_bytes": left.dtype == right.dtype and left.tobytes() == right.tobytes(),
            "exact_values": bool(np.array_equal(left, right)),
            "max_abs": maximum if math.isfinite(maximum) else None,
            "rmse": rmse if math.isfinite(rmse) else None,
            "metric_overflow": not math.isfinite(maximum) or not math.isfinite(rmse),
            "scope": "all archived public actions including warmup; descriptive, no quality threshold"}


def _compare(baseline: dict, candidate: dict, evidence: dict, *, cross_policy: bool = False) -> dict:
    left, right = evidence[baseline["id"]], evidence[candidate["id"]]
    result = {"baseline": baseline["id"], "candidate": candidate["id"],
              "kind": "changed_policy_latency_ratio" if cross_policy else "matched_policy_latency_ratio",
              "status": "passed", "task_quality_validated": False, "recommended": False}
    keys = ("model_id", "revision", "device", "torch", "action_shape", "requests", "primary_indices",
            "timing_scope", "input_contract")
    differences = [key for key in keys if left[key] != right[key]]
    if not cross_policy:
        differences.extend(key for key in ("schedule", "guidance") if left[key] != right[key])
    has_queue = "queue_requests" in left or "queue_requests" in right
    if has_queue and left.get("queue_requests") != right.get("queue_requests"):
        differences.append("queue_drain requests")
    if candidate["arm"] == "runtime_update":
        if left["actions"].dtype != right["actions"].dtype:
            differences.append("public action dtype")
        if has_queue and left["queue_actions"].dtype != right["queue_actions"].dtype:
            differences.append("queue_drain action dtype")
        if baseline["default_schedule"] != candidate["default_schedule"]:
            differences.append("default_schedule")
        if baseline["arm"] == "runtime_selected":
            differences.extend(key for key in ("precision", "runtime_kwargs", "optimizer_environment")
                               if baseline[key] != candidate[key])
    if differences:
        return result | {"status": "failed", "errors": ["incomparable fields: " + ", ".join(differences)]}
    indices = left["primary_indices"]
    result.update(
        ratio_of_p50=baseline["primary"]["p50_ms"] / candidate["primary"]["p50_ms"],
        median_paired_ratio=statistics.median([left["timings"][i] / right["timings"][i] for i in indices]),
        paired_samples=len(indices), actions=_action_difference(left["actions"], right["actions"]),
        same_sampling_policy=left["schedule"] == right["schedule"] and left["guidance"] == right["guidance"],
        baseline_schedule=left["schedule"], candidate_schedule=right["schedule"],
        interpretation=("changes sampling policy; latency ratio is not an architecture-only speedup or quality evidence"
                        if cross_policy else "matched declared sampling policy; numerical implementations may differ; no task-quality evidence"),
    )
    if has_queue:
        result["queue_drain_actions"] = _action_difference(left["queue_actions"], right["queue_actions"])
        result["queue_drain_total_ratio"] = baseline["queue_drain"]["total_ms"] / candidate["queue_drain"]["total_ms"]
    return result


def build_report(matrix_path: str | Path, root: str | Path | None = None) -> dict:
    """Validate a frozen matrix; report missing artifacts separately from failures.

    Relative receipt/archive/binding paths are resolved against ``root``, which
    defaults to the matrix directory. ``comparison_baseline`` identifies an
    operating-point native reference; main groups use their eager_native cell.
    ``runtime_update`` is an additional source candidate with an explicit
    ``baseline_cell`` in the same group; it never changes a group's main baseline.
    ``expected_operating_point_cells`` can bind the number of original extras.
    Optional ``cross_policy_baseline`` explicitly requests a separate tradeoff.
    """
    matrix_path = Path(matrix_path).resolve()
    root = Path(root).resolve() if root is not None else matrix_path.parent
    matrix = _json(matrix_path)
    _check(isinstance(matrix, dict) and matrix.get("schema") == 1, "matrix schema must be 1")
    cells = matrix.get("cells")
    _check(isinstance(cells, list) and bool(cells), "matrix cells must be a nonempty list")
    identifiers = [cell.get("id") for cell in cells]
    _check(all(isinstance(value, str) and value for value in identifiers), "cell ids must be nonempty strings")
    _check(len(set(identifiers)) == len(identifiers), "duplicate cell ids")
    for cell in cells:
        _check(cell.get("arm") in (*MAIN_ARMS, "operating_point", "runtime_update"), "unknown cell arm")
        _check(isinstance(cell.get("family"), str) and bool(cell["family"]), "missing cell family")
        _check(isinstance(cell.get("comparison_group"), str) and bool(cell["comparison_group"]),
               "missing comparison_group")
    report = {"schema": REPORT_SCHEMA, "status": "passed",
              "matrix": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
              "validator": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
              "task_quality_validated": False, "recommended": False,
              "scope": "CPU artifact/protocol validation on recorded requests; no simulator or task-quality admission",
              "errors": [], "pending": [], "bindings": [], "cells": [], "comparisons": []}
    require_observed = matrix.get("require_observed_schedule", False)
    _check(type(require_observed) is bool, "require_observed_schedule must be boolean")
    report["require_observed_schedule"] = require_observed
    main = [cell for cell in cells if cell["arm"] in MAIN_ARMS]
    operating_points = [cell for cell in cells if cell["arm"] == "operating_point"]
    updates = [cell for cell in cells if cell["arm"] == "runtime_update"]
    expected_main = matrix.get("expected_main_cells", 24)
    _integer(expected_main, "expected_main_cells", 1)
    if len(main) != expected_main:
        report["errors"].append(f"matrix declares {len(main)} main cells, expected {expected_main}")
    expected_extras = matrix.get("expected_operating_point_cells")
    if expected_extras is not None:
        _integer(expected_extras, "expected_operating_point_cells")
        if len(operating_points) != expected_extras:
            report["errors"].append(f"matrix declares {len(operating_points)} operating-point cells, expected {expected_extras}")
    expected_updates = matrix.get("expected_runtime_update_cells")
    if expected_updates is not None:
        _integer(expected_updates, "expected_runtime_update_cells")
        if len(updates) != expected_updates:
            report["errors"].append(f"matrix declares {len(updates)} runtime-update cells, expected {expected_updates}")
    families = set(matrix.get("expected_families", [cell["family"] for cell in main]))
    if set(cell["family"] for cell in main) != families:
        report["errors"].append("main family coverage does not match expected_families")
    for family in sorted(families):
        arms = [cell["arm"] for cell in main if cell["family"] == family]
        if sorted(arms) != sorted(MAIN_ARMS):
            report["errors"].append(f"{family}: exactly one cell for each main arm is required")
        groups = {cell["comparison_group"] for cell in main if cell["family"] == family}
        if len(groups) != 1:
            report["errors"].append(f"{family}: main arms must share a comparison_group")
    for binding in matrix.get("bindings", []):
        try:
            path = _path(root, binding["path"])
            digest = _sha256(path)
            _check(digest == _hash(binding["sha256"], "binding sha256"), f"binding SHA-256 mismatch: {path}")
            report["bindings"].append({"path": str(path.resolve()), "sha256": digest, "status": "passed"})
        except FileNotFoundError as error:
            report["pending"].append(f"missing binding: {error.filename}")
        except (ValueError, OSError, KeyError, TypeError) as error:
            report["errors"].append(str(error))
    evidence, rows = {}, {}
    for cell in cells:
        try:
            row, validated = _validate_cell(cell, root, require_observed_schedule=require_observed)
            evidence[cell["id"]] = validated
        except FileNotFoundError as error:
            row = {"id": cell["id"], "family": cell["family"], "arm": cell["arm"],
                   "status": "incomplete", "pending": [f"missing artifact: {error.filename}"]}
        except (ValueError, OSError, KeyError, TypeError, AttributeError, BadZipFile, EOFError) as error:
            row = {"id": cell["id"], "family": cell["family"], "arm": cell["arm"],
                   "status": "failed", "errors": [str(error)]}
        report["cells"].append(row)
        rows[cell["id"]] = row
    groups = {}
    for cell in cells:
        if cell["arm"] != "runtime_update":
            groups.setdefault(cell["comparison_group"], []).append(cell)
    for name, group in groups.items():
        defaults = [cell for cell in group if cell["arm"] == "eager_native" or cell.get("comparison_baseline") is True]
        referenced = {cell["baseline_cell"] for cell in group if cell.get("baseline_cell")}
        if not defaults and len(referenced) == 1:
            defaults = [cell for cell in group if cell["id"] in referenced]
        if len(defaults) != 1:
            report["errors"].append(f"comparison group {name} must have exactly one baseline")
            continue
        baseline = defaults[0]
        for cell in group:
            if cell["id"] == baseline["id"]:
                continue
            if cell.get("baseline_cell", baseline["id"]) != baseline["id"]:
                report["errors"].append(f"{cell['id']}: conflicting baseline_cell")
                continue
            if all(rows[value]["status"] == "passed" for value in (baseline["id"], cell["id"])):
                report["comparisons"].append(_compare(rows[baseline["id"]], rows[cell["id"]], evidence))
    by_id = {cell["id"]: cell for cell in cells}
    for cell in updates:
        baseline_id = cell.get("baseline_cell")
        if cell.get("experimental") is not False:
            report["errors"].append(f"{cell['id']}: runtime_update requires experimental=false")
            continue
        if cell.get("comparison_baseline") is True or cell.get("cross_policy_baseline") is not None:
            report["errors"].append(f"{cell['id']}: runtime_update cannot redefine a group baseline or request a cross-policy comparison")
            continue
        if not isinstance(baseline_id, str) or baseline_id not in by_id or baseline_id == cell["id"]:
            report["errors"].append(f"{cell['id']}: runtime_update requires an explicit existing baseline_cell")
            continue
        baseline = by_id[baseline_id]
        if baseline["arm"] not in ("eager_native", "runtime_selected"):
            report["errors"].append(f"{cell['id']}: runtime_update baseline must be eager_native or runtime_selected")
            continue
        if any(cell[key] != baseline[key] for key in ("family", "comparison_group")):
            report["errors"].append(f"{cell['id']}: runtime_update baseline must share family and comparison_group")
            continue
        if all(rows[value]["status"] == "passed" for value in (baseline_id, cell["id"])):
            report["comparisons"].append(_compare(rows[baseline_id], rows[cell["id"]], evidence))
    for cell in cells:
        baseline = cell.get("cross_policy_baseline")
        if baseline is None:
            continue
        if cell["arm"] != "operating_point" or baseline not in rows or baseline == cell["id"]:
            report["errors"].append(f"{cell['id']}: invalid cross_policy_baseline")
        elif rows[cell["id"]]["status"] == rows[baseline]["status"] == "passed":
            report["comparisons"].append(_compare(rows[baseline], rows[cell["id"]], evidence, cross_policy=True))
    failed = sum(row["status"] == "failed" for row in report["cells"])
    incomplete = sum(row["status"] == "incomplete" for row in report["cells"])
    comparison_failures = sum(value["status"] == "failed" for value in report["comparisons"])
    report["counts"] = {"expected_cells": len(cells), "main_cells": len(main),
                         "operating_point_cells": len(operating_points), "runtime_update_cells": len(updates),
                         "passed_cells": len(cells) - failed - incomplete, "failed_cells": failed,
                         "incomplete_cells": incomplete, "failed_comparisons": comparison_failures,
                         "finite_public_actions": sum(row.get("total_calls", 0) for row in report["cells"] if row["status"] == "passed"),
                         "finite_queue_actions": sum(51 for row in report["cells"] if row.get("queue_drain"))}
    if report["errors"] or failed or comparison_failures:
        report["status"] = "failed"
    elif report["pending"] or incomplete:
        report["status"] = "incomplete"
    return report


CSV_FIELDS = ("id", "family", "arm", "status", "model_id", "revision", "precision", "comparison_group",
              "experimental", "primary_metric", "primary_p50_ms", "primary_samples", "measured_samples",
              "first_predict_ms", "setup_seconds", "warmup_calls", "total_calls", "action_shape",
              "default_schedule", "effective_schedule", "guidance", "runtime_kwargs", "optimizer_environment",
              "matched_baseline", "matched_ratio_of_p50", "cross_policy_baseline", "cross_policy_ratio_of_p50",
              "queue_total_ms", "task_quality_validated", "recommended", "usability", "errors", "pending")


def write_csv(report: dict, path: str | Path) -> None:
    """Write one row per declared cell, retaining failed and pending cells."""
    comparisons = {}
    for comparison in report["comparisons"]:
        if comparison["status"] == "passed":
            comparisons[(comparison["candidate"], comparison["kind"])] = comparison
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for source in report["cells"]:
            row = dict(source)
            row.update(primary_p50_ms=source.get("primary", {}).get("p50_ms"),
                       primary_samples=source.get("primary", {}).get("count"),
                       measured_samples=source.get("measured", {}).get("count"),
                       queue_total_ms=source.get("queue_drain", {}).get("total_ms"),
                       task_quality_validated=False, recommended=False)
            for kind, prefix in (("matched_policy_latency_ratio", "matched"), ("changed_policy_latency_ratio", "cross_policy")):
                comparison = comparisons.get((source["id"], kind), {})
                row[f"{prefix}_baseline"] = comparison.get("baseline")
                row[f"{prefix}_ratio_of_p50"] = comparison.get("ratio_of_p50")
            for key, value in tuple(row.items()):
                if isinstance(value, (dict, list)):
                    row[key] = _canonical(value)
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.matrix, args.root)
    args.json_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(report, args.csv_output)
    print(json.dumps({"status": report["status"], **report["counts"]}, sort_keys=True))
    return {"passed": 0, "failed": 1, "incomplete": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())

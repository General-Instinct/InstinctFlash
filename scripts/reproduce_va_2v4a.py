"""Capture and compare three fresh VA 2V/4A cells with the public benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from uuid import UUID

from benchmarks.regression.reproduce import (
    capture_process,
    child_environment,
    encoded,
    make_plan,
    new_directory,
    require,
    require_installed,
    sha,
    validate_bundle,
    write_new,
)

NFE = {"video": 2, "action": 4}
CHECKPOINT_NFE = {"video": 25, "action": 50}
CELL_IDS = ("va-eager_native-2v4a", "va-2v4a-native", "va-2v4a-fp8")


def canonical_gpu_uuid(value):
    require(isinstance(value, str), "GPU UUID must be a string")
    return str(UUID(value.removeprefix("GPU-")))


def matched_matrix(native_plan, fp8_plan):
    """Keep checkpoint facts and public Runtime options; override only native NFE."""
    target = native_plan["target"]
    require(target["name"] in {"rtx4090", "rtx5090"} and fp8_plan["target"] == target,
            "Both preparations must use the same RTX target")
    require(native_plan["model"] == fp8_plan["model"] == "va", "Both preparations must be VA")
    require(native_plan["execution_mode"] == "2v4a-native"
            and fp8_plan["execution_mode"] == "2v4a-fp8", "Expected native and FP8 2V/4A preparations")
    for key in ("checkpoint", "fixture_sha256", "profiles_sha256"):
        require(native_plan[key] == fp8_plan[key], f"Prepared {key} differs")
    require(native_plan["matrix"]["cells"][:3] == fp8_plan["matrix"]["cells"][:3],
            "Prepared checkpoint-default cells differ")
    native = deepcopy(native_plan["matrix"]["cells"][0])
    runtimes = [deepcopy(plan["matrix"]["cells"][-1]) for plan in (native_plan, fp8_plan)]
    require(native["arm"] == "eager_native" and native["effective_schedule"]["nfe"] == CHECKPOINT_NFE,
            "The original native reference must retain the 25V/50A checkpoint declaration")
    for cell, cell_id, precision in zip(runtimes, CELL_IDS[1:], ("native", "fp8")):
        require(cell["id"] == cell_id and cell["effective_schedule"]["nfe"] == NFE
                and cell["expected_runtime_kwargs"]["precision"] == precision,
                "Public Runtime operating point differs from the requested 2V/4A recipe")
    native.update(id=CELL_IDS[0], receipt=f"cells/{CELL_IDS[0]}/receipt.json", native_nfe=dict(NFE),
                  effective_schedule=deepcopy(runtimes[0]["effective_schedule"]), experimental=True)
    cells = [native, *runtimes]
    for cell in cells:
        cell.update(default_schedule=dict(CHECKPOINT_NFE), comparison_group="va-2v4a-matched")
        cell.pop("comparison_baseline", None)
        cell.pop("cross_policy_baseline", None)
    matrix = deepcopy(native_plan["matrix"])
    matrix.update(cells=cells, expected_main_cells=1, expected_operating_point_cells=2,
                  supplementary_only=True, publication_ready=False, quality_certified=False,
                  matched_baseline=CELL_IDS[0], parent_full_schedule_recaptured=False)
    return matrix


def prepared_inputs(args, *, installed):
    directories = [args.native_prepared.resolve(), args.fp8_prepared.resolve()]
    pairs = [validate_bundle(path, check_libraries=installed) for path in directories]
    plans, preparations = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    for mode, plan in zip(("2v4a-native", "2v4a-fp8"), plans):
        require(plan["target"]["name"] == args.target, "Prepared target differs from --target")
        if installed:
            require(plan == make_plan("va", mode, target=args.target),
                    "Preparation differs from the current installed profile")
    require(preparations[0]["checkpoint_snapshot"] == preparations[1]["checkpoint_snapshot"],
            "Preparations reference different original checkpoint snapshots")
    return directories, plans, preparations, matched_matrix(*plans)


def report_capture(capture, plans, matrix):
    """Validate saved arrays on CPU; report ratios against the fresh native cell."""
    from benchmarks.regression.hardware import validate_device_receipt

    from benchmarks.regression import user_report

    capture = Path(capture).resolve(strict=True)
    matrix_path = capture / "matrix.json"
    require(json.loads(matrix_path.read_text()) == matrix, "Captured matrix differs from the prepared recipe")
    require(sha(capture / "inputs/recorded_inputs_v1.npz") == plans[0]["fixture_sha256"],
            "Captured fixture bytes differ")
    rows, evidence, identities, receipts = [], {}, [], []
    for cell in matrix["cells"]:
        receipt = json.loads((capture / cell["receipt"]).read_text())
        row, detail = user_report._validate_cell(cell, capture, require_observed_schedule=True)
        require(receipt["matrix_sha256"] == sha(matrix_path), "Receipt matrix SHA differs")
        require(receipt["input_archive_sha256"] == plans[0]["fixture_sha256"], "Receipt fixture SHA differs")
        require(receipt["target"] == matrix["target"], "Receipt target differs")
        validate_device_receipt(receipt["hardware"], matrix["target"])
        identities.append((canonical_gpu_uuid(receipt["hardware"]["uuid"]), receipt["numeric_environment"]))
        require(receipt["observed_nfe_before"] == receipt["observed_nfe_after"] == NFE
                and receipt["default_schedule"] == CHECKPOINT_NFE,
                "Actual 2V/4A execution or original 25V/50A declaration differs")
        require(detail["primary_indices"] == [i for i in range(3, 21) if i % 3],
                "Expected twelve measured continuation calls from the original 21-call protocol")
        if cell["arm"] == "eager_native":
            require(receipt["native_nfe_override"] == cell["native_nfe"]
                    and not receipt.get("e4m3_tensors"), "Expected original native execution with an explicit NFE override")
        else:
            precision = cell["expected_runtime_kwargs"]["precision"]
            require(receipt["precision"] == precision
                    and bool(receipt.get("e4m3_tensors")) == (precision == "fp8"), "Actual precision evidence differs")
        rows.append(row)
        evidence[cell["id"]] = detail
        receipts.append({"cell": cell["id"], "sha256": sha(capture / cell["receipt"])})
    require(all(identity == identities[0] for identity in identities), "GPU or numerical environment changed across cells")
    comparisons = [user_report._compare(rows[0], candidate, evidence) for candidate in rows[1:]]
    require(all(item["status"] == "passed" and item["same_sampling_policy"] for item in comparisons),
            "Original requests, schedule, guidance or paired action comparison differs")
    return {"schema": "instinctflash.va_2v4a_reproduction_report.v1", "status": "passed_recorded_input_comparison",
            "target": matrix["target"], "cells": rows, "comparisons": comparisons,
            "matched_baseline": CELL_IDS[0], "matrix_sha256": sha(matrix_path), "receipts": receipts,
            "checkpoint_nfe": CHECKPOINT_NFE, "effective_nfe": NFE, "GPU_uuid": identities[0][0],
            "numeric_environment": identities[0][1], "task_quality_validated": False,
            "full_schedule_recaptured": False, "GPU_used_by_report": False,
            "scope": "Three fresh 21-call API cells; ratios use twelve continuation calls. WebSocket checks are separate."}


def run_capture(args):
    require_installed()
    directories, plans, preparations, matrix = prepared_inputs(args, installed=True)
    output = new_directory(args.output)
    matrix_path = output / "matrix.json"
    fixture = output / "inputs/recorded_inputs_v1.npz"
    write_new(matrix_path, encoded(matrix))
    original_fixture = directories[0] / "inputs/recorded_inputs_v1.npz"
    require(sha(original_fixture) == plans[0]["fixture_sha256"], "Original fixture SHA differs")
    write_new(fixture, original_fixture.read_bytes())
    for index, cell in enumerate(matrix["cells"]):
        selected = 1 if index == 2 else 0
        command = [sys.executable, "-I", "-m", "benchmarks.regression.user_e2e",
                   "--matrix", str(matrix_path), "--cell", cell["id"],
                   "--output-root", str(output), "--fixture", str(fixture)]
        with (output / (cell["id"] + ".log")).open("xb") as log:
            process = capture_process(command, cwd=output,
                env=child_environment(plans[selected], cell, preparations[selected]), log=log, timeout=7200)
        write_new(output / (cell["id"] + ".process.json"), encoded(process))
        require(process["exit_code"] == 0 and not process["timed_out"],
                f"Capture failed for {cell['id']}; inspect its original log and receipt")
    result = report_capture(output, plans, matrix)
    write_new(output / "matched_report.json", encoded(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--target", required=True, choices=("rtx4090", "rtx5090"))
        command.add_argument("--native-prepared", type=Path, required=True)
        command.add_argument("--fp8-prepared", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "report":
            command.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_capture(args)
        else:
            require(not args.output.exists(), "Report output must be new; preserve prior evidence")
            _, plans, _, matrix = prepared_inputs(args, installed=False)
            result = report_capture(args.capture, plans, matrix)
            write_new(args.output, encoded(result))
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"VA 2V/4A reproduction failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "p50_ms": {
        row["id"]: row["primary"]["p50_ms"] for row in result["cells"]}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate the additive upstream VA 2V/4A capture without importing Torch."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from benchmarks.regression.user_report import _validate_cell


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(run, matrix_path, recipe_path):
    root = Path(run).resolve(strict=True)
    matrix_path, recipe_path = Path(matrix_path), Path(recipe_path)
    matrix = json.loads(matrix_path.read_text())
    recipe = json.loads(recipe_path.read_text())
    require(len(matrix["cells"]) == 1, "expected the single additive native cell")
    cell, = matrix["cells"]
    require(cell["id"] == "va-eager_native-2v4a", "unexpected cell")
    require(cell["arm"] == "eager_native" and cell["family"] == "va", "not upstream VA")
    require(cell["native_nfe"] == {"video": 2, "action": 4}, "wrong explicit native NFE")
    receipt_path = root / cell["receipt"]
    receipt = json.loads(receipt_path.read_text())
    row, evidence = _validate_cell(cell, root, require_observed_schedule=True)
    require(receipt["matrix_sha256"] == sha(matrix_path), "matrix binding differs")
    require(receipt["input_archive_sha256"] == recipe["fixture"]["sha256"], "fixture binding differs")
    require(receipt["cases"] == recipe["request_cases"], "request/seed/feedback protocol differs")
    require(receipt["native_nfe_override"] == cell["native_nfe"], "native override receipt differs")
    require(receipt["default_schedule"] == {"video": 25, "action": 50}, "checkpoint default was changed")
    require(receipt["numeric_environment"] == {
        "matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False}, "numeric environment differs")
    require(receipt["execution_policy"] == {
        "reference": "upstream eager native policy; shared I/O translation only",
        "checkpoint_nfe": {"video": 25, "action": 50},
        "native_nfe_override": {"video": 2, "action": 4},
        "nfe": {"video": 2, "action": 4}, "schedule_changed": True}, "native execution policy differs")
    require(receipt["plan"] == "Upstream eager; no optimization passes installed", "not the upstream reference")
    require(receipt["applied_passes"] == [] and receipt["graph_stats"] == {}
            and receipt["backend_stats"] == {} and not receipt.get("e4m3_tensors"),
            "unexpected accelerator metadata in upstream capture")
    primary = recipe["protocol"]["primary_indices"]
    require(evidence["primary_indices"] == primary and len(primary) == 12, "primary samples differ")
    p50 = statistics.median(receipt["calls"][index]["ms"] for index in primary)
    require(p50 == row["primary"]["p50_ms"], "independent median differs")
    return {
        "schema": "instinctflash.va_native_2v4a.validation.v1",
        "status": "passed_recorded_input_native_latency",
        "matrix": {"path": str(matrix_path), "sha256": sha(matrix_path)},
        "recipe": {"path": str(recipe_path), "sha256": sha(recipe_path)},
        "validator_sha256": sha(__file__), "native": row,
        "primary_p50_ms": p50, "primary_indices": primary,
        "quality_certified": False,
        "scope": "Upstream native BF16 with explicit 2V/4A; numerical/protocol/latency validation, not task quality.",
    }


def main():
    parent = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=parent / "matrix_v1.json")
    parser.add_argument("--recipe", type=Path, default=parent / "recipe_review_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.run, args.matrix, args.recipe)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "primary_p50_ms": result["primary_p50_ms"]}))


if __name__ == "__main__":
    main()

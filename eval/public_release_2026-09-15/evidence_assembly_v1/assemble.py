"""Assemble copied public qualification evidence without executing any model.

Each invocation creates a new output directory. Missing/failed cells remain
explicit; this is artifact reproduction, never a task-quality certificate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
STUDY = REPO / "eval/public_release_2026-09-15"
FAMILIES = ("pi05", "groot", "vla4", "vla2", "va", "edge", "nano", "dreamzero")
CORRESPONDENCE_SHA = "ade37fc6c6751f0eddca9a6881a7feb099652cc52ecf09d5dd534bf5d8a65abd"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def load_module(path, expected, name):
    require(sha(path) == expected, f"source drift before import: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arrays(path, expected=None):
    if expected is not None:
        require(sha(path) == expected, f"archive hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        result = {key: loaded[key].copy() for key in loaded.files}
    require(all(a.dtype.kind in "fiu" and np.isfinite(a).all() for a in result.values()),
            f"non-numeric or nonfinite arrays: {path}")
    return result


def guarded(function, *args):
    try:
        return function(*args)
    except FileNotFoundError as error:
        return {"status": "pending", "reason": f"missing copied artifact: {error.filename}"}
    except (ValueError, KeyError, TypeError, OSError, EOFError) as error:
        return {"status": "failed", "reason": f"{type(error).__name__}: {error}"}


def source_correspondence(stage, qualification, output):
    require(qualification == (STUDY / "qualification").resolve(),
            "source correspondence auditor is bound to the actual qualification root")
    path = STUDY / "qualified_stage_correspondence_v1/audit_v2.py"
    auditor = load_module(path, CORRESPONDENCE_SHA, "public_source_correspondence")
    result = auditor.audit(stage, output / "source_correspondence")
    require(result["status"] == "passed_source_and_plan_correspondence", "source correspondence failed")
    return {"status": "passed", "receipt": ref(output / "source_correspondence/receipt.json"),
            "covered_cells": sorted(item["cell"] for item in result["actual_passed_cells"])}


def foreign_status(catalog, results, framework, family):
    if family in catalog["unsupported"].get(framework, []):
        return "unsupported"
    if family in catalog["not_qualified"].get(framework, {}):
        return "not_qualified"
    matching = [row for row in catalog["cells"] if row["family"] == family and row["framework"] == framework]
    require(len(matching) == 1, "missing foreign support declaration")
    return results[matching[0]["id"]]["status"]


def validate_main(run, family, validator, source, output):
    plan, matrix = read(run / "plan.json"), read(run / "matrix.json")
    execution, prepared = read(run / "run.json"), read(run / "preparation.json")
    require(execution["status"] == execution["report_status"] == "passed", "native run did not pass")
    require(execution["task_quality_validated"] is False, "unexpected task-quality promotion")
    expected = [f"{family}-{arm}" for arm in ("eager_native", "runtime_default", "runtime_selected")]
    if family == "va":
        expected.append("va-2v4a-fp8")
    require([cell["id"] for cell in matrix["cells"]] == expected, "main cell coverage changed")
    require(plan["model"] == family and plan["matrix"] == matrix, "plan/matrix mismatch")
    require(execution["plan_sha256"] == prepared["plan_sha256"] == sha(run / "plan.json"), "plan hash mismatch")
    require(prepared["matrix_sha256"] == sha(run / "matrix.json"), "matrix hash mismatch")
    require(plan["profiles_sha256"] == prepared["profiles_sha256"] == sha(run / "profiles.json"), "profile hash mismatch")
    require(prepared["fixture_sha256"] == plan["fixture_sha256"] == sha(run / "inputs/recorded_inputs_v1.npz"), "fixture mismatch")
    require(prepared["checkpoint"] == plan["checkpoint"], "checkpoint preparation mismatch")
    attempts = execution["attempts"]
    require([item["cell"] for item in attempts] == expected, "extra, missing, duplicate or reordered model attempt")
    for attempt, cell in zip(attempts, matrix["cells"]):
        require(type(attempt["exit_code"]) is int and attempt["exit_code"] == 0, "native process failed")
        command = attempt["command"]
        require(command[0] == execution["interpreter"] and command[1:5] == ["-I", "-B", "-m", "benchmarks.regression.user_e2e"], "unexpected public entrypoint")
        require(command[command.index("--cell") + 1] == cell["id"], "wrong command cell")
        require(attempt["optimizer_environment"] == cell["expected_optimizer_environment"], "optimizer environment differs")
        require(sha(run / "logs" / f"{cell['id']}.log") == attempt["log_sha256"], "worker log hash mismatch")
        receipt = read(run / "cells" / cell["id"] / "receipt.json")
        require(receipt["matrix_sha256"] == sha(run / "matrix.json"), "receipt matrix mismatch")
        require(receipt["input_archive_sha256"] == plan["fixture_sha256"], "receipt fixture mismatch")
    replay = validator.build_report(run / "matrix.json", run)
    require(replay["status"] == "passed", f"CPU paired replay failed: {replay}")
    published = read(run / "validated_report/report.json")
    require(published["status"] == "passed" and published["validator"]["sha256"] == sha(source / "benchmarks/regression/user_report.py"), "published validator differs")
    require(published["counts"] == replay["counts"] and published["comparisons"] == replay["comparisons"], "published paired comparisons differ")
    rows = {item["id"]: item for item in published["cells"]}
    require(set(rows) == set(expected), "published rows differ")
    for item in replay["cells"]:
        original = rows[item["id"]]
        for key in ("primary", "measured", "effective_schedule", "precision", "runtime_kwargs", "optimizer_environment", "request_sequence_sha256"):
            require(original[key] == item[key], f"published metric/contract differs: {item['id']} {key}")
        require(original["receipt"]["sha256"] == item["receipt"]["sha256"], "published receipt differs")
        require(original["actions"]["sha256"] == item["actions"]["sha256"], "published action archive differs")
    save(output / f"{family}_paired_replay.json", replay)
    return {"status": "passed", "plan": ref(run / "plan.json"), "run": ref(run / "run.json"),
            "matrix": ref(run / "matrix.json"), "preparation": ref(run / "preparation.json"),
            "published_report": ref(run / "validated_report/report.json"),
            "replay_report": ref(output / f"{family}_paired_replay.json"),
            "cells": [{"id": row["id"], "p50_ms": row["primary"]["p50_ms"],
                       "samples": row["primary"]["count"], "metric": row["primary_metric"],
                       "precision": row["precision"], "schedule": row["effective_schedule"],
                       "receipt": row["receipt"], "action_sha256": row["actions"]["sha256"]} for row in replay["cells"]],
            "comparisons": replay["comparisons"], "task_quality_certified": False}


def validate_serving(directory, run, inputs_module):
    receipt = read(directory / "receipt.json")
    plan = read(run / "plan.json")
    require(receipt["status"] == "passed" and type(receipt["server_exit_code"]) is int and receipt["server_exit_code"] == 0,
            "serving process did not close successfully")
    require(receipt["task_quality_validated"] is False and receipt["latency_benchmark"] is False, "serving scope changed")
    require(receipt["plan_sha256"] == sha(run / "plan.json") and receipt["fixture_sha256"] == plan["fixture_sha256"], "serving plan/input differs")
    require(receipt["checkpoint"] == plan["checkpoint"], "serving checkpoint differs")
    require(receipt["config_sha256"] == sha(directory / "serve.json"), "serving config changed")
    matches = [cell for cell in plan["matrix"]["cells"] if cell["id"] == receipt["cell_id"]]
    require(len(matches) == 1, "serving cell is absent from paired plan")
    cell = matches[0]
    config = read(directory / "serve.json")
    require(config["runtime"] == cell["expected_runtime_kwargs"], "serving runtime options differ")
    metadata, options = receipt["metadata"], cell["expected_runtime_kwargs"]
    policy = metadata["execution_policy"]
    require(metadata["model_id"] == cell["model_id"] and metadata["precision"] == options.get("precision", "native"), "serving model/precision differs")
    require(policy["nfe"] == cell["effective_schedule"]["nfe"] and policy["tier_ceiling"] == options.get("tier_ceiling", "bitexact"), "serving execution policy differs")
    require(metadata["protocol"]["wire"] == "openpi-websocket-msgpack-numpy" and metadata["protocol"]["reset_extension"] is True, "serving wire/reset differs")
    payload = arrays(directory / "actions.npz", receipt["action_archive_sha256"])
    require(set(payload) == {"actions"} and list(payload["actions"].shape) == [6, *cell["action_shape"]], "serving action coverage differs")
    calls = receipt["calls"]
    require(len(calls) == 6, "serving did not complete six calls")
    inputs = inputs_module.RecordedInputs(run / "inputs/recorded_inputs_v1.npz")
    for i, call in enumerate(calls):
        episode, cycle = divmod(i, 3)
        require((call["episode"], call["cycle"]) == (episode, cycle), "serving episode/cycle differs")
        observation = inputs.observation(cell["family"], i, cycle)
        observation["prompt"] = inputs_module.prompt(episode)
        if cell["family"] == "va":
            observation["executed_action"] = inputs.feedback[cycle].copy()
        require(call["request_sha256"] == inputs_module.request_hash(observation), "serving request bytes differ")
        require(call["action_shape"] == cell["action_shape"] and call["action_dtype"] == str(payload["actions"].dtype), "serving dtype/shape differs")
        require(math.isfinite(call["roundtrip_ms"]) and call["roundtrip_ms"] > 0, "invalid serving elapsed time")
    return {"status": "passed", "cell": cell["id"], "receipt": ref(directory / "receipt.json"),
            "config": ref(directory / "serve.json"), "actions": ref(directory / "actions.npz"),
            "calls": 6, "episodes": 2, "reset_scope": "Frozen client sends and validates reset before each episode; individual reset acknowledgments are not separately persisted.",
            "server_exit_code": 0, "latency_benchmark": False, "task_quality_certified": False}


def validate_historical(family_root, history):
    run = family_root / "run"
    cells = read(run / "matrix.json")["cells"]
    published_path = family_root / "historical_action_comparison_v1.json"
    published = read(published_path)
    paired_report = read(run / "validated_report/report.json")
    metrics = {row["id"]: row for row in paired_report["cells"]}
    if isinstance(published, dict):
        require(published["status"] == "passed", "published historical status did not pass")
        paired = {row["candidate"]: row for row in paired_report["comparisons"]}
        require(len(published["comparisons"]) == len(paired)
                and {row["candidate"] for row in published["comparisons"]} == set(paired), "historical comparison coverage differs")
        for row in published["comparisons"]:
            expected = paired[row["candidate"]]
            require(row["ratio"] == expected["ratio_of_p50"]
                    and row["exact_native_actions"] == expected["actions"]["exact_bytes"],
                    "published historical timing/native-equivalence field differs")
    claims = published if isinstance(published, list) else published["cells"]
    claimed = {(row["cell"], row.get("file", "receipt.npz")): row for row in claims}
    require(len(claimed) == len(claims), "duplicate historical comparison claim")
    rows = []
    for cell in cells:
        current = run / "cells" / cell["id"]
        previous = history / "cells" / cell["id"]
        new_receipt, old_receipt = read(current / "receipt.json"), read(previous / "receipt.json")
        for key in ("cases", "model_id", "revision", "effective_schedule", "precision"):
            require(new_receipt[key] == old_receipt[key], f"historical {key} mismatch for {cell['id']}")
        files = ["receipt.npz"] + (["receipt.queue.npz"] if "queue_drain" in new_receipt else [])
        if "queue_drain" in new_receipt:
            require(new_receipt["queue_drain"]["cases"] == old_receipt["queue_drain"]["cases"], "historical queue request contract differs")
        for name in files:
            expected_new = new_receipt["actions_sha256"] if name == "receipt.npz" else new_receipt["queue_drain"]["actions_sha256"]
            expected_old = old_receipt["actions_sha256"] if name == "receipt.npz" else old_receipt["queue_drain"]["actions_sha256"]
            left, right = arrays(previous / name, expected_old), arrays(current / name, expected_new)
            require(set(left) == set(right) == {"actions"}, "historical action array keys differ")
            a, b = left["actions"], right["actions"]
            exact = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            require(exact, f"historical actions differ: {cell['id']} {name}")
            claim = claimed.pop((cell["id"], name))
            require(claim.get("historical_actions_match_bytes", claim.get("actions_match_historical_bytes")) is True, "historical claim mismatch")
            require(claim.get("max_abs", claim.get("max_abs_vs_historical")) == 0, "historical maximum mismatch")
            for key, expected in (("historical_file_sha256", expected_old), ("new_file_sha256", expected_new)):
                if key in claim:
                    require(claim[key] == expected, "published historical archive hash mismatch")
            if "shape" in claim:
                require(claim["shape"] == list(a.shape), "historical shape claim mismatch")
            if "historical_file" in claim:
                require((REPO / claim["historical_file"]).resolve() == (previous / name).resolve(), "historical path claim mismatch")
            if "p50_ms" in claim:
                require(claim["p50_ms"] == metrics[cell["id"]]["primary"]["p50_ms"], "historical timing field differs")
            rows.append({"cell": cell["id"], "file": name, "shape": list(a.shape), "dtype": str(a.dtype),
                         "historical": ref(previous / name), "new": ref(current / name),
                         "historical_receipt": ref(previous / "receipt.json"), "new_receipt": ref(current / "receipt.json"),
                         "exact_action_bytes": True})
    require(not claimed, "extra historical claims")
    return {"status": "passed", "published": ref(published_path), "arrays": rows,
            "scope": "Each fresh arm versus its own historical arm. Does not mean FP8/NUMERIC equals native or establish task success."}


def validate_foreign(cell_id, directory, prepared_root, helper, fixture):
    preparation, stored_plan = read(prepared_root / "preparation.json"), read(prepared_root / "plan.json")
    expected = helper.plan(cell_id)
    require(stored_plan == expected and preparation["plan_sha256"] == sha(prepared_root / "plan.json"), "foreign frozen plan mismatch")
    captured, closed = read(directory / "capture.json"), read(directory / "closed.json")
    completion, published = read(directory / "completion.json"), read(directory / "report.json")
    require(type(completion["exit_code"]) is int and completion["exit_code"] == 0 and completion["forced_kill"] is False and completion["timed_out"] is False, "foreign process failed or timed out")
    require(captured["status"] == "captured" and captured["cell"] == expected["cell"], "foreign capture/cell mismatch")
    require(captured["runner_sha256"] == completion["runner_sha256"] == sha(helper.__file__), "foreign source drift")
    require(captured["preparation_sha256"] == sha(prepared_root / "preparation.json"), "foreign preparation changed")
    for key in ("schema", "catalog_sha256", "source_inventory_sha256", "fixture_sha256", "packages"):
        require(captured[key] == expected[key], f"foreign {key} mismatch")
    require(closed["status"] == "closed" and closed["capture_sha256"] == sha(directory / "capture.json"), "foreign policy close mismatch")
    facts = read(Path(helper.__file__).parent / "fixtures/frameworks/sources.json")
    names = [expected["cell"]["framework"]]
    if names[0] == "vllm-omni":
        names.append("cosmos-framework")
        paths = helper.checkpoint_cache_paths(expected["cell"], preparation)
        require(captured["startup_checkpoint_paths"] == completion["startup_checkpoint_paths"] == paths, "foreign startup checkpoint path mismatch")
    require(set(captured["sources"]) == set(names), "foreign source coverage changed")
    for name in names:
        for key, source_key in (("revision", "revision"), ("files", "installed_inventory"), ("runtime_patch", "runtime_patch")):
            require(captured["sources"][name][key] == facts[name][source_key], "foreign installed source mismatch")
    payload = arrays(directory / "actions.npz", captured["actions_sha256"])
    inputs = helper.Inputs(fixture)
    cell = expected["cell"]
    keys = {f"action_{i}" for i in range(cell["total_calls"])}
    if cell["family"] == "pi05":
        keys |= {f"queued_actions_{i}" for i in range(cell["total_calls"])}
    require(set(payload) == keys, "foreign archive coverage differs")
    for i, row in enumerate(captured["calls"]):
        req, action = helper.request(cell, inputs, i), payload[f"action_{i}"]
        require(row["request_sha256"] == helper.request_hash(req), "foreign request byte mismatch")
        require(all(row[k] == req[k] for k in ("i", "episode", "cycle", "reset", "seed")), "foreign request/reset/seed mismatch")
        require(list(action.shape) == row["shape"] and hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest() == row["action_sha256"], "foreign action byte/shape mismatch")
        if cell["family"] == "pi05":
            require(payload[f"queued_actions_{i}"].shape == (49, 1, 7), "foreign pi05 queue shape differs")
    latency = helper.selected_report(cell, captured["calls"])
    require(captured["latency"] == published["latency"] == latency, "foreign latency arithmetic differs")
    require(published["status"] == "passed" and published["cell"] == cell, "foreign report cell differs")
    require(published["capture_sha256"] == sha(directory / "capture.json") and published["completion_sha256"] == sha(directory / "completion.json"), "foreign report hashes differ")
    require(published["task_quality_certified"] is False and captured["task_quality_certified"] is False, "foreign quality scope changed")
    return {"status": "passed", "cell": cell_id, "family": cell["family"], "framework": cell["framework"],
            "schedule": cell["effective_schedule"], "latency": latency, "plan": ref(prepared_root / "plan.json"),
            "preparation": ref(prepared_root / "preparation.json"), "capture": ref(directory / "capture.json"),
            "completion": ref(directory / "completion.json"), "closed": ref(directory / "closed.json"),
            "actions": ref(directory / "actions.npz"), "published_report": ref(directory / "report.json"),
            "limitation": "Prepared asset inventories are bound, not rehashed against remote checkpoint storage in this CPU assembly.",
            "task_quality_certified": False}


def assemble(stage, qualification, output):
    require(not output.exists(), "output already exists; select a new snapshot path")
    output.mkdir(parents=True)
    source = stage / "source"
    manifest = read(stage / "manifest.json")
    modules = {}
    for name in ("user_report", "user_e2e", "framework_compare"):
        relative = f"benchmarks/regression/{name}.py"
        modules[name] = load_module(source / relative, manifest["files"][relative]["sha256"], f"public_assembly_{name}")
    main, foreign = {}, {}
    for family in FAMILIES:
        root, run = qualification / family, qualification / family / "run"
        paired = guarded(validate_main, run, family, modules["user_report"], source, output)
        serving = [guarded(validate_serving, directory, run, modules["user_e2e"])
                   for directory in sorted(root.glob("serving*")) if directory.is_dir()]
        expected_serving = {f"{family}-runtime_selected"} | ({"va-2v4a-fp8"} if family == "va" else set())
        coverage = {item.get("cell") for item in serving if item["status"] == "passed"}
        serving_status = "passed" if expected_serving <= coverage else "failed" if any(item["status"] == "failed" for item in serving) else "pending"
        main[family] = {"paired": paired, "serving": {"status": serving_status, "expected_cells": sorted(expected_serving), "results": serving},
                        "historical_arrays": guarded(validate_historical, root, REPO / "eval/user_e2e_2026-09-14"),
                        "status": "passed" if paired["status"] == serving_status == "passed" else "failed" if "failed" in (paired["status"], serving_status) else "pending"}
    catalog = modules["framework_compare"].catalog()
    for cell in catalog["cells"]:
        name = cell["id"]
        foreign[name] = guarded(validate_foreign, name, qualification / "foreign" / name,
                                STUDY / "framework_compare_v1/prepared_assets_v2" / name,
                                modules["framework_compare"], source / "benchmarks/regression/fixtures/recorded_inputs_v1.npz")
    require(len(foreign) == 6, "foreign six-cell catalog changed")
    correspondence = guarded(source_correspondence, stage, qualification, output)
    rows = []
    for family in FAMILIES:
        paired = main[family]["paired"]
        cells = {row["id"]: row for row in paired.get("cells", [])}
        row = {"model": family, "main_status": main[family]["status"], "native_p50_ms": cells.get(f"{family}-eager_native", {}).get("p50_ms"),
               "InstinctFlash_p50_ms": cells.get(f"{family}-runtime_selected", {}).get("p50_ms"),
               "LeRobot_p50_ms": None, "vLLM_Omni_p50_ms": None,
               "LeRobot_status": foreign_status(catalog, foreign, "lerobot", family),
               "vLLM_Omni_status": foreign_status(catalog, foreign, "vllm-omni", family)}
        for foreign_row in foreign.values():
            if foreign_row["status"] == "passed" and foreign_row["family"] == family:
                column = "LeRobot_p50_ms" if foreign_row["framework"] == "lerobot" else "vLLM_Omni_p50_ms"
                row[column] = foreign_row["latency"]["p50_ms"]
        rows.append(row)
        if family == "va":
            rows.append({"model": "va-2v4a-fp8", "main_status": main[family]["status"],
                         "native_p50_ms": row["native_p50_ms"], "InstinctFlash_p50_ms": cells.get("va-2v4a-fp8", {}).get("p50_ms"),
                         "LeRobot_p50_ms": row["LeRobot_p50_ms"], "vLLM_Omni_p50_ms": None,
                         "LeRobot_status": row["LeRobot_status"], "vLLM_Omni_status": row["vLLM_Omni_status"]})
    counts = {"main_expected": 8, "main_passed": sum(row["status"] == "passed" for row in main.values()),
              "foreign_expected": 6, "foreign_passed": sum(row["status"] == "passed" for row in foreign.values()),
              "main_cells_passed": sum(len(row["paired"].get("cells", [])) for row in main.values()),
              "historical_families_passed": sum(row["historical_arrays"]["status"] == "passed" for row in main.values()),
              "serving_passed": sum(item["status"] == "passed" for row in main.values() for item in row["serving"]["results"])}
    failures = [f"main/{name}" for name, row in main.items() if row["status"] == "failed"] + [f"foreign/{name}" for name, row in foreign.items() if row["status"] == "failed"]
    historical_failures = [name for name, row in main.items() if row["historical_arrays"]["status"] == "failed"]
    if correspondence["status"] == "failed":
        failures.append("source_correspondence")
    inventory = {str(path.relative_to(qualification)): {"sha256": sha(path), "bytes": path.stat().st_size}
                 for path in sorted(qualification.rglob("*")) if path.is_file()}
    save(output / "copied_evidence_inventory.json", inventory)
    result = {"schema": "instinctflash.public_reproduction_evidence.v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "status": "failed_evidence" if failures or historical_failures else "complete_recorded_reproduction" if counts["main_passed"] == 8 and counts["foreign_passed"] == 6 and counts["historical_families_passed"] == 8 and correspondence["status"] == "passed" else "partial",
              "assembler": ref(Path(__file__)), "stage_manifest": ref(stage / "manifest.json"),
              "source_bindings": {name: ref(module.__file__) for name, module in modules.items()},
              "source_correspondence": correspondence,
              "copied_evidence_inventory": ref(output / "copied_evidence_inventory.json"), "counts": counts,
              "main": main, "foreign": foreign, "table": rows, "failures": failures, "historical_failures": historical_failures,
              "unsupported_foreign": catalog["unsupported"], "not_qualified_foreign": catalog["not_qualified"],
              "task_quality_certified": False,
              "limitations": [
                  "Current copied evidence only; missing cells remain pending and failed prior attempts stay in the inventory.",
                  "Foreign and InstinctFlash selected schedules differ (notably LeRobot pi05 NFE1 versus IFL NFE10); columns are declared operating points, not uniformly matched computation.",
                  "The InstinctFlash table column is the prospectively selected Runtime route. The separate per-cell records also retain default-route timings; this assembly does not choose a winner after seeing timing noise.",
                  "Historical parity compares each arm to its own historical implementation; it does not make lossy paths BITEXACT to native.",
                  "Serving validates six transport calls/two episode resets and process exit; it is not a latency benchmark or task-success test.",
                  "No GPU, remote checkpoint rehash, anonymous download, or simulator qualification is performed here.",
                  "Installed source/native wheel correspondence has its separately bound audit; logs and source inventories here preserve actual provenance without claiming every external dependency was re-executed.",
              ]}
    save(output / "summary.json", result)
    with (output / "table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=Path("/home/ubuntu/ifl-public-full-stage-20260915-v5"))
    parser.add_argument("--qualification", type=Path, default=STUDY / "qualification")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.stage.resolve(), args.qualification.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "counts": result["counts"], "failures": result["failures"], "historical_failures": result["historical_failures"], "summary": str(args.output / "summary.json")}))
    return 1 if result["status"] == "failed_evidence" else 0


if __name__ == "__main__":
    raise SystemExit(main())

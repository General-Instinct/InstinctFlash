"""V4: explicit additive Edge/Nano compiler rerun selection, preserving prior evidence.

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
CORRESPONDENCE_SHA = "af1f9ddd6c08cdbdc29729279e060acde8bb21108f17fa87ec0d7b05bd8870cc"
HISTORICAL_MATRIX_SHA = "346154c84a871fd48addd23f563935c29ec0f90a7cbd5b7ef813b7739bf8077d"
V3_ASSEMBLER_SHA = "732c5ed16562bf7e5bc2dbb6d4670fdd12ea59e2a1660f7e5f30facbaf04b7f8"
V2_ASSEMBLER_SHA = "8bd6a70fd3eb37430b19e21d39c77266ad1ed854dbe9fe0c70eefc01904b7f2b"
FOREIGN_STAGE = Path("/home/ubuntu/ifl-public-full-stage-20260915-v6")
FOREIGN_STAGE_SHA = "f496286c35bbe5ed7f576d1659a4fb29795251dd3c181612bf1b810d25812a70"
FOREIGN_SOURCE_SHA = "dedb8784927fde53d87ce0464f87070ef5cf150334bd76a537dbeb2e230d0e6a"
OMNI_CLOSURE_SHA = "f9bb92cbb03ae55ef5cd9b70c771e82c7146085c6e0392ef2140a91c042e9f95"
OMNI_CLOSURE_COPY_SHA = "4e6b68b5400aba949119d8770b063625caacc7464a03bcccefee52bde51f4b52"
PARENT_ASSEMBLER_SHA = "630e0820351d5e723b8a100c6c43dadd7373f61489e24784c0dfca9248974eae"
PRESERVED_INVENTORY_SHA = "0b45160306e2a0e99e3e449c384fecf0ec8d15e316ba46b6aa8ec7f5f03e4130"
COMPILER_ANALYZER_SHA = "0a5533abdd7502c4f19b75ebb8575fa1077e6d8bd3d89a3b11f2d6e6d313fd42"
COMPILER_WORKER_SHA = "d9d32f526d34e7fb5758c9e08ce8db9f2ff0483428af94a0ed20200b90e8615b"
COMPILER_BINDING_SHA = "41c536a582d3c318e07bbdaf1f4b90787eb25c4e6fcf387425d727a98f3ed211"
CORRECTED_PYTHON = "/dev/shm/ifl_public_envs_20260915_v1/cosmos_pytorch_triton_v1/bin/python"
SELECTION_FILES = {"run/run.json", "run/plan.json", "run/matrix.json", "run/preparation.json",
                   "run/profiles.json", "run/validated_report/report.json", "serving/receipt.json", "serving/serve.json"}


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


def checked_ref(value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "invalid evidence reference")
    path = Path(value["path"])
    require(path.is_absolute() and sha(path) == value["sha256"], f"bound evidence changed: {path}")
    return path


def preserve_original_evidence(qualification, inventory_ref):
    expected = STUDY / "evidence_assembly_v1/stage_v7_partial_v1/copied_evidence_inventory.json"
    path = checked_ref(inventory_ref)
    require(path.resolve() == expected.resolve() and sha(path) == PRESERVED_INVENTORY_SHA,
            "wrong preserved evidence inventory")
    selected = {name: row for name, row in read(path).items() if Path(name).parts[0] in {"edge", "nano"}}
    require(len(selected) == 68, "preserved Edge/Nano inventory coverage changed")
    for relative, item in selected.items():
        local = qualification / relative
        require(local.resolve().is_relative_to(qualification.resolve()), "preserved evidence leaves root")
        require(local.stat().st_size == item["bytes"] and sha(local) == item["sha256"],
                f"original evidence changed: {relative}")
    return {"inventory": ref(path), "original_files_rehashed": len(selected),
            "original_results_and_failures_retained": True}


def validate_compiler_diagnostic(binding):
    path = checked_ref(binding)
    require(path.resolve().is_relative_to((STUDY / "nano_compiler_diagnostic_v1").resolve()),
            "compiler diagnostic is outside the declared study")
    result = read(path)
    require(result["schema"] == "instinctflash.nano_compiler_diagnostic_comparison.v1"
            and result["status"] == "diagnostic_comparison_complete", "compiler diagnostic did not complete")
    for key, expected in (("source", COMPILER_ANALYZER_SHA), ("worker", COMPILER_WORKER_SHA),
                          ("compiler_binding", COMPILER_BINDING_SHA)):
        require(sha(checked_ref(result[key])) == expected, f"compiler diagnostic {key} changed")
    for key in ("current", "previous_control", "external_completion", "queue_config"):
        checked_ref(result[key])
    require(result["historical_action_bytes_recovered_on_tested_inputs"] is True
            and result["actions"]["corrected_vs_historical"]["exact_bytes"] is True,
            "compiler diagnostic did not recover historical action bytes")
    require(result["task_quality_certified"] is False and result["normal_benchmark_or_serving_qualified"] is False,
            "compiler diagnostic scope changed")
    return {"comparison": ref(path), "historical_bytes_recovered_on_fixed_diagnostic": True,
            "scope": "Hash-bound diagnostic analysis; normal paired/serving and historical validators still run separately."}


def load_selection(qualification, selection=None, selection_sha256=None):
    qualification = Path(qualification).resolve()
    roots = {family: qualification / family for family in FAMILIES}
    if selection is None:
        require(selection_sha256 is None, "selection hash supplied without a selection")
        return roots, {"mode": "original_qualification_roots", "overrides": {}}
    require(isinstance(selection_sha256, str) and len(selection_sha256) == 64
            and sha(selection) == selection_sha256, "explicit selection hash differs")
    data = read(selection)
    require(set(data) == {"schema", "template_only", "qualification_root", "preserved_evidence_inventory",
                          "compiler_diagnostic", "overrides"}, "selection fields differ")
    require(data["schema"] == "instinctflash.public_reproduction_selection.v1" and data["template_only"] is False,
            "selection is not an actual execution binding")
    require(Path(data["qualification_root"]).resolve() == qualification == (STUDY / "qualification").resolve(),
            "selection qualification root differs")
    overrides = data["overrides"]
    require(isinstance(overrides, dict) and 0 < len(overrides) <= 2
            and set(overrides) <= {"edge", "nano"}, "only explicit Edge/Nano overrides are allowed")
    preserved = preserve_original_evidence(qualification, data["preserved_evidence_inventory"])
    diagnostic = validate_compiler_diagnostic(data["compiler_diagnostic"])
    for family, entry in overrides.items():
        require(isinstance(entry, dict) and set(entry) == {"root", "files"}, "override fields differ")
        relative = f"{family}/pytorch_triton_v1"
        require(entry["root"] == relative, "only the declared additive nested root is allowed")
        root = qualification / relative
        require(root.resolve() == root.absolute(), "selected root may not redirect through symlinks")
        require(set(entry["files"]) == SELECTION_FILES, "selection artifact coverage differs")
        for name, expected in entry["files"].items():
            require((root / name).resolve().is_relative_to(root.resolve()), "selection artifact leaves selected root")
            require(sha(root / name) == expected, f"selected evidence changed: {family}/{name}")
        execution = read(root / "run/run.json")
        serving = read(root / "serving/receipt.json")
        require(execution["status"] == execution["report_status"] == serving["status"] == "passed",
                "selected normal benchmark/serving has not passed")
        require(execution["interpreter"] == serving["interpreter"] == CORRECTED_PYTHON,
                "selected run does not use the declared corrected compiler environment")
        roots[family] = root
    return roots, {"mode": "explicit_additive_compiler_requalification", "selection": ref(selection),
                   "overrides": {family: str(roots[family]) for family in sorted(overrides)},
                   "preserved_evidence": preserved, "compiler_diagnostic": diagnostic}


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


def source_correspondence(stage, qualification, output, selected_roots, selection_binding):
    require(qualification == (STUDY / "qualification").resolve(),
            "source correspondence auditor is bound to the actual qualification root")
    path = STUDY / "qualified_stage_correspondence_v1/audit_v3.py"
    auditor = load_module(path, CORRESPONDENCE_SHA, "public_source_correspondence")
    result = auditor.audit(stage, output / "source_correspondence", selected_roots, selection_binding)
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


def historical_receipt(history, cell):
    matrix_path = history / "matrix_final.json"
    require(sha(matrix_path) == HISTORICAL_MATRIX_SHA, "historical final matrix changed")
    matrix = read(matrix_path)
    matches = [row for row in matrix["cells"] if row["id"] == cell["id"]]
    require(len(matches) == 1, "historical matrix has no unique declared cell")
    row = matches[0]
    for key in ("id", "family", "arm", "model_id", "revision", "action_shape", "effective_schedule"):
        require(row[key] == cell[key], f"historical matrix cell contract differs: {key}")
    relative = Path(row["receipt"])
    require(not relative.is_absolute() and ".." not in relative.parts, "invalid historical receipt path")
    path = (history / relative).resolve(strict=True)
    require(path.is_relative_to(history.resolve()) and path.name == "receipt.json", "historical receipt leaves the declared root")
    receipt = read(path)
    require(receipt["ok"] is True and receipt["cell_id"] == cell["id"], "declared historical receipt is not successful")
    return path


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
        previous = historical_receipt(history, cell).parent
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
            "historical_matrix": ref(history / "matrix_final.json"),
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


def load_foreign_sources(stage):
    """Select original execution inputs prospectively, never by a passing result."""
    require(sha(stage / "manifest.json") == FOREIGN_STAGE_SHA, "foreign source stage manifest changed")
    manifest = read(stage / "manifest.json")
    require(manifest["status"] == "wheels_built_and_inspected_CPU_only", "foreign source stage incomplete")
    inventory_path = STUDY / "framework_compare_v1/source_inventory_v4.json"
    require(sha(inventory_path) == FOREIGN_SOURCE_SHA, "foreign V4 source binding changed")
    inventory = read(inventory_path)
    bound = {}
    for relative, expected in inventory["files"].items():
        path = stage / "source" / relative
        require(manifest["files"][relative]["sha256"] == expected, "foreign manifest/V4 source disagreement")
        require(sha(path) == expected, f"foreign V4 source drift before import: {relative}")
        bound[relative] = ref(path)
    relative = "benchmarks/regression/framework_compare.py"
    helper = load_module(stage / "source" / relative, inventory["files"][relative], "public_assembly_foreign_v4")
    return helper, {"manifest": ref(stage / "manifest.json"), "execution_source_inventory": ref(inventory_path),
                    "files": bound, "current_catalog_substitution_permitted": False,
                    "scope": "Exact original foreign V4 preparation/capture source, separate from the requested main source stage."}


def validate_omni_dependency_closure(root):
    """Replay copied CPU closure facts without rewriting prior execution catalogs."""
    inventory_path = root / "copied_native_evidence.json"
    require(sha(inventory_path) == OMNI_CLOSURE_COPY_SHA, "Omni closure copied inventory changed")
    inventory = read(inventory_path)
    require(inventory["status"] == "copied" and len(inventory["files"]) == 30, "Omni closure copy incomplete")
    directory = root / "native_evidence"
    require({p.name for p in directory.iterdir()} == set(inventory["files"]), "Omni immediate evidence coverage differs")
    for name, expected in inventory["files"].items():
        require(Path(name).name == name, "Omni closure path escape")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "Omni closure evidence is not a regular copied file")
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], f"Omni closure evidence changed: {name}")
    completion_path = directory / "completion_v2.json"
    require(sha(completion_path) == OMNI_CLOSURE_SHA, "Omni closure completion changed")
    completion = read(completion_path)
    require(completion["status"] == "passed" and completion["pip_check"] is True and completion["uv_pip_check"] is True,
            "Omni dependency checks did not pass")
    require(all(completion[k] is False for k in ("GPU_initialized", "model_constructed", "weights_accessed")),
            "Omni CPU closure scope differs")
    require(completion["original_source_unchanged"] is True and completion["preserved_native_source_files"] == 2305
            and completion["preserved_vendor_payload_members"] == 767, "Omni source preservation gate differs")
    before, after = read(directory / "packages_before.json"), completion["packages"]
    require(len(after) == completion["package_count"] == 268, "Omni package inventory count differs")
    added = {name: row["version"] for name, row in after.items() if name not in before}
    require(added == {"iopath": "0.1.10", "portalocker": "4.3.0"}, "Omni added package set differs")
    require(set(before) <= set(after) and all(after[name]["version"] == row["version"] for name, row in before.items()),
            "Omni existing package versions changed")
    metadata_changes = {name: {"before": row["metadata_sha256"], "after": after[name]["metadata_sha256"]}
                        for name, row in before.items() if row != after[name]}
    require(set(metadata_changes) == {"cosmos-framework"}, "Omni metadata changed outside the declared vendor wheel")
    guard_path = directory / "native_symbol_cpu_guard_v2.receipt.json"
    require(sha(guard_path) == completion["guard_receipt_sha256"], "Omni symbol guard hash differs")
    guard = read(guard_path)
    require(guard["status"] == "passed" and guard["symbol"] == "ActionTransformPipeline"
            and guard["blocked_operations"] == [] and guard["gpu_inference"] is False
            and guard["model_constructed"] is False, "Omni actual native symbol guard failed")
    for name in ("pip_check.json", "uv_pip_check.json"):
        require(read(directory / name)["returncode"] == 0, "Omni actual checker exit was nonzero")
    require(sha(directory / "wheel_gate.json") == completion["wheel_gate_sha256"], "Omni vendor wheel gate changed")
    require(sha(directory / "bootstrap_framework_compare_v2.py") == completion["bootstrap_source_sha256"], "Omni closure bootstrap source differs")
    require(sha(directory / "failure.json") == completion["first_cpu_guard_failure_preserved"]["sha256"], "Omni earlier failure not preserved")
    return {"status": "passed", "completion": ref(completion_path), "copied_inventory": ref(inventory_path),
            "added_package_versions": added, "existing_package_version_changes": [],
            "metadata_changes": metadata_changes, "actual_native_symbol_guard": ref(guard_path),
            "preserved_native_source_files": 2305, "preserved_vendor_payload_members": 767,
            "gpu_capture_qualified": False, "old_prepared_plan_or_catalog_modified": False,
            "scope": "Replayed hash-bound CPU closure only. The actual foreign captures retain their original package/catalog fields and independent process gates."}


def assemble(stage, qualification, output, foreign_stage=FOREIGN_STAGE, selection=None, selection_sha256=None):
    require(sha(Path(__file__).with_name("assemble_v3.py")) == V3_ASSEMBLER_SHA, "preserved V3 assembler changed")
    selected_roots, selection_binding = load_selection(qualification, selection, selection_sha256)
    require(sha(Path(__file__).with_name("assemble_v2.py")) == V2_ASSEMBLER_SHA, "preserved V2 assembler source changed")
    require(sha(Path(__file__).with_name("assemble.py")) == PARENT_ASSEMBLER_SHA, "preserved V1 assembler source changed")
    require(not output.exists(), "output already exists; select a new snapshot path")
    output.mkdir(parents=True)
    source = stage / "source"
    manifest = read(stage / "manifest.json")
    modules = {}
    for name in ("user_report", "user_e2e"):
        relative = f"benchmarks/regression/{name}.py"
        modules[name] = load_module(source / relative, manifest["files"][relative]["sha256"], f"public_assembly_{name}")
    foreign_helper, foreign_origin = load_foreign_sources(foreign_stage)
    modules["framework_compare"] = foreign_helper
    dependency_closure = guarded(validate_omni_dependency_closure, STUDY / "omni_missing_dependencies_v1")
    main, foreign = {}, {}
    for family in FAMILIES:
        root = selected_roots[family]
        run = root / "run"
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
                                modules["framework_compare"], foreign_stage / "source/benchmarks/regression/fixtures/recorded_inputs_v1.npz")
    require(len(foreign) == 6, "foreign six-cell catalog changed")
    correspondence = guarded(source_correspondence, stage, qualification, output, selected_roots, selection_binding)
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
    if dependency_closure["status"] == "failed":
        failures.append("omni_dependency_closure")
    if correspondence["status"] == "failed":
        failures.append("source_correspondence")
    inventory = {str(path.relative_to(qualification)): {"sha256": sha(path), "bytes": path.stat().st_size}
                 for path in sorted(qualification.rglob("*")) if path.is_file()}
    save(output / "copied_evidence_inventory.json", inventory)
    result = {"schema": "instinctflash.public_reproduction_evidence.v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "status": "failed_evidence" if failures or historical_failures else "complete_recorded_reproduction" if counts["main_passed"] == 8 and counts["foreign_passed"] == 6 and counts["historical_families_passed"] == 8 and correspondence["status"] == dependency_closure["status"] == "passed" else "partial",
              "assembler": ref(Path(__file__)), "stage_manifest": ref(stage / "manifest.json"),
              "assembler_parent": ref(Path(__file__).with_name("assemble_v3.py")),
              "assembler_v2": ref(Path(__file__).with_name("assemble_v2.py")),
              "qualification_selection": selection_binding,
              "selected_qualification_roots": {family: str(root) for family, root in sorted(selected_roots.items())},
              "assembler_v1": ref(Path(__file__).with_name("assemble.py")),
              "foreign_execution_source_origin": foreign_origin,
              "omni_additive_dependency_closure": dependency_closure,
              "historical_matrix": ref(REPO / "eval/user_e2e_2026-09-14/matrix_final.json"),
              "source_bindings": {name: ref(module.__file__) for name, module in modules.items()},
              "source_correspondence": correspondence,
              "copied_evidence_inventory": ref(output / "copied_evidence_inventory.json"), "counts": counts,
              "main": main, "foreign": foreign, "table": rows, "failures": failures, "historical_failures": historical_failures,
              "unsupported_foreign": catalog["unsupported"], "not_qualified_foreign": catalog["not_qualified"],
              "task_quality_certified": False,
              "limitations": [
                  "Foreign captures keep their exact original V4 runner/catalog/plans loaded from immutable stage V6; latest main source-stage metadata does not relabel those captures.",
                  "The separately bound Omni utility closure retains prior failure evidence and the exact added dependencies; its CPU import does not establish a successful GPU capture.",
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
    parser.add_argument("--stage", type=Path, default=Path("/home/ubuntu/ifl-public-full-stage-20260915-v6"))
    parser.add_argument("--foreign-stage", type=Path, default=FOREIGN_STAGE, help="Exact immutable V6 source bytes for original foreign V4 captures")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--qualification", type=Path, default=STUDY / "qualification")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.stage.resolve(), args.qualification.resolve(), args.output.resolve(), args.foreign_stage.resolve(), args.selection, args.selection_sha256)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "failures": result["failures"], "historical_failures": result["historical_failures"], "summary": str(args.output / "summary.json")}))
    return 1 if result["status"] == "failed_evidence" else 0


if __name__ == "__main__":
    raise SystemExit(main())

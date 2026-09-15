"""Qualify the declared current pipeline; report historical equivalence separately.

CPU/read-only evidence replay except create-only selection and report outputs.
The strict historical reproduction assemblers are imported by hash, never edited.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
CONTRACT_SHA = "1086a7e236219f6b8a073bee9e695d870981877cd52dda8646d75dace0146f0d"
ADDENDUM_SHA = "e751698fc99d3d4771a2940b02b33a0f1b99a267420ad89ea671a86dace7458a"
DREAMZERO_QUEUE_SHA = "9c86963b0adbd4b92f0465e04ca27c7b0775e7fdeb68517a4983d80500b18247"
FAMILIES = ("pi05", "groot", "vla4", "vla2", "va", "edge", "nano", "dreamzero")
CLAIM_LIMITS = {"complete_historical_reproduction": False, "task_quality_certified": False,
                "prior_cosmos_SCREEN_transferred": False, "diagnostic_latency_used": False,
                "foreign_operating_points_uniformly_matched": False}
REQUIRED_COSMOS = {"run/run.json", "run/plan.json", "run/matrix.json", "run/preparation.json",
                   "run/profiles.json", "run/validated_report/report.json",
                   "serving/receipt.json", "serving/serve.json"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate JSON key")
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


def checked(value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "invalid bound reference")
    path = Path(value["path"])
    require(path.is_absolute() and sha(path) == value["sha256"], "bound input changed")
    return path


def load_module(value, name):
    path = checked(value)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_roots(contract):
    require(contract["families"] == list(FAMILIES), "family coverage or order changed")
    expected = {family: family + "/pytorch_triton_v1" if family in {"edge", "nano"} else family
                for family in FAMILIES}
    require(contract["family_roots"] == expected, "prospective family roots changed")
    qualification = Path(contract["qualification_root"])
    roots = {family: qualification / relative for family, relative in expected.items()}
    for root in roots.values():
        require(root.resolve() == root.absolute(), "selected root redirects through symlinks")
    return roots


def load_contract(path, expected):
    require(expected == CONTRACT_SHA and sha(path) == expected, "prospective contract changed")
    contract = read(path)
    require(contract["schema"] == "instinctflash.current_public_pipeline_contract.v1"
            and contract["prospective"] is True, "invalid current-pipeline contract")
    require(contract["claim_limits"] == CLAIM_LIMITS, "claim scope changed")
    fixed_roots(contract)
    require(Path(contract["qualification_root"]).resolve() == (STUDY / "qualification").resolve(),
            "qualification root changed")
    for key in ("main_stage", "queue", "compiler_environment_preparation", "negative_compiler_diagnostic",
                "negative_diagnostic_independent_audit", "bounded_no_cause_review", "historical_matrix",
                "preserved_original_inventory"):
        checked(contract[key])
    for value in contract["validator_sources"].values():
        checked(value)
    diagnostic = read(checked(contract["negative_compiler_diagnostic"]))
    require(diagnostic["status"] == "diagnostic_comparison_complete"
            and diagnostic["historical_action_bytes_recovered_on_tested_inputs"] is False
            and diagnostic["actions"]["corrected_vs_historical"]["exact_bytes"] is False,
            "preserved negative historical result changed")
    require(read(checked(contract["negative_diagnostic_independent_audit"]))["status"] == "passed_evidence_replay",
            "negative diagnostic evidence audit failed")
    environment = read(checked(contract["compiler_environment_preparation"]))
    require(environment["status"] == "prepared" and environment["pip_check_exit_code"] == 0
            and environment["destination"] + "/bin/python" == contract["corrected_python"]
            and environment["wheel_sha256"] == "58d57d6796b0004076315433526fe9d4af42044d430afdee1e6cd42a76bd6d09"
            and environment["original_compiler_payload_unchanged"] is True,
            "complete compiler installation binding differs")
    helper = load_module(contract["validator_sources"]["assemble_v4"], "current_pipeline_frozen_v4")
    helper.preserve_original_evidence(Path(contract["qualification_root"]), contract["preserved_original_inventory"])
    return contract, helper


def source_addendum():
    path = HERE / "source_addendum_v5.json"
    require(sha(path) == ADDENDUM_SHA, "source-stage addendum changed")
    result = read(path)
    require(result["contract"]["sha256"] == CONTRACT_SHA, "addendum contract differs")
    checked(result["DreamZero_ptxas_probe"])
    checked(result["normal_queue_source"])
    checked(result["DreamZero_queue_config"])
    previous = result["preserved_DreamZero_V5_failure"]
    root = Path(previous["root"])
    paths = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    require(all(not p.is_symlink() and p.resolve() == p.absolute() for p in paths),
            "preserved DreamZeroV5 evidence contains a symlink")
    require({str(p.relative_to(root)): {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in paths} == previous["files"], "preserved DreamZeroV5 failure changed")
    return result


def validate_stage(contract, candidate):
    """Allow only the prospective installer/docs delta; all implementation stays exact."""
    original = read(checked(contract["main_stage"]))
    path = checked(candidate)
    current = read(path)
    require(current["status"] == "wheels_built_and_inspected_CPU_only"
            and set(current["files"]) == set(original["files"]), "source-stage coverage changed")
    changed = {name for name in original["files"]
               if original["files"][name]["sha256"] != current["files"][name]["sha256"]}
    require(changed <= set(source_addendum()["allowed_changed_source_paths"]),
            "source changed outside the prospective installer/docs scope")
    document = source_addendum()["SDE1_documentation_correction"]
    if document["path"] in changed:
        require(current["files"][document["path"]]["sha256"] == document["sha256"],
                "SDE1 documentation differs from the exact authorized correction")
    for name, entry in current["files"].items():
        source = path.parent / "source" / name
        require(source.resolve() == source.absolute() and sha(source) == entry["sha256"],
                "actual staged source differs from bound manifest")
    return {"manifest": candidate, "baseline": contract["main_stage"],
            "source_addendum": ref(HERE / "source_addendum_v5.json"), "changed_source_paths": sorted(changed)}


def validate_dreamzero_compiler(contract, completion_ref):
    addendum = source_addendum()
    probe = read(checked(addendum["DreamZero_ptxas_probe"]))
    queue_path = STUDY / "dreamzero_memory_preparation_v1/dreamzero_paired_serving_queue_v6.json"
    require(sha(queue_path) == DREAMZERO_QUEUE_SHA, "DreamZero prospective queue changed")
    config, completion = read(queue_path), read(checked(completion_ref))
    require(probe["status"] == "passed" and probe["returncode"] == 0
            and probe["GPU_used"] is False and probe["target"] == "sm_110a", "PTXAS CPU probe failed")
    expected_env = {key: probe["ptxas"] for key in ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH")}
    require(probe["environment"] == expected_env, "PTXAS probe activation differs")
    memory_env = addendum["DreamZero_compiler_memory_environment"]
    require(memory_env["TORCHINDUCTOR_COMPILE_THREADS"] == "1", "DreamZero compile-thread limit differs")
    require(completion["status"] == "passed" and len(completion["jobs"]) == len(config["jobs"]) == 2,
            "DreamZero normal queue is incomplete")
    root = fixed_roots(contract)["dreamzero"]
    endpoints = [root / "run/run.json", root / "serving/receipt.json"]
    for declared, actual, endpoint in zip(config["jobs"], completion["jobs"], endpoints):
        command = [declared["python"], "-I", "-m", declared["module"], *declared["arguments"]]
        require(actual["id"] == declared["id"] and actual["command"] == command
                and type(actual["returncode"]) is int and actual["returncode"] == 0
                and actual["receipt_status"] == "passed", "DreamZero actual process differs")
        require(actual["activation"] == declared["activation"], "DreamZero activation files differ")
        require(all(actual["applied_compiler_environment"].get(k) == v for k, v in expected_env.items())
                and actual["assembler_sha256"] == {k: probe["sha256"] for k in expected_env},
                "DreamZero applied compiler identity differs")
        require({key: declared["environment"][key] for key in memory_env} == memory_env,
                "DreamZero declared compiler memory environment differs")
        require(actual["applied_compiler_environment"] == {**expected_env, **memory_env},
                "DreamZero actual compiler memory environment differs")
        require(actual["started_unix"] < actual["completed_unix"]
                and read(endpoint)["interpreter"] == declared["python"], "DreamZero process/endpoint differs")
    return {"status": "passed", "CPU_probe": addendum["DreamZero_ptxas_probe"],
            "queue_source": addendum["normal_queue_source"], "queue_config": ref(queue_path),
            "queue_completion": completion_ref, "applied_environment": {**expected_env, **memory_env},
            "cache_scope": addendum["DreamZero_cache_scope"],
            "preserved_V5_failure": addendum["preserved_DreamZero_V5_failure"],
            "ptxas_sha256": probe["sha256"]}


def snapshot_inputs(contract):
    """Hash only the fixed run/serving and six foreign evidence trees; copy none."""
    qualification = Path(contract["qualification_root"])
    directories = [root / name for family, root in fixed_roots(contract).items()
                   for name in ["run", *contract["serving_directories"][family]]]
    directories += [qualification / "foreign" / name for name in contract["foreign_ids"]]
    result, total = {}, 0
    for directory in directories:
        require(directory.resolve() == directory.absolute(), "evidence directory redirects through symlinks")
        if not directory.exists():
            continue
        require(directory.is_dir(), "evidence root is not a directory")
        for path in sorted(directory.rglob("*")):
            require(not path.is_symlink(), "symlink in selected evidence")
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            size = path.stat().st_size
            total += size
            require(size <= 256 << 20 and total <= 2 << 30 and len(result) < 20000,
                    "selected evidence exceeds bounded metadata/action snapshot")
            relative = str(path.relative_to(qualification))
            require(relative not in result, "overlapping evidence selection")
            result[relative] = {"sha256": sha(path), "bytes": size}
    return result


def require_new_cosmos(contract, snapshot):
    for family in ("edge", "nano"):
        root = fixed_roots(contract)[family]
        prefix = contract["family_roots"][family] + "/"
        require(all(prefix + name in snapshot for name in REQUIRED_COSMOS),
                "new Cosmos paired/serving evidence is not complete")
        run, serving = read(root / "run/run.json"), read(root / "serving/receipt.json")
        require(run["status"] == run["report_status"] == serving["status"] == "passed",
                "new Cosmos normal pipeline has not passed")
        require(run["interpreter"] == serving["interpreter"] == contract["corrected_python"],
                "new Cosmos evidence used another interpreter")


def foreign_resolution_context(contract):
    """Consume the actual Thor strict-resolution proof; never inspect remote paths locally."""
    addendum = source_addendum()
    spec = addendum["foreign_checkpoint_resolution"]
    proof = read(checked(spec["proof"]))
    request = read(checked(spec["request"]))
    collector = checked(spec["collector"])
    original = checked(spec["original_runner"])
    resolver = next(node for node in ast.parse(original.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "checkpoint_cache_paths")
    require(hashlib.sha256(ast.dump(resolver, include_attributes=False).encode()).hexdigest()
            == spec["original_resolver_function_ast_sha256"], "original strict resolver differs")
    require(proof["schema"] == "instinctflash.foreign_checkpoint_resolution.v1"
            and proof["status"] == "passed_actual_remote_strict_resolution"
            and request["schema"] == "instinctflash.foreign_checkpoint_resolution_request.v1",
            "actual strict-resolution proof required")
    require(proof["executed_script_sha256"] == sha(collector)
            and proof["request_sha256"] == spec["request"]["sha256"]
            and proof["original_runner"] == request["original_runner"] == spec["original_runner"],
            "strict-resolution collector/request/source binding differs")
    require(proof["operator"] == "str(Path(recorded_checkpoint_path).resolve(strict=True))"
            and all(proof[key] is False for key in ("GPU_queried", "weights_read", "files_mutated")),
            "strict-resolution operation or scope differs")
    expected = [name for name in contract["foreign_ids"] if name.startswith("vllm-omni-")]
    require(len(expected) == 3 and [row["cell"] for row in proof["rows"]]
            == [row["cell"] for row in request["rows"]] == expected, "strict-resolution cell coverage differs")
    for row, declared in zip(proof["rows"], request["rows"]):
        require({key: row[key] for key in declared} == declared, "strict-resolution request row differs")
        cell = row["cell"]
        local = Path(contract["qualification_root"]) / "foreign" / cell
        prepared = STUDY / "framework_compare_v1/prepared_assets_v2" / cell / "preparation.json"
        refs = {"preparation": ref(prepared), "capture": ref(local / "capture.json"),
                "completion": ref(local / "completion.json")}
        require(row["raw_evidence"] == refs, "strict-resolution raw evidence changed")
        values = {key: read(checked(value)) for key, value in refs.items()}
        value = row["resolved_checkpoint_path"]
        path = Path(value)
        require(path.is_absolute() and str(path) == value and ".." not in path.parts
                and not any(char in value for char in "\x00\r\n"), "noncanonical recorded checkpoint path")
        require(row["is_directory"] is True and row["prepared_checkpoint_path"] == value
                == values["preparation"]["assets"]["checkpoint"]["path"]
                and values["capture"]["startup_checkpoint_paths"]
                == values["completion"]["startup_checkpoint_paths"] == [value],
                "actual strict-resolution three-way path equality differs")
    preserved = addendum["preserved_V4_relocation_failure"]
    for value in preserved.values():
        checked(value)
    prior = read(checked(preserved["snapshot_v4/summary.json"]))
    require(prior["status"] == "partial_current_pipeline" and prior["counts"]["main_passed"] == 8
            and prior["counts"]["foreign_passed"] == 3, "preserved V4 relocation failure differs")
    return {"status": "passed_actual_remote_strict_resolution_replay", **spec, "rows": proof["rows"],
            "preserved_V4_relocation_failure": preserved, "new_remote_or_weight_reads": False}


class ForeignResolutionAdapter:
    """Delegate the frozen helper unchanged except its externally proven path result."""
    def __init__(self, original, context):
        require(ref(original.__file__) == context["original_runner"], "foreign helper source differs")
        self._original = original
        self._rows = {row["cell"]: row for row in context["rows"]}

    def __getattr__(self, name):
        return getattr(self._original, name)

    def checkpoint_cache_paths(self, cell, prepared):
        if cell["framework"] != "vllm-omni":
            return self._original.checkpoint_cache_paths(cell, prepared)
        require(cell["id"] in self._rows, "unproven foreign checkpoint cell")
        row = self._rows[cell["id"]]
        require(prepared == read(checked(row["raw_evidence"]["preparation"])),
                "foreign preparation differs from strict-resolution proof")
        return [row["resolved_checkpoint_path"]]


def bind(contract_path, contract_sha256, output, stage_ref, dreamzero_completion):
    require(output.resolve().is_relative_to(HERE) and not output.exists(), "selection output must be new inside this study")
    contract, _ = load_contract(contract_path, contract_sha256)
    validate_stage(contract, stage_ref)
    snapshot = snapshot_inputs(contract)
    require_new_cosmos(contract, snapshot)
    validate_dreamzero_compiler(contract, dreamzero_completion)
    resolution = foreign_resolution_context(contract)
    selection = {"schema": "instinctflash.current_public_pipeline_selection.v1", "template_only": False,
                 "contract": ref(contract_path), "source_stage_manifest": stage_ref,
                 "source_addendum": ref(HERE / "source_addendum_v5.json"),
                 "dreamzero_queue_completion": dreamzero_completion,
                 "foreign_checkpoint_resolution": resolution["proof"],
                 "qualification_root": contract["qualification_root"], "family_roots": contract["family_roots"],
                 "snapshot": snapshot}
    require(snapshot_inputs(contract) == snapshot, "evidence changed while binding")
    save(output, selection)
    return selection


def validate_selection(contract, contract_path, selection_path, expected):
    require(isinstance(expected, str) and len(expected) == 64 and sha(selection_path) == expected,
            "actual selection hash differs")
    selection = read(selection_path)
    require(set(selection) == {"schema", "template_only", "contract", "source_stage_manifest",
                               "source_addendum", "dreamzero_queue_completion", "foreign_checkpoint_resolution",
                               "qualification_root", "family_roots", "snapshot"}, "selection fields differ")
    require(selection["schema"] == "instinctflash.current_public_pipeline_selection.v1"
            and selection["template_only"] is False, "actual selection required")
    require(selection["contract"] == ref(contract_path)
            and selection["source_addendum"] == ref(HERE / "source_addendum_v5.json")
            and selection["qualification_root"] == contract["qualification_root"]
            and selection["family_roots"] == contract["family_roots"], "selection identity differs")
    require(selection["snapshot"] == snapshot_inputs(contract), "selected evidence changed after binding")
    require_new_cosmos(contract, selection["snapshot"])
    validate_stage(contract, selection["source_stage_manifest"])
    validate_dreamzero_compiler(contract, selection["dreamzero_queue_completion"])
    require(selection["foreign_checkpoint_resolution"] == foreign_resolution_context(contract)["proof"],
            "selection strict-resolution proof differs")
    return selection


def published_selected_cell(family):
    mapping = source_addendum()["published_selected_cell_ids"]
    require(set(mapping) == set(FAMILIES), "published family selection coverage differs")
    return mapping[family]


def expected_paired_cells(family):
    require(family in FAMILIES, "unknown paired family")
    return [f"{family}-{arm}" for arm in ("eager_native", "runtime_default", "runtime_selected")] + list(
        source_addendum()["required_operating_point_ids"].get(family, []))


def expected_serving_cells(family):
    return {published_selected_cell(family)} | ({"va-2v4a-fp8"} if family == "va" else set())


def validate_requested_plan(family, plan):
    if family == "dreamzero":
        original = read(checked(source_addendum()["DreamZero_requested_plan"]))
        require(plan == original and plan["execution_mode"] == "dynamic-fp8",
                "DreamZero requested four-arm plan differs")
        require(plan["matrix"]["cells"][-1]["id"] == published_selected_cell(family),
                "DreamZero published selection differs from requested operating point")


def validate_main_current(run, family, validator, source, output):
    plan, matrix = read(run / "plan.json"), read(run / "matrix.json")
    execution, prepared = read(run / "run.json"), read(run / "preparation.json")
    require(execution["status"] == execution["report_status"] == "passed", "native run did not pass")
    require(execution["task_quality_validated"] is False, "unexpected task-quality promotion")
    expected = expected_paired_cells(family)
    validate_requested_plan(family, plan)
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


def assess_dreamzero_historical(helper, root, selection_ref, source):
    """Compare three unchanged historical routes; label the new native control explicitly."""
    import numpy as np

    run = root / "run"
    matrix, execution = read(run / "matrix.json"), read(run / "run.json")
    validate_requested_plan("dreamzero", read(run / "plan.json"))
    require([cell["id"] for cell in matrix["cells"]] == expected_paired_cells("dreamzero"),
            "historical assessment four-arm coverage differs")
    require(execution["status"] == execution["report_status"] == "passed"
            and [row["cell"] for row in execution["attempts"]] == expected_paired_cells("dreamzero")
            and all(type(row["exit_code"]) is int and row["exit_code"] == 0 for row in execution["attempts"]),
            "DreamZero normal four-arm process did not pass")
    policy = source_addendum()["DreamZero_historical_comparability"]
    rows = []
    for cell in matrix["cells"]:
        current_path = run / "cells" / cell["id"] / "receipt.json"
        old_path = helper.historical_receipt(helper.REPO / "eval/user_e2e_2026-09-14", cell)
        current, old = read(current_path), read(old_path)
        require(current["ok"] is True and current["cell_id"] == cell["id"], "current DreamZero receipt differs")
        for key in ("cases", "model_id", "revision", "effective_schedule"):
            require(current[key] == old[key], "historical DreamZero request contract differs: " + key)
        comparable = cell["id"] in policy["same_ID_comparable"]
        if comparable:
            require(current["precision"] == old["precision"], "comparable historical precision differs")
        else:
            require(cell["id"] == policy["new_native_control"]
                    and current["precision"] == policy["new_same_ID_precision"]
                    and old["precision"] == policy["old_same_ID_precision"], "native-control historical scope differs")
        a_path, b_path = current_path.with_suffix(".npz"), old_path.with_suffix(".npz")
        a_hash, b_hash = current["actions_sha256"], old["actions_sha256"]
        fresh, previous = helper.arrays(a_path, a_hash), helper.arrays(b_path, b_hash)
        require(set(fresh) == set(previous) == {"actions"}, "historical DreamZero action keys differ")
        a, b = fresh["actions"], previous["actions"]
        same_shape, same_dtype = a.shape == b.shape, a.dtype == b.dtype
        exact = same_shape and same_dtype and a.tobytes() == b.tobytes() if comparable else None
        maximum = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if comparable and same_shape else None
        rows.append({"cell": cell["id"], "file": "receipt.npz", "shape": list(a.shape), "dtype": str(a.dtype),
                     "historical_shape": list(b.shape), "historical_dtype": str(b.dtype),
                     "same_shape": same_shape, "same_dtype": same_dtype,
                     "historically_comparable": comparable,
                     "historical_comparability": "same_declared_route" if comparable else "not_comparable_native_control_vs_old_FP8",
                     "current_precision": current["precision"], "historical_precision": old["precision"],
                     "historical_actions_match_bytes": exact, "max_abs": maximum,
                     "new_receipt": ref(current_path), "historical_receipt": ref(old_path),
                     "new_file": str(a_path.relative_to(helper.REPO)), "new_file_sha256": a_hash,
                     "historical_file": str(b_path.relative_to(helper.REPO)), "historical_file_sha256": b_hash,
                     "historical_matrix": ref(helper.REPO / "eval/user_e2e_2026-09-14/matrix_final.json"),
                     "comparison_source_sha256": sha(Path(__file__)), "assembler_source_sha256": sha(source),
                     "qualification_selection": selection_ref,
                     "scope": "Exact own-route bytes for comparable cells; the added native control remains under the normal paired BITEXACT gate."})
    comparable_rows = [row for row in rows if row["historically_comparable"]]
    require(len(comparable_rows) == 3 and len(rows) == 4, "historical comparability coverage differs")
    all_comparable_exact = all(row["historical_actions_match_bytes"] for row in comparable_rows)
    return {"status": "historical_routes_partially_comparable" if all_comparable_exact else "failed_historical_equivalence",
            "assessment_complete": True, "exact_action_equivalence": False,
            "comparable_routes_exact": all_comparable_exact, "comparable_cells": 3, "not_comparable_cells": 1,
            "arrays": rows, "scope": policy["scope"]}


def assess_historical(helper, comparator, family, root, selection_ref, source):
    if family == "dreamzero":
        return assess_dreamzero_historical(helper, root, selection_ref, source)
    # Retain the strict queue request-identity check alongside the frozen comparator.
    for cell in read(root / "run/matrix.json")["cells"]:
        current = read(root / "run/cells" / cell["id"] / "receipt.json")
        old = read(helper.historical_receipt(helper.REPO / "eval/user_e2e_2026-09-14", cell))
        if "queue_drain" in current:
            require(current["queue_drain"]["cases"] == old["queue_drain"]["cases"],
                    "historical queue request identity changed")
    rows = comparator.compare(helper, family, root, {"selection": selection_ref}, source)
    equivalent = all(row["historical_actions_match_bytes"] for row in rows)
    return {"status": "passed" if equivalent else "failed_historical_equivalence",
            "assessment_complete": True, "exact_action_equivalence": equivalent, "arrays": rows,
            "scope": "Each current arm versus its own frozen historical arm, exact bytes without tolerance changes."}


def summarize(main, foreign, correspondence, closure):
    counts = {"main_expected": 8, "main_passed": sum(x["status"] == "passed" for x in main.values()),
              "foreign_expected": 6, "foreign_passed": sum(x["status"] == "passed" for x in foreign.values()),
              "historical_families_assessed": sum(x["historical_equivalence"].get("assessment_complete", False)
                                                  for x in main.values()),
              "historical_families_equivalent": sum(x["historical_equivalence"].get("exact_action_equivalence", False)
                                                    for x in main.values())}
    failures = ["main/" + name for name, row in main.items() if row["status"] == "failed"]
    failures += ["foreign/" + name for name, row in foreign.items() if row["status"] == "failed"]
    failures += ["historical_inputs/" + name for name, row in main.items()
                 if row["historical_equivalence"]["status"] == "failed"]
    for label, row in (("source_correspondence", correspondence), ("omni_dependency_closure", closure)):
        if row["status"] == "failed":
            failures.append(label)
    complete = (counts["main_passed"] == 8 and counts["foreign_passed"] == 6
                and counts["historical_families_assessed"] == 8
                and correspondence["status"] == closure["status"] == "passed")
    status = "failed_current_evidence" if failures else "current_pipeline_qualified" if complete else "partial_current_pipeline"
    return status, counts, failures


def table_rows(main, foreign, catalog, helper):
    """Keep measured operating points distinct from comparison references."""
    rows = []
    for family in FAMILIES:
        cells = {row["id"]: row for row in main[family]["paired"].get("cells", [])}
        names = [family] + (["va-2v4a-fp8"] if family == "va" else [])
        for name in names:
            selected = published_selected_cell(family) if name == family else name
            row = {"model": name, "published_cell": selected, "current_pipeline_status": main[family]["status"],
                   "historical_equivalence": main[family]["historical_equivalence"]["status"],
                   "native_p50_ms": cells.get(family + "-eager_native", {}).get("p50_ms"),
                   "runtime_default_p50_ms": cells.get(family + "-runtime_default", {}).get("p50_ms"),
                   "InstinctFlash_p50_ms": cells.get(selected, {}).get("p50_ms"),
                   "comparison_reference_native_p50_ms": None,
                   "comparison_reference_default_p50_ms": None,
                   "comparison_reference_scope": None}
            if name == "va-2v4a-fp8":
                row["comparison_reference_native_p50_ms"] = row["native_p50_ms"]
                row["comparison_reference_default_p50_ms"] = row["runtime_default_p50_ms"]
                row["comparison_reference_scope"] = "Full VA 25V50A; not a native/default 2V4A measurement"
                row["native_p50_ms"] = row["runtime_default_p50_ms"] = None
            for framework, column in (("lerobot", "LeRobot"), ("vllm-omni", "vLLM_Omni")):
                if family == name == "va" and framework == "lerobot":
                    row[column + "_status"] = "unmeasured"
                    row[column + "_p50_ms"] = None
                    continue
                row[column + "_status"] = helper.foreign_status(catalog, foreign, framework, family)
                matching = [f for f in foreign.values() if f["status"] == "passed"
                            and f["family"] == family and f["framework"] == framework]
                row[column + "_p50_ms"] = matching[0]["latency"]["p50_ms"] if matching else None
            rows.append(row)
    return rows


def assemble(contract_path, contract_sha256, selection_path, selection_sha256, output):
    contract, helper = load_contract(contract_path, contract_sha256)
    selection = validate_selection(contract, contract_path, selection_path, selection_sha256)
    require(not output.exists() and not output.is_symlink(), "output already exists")
    require(output.resolve().is_relative_to(HERE), "report output must be inside this additive study")
    output.mkdir(parents=True)
    roots = fixed_roots(contract)
    qualification = Path(contract["qualification_root"])
    stage = checked(selection["source_stage_manifest"]).parent
    manifest = read(stage / "manifest.json")
    source = stage / "source"
    modules = {name: helper.load_module(source / (rel := f"benchmarks/regression/{name}.py"),
                                      manifest["files"][rel]["sha256"], "current_pipeline_" + name)
               for name in ("user_report", "user_e2e")}
    comparator = load_module(contract["validator_sources"]["historical_comparator_v2"], "current_pipeline_historical_v2")
    foreign_helper, foreign_origin = helper.load_foreign_sources(helper.FOREIGN_STAGE)
    resolution = foreign_resolution_context(contract)
    foreign_helper = ForeignResolutionAdapter(foreign_helper, resolution)
    closure = helper.guarded(helper.validate_omni_dependency_closure, STUDY / "omni_missing_dependencies_v1")
    main, foreign = {}, {}
    for family, root in roots.items():
        run = root / "run"
        paired = helper.guarded(validate_main_current, run, family, modules["user_report"], source, output)
        serving = [helper.guarded(helper.validate_serving, root / name, run, modules["user_e2e"])
                   for name in contract["serving_directories"][family]]
        expected = expected_serving_cells(family)
        covered = {row.get("cell") for row in serving if row["status"] == "passed"}
        serving_status = ("failed" if any(row["status"] == "failed" for row in serving)
                          else "passed" if covered == expected and len(covered) == len(serving) else "pending")
        historical = helper.guarded(assess_historical, helper, comparator, family, root,
                                    ref(selection_path), checked(contract["validator_sources"]["assemble_v4"]))
        main[family] = {"paired": paired, "serving": {"status": serving_status, "results": serving},
                        "historical_equivalence": historical,
                        "status": "passed" if paired["status"] == serving_status == "passed"
                        else "failed" if "failed" in (paired["status"], serving_status) else "pending"}
    catalog = foreign_helper.catalog()
    require([cell["id"] for cell in catalog["cells"]] == contract["foreign_ids"], "foreign selection changed")
    for name in contract["foreign_ids"]:
        foreign[name] = helper.guarded(helper.validate_foreign, name, qualification / "foreign" / name,
                                      STUDY / "framework_compare_v1/prepared_assets_v2" / name, foreign_helper,
                                      helper.FOREIGN_STAGE / "source/benchmarks/regression/fixtures/recorded_inputs_v1.npz")
    selection_binding = {"mode": "current_public_pipeline_fixed_prospective_roots", "selection": ref(selection_path)}
    correspondence = helper.guarded(helper.source_correspondence, stage, qualification, output, roots, selection_binding)
    if correspondence["status"] == "passed":
        expected_cells = {cell["id"] for row in main.values() for cell in row["paired"].get("cells", [])}
        if set(correspondence["covered_cells"]) != expected_cells:
            correspondence = {"status": "failed", "reason": "source audit missed or added selected paired cells",
                              "audit": correspondence}
    require(snapshot_inputs(contract) == selection["snapshot"], "evidence changed during CPU replay")
    helper.preserve_original_evidence(qualification, contract["preserved_original_inventory"])
    rows = table_rows(main, foreign, catalog, helper)
    status, counts, failures = summarize(main, foreign, correspondence, closure)
    result = {"schema": "instinctflash.current_public_pipeline_qualification.v1", "status": status,
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "assembler": ref(__file__),
              "contract": ref(contract_path), "selection": ref(selection_path),
              "source_stage_manifest": selection["source_stage_manifest"],
              "source_stage_admission": validate_stage(contract, selection["source_stage_manifest"]),
              "DreamZero_compiler_activation": validate_dreamzero_compiler(contract, selection["dreamzero_queue_completion"]),
              "selected_qualification_roots": {k: str(v) for k, v in roots.items()},
              "published_selected_cell_ids": source_addendum()["published_selected_cell_ids"],
              "counts": counts, "failures": failures, "main": main, "foreign": foreign, "table": rows,
              "source_correspondence": correspondence, "omni_dependency_closure": closure,
              "foreign_execution_source_origin": foreign_origin, "historical_matrix": contract["historical_matrix"],
              "foreign_checkpoint_resolution": resolution,
              "historical_equivalence": {family: row["historical_equivalence"]["status"] for family, row in main.items()},
              "preserved_prior_nano_diagnostic": {"status": "failed_historical_equivalence",
                  "comparison": contract["negative_compiler_diagnostic"],
                  "independent_audit": contract["negative_diagnostic_independent_audit"],
                  "no_established_cause": contract["bounded_no_cause_review"]},
              **CLAIM_LIMITS,
              "limitations": ["This qualifies only the declared current public pipeline, not complete historical reproduction.",
                              "Historical numerical drift is retained per arm without modified thresholds or task-quality transfer.",
                              "Foreign columns retain their original pinned sources and differing operating points; no uniform matched-compute claim.",
                              "WS checks are six calls/two episodes with close proof, not a latency benchmark or task test.",
                              "No GPU, remote checkpoint rehash, publication or capture was performed by this assembler."]}
    save(output / "summary.json", result)
    with (output / "table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("bind", "assemble"))
    parser.add_argument("--contract", type=Path, default=HERE / "contract_v1.json")
    parser.add_argument("--contract-sha256", default=CONTRACT_SHA)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--stage", type=Path, help="Reviewed later source stage (bind only)")
    parser.add_argument("--stage-manifest-sha256")
    parser.add_argument("--dreamzero-queue-completion", type=Path)
    parser.add_argument("--dreamzero-queue-completion-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "bind":
        require(args.selection is None and args.selection_sha256 is None, "bind takes no existing selection")
        require(args.stage is not None and args.stage_manifest_sha256 is not None
                and args.dreamzero_queue_completion is not None
                and args.dreamzero_queue_completion_sha256 is not None, "bind requires actual stage and completed DreamZero queue")
        bind(args.contract, args.contract_sha256, args.output,
             {"path": str((args.stage / "manifest.json").resolve()), "sha256": args.stage_manifest_sha256},
             {"path": str(args.dreamzero_queue_completion.resolve()), "sha256": args.dreamzero_queue_completion_sha256})
        print(json.dumps({"selection": ref(args.output)}))
        return 0
    require(args.selection is not None and args.selection_sha256 is not None, "assemble requires an exact actual selection")
    require(all(value is None for value in (args.stage, args.stage_manifest_sha256,
                                            args.dreamzero_queue_completion,
                                            args.dreamzero_queue_completion_sha256)),
            "assemble takes stage/compiler facts only from its frozen selection")
    result = assemble(args.contract, args.contract_sha256, args.selection, args.selection_sha256, args.output)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "summary": ref(args.output / "summary.json")}))
    return 1 if result["status"] == "failed_current_evidence" else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Qualify the declared current pipeline; report historical equivalence separately.

CPU/read-only evidence replay except create-only selection and report outputs.
The strict historical reproduction assemblers are imported by hash, never edited.
"""
from __future__ import annotations

import argparse
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
ADDENDUM_SHA = "ce5cceaf3d3645bca54daceec7448a778a31a55417d7157cc8c69332d74005d1"
DREAMZERO_QUEUE_SHA = "dbbaaf7f391c30dd78b5c9de10738780bc3347b5f0d8352e998d3d187a13e279"
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
    path = HERE / "source_addendum_v1.json"
    require(sha(path) == ADDENDUM_SHA, "source-stage addendum changed")
    result = read(path)
    require(result["contract"]["sha256"] == CONTRACT_SHA, "addendum contract differs")
    checked(result["DreamZero_ptxas_probe"])
    checked(result["normal_queue_source"])
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
    for name, entry in current["files"].items():
        source = path.parent / "source" / name
        require(source.resolve() == source.absolute() and sha(source) == entry["sha256"],
                "actual staged source differs from bound manifest")
    return {"manifest": candidate, "baseline": contract["main_stage"],
            "source_addendum": ref(HERE / "source_addendum_v1.json"), "changed_source_paths": sorted(changed)}


def validate_dreamzero_compiler(contract, completion_ref):
    addendum = source_addendum()
    probe = read(checked(addendum["DreamZero_ptxas_probe"]))
    queue_path = STUDY / "dreamzero_paired_serving_queue_v5.json"
    require(sha(queue_path) == DREAMZERO_QUEUE_SHA, "DreamZero prospective queue changed")
    config, completion = read(queue_path), read(checked(completion_ref))
    require(probe["status"] == "passed" and probe["returncode"] == 0
            and probe["GPU_used"] is False and probe["target"] == "sm_110a", "PTXAS CPU probe failed")
    expected_env = {key: probe["ptxas"] for key in ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH")}
    require(probe["environment"] == expected_env, "PTXAS probe activation differs")
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
        for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            require(actual["applied_compiler_environment"].get(key) == declared["environment"][key],
                    "DreamZero selected compiler cache differs")
        require(actual["started_unix"] < actual["completed_unix"]
                and read(endpoint)["interpreter"] == declared["python"], "DreamZero process/endpoint differs")
    return {"status": "passed", "CPU_probe": addendum["DreamZero_ptxas_probe"],
            "queue_source": addendum["normal_queue_source"], "queue_config": ref(queue_path),
            "queue_completion": completion_ref, "applied_environment": expected_env,
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


def bind(contract_path, contract_sha256, output, stage_ref, dreamzero_completion):
    require(output.resolve().is_relative_to(HERE) and not output.exists(), "selection output must be new inside this study")
    contract, _ = load_contract(contract_path, contract_sha256)
    validate_stage(contract, stage_ref)
    snapshot = snapshot_inputs(contract)
    require_new_cosmos(contract, snapshot)
    validate_dreamzero_compiler(contract, dreamzero_completion)
    selection = {"schema": "instinctflash.current_public_pipeline_selection.v1", "template_only": False,
                 "contract": ref(contract_path), "source_stage_manifest": stage_ref,
                 "source_addendum": ref(HERE / "source_addendum_v1.json"),
                 "dreamzero_queue_completion": dreamzero_completion,
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
                               "source_addendum", "dreamzero_queue_completion",
                               "qualification_root", "family_roots", "snapshot"}, "selection fields differ")
    require(selection["schema"] == "instinctflash.current_public_pipeline_selection.v1"
            and selection["template_only"] is False, "actual selection required")
    require(selection["contract"] == ref(contract_path)
            and selection["source_addendum"] == ref(HERE / "source_addendum_v1.json")
            and selection["qualification_root"] == contract["qualification_root"]
            and selection["family_roots"] == contract["family_roots"], "selection identity differs")
    require(selection["snapshot"] == snapshot_inputs(contract), "selected evidence changed after binding")
    require_new_cosmos(contract, selection["snapshot"])
    validate_stage(contract, selection["source_stage_manifest"])
    validate_dreamzero_compiler(contract, selection["dreamzero_queue_completion"])
    return selection


def assess_historical(helper, comparator, family, root, selection_ref, source):
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
    closure = helper.guarded(helper.validate_omni_dependency_closure, STUDY / "omni_missing_dependencies_v1")
    main, foreign = {}, {}
    for family, root in roots.items():
        run = root / "run"
        paired = helper.guarded(helper.validate_main, run, family, modules["user_report"], source, output)
        serving = [helper.guarded(helper.validate_serving, root / name, run, modules["user_e2e"])
                   for name in contract["serving_directories"][family]]
        expected = {family + "-runtime_selected"} | ({"va-2v4a-fp8"} if family == "va" else set())
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
    rows = []
    for family in FAMILIES:
        cells = {row["id"]: row for row in main[family]["paired"].get("cells", [])}
        names = [family] + (["va-2v4a-fp8"] if family == "va" else [])
        for name in names:
            selected = family + "-runtime_selected" if name == family else name
            row = {"model": name, "current_pipeline_status": main[family]["status"],
                   "historical_equivalence": main[family]["historical_equivalence"]["status"],
                   "native_p50_ms": cells.get(family + "-eager_native", {}).get("p50_ms"),
                   "runtime_default_p50_ms": cells.get(family + "-runtime_default", {}).get("p50_ms"),
                   "InstinctFlash_p50_ms": cells.get(selected, {}).get("p50_ms")}
            for framework, column in (("lerobot", "LeRobot"), ("vllm-omni", "vLLM_Omni")):
                row[column + "_status"] = helper.foreign_status(catalog, foreign, framework, family)
                matching = [f for f in foreign.values() if f["status"] == "passed"
                            and f["family"] == family and f["framework"] == framework]
                row[column + "_p50_ms"] = matching[0]["latency"]["p50_ms"] if matching else None
            rows.append(row)
    status, counts, failures = summarize(main, foreign, correspondence, closure)
    result = {"schema": "instinctflash.current_public_pipeline_qualification.v1", "status": status,
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "assembler": ref(__file__),
              "contract": ref(contract_path), "selection": ref(selection_path),
              "source_stage_manifest": selection["source_stage_manifest"],
              "source_stage_admission": validate_stage(contract, selection["source_stage_manifest"]),
              "DreamZero_compiler_activation": validate_dreamzero_compiler(contract, selection["dreamzero_queue_completion"]),
              "selected_qualification_roots": {k: str(v) for k, v in roots.items()},
              "counts": counts, "failures": failures, "main": main, "foreign": foreign, "table": rows,
              "source_correspondence": correspondence, "omni_dependency_closure": closure,
              "foreign_execution_source_origin": foreign_origin, "historical_matrix": contract["historical_matrix"],
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

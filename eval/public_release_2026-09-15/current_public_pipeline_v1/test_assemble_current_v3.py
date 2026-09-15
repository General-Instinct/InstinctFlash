"""Prospective selection/status integrity; no native execution or evidence admission."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("current_pipeline_test_module", HERE / "assemble_current_v3.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def contract(tmp_path):
    value = m.read(HERE / "contract_v1.json")
    value["qualification_root"] = str(tmp_path / "qualification")
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def full_results():
    main = {family: {"status": "passed", "historical_equivalence": {
        "status": "passed", "assessment_complete": True, "exact_action_equivalence": True}}
        for family in m.FAMILIES}
    foreign = {str(i): {"status": "passed"} for i in range(6)}
    return main, foreign


def test_current_success_is_independent_of_validated_historical_nano_drift():
    main, foreign = full_results()
    main["nano"]["historical_equivalence"].update(status="failed_historical_equivalence", exact_action_equivalence=False)
    status, counts, failures = m.summarize(main, foreign, {"status": "passed"}, {"status": "passed"})
    assert status == "current_pipeline_qualified" and not failures
    assert counts["main_passed"] == counts["historical_families_assessed"] == 8
    assert counts["historical_families_equivalent"] == 7 and counts["foreign_passed"] == 6
    assert not any(m.CLAIM_LIMITS.values())


@pytest.mark.parametrize("where", ["paired", "foreign", "history", "source", "closure"])
def test_actual_failed_or_corrupt_evidence_cannot_qualify(where):
    main, foreign = full_results()
    source, closure = {"status": "passed"}, {"status": "passed"}
    if where == "paired":
        main["nano"]["status"] = "failed"
    elif where == "foreign":
        foreign["0"]["status"] = "failed"
    elif where == "history":
        main["nano"]["historical_equivalence"] = {"status": "failed", "reason": "archive hash differs"}
    else:
        (source if where == "source" else closure)["status"] = "failed"
    assert m.summarize(main, foreign, source, closure)[0] == "failed_current_evidence"


def test_missing_current_row_stays_partial_and_no_historical_certificate_even_if_all_exact():
    main, foreign = full_results()
    main["dreamzero"]["status"] = "pending"
    assert m.summarize(main, foreign, {"status": "passed"}, {"status": "passed"})[0] == "partial_current_pipeline"
    assert m.CLAIM_LIMITS["complete_historical_reproduction"] is False


@pytest.mark.parametrize("mutation", ["old_cosmos", "another_rerun", "missing_family"])
def test_prospective_roots_cannot_fallback_or_select_another_outcome(tmp_path, mutation):
    c = contract(tmp_path)
    if mutation == "missing_family":
        del c["family_roots"]["edge"]
    else:
        c["family_roots"]["nano"] = "nano" if mutation == "old_cosmos" else "nano/pytorch_triton_v2"
    with pytest.raises(ValueError, match="prospective family roots"):
        m.fixed_roots(c)


def test_snapshot_detects_new_changed_files_and_rejects_symlink_targets(tmp_path):
    c = contract(tmp_path)
    root = m.fixed_roots(c)["nano"] / "run"
    write(root / "run.json", {"value": 1})
    first = m.snapshot_inputs(c)
    write(root / "run.json", {"value": 2})
    assert m.snapshot_inputs(c) != first
    write(root / "new.json", {})
    assert len(m.snapshot_inputs(c)) == len(first) + 1
    (root / "alias.json").symlink_to(root / "run.json")
    with pytest.raises(ValueError, match="symlink"):
        m.snapshot_inputs(c)


def test_old_complete_cosmos_evidence_does_not_satisfy_new_binding(tmp_path):
    c = contract(tmp_path)
    for family in ("edge", "nano"):
        for relative in m.REQUIRED_COSMOS:
            write(Path(c["qualification_root"]) / family / relative, {"status": "passed"})
    with pytest.raises(ValueError, match="new Cosmos"):
        m.require_new_cosmos(c, m.snapshot_inputs(c))


@pytest.mark.parametrize("interpreter", ["old/bin/python", None])
def test_new_cosmos_interpreter_and_completion_are_mandatory(tmp_path, interpreter):
    c = contract(tmp_path)
    for family in ("edge", "nano"):
        for relative in m.REQUIRED_COSMOS:
            write(m.fixed_roots(c)[family] / relative,
                  {"status": "passed", "report_status": "passed", "interpreter": interpreter})
    with pytest.raises(ValueError, match="another interpreter"):
        m.require_new_cosmos(c, m.snapshot_inputs(c))


def test_actual_selection_rejects_hash_field_and_snapshot_drift(tmp_path, monkeypatch):
    c = contract(tmp_path)
    cp = tmp_path / "contract.json"
    write(cp, c)
    snapshot = {"nano/pytorch_triton_v1/run/run.json": {"sha256": "a" * 64, "bytes": 5}}
    selection = {"schema": "instinctflash.current_public_pipeline_selection.v1", "template_only": False,
                 "contract": m.ref(cp), "source_stage_manifest": c["main_stage"],
                 "source_addendum": m.ref(HERE / "source_addendum_v3.json"),
                 "dreamzero_queue_completion": {"path": "/bound", "sha256": "c" * 64},
                 "qualification_root": c["qualification_root"], "family_roots": c["family_roots"], "snapshot": snapshot}
    path = tmp_path / "selection.json"
    write(path, selection)
    monkeypatch.setattr(m, "snapshot_inputs", lambda c: snapshot)
    monkeypatch.setattr(m, "require_new_cosmos", lambda *a: None)
    monkeypatch.setattr(m, "validate_stage", lambda *a: None)
    monkeypatch.setattr(m, "validate_dreamzero_compiler", lambda *a: None)
    assert m.validate_selection(c, cp, path, m.sha(path)) == selection
    with pytest.raises(ValueError, match="selection hash"):
        m.validate_selection(c, cp, path, "0" * 64)
    forged = copy.deepcopy(selection)
    forged["family_roots"]["nano"] = "nano"
    write(path, forged)
    with pytest.raises(ValueError, match="selection identity"):
        m.validate_selection(c, cp, path, m.sha(path))
    write(path, selection)
    monkeypatch.setattr(m, "snapshot_inputs", lambda c: {})
    with pytest.raises(ValueError, match="changed after binding"):
        m.validate_selection(c, cp, path, m.sha(path))


def test_stage_extension_allows_installer_docs_but_rejects_runtime_change(tmp_path):
    files = {"scripts/bootstrap_vendor.py": "print('old')", "instinctflash/runtime/example.py": "VALUE = 1"}
    manifests = []
    for version in ("base", "next"):
        root = tmp_path / version
        entries = {}
        for name, text in files.items():
            path = root / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text if version == "base" or name.startswith("instinctflash") else "print('new')")
            entries[name] = {"sha256": m.sha(path)}
        path = root / "manifest.json"
        write(path, {"status": "wheels_built_and_inspected_CPU_only", "files": entries})
        manifests.append(path)
    c = {"main_stage": m.ref(manifests[0])}
    assert m.validate_stage(c, m.ref(manifests[1]))["changed_source_paths"] == ["scripts/bootstrap_vendor.py"]
    current = m.read(manifests[1])
    current["files"]["instinctflash/runtime/example.py"]["sha256"] = "0" * 64
    write(manifests[1], current)
    with pytest.raises(ValueError, match="outside.*scope"):
        m.validate_stage(c, m.ref(manifests[1]))


def test_frozen_real_contract_preserves_negative_result_and_original_files():
    c, helper = m.load_contract(HERE / "contract_v1.json", m.CONTRACT_SHA)
    assert c["claim_limits"] == m.CLAIM_LIMITS
    assert m.sha(helper.__file__) == "45b8abe781a1141deacf4c1f7f100f176844ec8ab64ada526286d3a50dd2085b"
    assert m.read(m.checked(c["negative_compiler_diagnostic"]))["actions"]["corrected_vs_historical"]["exact_bytes"] is False


def test_only_exact_sde1_documentation_correction_is_admitted(tmp_path):
    document = m.source_addendum()["SDE1_documentation_correction"]
    relative = document["path"]
    approved = (m.STUDY.parents[1] / relative).read_bytes()
    manifests = []
    for version, content in (("old", b"Old description\n"), ("corrected", approved),
                             ("unapproved", approved + b"An extra claim.\n")):
        root = tmp_path / version
        path = root / "source" / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        manifest = root / "manifest.json"
        write(manifest, {"status": "wheels_built_and_inspected_CPU_only",
                         "files": {relative: {"sha256": m.sha(path)}}})
        manifests.append(manifest)
    c = {"main_stage": m.ref(manifests[0])}
    assert m.validate_stage(c, m.ref(manifests[1]))["changed_source_paths"] == [relative]
    with pytest.raises(ValueError, match="exact authorized correction"):
        m.validate_stage(c, m.ref(manifests[2]))


def test_strict_historical_validator_and_comparator_sources_remain_unmodified():
    c = m.read(HERE / "contract_v1.json")
    for value in c["validator_sources"].values():
        assert m.sha(m.checked(value)) == value["sha256"]


def test_actual_preserved_nano_arrays_report_drift_without_waiving_or_relabeling_it():
    c, helper = m.load_contract(HERE / "contract_v1.json", m.CONTRACT_SHA)
    comparator = m.load_module(c["validator_sources"]["historical_comparator_v2"], "test_old_nano_comparator")
    result = m.assess_historical(helper, comparator, "nano", m.STUDY / "qualification/nano",
                                 m.ref(HERE / "contract_v1.json"), Path(helper.__file__))
    assert result["status"] == "failed_historical_equivalence" and result["assessment_complete"]
    rows = {row["cell"]: row for row in result["arrays"]}
    assert rows["nano-eager_native"]["historical_actions_match_bytes"] is True
    assert rows["nano-runtime_default"]["historical_actions_match_bytes"] is True
    assert rows["nano-runtime_selected"]["historical_actions_match_bytes"] is False
    assert rows["nano-runtime_selected"]["max_abs"] > 0


@pytest.mark.parametrize("mutation", [None, "compiler_path", "compiler_hash", "missing_threads", "parallel_threads", "old_cache", "extra_env"])
def test_applied_dreamzero_compiler_must_match_cpu_probe(tmp_path, monkeypatch, mutation):
    c = contract(tmp_path)
    probe = m.read(m.checked(m.source_addendum()["DreamZero_ptxas_probe"]))
    queue = m.read(m.STUDY / "dreamzero_memory_preparation_v1/dreamzero_paired_serving_queue_v6.json")
    root = m.fixed_roots(c)["dreamzero"]
    rows = []
    for job, endpoint in zip(queue["jobs"], [root / "run/run.json", root / "serving/receipt.json"]):
        write(endpoint, {"interpreter": job["python"]})
        rows.append({"id": job["id"], "command": [job["python"], "-I", "-m", job["module"], *job["arguments"]],
                     "returncode": 0, "receipt_status": "passed", "activation": job["activation"],
                     "applied_compiler_environment": {**probe["environment"], **{
                         key: job["environment"][key] for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_COMPILE_THREADS")}},
                     "assembler_sha256": {key: probe["sha256"] for key in probe["environment"]},
                     "started_unix": 1, "completed_unix": 2})
    if mutation == "compiler_path":
        rows[0]["applied_compiler_environment"]["TRITON_PTXAS_PATH"] = "/old/bundled/ptxas"
    elif mutation == "compiler_hash":
        rows[1]["assembler_sha256"]["TRITON_PTXAS_PATH"] = "0" * 64
    elif mutation == "missing_threads":
        del rows[0]["applied_compiler_environment"]["TORCHINDUCTOR_COMPILE_THREADS"]
    elif mutation == "parallel_threads":
        rows[0]["applied_compiler_environment"]["TORCHINDUCTOR_COMPILE_THREADS"] = "16"
    elif mutation == "old_cache":
        rows[1]["applied_compiler_environment"]["TRITON_CACHE_DIR"] = "/old/cache"
    elif mutation == "extra_env":
        rows[0]["applied_compiler_environment"]["UNDECLARED_COMPILER_MODE"] = "1"
    path = tmp_path / "completion.json"
    write(path, {"status": "passed", "jobs": rows})
    if mutation:
        with pytest.raises(ValueError, match="compiler (identity|memory environment)"):
            m.validate_dreamzero_compiler(c, m.ref(path))
    else:
        assert m.validate_dreamzero_compiler(c, m.ref(path))["status"] == "passed"


def test_duplicate_json_and_nonfinite_values_rejected(tmp_path):
    path = tmp_path / "bad.json"
    for text in ('{"a":1,"a":2}', '{"a":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            m.read(path)


def test_foreign_va_measurement_only_labels_its_actual_2v4a_operating_point():
    from types import SimpleNamespace

    main, _ = full_results()
    for family, row in main.items():
        row["paired"] = {"cells": [{"id": family + "-eager_native", "p50_ms": 8000},
                                    {"id": family + "-runtime_default", "p50_ms": 4000},
                                    {"id": family + "-runtime_selected", "p50_ms": 2000}]}
    main["va"]["paired"]["cells"].append({"id": "va-2v4a-fp8", "p50_ms": 500})
    foreign = {"lerobot-va": {"status": "passed", "family": "va", "framework": "lerobot",
                               "latency": {"p50_ms": 1171}}}
    helper = SimpleNamespace(foreign_status=lambda *a: "passed")
    rows = {row["model"]: row for row in m.table_rows(main, foreign, {}, helper)}
    assert rows["va"]["LeRobot_status"] == "unmeasured"
    assert rows["va"]["LeRobot_p50_ms"] is None
    assert rows["va-2v4a-fp8"]["LeRobot_p50_ms"] == 1171
    assert rows["va-2v4a-fp8"]["native_p50_ms"] is None
    assert rows["va-2v4a-fp8"]["runtime_default_p50_ms"] is None
    assert rows["va-2v4a-fp8"]["comparison_reference_native_p50_ms"] == 8000
    assert rows["va-2v4a-fp8"]["comparison_reference_default_p50_ms"] == 4000
    assert "25V50A" in rows["va-2v4a-fp8"]["comparison_reference_scope"]


def test_actual_v6_queue_only_changes_memory_environment_and_owned_output():
    old = m.read(m.STUDY / "dreamzero_paired_serving_queue_v5.json")
    new = m.read(m.STUDY / "dreamzero_memory_preparation_v1/dreamzero_paired_serving_queue_v6.json")
    assert new["parent_queue_sha256"] == m.sha(m.STUDY / "dreamzero_paired_serving_queue_v5.json")
    normalized = copy.deepcopy(new)
    normalized["parent_queue_sha256"] = old["parent_queue_sha256"]
    normalized["output"] = old["output"]
    for before, after in zip(old["jobs"], normalized["jobs"]):
        assert after["environment"]["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
        assert after["environment"]["TRITON_CACHE_DIR"] == "/dev/shm/ifl_public_dreamzero_compile_v6/triton"
        after["environment"] = before["environment"]
        after["arguments"] = [value.replace("dreamzero_dynamic-fp8_run_v6", "dreamzero_dynamic-fp8_run_v5")
                              for value in after["arguments"]]
        after["receipt"] = after["receipt"].replace("dreamzero_dynamic-fp8_run_v6", "dreamzero_dynamic-fp8_run_v5")
    assert normalized == old


def test_v6_scientific_and_table_functions_remain_ast_exact_v2():
    import ast

    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef)}
    before, after = functions(HERE / "assemble_current_v2.py"), functions(HERE / "assemble_current_v3.py")
    for name in ("load_contract", "fixed_roots", "snapshot_inputs", "require_new_cosmos",
                 "assess_historical", "summarize", "table_rows", "assemble"):
        assert before[name] == after[name], name


def test_actual_queue_worker_records_threads_without_changing_launch_code():
    old = (m.STUDY / "thor_queue_v4.py").read_text()
    new = m.checked(m.source_addendum()["normal_queue_source"]).read_text()
    assert new.replace(', "TORCHINDUCTOR_COMPILE_THREADS")', ')') == old

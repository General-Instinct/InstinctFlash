import importlib.util
import json
from pathlib import Path
import shutil

import pytest


HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("evidence_assembly_v3", HERE / "assemble_v3.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def copy_foreign_stage(destination):
    destination.mkdir()
    shutil.copyfile(m.FOREIGN_STAGE / "manifest.json", destination / "manifest.json")
    inventory = m.read(m.STUDY / "framework_compare_v1/source_inventory_v4.json")
    for relative in inventory["files"]:
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m.FOREIGN_STAGE / "source" / relative, target)
    return destination


def test_exact_v4_foreign_source_resolves_all_six_original_preparations():
    helper, origin = m.load_foreign_sources(m.FOREIGN_STAGE)
    cells = helper.catalog()["cells"]
    assert len(cells) == 6
    for cell in cells:
        prepared = m.STUDY / "framework_compare_v1/prepared_assets_v2" / cell["id"]
        assert helper.plan(cell["id"]) == m.read(prepared / "plan.json")
    assert len(origin["files"]) == 11
    assert origin["current_catalog_substitution_permitted"] is False
    assert m.sha(HERE / "assemble_v2.py") == m.V2_ASSEMBLER_SHA


def test_actual_three_completed_lerobot_captures_keep_old_runner_catalog():
    helper, _ = m.load_foreign_sources(m.FOREIGN_STAGE)
    cells = [row for row in helper.catalog()["cells"] if row["framework"] == "lerobot"]
    assert len(cells) == 3
    for cell in cells:
        result = m.validate_foreign(
            cell["id"], m.STUDY / "qualification/foreign" / cell["id"],
            m.STUDY / "framework_compare_v1/prepared_assets_v2" / cell["id"], helper,
            m.FOREIGN_STAGE / "source/benchmarks/regression/fixtures/recorded_inputs_v1.npz")
        assert result["status"] == "passed"
        assert result["task_quality_certified"] is False


def test_changed_foreign_catalog_rejected_before_any_code_import(tmp_path, monkeypatch):
    stage = copy_foreign_stage(tmp_path / "stage")
    catalog = stage / "source/benchmarks/regression/fixtures/frameworks/catalog.json"
    data = m.read(catalog)
    data["unbound_package_addition"] = {"iopath": "0.1.10"}
    catalog.write_text(json.dumps(data))
    monkeypatch.setattr(m, "load_module", lambda *_: pytest.fail("source executed before all input hashes passed"))
    with pytest.raises(ValueError, match="foreign V4 source drift before import"):
        m.load_foreign_sources(stage)


def test_revised_manifest_cannot_admit_revised_old_catalog(tmp_path, monkeypatch):
    stage = copy_foreign_stage(tmp_path / "stage")
    manifest = m.read(stage / "manifest.json")
    manifest["files"]["benchmarks/regression/fixtures/frameworks/catalog.json"]["sha256"] = "0" * 64
    (stage / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(m, "load_module", lambda *_: pytest.fail("unbound stage imported"))
    with pytest.raises(ValueError, match="foreign source stage manifest changed"):
        m.load_foreign_sources(stage)


def test_actual_omni_closure_only_adds_two_utilities_preserves_failure_and_cpu_scope():
    result = m.validate_omni_dependency_closure(m.STUDY / "omni_missing_dependencies_v1")
    assert result["status"] == "passed"
    assert result["added_package_versions"] == {"iopath": "0.1.10", "portalocker": "4.3.0"}
    assert result["existing_package_version_changes"] == []
    assert set(result["metadata_changes"]) == {"cosmos-framework"}
    assert result["gpu_capture_qualified"] is False
    assert result["old_prepared_plan_or_catalog_modified"] is False


def test_omni_closure_file_drift_is_not_hidden_by_passed_completion(tmp_path):
    original = m.STUDY / "omni_missing_dependencies_v1"
    shutil.copyfile(original / "copied_native_evidence.json", tmp_path / "copied_native_evidence.json")
    shutil.copytree(original / "native_evidence", tmp_path / "native_evidence")
    (tmp_path / "native_evidence/pip_check.json").write_text('{"returncode": 1}\n')
    with pytest.raises(ValueError, match="Omni closure evidence changed: pip_check.json"):
        m.validate_omni_dependency_closure(tmp_path)


def test_old_captured_package_list_not_rewritten_by_new_closure():
    helper, _ = m.load_foreign_sources(m.FOREIGN_STAGE)
    cell = next(row for row in helper.catalog()["cells"] if row["framework"] == "vllm-omni")
    plan = helper.plan(cell["id"])
    before = json.dumps(plan, sort_keys=True)
    m.validate_omni_dependency_closure(m.STUDY / "omni_missing_dependencies_v1")
    assert json.dumps(helper.plan(cell["id"]), sort_keys=True) == before


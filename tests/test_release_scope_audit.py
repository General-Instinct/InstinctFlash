"""Source-boundary review must expose hidden training dependencies and packaging leaks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location(
    "release_scope_audit", Path(__file__).resolve().parents[1] / "scripts/audit_release_scope.py")
auditor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(auditor)


def scope():
    return {
        "schema": "instinctflash.oss_scope.v1", "status": "proposal",
        "inventory_roots": ["instinctflash", "benchmarks", "pyproject.toml"],
        "package_roots": [{"path": "instinctflash", "module": "instinctflash"},
                          {"path": "benchmarks", "module": "benchmarks"}],
        "rules": [
            {"prefix": "instinctflash", "classification": "public_candidate", "group": "runtime"},
            {"prefix": "instinctflash/distill", "classification": "hold", "group": "trainer"},
            {"prefix": "benchmarks", "classification": "public_candidate", "group": "eval"},
            {"prefix": "eval", "classification": "hold", "group": "study"},
            {"prefix": "pyproject.toml", "classification": "release_review", "group": "package"},
        ],
    }


def put(root, relative, content=""):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return relative


def test_nested_hold_wins_without_matching_similarly_named_public_module():
    assert auditor.classify("instinctflash/distill/core.py", scope())["classification"] == "hold"
    assert auditor.classify("instinctflash/distillation_metadata.py", scope())["classification"] == "public_candidate"
    assert auditor.classify("new_package/main.py", scope())["classification"] == "unclassified"


@pytest.mark.parametrize("path", ["/tmp/source", "../source", "instinctflash/../private", "x//y", "x\\y"])
def test_scope_cannot_escape_or_ambiguously_name_the_repository(path):
    manifest = scope()
    manifest["rules"][0]["prefix"] = path
    with pytest.raises(ValueError):
        auditor.validate_scope(manifest)


@pytest.mark.parametrize("code,placement", [
    ("from instinctflash.distill.pipeline import rank_candidates", "module_level"),
    ("def run():\n    from ..distill.pipeline import rank_candidates", "deferred_or_conditional"),
    ("try:\n    import instinctflash.distill.pipeline\nexcept ImportError:\n    pass", "deferred_or_conditional"),
    ("import importlib\nimportlib.import_module('instinctflash.distill.pipeline')", "deferred_or_conditional"),
    ("from importlib import import_module as load\nload('instinctflash.distill.pipeline')", "deferred_or_conditional"),
])
def test_public_entrypoint_records_direct_relative_optional_and_dynamic_trainer_edges(tmp_path, code, placement):
    paths = [put(tmp_path, "instinctflash/runtime/entry.py", code),
             put(tmp_path, "instinctflash/distill/pipeline.py", "raise RuntimeError('never import me')")]
    report = auditor.audit(tmp_path, scope(), paths)
    edges = [f for f in report["findings"] if f["kind"] == "import_crosses_boundary"]
    assert len(edges) == 1
    assert edges[0]["target"] == "instinctflash/distill/pipeline.py"
    assert edges[0]["placement"] == placement
    assert edges[0]["severity"] == "blocker"
    assert report["publication_ready"] is False


def test_current_wheel_discovery_exposes_trainer_until_explicitly_excluded(tmp_path):
    paths = [put(tmp_path, "instinctflash/distill/__init__.py"),
             put(tmp_path, "instinctflash/distill/pipeline.py")]
    paths.append(put(tmp_path, "pyproject.toml",
                     '[tool.setuptools.packages.find]\ninclude = ["instinctflash*"]\n'))
    before = auditor.audit(tmp_path, scope(), paths)
    leaks = [f for f in before["findings"] if f["kind"] == "current_core_discovery_includes_nonpublic_candidate"]
    assert {f["path"] for f in leaks} == set(paths[:2])
    with (tmp_path / "pyproject.toml").open("a") as config:
        config.write('exclude = ["instinctflash.distill*"]\n')
    after = auditor.audit(tmp_path, scope(), paths)
    assert not after["findings"]
    assert after["publication_ready"] is False  # Static success never means release qualification.


def test_worker_symlink_into_withheld_study_is_not_a_public_source_closure(tmp_path):
    put(tmp_path, "eval/study/worker.py", "value = 1\n")
    worker = tmp_path / "instinctflash/runtime/worker.py"
    worker.parent.mkdir(parents=True)
    worker.symlink_to("../../eval/study/worker.py")
    report = auditor.audit(tmp_path, scope(), ["instinctflash/runtime/worker.py"])
    assert report["files"]["instinctflash/runtime/worker.py"]["target_classification"] == "hold"
    assert report["findings"][0]["kind"] == "public_symlink_crosses_boundary"


def test_external_symlink_is_reported_without_reading_its_contents(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"\xffprivate content not to decode or hash")
    link = repository / "instinctflash/runtime.py"
    link.parent.mkdir()
    link.symlink_to(outside)
    report = auditor.audit(repository, scope(), ["instinctflash/runtime.py"])
    assert report["findings"][0]["kind"] == "symlink_outside_repository"
    entry = report["files"]["instinctflash/runtime.py"]
    assert "sha256" not in entry and "bytes" not in entry


def test_computed_import_and_unclassified_source_remain_visible(tmp_path):
    paths = [put(tmp_path, "instinctflash/runtime.py", "import importlib\nimportlib.import_module(selected_backend)\n"),
             put(tmp_path, "unknown/helper.py", "value = 1\n")]
    report = auditor.audit(tmp_path, scope(), paths)
    assert {f["kind"] for f in report["findings"]} == {"dynamic_import_review", "unclassified_source"}
    assert report["reviews"] == 1 and report["blockers"] == 1


def test_public_ranking_dependency_can_be_audited_without_executing_source(tmp_path):
    marker = tmp_path / "unexpected_import"
    paths = [put(tmp_path, "benchmarks/screen.py", "from instinctflash.verify.ranking import rank_candidates\n"),
             put(tmp_path, "instinctflash/verify/ranking.py",
                 f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")]
    report = auditor.audit(tmp_path, scope(), paths)
    assert not report["findings"]
    assert not marker.exists()
    assert len(report["files"][paths[0]]["sha256"]) == 64


def test_symlink_directory_and_project_config_never_read_external_bytes(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "entry.py").write_bytes(b"\xffnot to decode")
    (outside / "pyproject.toml").write_bytes(b"\xffnot to parse")
    (repository / "instinctflash").symlink_to(outside, target_is_directory=True)
    (repository / "pyproject.toml").symlink_to(outside / "pyproject.toml")
    report = auditor.audit(repository, scope(), ["instinctflash/entry.py", "pyproject.toml"])
    assert report["blockers"] == 2
    assert {f["kind"] for f in report["findings"]} == {"symlink_outside_repository"}
    assert all("sha256" not in entry for entry in report["files"].values())


def test_import_initializes_held_parent_even_when_child_is_public(tmp_path):
    manifest = scope()
    manifest["rules"].append({"prefix": "instinctflash/distill/public.py",
                              "classification": "public_candidate", "group": "public_helper"})
    paths = [put(tmp_path, "benchmarks/screen.py", "from instinctflash.distill.public import rank\n"),
             put(tmp_path, "instinctflash/distill/__init__.py", "import commercial_trainer\n"),
             put(tmp_path, "instinctflash/distill/public.py", "rank = 1\n")]
    report = auditor.audit(tmp_path, manifest, paths)
    assert any(f.get("target") == "instinctflash/distill/__init__.py"
               and f["severity"] == "blocker" for f in report["findings"])


def test_missing_held_module_does_not_resolve_as_public_parent(tmp_path):
    paths = [put(tmp_path, "instinctflash/__init__.py"),
             put(tmp_path, "benchmarks/screen.py", "import instinctflash.distill.absent\n")]
    report = auditor.audit(tmp_path, scope(), paths)
    assert report["findings"][0]["kind"] == "uninventoried_held_import"
    assert report["findings"][0]["severity"] == "blocker"


def test_explicit_package_list_does_not_include_unlisted_training_packages(tmp_path):
    paths = [put(tmp_path, "instinctflash/__init__.py"),
             put(tmp_path, "instinctflash/distill/__init__.py"),
             put(tmp_path, "pyproject.toml", '[tool.setuptools]\npackages = ["instinctflash"]\n')]
    report = auditor.audit(tmp_path, scope(), paths)
    assert not report["findings"]

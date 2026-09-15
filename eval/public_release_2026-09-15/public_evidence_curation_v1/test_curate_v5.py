import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).with_name("curate_v5.py")
SPEC = importlib.util.spec_from_file_location("bounded_public_curation", PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("name", ["../secret", "/tmp/private", "-option", "a\nb", "a/../b", "a//b"])
def test_noncanonical_or_escaping_paths_rejected(name):
    with pytest.raises(ValueError):
        m.relative(name)


def test_both_file_and_parent_symlink_targets_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "proof.json").write_text("{}")
    root = tmp_path / "root"
    root.mkdir()
    (root / "parent").symlink_to(outside)
    (root / "alias.json").symlink_to(outside / "proof.json")
    for name in ("parent/proof.json", "alias.json", "parent/new.json"):
        with pytest.raises(ValueError, match="symlink"):
            m.safe(root, name, exists=False)


def test_secret_candidate_rejected_without_echoing_value(tmp_path):
    value = "hf_" + "a" * 34
    (tmp_path / "receipt.json").write_text(json.dumps({"token": value}))
    with pytest.raises(ValueError, match="credential candidate") as error:
        m.inspect_file(tmp_path, "receipt.json", m.policy())
    assert value not in str(error.value)


@pytest.mark.parametrize("mutation", ["weight", "large", "pickle", "pixels", "nonfinite"])
def test_only_bounded_safe_actions_and_exact_public_fixture_are_admitted(tmp_path, mutation):
    rules = m.policy()
    name = "receipt.npz"
    if mutation == "weight":
        name = "checkpoint.safetensors"
        (tmp_path / name).write_bytes(b"not selected")
    elif mutation == "large":
        name = "receipt.json"
        rules["max_file_bytes"] = 5
        (tmp_path / name).write_text("123456")
    else:
        values = np.asarray([object()], dtype=object) if mutation == "pickle" else np.asarray([np.nan if mutation == "nonfinite" else 0], dtype=np.float32)
        data = io.BytesIO()
        np.savez(data, **{"pixels" if mutation == "pixels" else "actions": values})
        (tmp_path / name).write_bytes(data.getvalue())
    with pytest.raises(ValueError):
        m.inspect_file(tmp_path, name, rules)


def test_valid_actions_and_named_fixture_hash_gate(tmp_path):
    data = io.BytesIO()
    np.savez(data, action_0=np.zeros(7, dtype=np.float32),
             queued_actions_0=np.zeros((49, 1, 7), dtype=np.float32))
    (tmp_path / "actions.npz").write_bytes(data.getvalue())
    assert m.inspect_file(tmp_path, "actions.npz", m.policy())["sha256"] == m.digest(data.getvalue())
    (tmp_path / "recorded_inputs_v1.npz").write_bytes(data.getvalue())
    with pytest.raises(ValueError, match="fixture changed"):
        m.inspect_file(tmp_path, "recorded_inputs_v1.npz", m.policy())


def test_provenance_paths_are_marked_not_rewritten_or_fetched(tmp_path):
    repo, public = tmp_path / "repo", tmp_path / "public"
    repo.mkdir()
    public.mkdir()
    (repo / "raw.log").write_text("raw")
    (public / "old.json").write_text("old")
    rows = [{"path": str(repo / "raw.log"), "sha256": m.digest(b"raw")},
            {"path": str(repo / "old.json"), "sha256": m.digest(b"old")},
            {"path": "/dev/shm/private-native-cache/source.py", "sha256": "a" * 64}]
    write(repo / "receipt.json", rows)
    result = m.provenance_index(repo, public, {"receipt.json", "raw.log"})
    assert [x["availability"] for x in result["references"]] == ["bundled_exact", "already_public_exact", "local_only_provenance"]
    assert result["references"][2]["public_relative_path"] is None
    assert json.loads((repo / "receipt.json").read_text()) == rows


def test_preview_and_conflicting_public_files_never_copy(tmp_path, monkeypatch):
    repo, public = tmp_path / "repo", tmp_path / "public"
    repo.mkdir()
    public.mkdir()
    (repo / "first.json").write_text("{}")
    (repo / "second.json").write_text("{}")
    (public / "second.json").write_text("preserved")
    manifest = {"status": "preview_only_waiting_current_qualification", "policy_sha256": m.POLICY_SHA,
                "curator_sha256": m.digest(PATH.read_bytes()), "repository": str(repo),
                "report_renderers": None, "SDE1": None, "require_completed_sde1": False,
                "current_qualification": {"summary": {"path": "/unused", "sha256": "a" * 64}},
                "files": {name: m.inspect_file(repo, name, m.policy()) for name in ("first.json", "second.json")}}
    path = repo / "manifest.json"
    write(path, manifest)
    monkeypatch.setattr(m, "validate_current", lambda *args: None)
    monkeypatch.setattr(m, "validate_renderers", lambda *args: None)
    monkeypatch.setattr(m, "sde1_evidence", lambda *args: None)
    with pytest.raises(ValueError, match="preview"):
        m.copy_plan(path, m.digest(path.read_bytes()), public, tmp_path / "receipt.json")
    manifest["status"] = "ready_for_create_only_copy"
    write(path, manifest)
    with pytest.raises(ValueError, match="conflicts"):
        m.copy_plan(path, m.digest(path.read_bytes()), public, tmp_path / "receipt.json")
    assert not (public / "first.json").exists()
    assert (public / "second.json").read_text() == "preserved"


def test_create_only_copy_keeps_identical_public_bytes_and_rejects_source_drift(tmp_path, monkeypatch):
    repo, public = tmp_path / "repo", tmp_path / "public"
    repo.mkdir()
    public.mkdir()
    (repo / "raw.json").write_text("{}")
    (public / "raw.json").write_text("{}")
    path = repo / "manifest.json"
    write(path, {"status": "ready_for_create_only_copy", "policy_sha256": m.POLICY_SHA,
                 "curator_sha256": m.digest(PATH.read_bytes()), "repository": str(repo),
                 "report_renderers": None, "SDE1": None, "require_completed_sde1": False,
                "current_qualification": {"summary": {"path": "/unused", "sha256": "a" * 64}},
                 "files": {"raw.json": m.inspect_file(repo, "raw.json", m.policy())}})
    monkeypatch.setattr(m, "validate_current", lambda *args: None)
    monkeypatch.setattr(m, "validate_renderers", lambda *args: None)
    monkeypatch.setattr(m, "sde1_evidence", lambda *args: None)
    original = (public / "raw.json").stat().st_mtime_ns
    result = m.copy_plan(path, m.digest(path.read_bytes()), public, tmp_path / "copy.json")
    assert result["already_identical"] == result["copied"] == 1
    assert (public / "raw.json").stat().st_mtime_ns == original
    assert (public / "manifest.json").read_bytes() == path.read_bytes()
    (repo / "raw.json").write_text("changed")
    with pytest.raises(ValueError, match="source bytes changed"):
        m.copy_plan(path, m.digest(path.read_bytes()), public, tmp_path / "second-copy.json")


@pytest.mark.parametrize("mutation", ["pending", "coverage", "quality"])
def test_partial_current_or_quality_promotion_cannot_make_final_plan(tmp_path, mutation):
    summary = tmp_path / "eval/public_release_2026-09-15/current_public_pipeline_v1/snapshot_v2/summary.json"
    value = {"schema": "instinctflash.current_public_pipeline_qualification.v1", "status": "current_pipeline_qualified",
             "counts": {"main_passed": 8, "foreign_passed": 6, "historical_families_assessed": 8},
             "complete_historical_reproduction": False, "task_quality_certified": False, "prior_cosmos_SCREEN_transferred": False}
    if mutation == "pending":
        value["status"] = "partial_current_pipeline"
    elif mutation == "coverage":
        value["counts"]["main_passed"] = 7
    else:
        value["task_quality_certified"] = True
    write(summary, value)
    with pytest.raises(ValueError):
        m.validate_current(tmp_path, summary, ref(summary)["sha256"])


def test_complete_current_binding_still_requires_exact_raw_snapshot(tmp_path, monkeypatch):
    study = tmp_path / "eval/public_release_2026-09-15"
    here = study / "current_public_pipeline_v1"
    for name in ("assembler.py", "contract.json", "stage.json"):
        write(here / name, {"identity": name})
    monkeypatch.setattr(m, "ASSEMBLER_SHA", ref(here / "assembler.py")["sha256"])
    monkeypatch.setattr(m, "CONTRACT_SHA", ref(here / "contract.json")["sha256"])
    monkeypatch.setattr(m, "STAGE_SHA", ref(here / "stage.json")["sha256"])
    raw = study / "qualification/nano/pytorch_triton_v1/run/run.json"
    write(raw, {"status": "passed"})
    selection = here / "selection.json"
    write(selection, {"template_only": False, "source_stage_manifest": ref(here / "stage.json"),
                      "snapshot": {"nano/pytorch_triton_v1/run/run.json": {
                          "sha256": ref(raw)["sha256"], "bytes": raw.stat().st_size}}})
    summary = here / "snapshot_v2/summary.json"
    write(summary, {"schema": "instinctflash.current_public_pipeline_qualification.v1", "status": "current_pipeline_qualified",
                    "counts": {"main_passed": 8, "foreign_passed": 6, "historical_families_assessed": 8},
                    "complete_historical_reproduction": False, "task_quality_certified": False,
                    "prior_cosmos_SCREEN_transferred": False, "assembler": ref(here / "assembler.py"),
                    "contract": ref(here / "contract.json"), "source_stage_manifest": ref(here / "stage.json"),
                    "selection": ref(selection)})
    assert m.validate_current(tmp_path, summary, ref(summary)["sha256"])[1] == selection
    write(raw, {"status": "changed"})
    with pytest.raises(ValueError, match="raw evidence changed"):
        m.validate_current(tmp_path, summary, ref(summary)["sha256"])


def sde1_fixture(tmp_path):
    """Synthetic receipt chain using the actual frozen pure command constructor."""
    rules = m.policy()
    spec = rules["optional_SDE1"]
    original_study = PATH.parent.parent
    study = tmp_path / rules["study_relative"]
    for key in ("queue_source", "queue_template"):
        target = study / spec[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original_study / spec[key]).read_bytes())
    for control in spec["queue_control_files"].values():
        target = study / control["path"]
        target.write_bytes((original_study / control["path"]).read_bytes())
    queue_spec = importlib.util.spec_from_file_location("synthetic_SDE1_commands", study / spec["queue_source"])
    queue = importlib.util.module_from_spec(queue_spec)
    queue_spec.loader.exec_module(queue)
    config = m.read(study / spec["queue_template"])
    config["template_only"] = False
    base = study / spec["root"]
    write(base / "config.json", config)
    prefix = Path(config["python"]).parent.parent / "lib/python3.13/site-packages"
    source_guard = {str(prefix / name): value for files in config["installed_sources"].values() for name, value in files.items()}
    write(base / "source_guard.json", source_guard)
    operations = queue.commands(config)
    done = {"status": spec["completed_status"], "category": "SCREEN", "task_quality_certified": False,
            "automatic_retry": False, "operations": [{"id": row["id"], "process": {"command": row["argv"], "returncode": 0}}
                                                       for row in operations], "screens": []}
    def remote(path, original):
        return {"path": str(original), "bytes": path.stat().st_size, "sha256": m.digest(path.read_bytes())}
    for i, name in enumerate(spec["recipes"]):
        archive = base / "archives" / name
        planned = next(row for row in operations if row["id"] == name + "-run")
        original = Path(config["run_root"]) / "runs" / name
        count = 36 if name == "nano-original" else 16
        extra = "budget_manifest.json" if name == "nano-original" else "instinctcompress_manifest.json"
        write(archive / "preparation" / extra, {"synthetic": name})
        write(archive / "preparation/public_preparation.json", {"recipe": name, "category": "SCREEN", "task_quality_certified": False})
        (archive / "run").mkdir()
        np.savez(archive / "run/receipt.npz", actions=np.zeros((count, 32, 8), dtype=np.float32))
        write(archive / "run/receipt.json", {"ok": True, "category": "SCREEN", "task_quality_certified": False,
              "p50_ms": 123.5 + i, "actions_sha256": m.digest((archive / "run/receipt.npz").read_bytes()),
              "checkpoint_manifest_sha256": m.digest((archive / "preparation" / extra).read_bytes())})
        write(archive / "run/completion.json", {"status": "passed", "returncode": 0, "category": "SCREEN",
              "task_quality_certified": False, "p50_ms": 123.5 + i})
        write(archive / "run/invocation.json", {"synthetic": name})
        (archive / "run/run.log").write_text("synthetic only\n")
        screen = {"status": "passed", "recipe": name, "category": "SCREEN", "task_quality_certified": False,
                  "p50_ms": 123.5 + i, "requests": count, "action_shape": [count, 32, 8]}
        for key, filename in (("completion", "completion.json"), ("receipt", "receipt.json"), ("actions", "receipt.npz")):
            screen[key] = remote(archive / "run" / filename, original / filename)
        write(archive / "validation.json", screen)
        done["screens"].append(screen)
        next(row for row in done["operations"] if row["id"] == name + "-run")["screen"] = screen
        inventory = []
        for section in ("preparation", "run"):
            origin = original if section == "run" else Path(planned["prepared"])
            for path in sorted((archive / section).iterdir()):
                inventory.append({"source": remote(path, origin / path.name),
                                  "copy": remote(path, Path(config["run_root"]) / "archives" / name / section / path.name)})
        write(archive / "archive.json", {"status": "archived_small_evidence", "large_weight_bytes_copied": 0, "files": inventory})
    write(base / "completion.json", done)
    return rules, base


def test_optional_candidates_never_walk_tokenizers_media_or_weights(tmp_path):
    rules, base = sde1_fixture(tmp_path)
    excluded = base / "archives/edge-seed12031/preparation/tokenizer.json"
    excluded.write_text("not selected")
    (excluded.parent / "model.safetensors").write_bytes(b"not selected")
    found, missing = m.sde1_candidates(tmp_path, rules)
    assert len(found) == 30 and len(missing) == 35
    assert all("tokenizer" not in path and not path.endswith(".safetensors") for path in found)
    result = m.sde1_evidence(tmp_path, rules)
    assert result["completed"] is True and len(result["validations"]) == 3
    assert result["task_quality_certified"] is result["belongs_to_main_8_plus_6"] is False
    assert result["independent_scientific_replay_performed"] is False


@pytest.mark.parametrize("mutation", ["source_guard", "template", "failed_operation", "wrong_recipe", "validation", "actions", "quality",
                                     "protected_config", "protected_source"])
def test_completed_sde1_requires_exact_final_source_config_and_all_recipe_chains(tmp_path, mutation):
    rules, base = sde1_fixture(tmp_path)
    if mutation == "source_guard":
        path = base / "source_guard.json"
        value = m.read(path)
        value[next(iter(value))] = "0" * 64
    elif mutation == "template":
        path = base / "config.json"
        value = m.read(path)
        value["template_only"] = True
    elif mutation == "validation":
        path = base / "archives/edge-seed12031/validation.json"
        value = m.read(path)
        value["p50_ms"] = 1
    elif mutation == "actions":
        path = base / "archives/nano-original/run/receipt.npz"
        path.write_bytes(path.read_bytes() + b"changed")
    elif mutation == "protected_config":
        path = base / "config.json"
        value = m.read(path)
        value["handle_inspector"]["sha256"] = "0" * 64
    elif mutation == "protected_source":
        control = rules["optional_SDE1"]["queue_control_files"]["protected_service_manifest"]
        path = tmp_path / rules["study_relative"] / control["path"]
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = base / "completion.json"
        value = m.read(path)
        if mutation == "failed_operation":
            value["operations"][0]["process"]["returncode"] = 1
        elif mutation == "wrong_recipe":
            value["screens"][2]["recipe"] = "edge-seed12031"
        else:
            value["task_quality_certified"] = True
    if mutation not in ("actions", "protected_source"):
        write(path, value)
    with pytest.raises(ValueError):
        m.sde1_evidence(tmp_path, rules)


def test_absent_partial_and_failed_sde1_never_become_completed(tmp_path):
    assert m.sde1_evidence(tmp_path, m.policy())["status"] == "not_present"
    rules, base = sde1_fixture(tmp_path)
    path = base / "archives/nano-original/validation.json"
    path.unlink()
    partial = m.sde1_evidence(tmp_path, rules)
    assert partial["completed"] is False and partial["missing_completion_references"]
    done = m.read(base / "completion.json")
    done["status"] = "failed_preserved_no_retry"
    write(base / "completion.json", done)
    failed = m.sde1_evidence(tmp_path, rules)
    assert failed["completed"] is False and failed["native_reported_status"] == "failed_preserved_no_retry"


def test_renderer_bindings_require_same_summary_and_preserve_edited_final_prose(tmp_path):
    rules = m.policy()
    original_study = PATH.parent.parent
    study = tmp_path / rules["study_relative"]
    summary = study / "current_public_pipeline_v1/snapshot_v4/summary.json"
    write(summary, {"synthetic": True})
    expected = ref(summary)["sha256"]
    for kind, spec in rules["renderer_bindings"].items():
        source, candidate = study / spec["source"], study / spec["candidate"]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((original_study / spec["source"]).read_bytes())
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("generated " + kind)
        write(study / spec["binding"], {"schema": spec["schema"], "status": "candidate_from_completed_qualification",
              "summary": {"path": str(summary), "sha256": expected}, "renderer": ref(source), "candidate": ref(candidate)})
    (study / "results.rst").write_text("Root reviewed and edited final prose")
    result = m.validate_renderers(tmp_path, summary, expected, rules)
    assert result["reader_entry_matches_generated_candidate"] is False
    with pytest.raises(ValueError, match="summary differs"):
        m.validate_renderers(tmp_path, summary, "0" * 64, rules)


def test_frozen_v3_caps_path_credentials_and_current_evidence_guards_are_unchanged():
    previous = ast.parse(PATH.with_name("curate_v3.py").read_text())
    current = ast.parse(PATH.read_text())
    names = {"read", "relative", "safe", "checked_ref", "inspect_file", "walk", "named_files", "validate_current", "provenance_index"}
    def selected(tree):
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names}
    assert selected(previous) == selected(current)
    def credentials(tree):
        return next(ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(name, ast.Name) and name.id == "SECRET_PATTERNS" for name in node.targets))
    assert credentials(previous) == credentials(current)
    old = m.read(PATH.with_name("policy_v3.json"))
    new = m.policy()
    for key in ("max_file_bytes", "max_total_bytes", "max_files", "allowed_extensions", "fixture_sha256", "ignored_directory_names"):
        assert new[key] == old[key]
    assert len(new["optional_SDE1"]["candidate_files"]) == 65

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).with_name("curate.py")
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
                "current_qualification": {"summary": {"path": "/unused", "sha256": "a" * 64}},
                "files": {name: m.inspect_file(repo, name, m.policy()) for name in ("first.json", "second.json")}}
    path = repo / "manifest.json"
    write(path, manifest)
    monkeypatch.setattr(m, "validate_current", lambda *args: None)
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
                 "current_qualification": {"summary": {"path": "/unused", "sha256": "a" * 64}},
                 "files": {"raw.json": m.inspect_file(repo, "raw.json", m.policy())}})
    monkeypatch.setattr(m, "validate_current", lambda *args: None)
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

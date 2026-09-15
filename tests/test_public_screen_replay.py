"""CPU fixtures for exact SCREEN input replay; no native/model execution."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from benchmarks.regression import screen_replay as replay
from benchmarks.regression.user_e2e import request_hash
from benchmarks.vla.cosmos_quality_policy import frozen_cell


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "CELLS", {name: 6 for name in replay.CELLS})
    root = tmp_path / "fixture"
    root.mkdir()
    arrays = {"images": np.zeros((6, 540, 640, 3), np.uint8),
              "joints": np.zeros((6, 7), np.float32), "grippers": np.zeros((6, 1), np.float32),
              "actions": np.zeros((6, 32, 8), np.float32)}
    rows = [{"episode_id": f"formal/fixture/{i:04d}", "request_id": 0,
             "request_seed": 42 + i, "benchmark_seed": 42 + i, "max_policy_chunks": 1,
             "prompt": "move the object", "historical_predict_host_seconds": 2.0,
             "action_sha256": request_hash(arrays["actions"][i])} for i in range(6)]
    for i, row in enumerate(rows):
        row["input_sha256"] = request_hash(replay.observation(arrays, row, i))
    np.savez_compressed(root / "arrays.npz", **arrays)
    entry = {"file": "arrays.npz", "bytes": (root / "arrays.npz").stat().st_size,
             "sha256": replay.sha(root / "arrays.npz"), "requests": rows}
    replay.write(root / "manifest.json", {"schema": replay.SCHEMA,
                 "source_latency_sha256": replay.SOURCE_LATENCY_SHA, "task_quality_certified": False,
                 "cells": {name: entry for name in replay.CELLS}})
    return root, replay.sha(root / "manifest.json"), arrays, entry


@pytest.mark.parametrize("name", list(replay.CELLS))
def test_exact_matrix_contract_accepted_by_existing_policy(name, tmp_path):
    library = tmp_path / "bf16.so"
    library.write_bytes(b"CPU binding fixture")
    cell = replay.matrix_cell(name, library if name == "edge-runtime_selected" else None)
    path = tmp_path / "matrix.json"
    replay.write(path, {"cells": [cell]})
    assert frozen_cell(path, name, matrix_sha256=replay.sha(path)) == cell
    assert cell["effective_schedule"]["steps"] == 4
    assert cell["effective_schedule"]["guidance"] == 3.0


def test_plan_has_no_quality_or_outcome_claim(fixture):
    root, digest, _, _ = fixture
    result = replay.plan(root, digest, "edge-eager_native")
    assert result["request_count"] == 6
    assert result["task_quality_certified"] is False
    assert result["historical_outcomes_replayed"] is False


def test_bitwise_action_comparison_retains_signed_zero_distinction():
    left = np.zeros((32, 8), np.float32)
    right = left.copy()
    right[0, 0] = -0.0
    assert np.array_equal(left, right)
    assert replay.action_comparison(left, right) == {"bitwise_equal": False, "max_abs_error": 0.0}
    assert replay.action_comparison(left, left.copy())["bitwise_equal"] is True


def test_observation_roundtrip_binds_full_pixels_state_and_actions(fixture):
    root, _, arrays, entry = fixture
    actual = replay.load_arrays(root, entry)
    replay.verify_inputs(actual, entry)
    for name in arrays:
        assert np.array_equal(arrays[name], actual[name])
    actual["images"][0, 1, 2, 0] = 1
    with pytest.raises(ValueError, match="observation hash"):
        replay.verify_inputs(actual, entry)


@pytest.mark.parametrize("field,value,match", [("request_seed", 99, "seed"),
                                               ("episode_id", "formal/fixture/0001", "order"),
                                               ("request_id", 1, "reset")])
def test_reordered_reset_or_seed_rejected(fixture, field, value, match):
    _, _, arrays, original = fixture
    entry = copy.deepcopy(original)
    entry["requests"][0][field] = value
    with pytest.raises(ValueError, match=match):
        replay.verify_inputs(arrays, entry)


def test_changed_manifest_and_archive_rejected(fixture):
    root, digest, _, entry = fixture
    with pytest.raises(ValueError, match="manifest hash"):
        replay.fixture_catalog(root, "0" * 64)
    with (root / entry["file"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        replay.load_arrays(root, entry)
    assert replay.fixture_catalog(root, digest)["task_quality_certified"] is False


def test_symlink_fixture_escape_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    other = tmp_path / "outside.npz"
    other.write_bytes(b"not data")
    (root / "escape.npz").symlink_to(other)
    with pytest.raises(ValueError, match="leaves"):
        replay.contained(root, "escape.npz")


def test_prepare_accepts_exact_hf_reference_symlink_and_preserves_binding(fixture, tmp_path, monkeypatch):
    root, digest, _, _ = fixture
    physical = tmp_path / "physical-checkpoint"
    physical.mkdir()
    reference = tmp_path / replay.CHECKPOINTS["edge"][1]
    reference.symlink_to(physical, target_is_directory=True)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **k: str(reference)))
    output = tmp_path / "prepared"
    replay.prepare(root, digest, "edge-eager_native", output)
    value, receipt = replay.loaded_plan(output)
    assert Path(receipt["checkpoint_reference"]) == reference
    assert Path(receipt["checkpoint_path"]) == physical
    assert value["cell"]["revision"] == reference.name
    with pytest.raises(ValueError, match="exists"):
        replay.prepare(root, digest, "edge-eager_native", output)


def test_prepare_wrong_revision_retains_failure(fixture, tmp_path, monkeypatch):
    root, digest, _, _ = fixture
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **k: str(wrong)))
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="checkpoint reference"):
        replay.prepare(root, digest, "edge-eager_native", output)
    assert (output / "failure.json").is_file()
    assert not (output / "prepared.json").exists()


def test_partial_capture_cannot_publish_report(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "loaded_plan", lambda _: ({}, {}))
    output = tmp_path / "partial"
    output.mkdir()
    with pytest.raises(ValueError):
        replay.report(tmp_path, output)


def test_cli_plan_uses_explicit_fixture_binding(fixture, capsys):
    root, digest, _, _ = fixture
    replay.main(["plan", "--fixture", str(root), "--fixture-sha256", digest,
                 "--cell", "nano-eager_native"])
    assert json.loads(capsys.readouterr().out)["cell"]["family"] == "nano"


def test_run_scrubs_shell_overrides_and_uses_fresh_isolated_child(tmp_path, monkeypatch):
    cell = replay.matrix_cell("edge-eager_native")
    monkeypatch.setattr(replay, "loaded_plan", lambda _: ({"cell": cell}, {"cache_dir": None}))
    monkeypatch.setenv("IFL_UNDECLARED_TEST_OVERRIDE", "1")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/source")
    output = tmp_path / "run"
    observed = {}

    class Child:
        def __init__(self, command, **kwargs):
            observed.update(command=command, **kwargs)
            replay.write(output / "capture_complete.json", {"task_quality_certified": False})

        def wait(self, timeout):
            assert timeout == 60
            return 0

    monkeypatch.setattr(replay.subprocess, "Popen", Child)
    monkeypatch.setattr(replay, "report", lambda *_: {"task_quality_certified": False})
    assert replay.run(tmp_path, output, timeout=60, lock_path=tmp_path / "lock") == {"task_quality_certified": False}
    assert observed["command"][1:4] == ["-I", "-B", "-m"]
    assert observed["start_new_session"] is True
    assert not any(k.startswith("IFL_") for k in observed["env"])
    assert "PYTHONPATH" not in observed["env"]
    assert observed["env"]["TORCHDYNAMO_DISABLE"] == "1"
    assert observed["env"]["HF_HUB_OFFLINE"] == "1"
    with pytest.raises(ValueError, match="exists"):
        replay.run(tmp_path, output, timeout=60, lock_path=tmp_path / "lock")


def test_run_interrupt_reaps_owned_child_before_releasing_lock(tmp_path, monkeypatch):
    cell = replay.matrix_cell("nano-eager_native")
    monkeypatch.setattr(replay, "loaded_plan", lambda _: ({"cell": cell}, {}))
    events = []

    class Child:
        pid = 12345

        def __init__(self, *args, **kwargs):
            self.calls = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.calls += 1
            events.append(("wait", timeout))
            if self.calls == 1:
                raise KeyboardInterrupt
            return 0

    monkeypatch.setattr(replay.subprocess, "Popen", Child)
    monkeypatch.setattr(replay.os, "killpg", lambda pid, sig: events.append(("signal", pid, sig)))
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        replay.run(tmp_path, output, timeout=60, lock_path=tmp_path / "lock")
    assert events == [("wait", 60), ("signal", 12345, replay.signal.SIGTERM), ("wait", 30)]
    assert (output / "failure.json").is_file()
    assert not (output / "completion.json").exists()

"""Portable benchmark frames preserve historical inputs without decoding arbitrary pickle."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regression.user_e2e import RecordedInputs, request_hash


def test_public_fixture_preserves_all_family_requests_and_feedback():
    root = Path(__file__).resolve().parents[1]
    public = root / "benchmarks/regression/fixtures/recorded_inputs_v1.npz"
    historical = root / "eval/native_total_2026-09-10/fixtures/va_eval_obs.npz"
    current = RecordedInputs(public)
    if not historical.is_file():
        pytest.skip("Historical source fixture is internal conversion evidence, not a public dependency")
    previous = RecordedInputs(historical)
    for family in ("pi05", "vla4", "vla2", "groot", "edge", "nano", "va", "dreamzero"):
        for i in range(25):
            assert request_hash(current.observation(family, i, i % 3)) == request_hash(
                previous.observation(family, i, i % 3)), (family, i)
    assert np.array_equal(current.feedback, previous.feedback)


def test_public_fixture_loads_without_pickle():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/regression/fixtures/recorded_inputs_v1.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "37843e22fa6dd9a2abf2bae390ccb8e5c4319446e3d3fab5d0d477860065f411"
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"frames", "feedback"}
        assert data["frames"].dtype == np.uint8
        assert data["feedback"].dtype == np.float32
    fixture = RecordedInputs(path)
    assert len(fixture.frames) == 13


def test_unknown_object_archive_is_rejected_before_pickle_decode(tmp_path, monkeypatch):
    path = tmp_path / "untrusted.npz"
    np.savez(path, jpeg_0=np.array([b"untrusted"], dtype=object))
    original = np.load
    calls = []

    def restricted(*args, **kwargs):
        calls.append(kwargs.get("allow_pickle"))
        assert kwargs.get("allow_pickle") is False
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "load", restricted)
    with pytest.raises(ValueError, match="non-pickle"):
        RecordedInputs(path)
    assert calls == [False]

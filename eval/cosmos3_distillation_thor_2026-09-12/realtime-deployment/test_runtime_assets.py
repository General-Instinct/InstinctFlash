"""Unknown or changing external weights must not pass qualification."""
import hashlib
from types import SimpleNamespace

import pytest
import runtime_assets as assets


def fixture(tmp_path, monkeypatch, loader=None):
    path=tmp_path/'Wan2.2_VAE.pth'; path.write_bytes(b'fixture weights')
    monkeypatch.setattr(assets,'WAN_BYTES',path.stat().st_size)
    monkeypatch.setattr(assets,'WAN_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
    return path,SimpleNamespace(load=loader or (lambda file, **kwargs: 'loaded'))


def test_capture_actual_load_and_restore(tmp_path, monkeypatch):
    path,io=fixture(tmp_path,monkeypatch); original=io.load
    with assets.audit_vae_load(io) as records:
        assert io.load(file=path)=='loaded'
    assert io.load is original and records==[assets.identify(path)]


def test_unknown_bytes_rejected_before_loader(tmp_path,monkeypatch):
    calls=[]
    path,io=fixture(tmp_path,monkeypatch,lambda *a,**k:calls.append(a));original=io.load
    path.write_bytes(b'unknown weights')
    with pytest.raises(ValueError,match='anchor'),assets.audit_vae_load(io): io.load(path)
    assert not calls and io.load is original


def test_changed_during_load_rejected(tmp_path,monkeypatch):
    path,io=fixture(tmp_path,monkeypatch)
    io.load=lambda *a,**k:path.write_bytes(b'changed payload')
    original=io.load
    with pytest.raises(ValueError,match='changed while'),assets.audit_vae_load(io): io.load(path)
    assert io.load is original


def test_missing_load_and_native_failure_restore(tmp_path,monkeypatch):
    path,io=fixture(tmp_path,monkeypatch);original=io.load
    with pytest.raises(RuntimeError,match='exactly one'),assets.audit_vae_load(io):
        assert io.load('unrelated.json')=='loaded'
    assert io.load is original
    def fail(*a,**k): raise RuntimeError('native failure')
    io.load=fail
    with pytest.raises(RuntimeError,match='native failure'),assets.audit_vae_load(io): io.load(path)
    assert io.load is fail

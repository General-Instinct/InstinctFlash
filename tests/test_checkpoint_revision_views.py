"""Pinned loads must not be redirected by a later load of the same Hub id."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from instinctflash.descriptors.known import lookup
from instinctflash.descriptors.package import _declared_view


@pytest.mark.parametrize("offline", ["1", "yes", "TRUE", "On"])
def test_offline_pinned_hub_load_never_requests_a_remote_tree(tmp_path, monkeypatch, offline):
    import sys
    from instinctflash.descriptors.package import from_pretrained

    monkeypatch.setenv("HF_HUB_OFFLINE", offline)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    revision = "a" * 40
    source = snapshot(tmp_path, revision, "weights")
    (source / "config.json").write_text('{"type":"pi05"}')

    def download(repo_id, **options):
        assert repo_id == "lerobot/pi05_libero_finetuned_v044"
        assert options["revision"] == revision
        if not options.get("local_files_only"):
            raise AssertionError("Pinned snapshot attempted a network tree request")
        return str(source)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    checkpoint = from_pretrained("lerobot/pi05_libero_finetuned_v044", revision=revision)
    assert (Path(checkpoint.path) / "model.safetensors").read_bytes() == b"weights"


@pytest.mark.parametrize("offline", [None, "0", "false"])
def test_online_hub_load_keeps_revision_resolution(tmp_path, monkeypatch, offline):
    import sys
    from instinctflash.descriptors.package import from_pretrained

    if offline is None:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    else:
        monkeypatch.setenv("HF_HUB_OFFLINE", offline)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source = snapshot(tmp_path, "snapshot", "weights")
    (source / "config.json").write_text('{"type":"pi05"}')

    def download(repo_id, *, revision):
        assert revision == "main"
        return str(source)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    assert from_pretrained("lerobot/pi05_libero_finetuned_v044", revision="main").path


def snapshot(root, name, value):
    path = root / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"revision": value}))
    (path / "model.safetensors").write_bytes(value.encode())
    return path


def test_revisions_and_declarations_do_not_retarget_live_views(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    model = "lerobot/pi05_libero_finetuned_v044"
    first = snapshot(tmp_path, "first", "a")
    second = snapshot(tmp_path, "second", "b")
    declaration = lookup(model)
    a = _declared_view(first, model, declaration)
    b = _declared_view(second, model, declaration)
    changed = copy.deepcopy(declaration)
    changed["execution"]["nfe"]["action"] = 4
    c = _declared_view(first, model, changed)
    assert len({a, b, c}) == 3
    assert (a / "model.safetensors").read_bytes() == b"a"
    assert (b / "model.safetensors").read_bytes() == b"b"
    assert json.loads((a / "instinctflash.json").read_text()) == declaration
    assert json.loads((c / "instinctflash.json").read_text()) == changed


def test_concurrent_loads_publish_complete_views(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    model = "lerobot/pi05_libero_finetuned_v044"
    source = snapshot(tmp_path, "source", "weights")
    with ThreadPoolExecutor(max_workers=8) as pool:
        views = list(pool.map(lambda _: _declared_view(source, model, lookup(model)), range(24)))
    assert len(set(views)) == 1
    assert (views[0] / "model.safetensors").read_bytes() == b"weights"
    assert not list(views[0].parent.glob(".building-*"))


def test_pi05_uses_pinned_local_native_weights(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/pi05_vla"))
    from pi05_iwm.adapter import _resolve_weights
    model = "lerobot/pi05_libero_finetuned_v044"
    source = snapshot(tmp_path, "source", "weights")
    checkpoint = SimpleNamespace(path=str(source), execution=SimpleNamespace(
        model_id=model, extra={"base_weights": model}))
    assert _resolve_weights(checkpoint) == str(source)
    checkpoint.execution.extra["base_weights"] = "another/explicit-pointer"
    assert _resolve_weights(checkpoint) == "another/explicit-pointer"

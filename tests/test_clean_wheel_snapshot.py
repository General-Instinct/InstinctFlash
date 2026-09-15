"""A wheel snapshot must preserve published Markdown and exclude local generated assets."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location(
    "clean_wheels", Path(__file__).resolve().parents[1] / "scripts" / "build_clean_wheels.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def test_snapshot_keeps_head_markdown_and_current_source_only(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    current = {
        "README.md": b"unpublished edits",
        "new.md": b"unpublished document",
        "pyproject.toml": b"current package config",
        "instinctflash/runtime/new.py": b"current source",
        "instinctflash/runtime/build/generated.py": b"generated",
        "instinctflash/native/kernel.so": b"binary",
        "eval/results.json": b"measurement",
        "serving/build/generated.py": b"generated",
        "benchmarks/regression/runtime_bundle.py": b"source",
        ".env": b"not package input",
    }
    for relative, content in current.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (repository / "eval" / "worker.py").write_bytes(b"worker module")
    (repository / "instinctflash" / "runtime" / "worker.py").symlink_to("../../eval/worker.py")
    current["instinctflash/runtime/worker.py"] = b""

    def git(repo, *args):
        if args == ("rev-parse", "HEAD"):
            return b"commit\n"
        if args[0] == "ls-tree":
            return b"README.md\0deleted.md\0pyproject.toml\0"
        if args[0] == "ls-files":
            return "\0".join(current).encode() + b"\0"
        if args[0] == "show":
            return {"HEAD:README.md": b"published README", "HEAD:deleted.md": b"published deleted document"}[args[1]]
        raise AssertionError(args)

    monkeypatch.setattr(builder, "_git", git)
    manifest = builder.create_snapshot(repository, tmp_path / "snapshot")
    assert set(manifest["files"]) == {"README.md", "deleted.md", "pyproject.toml",
                                      "instinctflash/runtime/new.py", "instinctflash/runtime/worker.py",
                                      "benchmarks/regression/runtime_bundle.py"}
    source = Path(manifest["snapshot"])
    assert (source / "README.md").read_bytes() == b"published README"
    assert (source / "instinctflash/runtime/new.py").read_bytes() == b"current source"
    assert (source / "instinctflash/runtime/worker.py").read_bytes() == b"worker module"
    assert not (source / "instinctflash/runtime/worker.py").is_symlink()
    assert (repository / "README.md").read_bytes() == b"unpublished edits"


@pytest.mark.parametrize("relative", ["/tmp/lib.so", "../lib.so", "serving/flash_rt/../lib.so",
                                      "serving/flash_rt/weights.bin", "other/lib.so"])
def test_native_injection_cannot_escape_package_or_copy_weights(relative):
    assert not builder._native_destination(relative)


def test_native_injection_is_explicit_and_hash_bound(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    native = tmp_path / "cached.so"
    native.write_bytes(b"native bytes for a CPU packaging fixture")
    monkeypatch.setattr(builder, "_git", lambda *args: b"abc\n" if args[1] == "rev-parse" else b"")
    destination = "serving/flash_rt/flash_rt_kernels.so"
    manifest = builder.create_snapshot(repository, tmp_path / "snapshot", native_files=[(native, destination)])
    assert manifest["native_files"] == [destination]
    assert manifest["files"][destination]["origin"] == "explicit_native_file"
    assert len(manifest["files"][destination]["sha256"]) == 64
    assert (Path(manifest["snapshot"]) / destination).read_bytes() == native.read_bytes()


def test_snapshot_refuses_output_in_checkout(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        builder.create_snapshot(tmp_path, tmp_path / "output")

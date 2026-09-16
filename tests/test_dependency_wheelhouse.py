"""Transported archives are verified against the original recipe before use."""
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("wheelhouse_bootstrap", ROOT / "scripts/bootstrap_vendor.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


@pytest.fixture
def wheelhouse(tmp_path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    filename = "fixture-1.0-py3-none-any.whl"
    path = wheels / filename
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fixture-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: fixture\nVersion: 1.0\n")
        # Real setuptools wheels also carry vendored dependency metadata.
        archive.writestr("fixture/vendor/nested.dist-info/METADATA", "Name: nested\nVersion: 2\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    url = f"https://files.pythonhosted.org/packages/test/{filename}"
    profile = copy.deepcopy(bootstrap.load_profile("pi05", target="rtx4090"))
    profile["public_wheel_overrides"] = {"fixture": url + "#sha256=" + digest}
    manifest = {
        "schema": "instinctflash.dependency_wheelhouse.v1", "family": "pi05", "target": "rtx4090",
        "wheel_directory": "wheels",
        "inputs": {"requirements_sha256": profile["requirements"]["sha256"],
                   "constraints_sha256": profile["constraints"]["sha256"],
                   "public_wheel_overrides": profile["public_wheel_overrides"]},
        "wheels": [{"filename": filename, "name": "fixture", "version": "1.0", "url": url,
                    "sha256": digest, "bytes": path.stat().st_size}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, profile, manifest, path


def test_verified_cache_resolves_exact_public_pin_without_loading_package(wheelhouse):
    directory, profile, _, path = wheelhouse
    receipt = bootstrap.admit_dependency_wheelhouse(directory, profile)
    assert receipt["verified_wheels"] == 1
    assert receipt["explicit_override_paths"] == {"fixture": str(path)}
    assert receipt["wheel_directory"] == str(directory / "wheels")
    assert "fixture" not in sys.modules


@pytest.mark.parametrize("mutation", ["bytes", "hash", "target", "pin", "extra", "traversal",
                                      "linked", "metadata", "source"])
def test_tampered_cache_cannot_replace_a_pinned_dependency(wheelhouse, mutation):
    directory, profile, manifest, path = wheelhouse
    if mutation == "bytes":
        path.write_bytes(path.read_bytes() + b"tamper")
    elif mutation == "hash":
        content = bytearray(path.read_bytes())
        content[20] ^= 1
        path.write_bytes(content)
    elif mutation == "target":
        manifest["target"] = "jetson_thor"
    elif mutation == "pin":
        # Even a self-consistent edited manifest cannot raise the original pin.
        manifest["wheels"][0]["sha256"] = "a" * 64
        manifest["inputs"]["public_wheel_overrides"] = {"fixture": "https://files.pythonhosted.org/new#sha256=" + "a" * 64}
    elif mutation == "extra":
        (path.parent / "unlisted-1-py3-none-any.whl").write_bytes(b"not admitted")
    elif mutation == "traversal":
        manifest["wheel_directory"] = "../outside"
    elif mutation == "linked":
        original = directory / "outside.whl"
        path.rename(original)
        path.symlink_to(original)
    elif mutation == "metadata":
        manifest["wheels"][0]["name"] = "another-package"
    elif mutation == "source":
        manifest["wheels"][0]["url"] = "https://unrelated.invalid/" + path.name
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bootstrap.admit_dependency_wheelhouse(directory, profile)


def test_two_transported_copies_use_the_same_verified_artifact_url(wheelhouse):
    directory, profile, manifest, source = wheelhouse
    first = bootstrap.admit_dependency_wheelhouse(directory, profile)
    cache = directory / "shared"
    one = bootstrap.share_dependency_artifacts(first, cache)
    other = directory / "second-family-cache"
    shutil.copytree(source.parent, other / "wheels")
    (other / "manifest.json").write_text(json.dumps(manifest))
    two = bootstrap.share_dependency_artifacts(bootstrap.admit_dependency_wheelhouse(other, profile), cache)
    assert one == two
    target = Path(one["paths"]["fixture"])
    assert os.stat(target).st_ino != os.stat(source).st_ino
    assert one["constraints"] == f"fixture @ {target.as_uri()}\n"
    assert one["explicit_override_paths"] == {"fixture": str(target)}


@pytest.mark.parametrize("mutation", ["content", "symlink", "manifest"])
def test_changed_shared_artifact_is_rejected_without_replacement(wheelhouse, mutation):
    directory, profile, _, _ = wheelhouse
    admission = bootstrap.admit_dependency_wheelhouse(directory, profile)
    shared = bootstrap.share_dependency_artifacts(admission, directory / "shared")
    target = Path(shared["paths"]["fixture"])
    if mutation == "content":
        target.write_bytes(b"changed")
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to(directory / "missing")
    else:
        (directory / "manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        bootstrap.share_dependency_artifacts(admission, directory / "shared")


def test_shared_artifact_never_links_the_callers_archive_inode(wheelhouse, monkeypatch):
    directory, profile, _, source = wheelhouse
    admission = bootstrap.admit_dependency_wheelhouse(directory, profile)
    original_link = os.link

    def refuse_caller_link(src, dst):
        if src == source:
            pytest.fail("canonical archive must own an independent inode")
        return original_link(src, dst)

    monkeypatch.setattr(bootstrap.os, "link", refuse_caller_link)
    result = bootstrap.share_dependency_artifacts(admission, directory / "shared")
    target = Path(result["paths"]["fixture"])
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_ino != source.stat().st_ino
    assert list(target.parent.iterdir()) == [target]


def test_caller_links_and_copied_family_views_preserve_canonical_ctime(wheelhouse):
    directory, profile, manifest, source = wheelhouse
    shared = bootstrap.share_dependency_artifacts(
        bootstrap.admit_dependency_wheelhouse(directory, profile), directory / "shared")
    target = Path(shared["paths"]["fixture"])
    before = target.stat()
    assert before.st_nlink == 1
    os.link(source, directory / "another-caller-link.whl")
    other = directory / "family-two"
    (other / "wheels").mkdir(parents=True)
    # Future family views copy canonical bytes without adding a canonical link.
    shutil.copyfile(target, other / "wheels" / source.name)
    (other / "manifest.json").write_text(json.dumps(manifest))
    bootstrap.share_dependency_artifacts(
        bootstrap.admit_dependency_wheelhouse(other, profile), directory / "shared")
    after = target.stat()
    assert after.st_ino == before.st_ino
    assert after.st_nlink == before.st_nlink == 1
    assert after.st_ctime_ns == before.st_ctime_ns
    assert after.st_mtime_ns == before.st_mtime_ns


@pytest.fixture
def original_inputs(wheelhouse, monkeypatch):
    directory, profile, manifest, _ = wheelhouse
    checkout = directory / "checkout"
    rule_directory = checkout / "release/vendor/fixture"
    rule_directory.mkdir(parents=True)
    overlay_directory = directory / "overlay_inputs"
    overlay_directory.mkdir()
    original = overlay_directory / "overlay-1.0-py3-none-any.whl"
    member = "overlay-1.0.dist-info/METADATA"
    with zipfile.ZipFile(original, "w") as archive:
        archive.writestr(member, "Metadata-Version: 2.1\nName: overlay\nVersion: 1.0\n")
    rule = {"filename": original.name, "url": "https://files.pythonhosted.org/packages/test/" + original.name,
            "bytes": original.stat().st_size, "sha256": bootstrap.sha(original), "metadata_member": member}
    rule_path = rule_directory / "overlay.json"
    rule_path.write_text(json.dumps(rule))
    reference = {"path": "fixture/overlay.json", "sha256": bootstrap.sha(rule_path)}
    profile["python_metadata_overlays"] = [reference]
    source_directory = directory / "source_archives"
    source_directory.mkdir()
    source = source_directory / "antlr4-python3-runtime-4.9.3.tar.gz"
    source.write_bytes(b"fixture audited original bytes")
    source_sha = bootstrap.sha(source)
    url = "https://files.pythonhosted.org/packages/test/" + source.name
    profile["audited_pure_python_sdists"] = {"antlr4-python3-runtime": {
        "version": "4.9.3", "url": url + "#sha256=" + source_sha, "bytes": source.stat().st_size}}
    monkeypatch.setitem(bootstrap.PURE_SDISTS, "antlr4-python3-runtime", ("4.9.3", source_sha))
    manifest["schema"] = "instinctflash.dependency_wheelhouse.v2"
    manifest["inputs"].update(python_metadata_overlays=copy.deepcopy(profile["python_metadata_overlays"]),
                              audited_pure_python_sdists=copy.deepcopy(profile["audited_pure_python_sdists"]))
    manifest["overlay_inputs"] = [{**rule, "name": "overlay", "version": "1.0",
                                    "rule_path": reference["path"], "rule_sha256": reference["sha256"]}]
    manifest["source_archives"] = [{"name": "antlr4-python3-runtime", "version": "4.9.3",
                                    "filename": source.name, "url": url,
                                    "sha256": source_sha, "bytes": source.stat().st_size}]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, profile, manifest, checkout, original, source


def test_original_sources_have_stable_paths_without_constraining_overlay_wheel(original_inputs):
    directory, profile, _, checkout, original, source = original_inputs
    admitted = bootstrap.admit_dependency_wheelhouse(directory, profile, checkout=checkout)
    assert admitted["verified_overlay_inputs"] == admitted["verified_source_archives"] == 1
    shared = bootstrap.share_dependency_artifacts(admitted, directory / "canonical")
    assert set(shared["paths"]) == {"fixture", "antlr4-python3-runtime"}
    assert "overlay @" not in shared["constraints"]
    assert shared["source_archive_paths"]["antlr4-python3-runtime"].endswith(source.name)
    canonical_original = Path(shared["overlay_input_paths"]["fixture/overlay.json"])
    assert canonical_original.stat().st_ino != original.stat().st_ino
    assert canonical_original.read_bytes() == original.read_bytes()


@pytest.mark.parametrize("mutation", ["source_bytes", "source_binding", "overlay_bytes", "overlay_rule",
                                      "missing", "unlisted", "linked", "input_binding", "wheel_collision"])
def test_original_dependency_inputs_cannot_replace_pinned_sources(original_inputs, mutation):
    directory, profile, manifest, checkout, original, source = original_inputs
    if mutation == "source_bytes":
        source.write_bytes(b"modified")
    elif mutation == "source_binding":
        source.write_bytes(b"self-consistent replacement")
        manifest["source_archives"][0].update(bytes=source.stat().st_size, sha256=bootstrap.sha(source))
    elif mutation == "overlay_bytes":
        original.write_bytes(b"modified")
    elif mutation == "overlay_rule":
        manifest["overlay_inputs"][0]["rule_sha256"] = "0" * 64
    elif mutation == "missing":
        manifest["source_archives"] = []
    elif mutation == "unlisted":
        (source.parent / "unexpected.tar.gz").write_bytes(b"unlisted")
    elif mutation == "linked":
        saved = directory / "source-original"
        source.rename(saved)
        source.symlink_to(saved)
    elif mutation == "input_binding":
        manifest["inputs"]["audited_pure_python_sdists"] = {}
    elif mutation == "wheel_collision":
        entry = manifest["overlay_inputs"][0]
        shutil.copyfile(original, directory / "wheels" / original.name)
        manifest["wheels"].append({key: entry[key] for key in ("name", "version", "filename", "url", "sha256", "bytes")})
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bootstrap.admit_dependency_wheelhouse(directory, profile, checkout=checkout)


def test_v1_cannot_silently_enable_network_for_original_input_recipe(original_inputs):
    directory, profile, manifest, checkout, _, _ = original_inputs
    manifest["schema"] = "instinctflash.dependency_wheelhouse.v1"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requires wheelhouse v2"):
        bootstrap.admit_dependency_wheelhouse(directory, profile, checkout=checkout)


def test_offline_overlay_uses_admitted_original_and_existing_preparer(tmp_path, monkeypatch):
    from types import SimpleNamespace

    profile = bootstrap.load_profile("vla4", target="rtx4090")
    original = tmp_path / "verified-original.whl"
    original.write_bytes(b"preparer is checked separately against real original RECORD")
    instance = bootstrap.Bootstrap(SimpleNamespace(root=tmp_path / "run", env_dir=None,
        vendor_dir=None, cache_dir=None, checkout=ROOT), profile)
    instance.root.mkdir()
    instance.dependency_wheelhouse = {"overlay_input_paths": {
        profile["python_metadata_overlays"][0]["path"]: str(original)}}
    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", lambda *a, **k: pytest.fail("offline input must not download"))
    calls = []

    def prepare(source, output, rule):
        calls.append((source, rule["sha256"]))
        output.parent.mkdir(parents=True)
        output.write_bytes(b"prepared output")
        return {"status": "test-preparation"}

    monkeypatch.setattr(bootstrap, "prepare_inference_metadata", prepare)
    output, = instance.auxiliary_python_wheels()
    assert calls == [(original, "e53800ead8216861540ad3aebaf12e3cf87a399b3c1f234eeead33716c9c24fd")]
    assert output.read_bytes() == b"prepared output"


def test_offline_source_command_keeps_source_build_and_pins_isolated_build_tools(original_inputs, monkeypatch):
    from types import SimpleNamespace

    directory, profile, _, checkout, _, source = original_inputs
    admitted = bootstrap.admit_dependency_wheelhouse(directory, profile, checkout=checkout)
    args = SimpleNamespace(root=directory / "run", env_dir=None, vendor_dir=None, cache_dir=None,
        checkout=ROOT, timeout=10, python=sys.executable, uv="uv", repaired_wheel=None,
        repair_receipt=None, package_wheel_dir=None, dependency_wheelhouse=directory,
        dependency_artifact_cache=directory / "canonical")
    instance = bootstrap.Bootstrap(args, profile)
    monkeypatch.setattr(bootstrap, "admit_dependency_wheelhouse", lambda *a, **k: admitted)
    monkeypatch.setattr(bootstrap.subprocess, "check_output", lambda *a, **k: '["Linux","x86_64","3.12"]')
    monkeypatch.setattr(instance, "prepare_source", lambda: None)
    monkeypatch.setattr(instance, "auxiliary_python_wheels", lambda: [])
    commands = []

    class StopBeforeInstall(Exception):
        pass

    def command(label, argv, **kwargs):
        commands.append((label, argv))
        if label == "install_source_wheels":
            raise StopBeforeInstall

    monkeypatch.setattr(instance, "command", command)
    with pytest.raises(StopBeforeInstall):
        instance.install()
    argv = next(argv for label, argv in commands if label == "dependency_resolution")
    assert {"--offline", "--no-index", "--no-binary", "--build-constraints"} <= set(argv)
    selected = next(x for x in argv if x.startswith("antlr4-python3-runtime @ "))
    assert selected.startswith("antlr4-python3-runtime @ file://") and selected.endswith(source.name)
    constraints = Path(argv[argv.index("--build-constraints") + 1]).read_text()
    assert constraints.startswith("fixture @ file://")
    assert "antlr4-python3-runtime" not in constraints and "overlay" not in constraints
    assert not any("https://" in item for item in argv)
    for label, later in commands:
        if label in {"build_tools", "install_source_wheels"}:
            assert later[later.index("--no-binary") + 1] == "antlr4-python3-runtime"
            assert later[later.index("--build-constraints") + 1] == argv[argv.index("--build-constraints") + 1]
    assert not instance.env_dir.exists()

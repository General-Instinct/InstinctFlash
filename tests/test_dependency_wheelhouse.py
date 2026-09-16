"""Transported archives are verified against the original recipe before use."""
import copy
import errno
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
    assert os.stat(target).st_ino == os.stat(source).st_ino
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


def test_shared_artifact_cross_filesystem_copy_is_still_verified(wheelhouse, monkeypatch):
    directory, profile, _, source = wheelhouse
    admission = bootstrap.admit_dependency_wheelhouse(directory, profile)
    original_link = os.link

    def simulate_cross_filesystem(src, dst):
        if src == source:
            raise OSError(errno.EXDEV, "different filesystem")
        return original_link(src, dst)

    monkeypatch.setattr(bootstrap.os, "link", simulate_cross_filesystem)
    result = bootstrap.share_dependency_artifacts(admission, directory / "shared")
    target = Path(result["paths"]["fixture"])
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_ino != source.stat().st_ino
    assert list(target.parent.iterdir()) == [target]

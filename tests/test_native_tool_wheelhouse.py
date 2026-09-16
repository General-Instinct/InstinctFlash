import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("native_wheelhouse", ROOT / "scripts/prepare_native_tools.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
import bootstrap_vendor as bootstrap


def save_manifest(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest))


def wheelhouse(root):
    root.mkdir()
    (root / "wheels").mkdir()
    catalog = tools.load_catalog(tools.CATALOG)
    rows = []
    for name, version in catalog["packages"].items():
        filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        path = root / "wheels" / filename
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(f"{name.replace('-', '_')}-{version}.dist-info/METADATA",
                       f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        rows.append({"name": name, "version": version, "filename": filename,
                     "url": f"https://files.pythonhosted.org/packages/aa/bb/{filename}",
                     "bytes": path.stat().st_size, "sha256": tools.sha(path)})
    manifest = {"schema": "instinctflash.native_tool_wheelhouse.v1",
                "inputs": {"catalog_sha256": tools.sha(tools.CATALOG),
                           "constraints_sha256": catalog["constraints"]["sha256"],
                           "uv_version": catalog["uv_version"], "python_minor": "3.13",
                           "native_argv_prefix": catalog["native_argv_prefix"]},
                "target": "rtx4090", "python_minor": "3.13", "wheel_directory": "wheels",
                "package_versions": catalog["packages"], "wheels": rows}
    save_manifest(root, manifest)
    return manifest


def test_exact_catalog_full_hash_and_metadata_admission(tmp_path):
    cache = tmp_path / "cache"
    manifest = wheelhouse(cache)
    result = tools.admit_wheelhouse(cache, tools.CATALOG, target="rtx4090")
    assert result["status"] == "admitted" and len(result["wheels"]) == 23
    assert result["manifest_sha256"] == tools.sha(cache / "manifest.json")
    assert result["package_versions"] == manifest["package_versions"]
    assert all(len(x["metadata_sha256"]) == 64 for x in result["wheels"])


def test_5090_tool_reuse_requires_explicit_target_and_retains_all_23_pins(tmp_path):
    cache = tmp_path / "cache"
    manifest = wheelhouse(cache)
    with pytest.raises(ValueError, match="catalog or target"):
        tools.admit_wheelhouse(cache, tools.CATALOG, target="rtx5090")
    manifest["target"] = "rtx5090"
    save_manifest(cache, manifest)
    result = tools.admit_wheelhouse(cache, tools.CATALOG, target="rtx5090")
    assert result["target"] == "rtx5090" and len(result["wheels"]) == 23
    assert result["all_original_file_hashes_checked"] is True
    assert result["package_versions"] == tools.load_catalog(tools.CATALOG)["packages"]
    with pytest.raises(ValueError, match="catalog or target"):
        tools.admit_wheelhouse(cache, tools.CATALOG, target="rtx4090")


@pytest.mark.parametrize("fault", ["catalog", "constraints", "argv", "versions", "target", "minor",
                                   "missing", "duplicate", "entry-version", "entry-name", "path",
                                   "url", "url-auth", "url-query", "sha", "partial-sha", "size",
                                   "linked-wheel", "linked-directory", "extra-wheel"])
def test_untrusted_or_incomplete_tool_inputs_fail_before_subprocess(tmp_path, monkeypatch, fault):
    cache = tmp_path / "cache"
    m = wheelhouse(cache)
    entry = m["wheels"][0]
    path = cache / "wheels" / entry["filename"]
    if fault == "catalog":
        m["inputs"]["catalog_sha256"] = "0" * 64
    elif fault == "constraints":
        m["inputs"]["constraints_sha256"] = "0" * 64
    elif fault == "argv":
        m["inputs"]["native_argv_prefix"] = ["other"]
    elif fault == "versions":
        m["package_versions"]["hf"] = "99"
    elif fault == "target":
        m["target"] = "jetson_thor"
    elif fault == "minor":
        m["python_minor"] = "3.12"
    elif fault == "missing":
        m["wheels"].pop()
    elif fault == "duplicate":
        m["wheels"][-1] = dict(entry)
    elif fault == "entry-version":
        entry["version"] = "99"
    elif fault == "entry-name":
        entry["name"] = "torch"
    elif fault == "path":
        entry["filename"] = "../outside.whl"
    elif fault == "url":
        entry["url"] = entry["url"].replace("files.pythonhosted.org", "example.invalid")
    elif fault == "url-auth":
        entry["url"] = entry["url"].replace("https://", "https://username:password@")
    elif fault == "url-query":
        entry["url"] += "?token=unused"
    elif fault == "sha":
        entry["sha256"] = "0" * 64
    elif fault == "partial-sha":
        entry["sha256"] = entry["sha256"][:16]
    elif fault == "size":
        entry["bytes"] += 1
    elif fault == "linked-wheel":
        outside = tmp_path / "original.whl"
        path.rename(outside)
        path.symlink_to(outside)
    elif fault == "linked-directory":
        (cache / "wheels").rename(cache / "originals")
        (cache / "wheels").symlink_to(cache / "originals", target_is_directory=True)
    else:
        (cache / "wheels/extra.whl").write_bytes(b"not admitted")
    save_manifest(cache, m)
    root = tmp_path / "new-tool"
    monkeypatch.setattr(tools, "execute", lambda *a, **k: pytest.fail("unadmitted code must not run"))
    with pytest.raises(ValueError):
        tools.prepare(root, Path(__file__), Path(__file__), tools.CATALOG,
                      wheelhouse=cache, target="rtx4090")
    assert not root.exists()


@pytest.mark.parametrize("field", ["Name", "Version"])
def test_rehashed_wheel_cannot_change_catalog_metadata(tmp_path, field):
    cache = tmp_path / "cache"
    m = wheelhouse(cache)
    entry = m["wheels"][0]
    path = cache / "wheels" / entry["filename"]
    name = "other-package" if field == "Name" else entry["name"]
    version = "99" if field == "Version" else entry["version"]
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("example.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    entry.update(bytes=path.stat().st_size, sha256=tools.sha(path))
    save_manifest(cache, m)
    with pytest.raises(ValueError, match="metadata differs"):
        tools.admit_wheelhouse(cache, tools.CATALOG)


@pytest.mark.parametrize("offline", [False, True])
def test_prepare_keeps_native_argv_and_default_online_behavior(tmp_path, monkeypatch, offline):
    cache = tmp_path / "cache"
    m = wheelhouse(cache)
    before = {x["filename"]: (cache / "wheels" / x["filename"]).stat() for x in m["wheels"]}
    root = tmp_path / "tool"
    uvx = tmp_path / "uvx"
    uvx.write_bytes(b"fixed uvx")
    calls = []
    catalog = tools.load_catalog(tools.CATALOG)

    def execute(root, name, command, env, **kwargs):
        calls.append((name, command, env.copy()))
        if name == "uvx_version":
            return "uvx 0.12.5 (fixture)"
        if name == "python_version":
            return "3.13"
        if name == "python_machine":
            return "x86_64"
        if name in {"public_tool_install", "offline_tool_install", "offline_tool_reuse"}:
            assert command == [str(uvx), "--with", "click", "hf@1.16.4", "--help"]
            if name != "offline_tool_reuse":
                assert (env.get("UV_OFFLINE") == "1") is offline
                site = root / "cache/archive-v0/tool/lib/python3.13/site-packages"
                for package, version in catalog["packages"].items():
                    p = site / f"{package}-{version}.dist-info/METADATA"
                    p.parent.mkdir(parents=True)
                    p.write_text(f"Name: {package}\nVersion: {version}\n")
            else:
                assert env["UV_OFFLINE"] == "1"
        return ""

    monkeypatch.setattr(tools, "execute", execute)
    result = tools.prepare(root, Path(__file__), uvx, tools.CATALOG,
                           wheelhouse=cache if offline else None, target="rtx4090")
    assert result["status"] == "tool_packages_checked_offline_reuse_passed"
    assert result["first_preparation_offline"] is result["local_wheels"] is offline
    assert result["package_versions"] == catalog["packages"]
    if offline:
        assert result["environment"]["UV_FIND_LINKS"] == str(root / "wheels")
        assert result["environment"]["UV_NO_INDEX"] == "1"
        assert (root / "catalog_constraints.txt").read_bytes() == (
            tools.CATALOG.parent / catalog["constraints"]["path"]).read_bytes()
        for entry in m["wheels"]:
            source = cache / "wheels" / entry["filename"]
            destination = root / "wheels" / entry["filename"]
            after = source.stat()
            old = before[entry["filename"]]
            assert (old.st_ino, old.st_nlink, old.st_ctime_ns, old.st_mtime_ns) == (
                after.st_ino, after.st_nlink, after.st_ctime_ns, after.st_mtime_ns)
            assert destination.stat().st_ino != source.stat().st_ino
            assert tools.sha(destination) == entry["sha256"]
            assert destination.as_uri() in (root / "constraints.txt").read_text()
        shutil.rmtree(cache)
        assert all((root / "wheels" / x["filename"]).is_file() for x in m["wheels"])
    else:
        assert not (root / "wheels").exists()
        assert "UV_NO_INDEX" not in result["environment"] and "UV_FIND_LINKS" not in result["environment"]
        assert any(name == "public_tool_install" for name, _, _ in calls)


def test_archive_changed_during_copy_fails_before_any_native_execution(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    wheelhouse(cache)
    copyfile = tools.shutil.copyfile

    def corrupt(source, destination):
        result = copyfile(source, destination)
        if Path(destination).suffix == ".whl":
            with Path(destination).open("ab") as f:
                f.write(b"changed")
        return result

    monkeypatch.setattr(tools.shutil, "copyfile", corrupt)
    monkeypatch.setattr(tools, "execute", lambda *a, **k: pytest.fail("archive copy must be admitted first"))
    root = tmp_path / "new-tool"
    with pytest.raises(ValueError, match="changed during owned archive copy"):
        tools.prepare(root, Path(__file__), Path(__file__), tools.CATALOG, wheelhouse=cache)
    assert json.loads((root / "failure.json").read_text())["automatic_retry"] is False


def test_interpreter_machine_mismatch_stops_before_tool_import(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    wheelhouse(cache)
    calls = []

    def execute(root, name, command, env, **kwargs):
        calls.append(name)
        return {"uvx_version": "uvx 0.12.5 (fixture)", "python_version": "3.13",
                "python_machine": "aarch64"}[name]

    monkeypatch.setattr(tools, "execute", execute)
    root = tmp_path / "new-tool"
    with pytest.raises(ValueError, match="interpreter does not match"):
        tools.prepare(root, Path(__file__), Path(__file__), tools.CATALOG, wheelhouse=cache)
    assert calls == ["uvx_version", "python_version", "python_machine"]
    assert not (root / "completion.json").exists()


def test_bootstrap_plan_reports_tool_admission_without_creating_run_root(tmp_path, capsys):
    cache = tmp_path / "cache"
    wheelhouse(cache)
    root = tmp_path / "uncreated"
    assert bootstrap.main(["plan", "edge", "--target", "rtx4090", "--root", str(root),
                           "--native-tool-wheelhouse", str(cache)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["native_tool_wheelhouse_admission"]["status"] == "admitted"
    assert not root.exists()


def test_bootstrap_forwards_tool_cache_and_target_without_changing_main_dependencies(tmp_path, monkeypatch):
    cache = tmp_path / "tool-cache"
    wheelhouse(cache)
    args = SimpleNamespace(root=tmp_path / "bootstrap", env_dir=None, vendor_dir=None, cache_dir=None,
                           checkout=ROOT, timeout=10, python=sys.executable, uv="uv", repaired_wheel=None,
                           repair_receipt=None, package_wheel_dir=None, native_tool_wheelhouse=cache)
    instance = bootstrap.Bootstrap(args, bootstrap.load_profile("edge", target="rtx4090"))
    monkeypatch.setattr(instance, "prepare_source", lambda: None)
    monkeypatch.setattr(instance, "command", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap.subprocess, "check_output", lambda *a, **k: '["Linux","x86_64","3.13"]')

    class NativeReached(Exception):
        pass

    def prepare(root, python, uvx, catalog, *, wheelhouse, target):
        assert wheelhouse == cache and target == "rtx4090"
        assert root == instance.root / "native_tools"
        assert catalog == ROOT / "release/vendor/cosmos/hf_tool.json"
        raise NativeReached

    monkeypatch.setattr(bootstrap.prepare_native_tools, "prepare", prepare)
    with pytest.raises(NativeReached):
        instance.install()
    admission = json.loads((instance.root / "native_tool_wheelhouse_admission.json").read_text())
    assert admission["status"] == "admitted" and len(admission["wheels"]) == 23


def test_plan_rejects_native_tool_cache_for_non_cosmos_family(tmp_path, capsys):
    assert bootstrap.main(["plan", "pi05", "--target", "rtx4090",
                           "--native-tool-wheelhouse", str(tmp_path / "absent")]) == 1
    assert "native tool catalog" in json.loads(capsys.readouterr().out)["error"]

"""Real Git transport fixtures; no network, package installation, or GPU."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("bootstrap_vendor", ROOT / "scripts/bootstrap_vendor.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def git(path, *args):
    return subprocess.check_output(
        ["git", "-c", "core.hooksPath=/dev/null", *args], cwd=path,
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
        stderr=subprocess.PIPE, text=True).strip()


def snapshot(path):
    return {str(p.relative_to(path)): (p.stat().st_mode, p.read_bytes())
            for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def source_fixture(tmp_path):
    source = tmp_path / "original repository"
    source.mkdir()
    git(source, "init")
    git(source, "config", "user.email", "fixture@example.invalid")
    git(source, "config", "user.name", "Fixture")
    (source / "model.py").write_text('MODEL = "original"\n')
    (source / "pyproject.toml").write_text('dependencies = ["training"]\n')
    git(source, "add", ".")
    git(source, "commit", "-m", "Pinned revision")
    revision = git(source, "rev-parse", "HEAD")
    (source / "model.py").write_text('MODEL = "later revision"\n')
    git(source, "commit", "-am", "Unselected revision")
    # Neither the current branch nor local worktree modifications are inputs.
    (source / "model.py").write_text('MODEL = "dirty worktree"\n')
    (source / "untracked.txt").write_text("must not be copied")
    checkout = tmp_path / "checkout"
    patches = checkout / "release/vendor/fixture"
    patches.mkdir(parents=True)
    recorded = patches / "recorded.patch"
    recorded.write_text('diff --git a/model.py b/model.py\n--- a/model.py\n+++ b/model.py\n'
                        '@@ -1 +1 @@\n-MODEL = "original"\n+MODEL = "recorded patch"\n')
    packaging = patches / "packaging.patch"
    packaging.write_text('diff --git a/pyproject.toml b/pyproject.toml\n--- a/pyproject.toml\n'
                         '+++ b/pyproject.toml\n@@ -1 +1 @@\n-dependencies = ["training"]\n'
                         '+dependencies = ["inference"]\n')
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    profile = {
        "source_manifest": "fixture/source.json",
        "source": {
            "source": {"repository": "https://github.com/example/frozen-vendor", "revision": revision},
            "patch": {"path": "recorded.patch", "sha256": bootstrap.sha(recorded)},
            "expected_patched_files": {"model.py": {"sha256": digest('MODEL = "recorded patch"\n')}},
        },
        "packaging_patch": {
            "path": "fixture/packaging.patch", "sha256": bootstrap.sha(packaging),
            "before": {"pyproject.toml": digest('dependencies = ["training"]\n')},
            "after": {"pyproject.toml": digest('dependencies = ["inference"]\n')},
        },
    }
    args = SimpleNamespace(root=tmp_path / "fresh bootstrap", env_dir=None, vendor_dir=None,
                           cache_dir=None, checkout=checkout, timeout=15, vendor_source=source)
    return source, profile, args


@pytest.mark.parametrize("kind", ["repository", "bare", "bundle"])
def test_local_transport_fetches_pinned_commit_and_runs_original_patch_gates(source_fixture, kind):
    source, profile, args = source_fixture
    if kind == "bare":
        bare = source.parent / "bare repository.git"
        git(source, "clone", "--bare", str(source), str(bare))
        args.vendor_source = bare
    elif kind == "bundle":
        bundle = source.parent / "original.bundle"
        git(source, "bundle", "create", str(bundle), "--all")
        args.vendor_source = bundle
    before = snapshot(source)
    input_before = (snapshot(args.vendor_source) if args.vendor_source.is_dir()
                    else args.vendor_source.read_bytes())
    b = bootstrap.Bootstrap(args, profile)
    b.prepare_roots()
    b.prepare_source()
    receipt = json.loads((b.receipts / "vendor_source.json").read_text())
    assert receipt["repository"] == profile["source"]["source"]["repository"]
    assert receipt["revision"] == git(b.vendor, "rev-parse", "HEAD") == profile["source"]["source"]["revision"]
    assert (b.vendor / "model.py").read_text() == 'MODEL = "recorded patch"\n'
    assert (b.vendor / "pyproject.toml").read_text() == 'dependencies = ["inference"]\n'
    assert not (b.vendor / "untracked.txt").exists()
    assert not (b.vendor / ".git/objects/info/alternates").exists()
    assert git(b.vendor, "for-each-ref") == ""
    assert snapshot(source) == before
    assert (snapshot(args.vendor_source) if args.vendor_source.is_dir()
            else args.vendor_source.read_bytes()) == input_before
    assert receipt["transport"]["path"] == str(args.vendor_source.resolve())
    assert b.environment["GIT_ALLOW_PROTOCOL"] == "file"
    fetch = next(c["argv"] for c in b.commands if c["label"] == "git_fetch")
    assert fetch[-2:] == [str(args.vendor_source.resolve()), receipt["revision"]]
    assert "--no-tags" in fetch
    if kind == "bundle":
        assert receipt["transport"]["sha256"] == bootstrap.sha(args.vendor_source)
        assert "--depth=1" not in fetch
    assert [c["label"] for c in b.commands][-4:] == [
        "recorded_patch_check", "recorded_patch", "inference_metadata_check", "inference_metadata_patch"]


def test_missing_commit_fails_locally_without_fallback_or_input_mutation(source_fixture):
    source, profile, args = source_fixture
    profile["source"]["source"]["revision"] = "0" * 40
    before = snapshot(source)
    b = bootstrap.Bootstrap(args, profile)
    b.prepare_roots()
    with pytest.raises(RuntimeError, match="git_fetch failed"):
        b.prepare_source()
    assert [c["label"] for c in b.commands] == ["git_init", "git_fetch"]
    assert snapshot(source) == before
    assert not (b.receipts / "vendor_source.json").exists()


@pytest.mark.parametrize("gate", ["runtime", "packaging_before", "packaging_after"])
def test_local_source_cannot_bypass_expected_file_hashes(source_fixture, gate):
    _, profile, args = source_fixture
    if gate == "runtime":
        profile["source"]["expected_patched_files"]["model.py"]["sha256"] = "0" * 64
    else:
        profile["packaging_patch"][gate.removeprefix("packaging_")]["pyproject.toml"] = "0" * 64
    b = bootstrap.Bootstrap(args, profile)
    b.prepare_roots()
    with pytest.raises(ValueError, match="source input hash mismatch"):
        b.prepare_source()
    assert not (b.receipts / "vendor_source.json").exists()


def test_incremental_bundle_requires_missing_base_and_is_rejected(source_fixture):
    source, profile, args = source_fixture
    bundle = source.parent / "incremental.bundle"
    git(source, "bundle", "create", str(bundle), profile["source"]["source"]["revision"] + "..HEAD")
    args.vendor_source = bundle
    b = bootstrap.Bootstrap(args, profile)
    b.prepare_roots()
    with pytest.raises(RuntimeError, match="git_bundle_verify failed"):
        b.prepare_source()
    assert not any(c["label"] == "git_fetch" for c in b.commands)


def test_packaging_patch_cannot_change_an_undeclared_runtime_file(source_fixture):
    _, profile, args = source_fixture
    patch = args.checkout / "release/vendor" / profile["packaging_patch"]["path"]
    with patch.open("a") as f:
        f.write('diff --git a/model.py b/model.py\n--- a/model.py\n+++ b/model.py\n'
                '@@ -1 +1 @@\n-MODEL = "recorded patch"\n+MODEL = "undeclared"\n')
    profile["packaging_patch"]["sha256"] = bootstrap.sha(patch)
    b = bootstrap.Bootstrap(args, profile)
    b.prepare_roots()
    with pytest.raises(ValueError, match="packaging patch touches undeclared files"):
        b.prepare_source()
    assert (b.vendor / "model.py").read_text() == 'MODEL = "recorded patch"\n'


def test_bundle_change_during_fetch_is_preserved_and_rejected(source_fixture, monkeypatch):
    source, profile, args = source_fixture
    bundle = source.parent / "changing.bundle"
    git(source, "bundle", "create", str(bundle), "--all")
    args.vendor_source = bundle
    b = bootstrap.Bootstrap(args, profile)
    b.prepare_roots()
    command = b.command

    def mutate_after_fetch(label, argv, **kwargs):
        result = command(label, argv, **kwargs)
        if label == "git_fetch":
            with bundle.open("ab") as f:
                f.write(b"changed")
        return result

    monkeypatch.setattr(b, "command", mutate_after_fetch)
    with pytest.raises(ValueError, match="bundle changed"):
        b.prepare_source()
    assert not (b.receipts / "vendor_source.json").exists()


def test_local_input_ignores_foreign_git_environment(source_fixture, monkeypatch):
    _, profile, args = source_fixture
    monkeypatch.setenv("GIT_DIR", "/nonexistent/foreign")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "url.https://invalid.example/.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/")
    b = bootstrap.Bootstrap(args, profile)
    assert "GIT_DIR" not in b.environment and "GIT_CONFIG_COUNT" not in b.environment
    b.prepare_roots()
    b.prepare_source()


def test_source_repository_cannot_contain_new_destinations(source_fixture):
    source, profile, args = source_fixture
    args.root = source / "nested-run"
    before = snapshot(source)
    b = bootstrap.Bootstrap(args, profile)
    with pytest.raises(ValueError, match="outside the local vendor source"):
        b.prepare_roots()
    assert snapshot(source) == before and not args.root.exists()


def test_invalid_local_source_and_non_git_family_are_rejected(tmp_path):
    profile = bootstrap.load_profile("va")
    with pytest.raises(FileNotFoundError):
        bootstrap.vendor_source_transport(profile, tmp_path / "missing")
    invalid = tmp_path / "not-a-bundle"
    invalid.write_text("not Git")
    with pytest.raises(ValueError, match="Git repository or Git bundle"):
        bootstrap.vendor_source_transport(profile, invalid)
    with pytest.raises(ValueError, match="family with a Git source profile"):
        bootstrap.vendor_source_transport(bootstrap.load_profile("pi05"), tmp_path)


def test_online_transport_retains_original_repository_and_plan_is_read_only(tmp_path, capsys):
    profile = bootstrap.load_profile("va")
    assert bootstrap.vendor_source_transport(profile, None) == {
        "kind": "public_https", "url": profile["source"]["source"]["repository"]}
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init")
    assert bootstrap.main(["plan", "va", "--vendor-source", str(repo),
                           "--root", str(tmp_path / "absent")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["source"] == profile["source"]
    assert result["vendor_source_transport"] == {"kind": "local_git_repository", "path": str(repo)}
    assert not (tmp_path / "absent").exists()


def test_default_source_command_keeps_public_exact_commit(source_fixture, monkeypatch):
    _, profile, args = source_fixture
    args.vendor_source = None
    b = bootstrap.Bootstrap(args, profile)
    b.prepare_roots()
    commands = []

    class StopBeforeNetwork(Exception):
        pass

    def record(label, argv, **kwargs):
        commands.append((label, argv))
        if label == "git_fetch":
            raise StopBeforeNetwork

    monkeypatch.setattr(b, "command", record)
    with pytest.raises(StopBeforeNetwork):
        b.prepare_source()
    assert commands[-1] == ("git_fetch", [
        "git", "-c", "core.hooksPath=/dev/null", "fetch", "--depth=1",
        profile["source"]["source"]["repository"], profile["source"]["source"]["revision"]])

"""Public staging enforces a bounded source/data selection without model imports."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("public_builder", Path(__file__).resolve().parents[1] / "scripts/build_public_release.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
CURRENT_CONTROLS = dict(builder.SELECTED_CONTROLS)
CURRENT_VENDOR_FILES = dict(builder.PUBLIC_VENDOR_FILES)


def put(root, relative, content=b""):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    return path


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    rules = [{"prefix": prefix, "classification": classification} for prefix, classification in [
        ("instinctflash", "public_candidate"), ("instinctflash/train", "hold"), ("instinctflash/distill", "hold"),
        ("instinctflash/native", "implementation_review"), ("instinctflash/native/__init__.py", "public_candidate"),
        ("benchmarks", "public_candidate"), ("scripts", "public_candidate"), ("examples", "public_candidate"),
        ("eval", "hold"), ("release", "release_review")]]
    put(root, builder.SCOPE_PATH, json.dumps({"schema": "instinctflash.oss_scope.v1", "rules": rules}))
    project = '[build-system]\nrequires = ["setuptools>=64"]\nbuild-backend = "setuptools.build_meta"\n[project]\nname = "{}"\nversion = "0.1.0"\n'
    for distribution, (directory, module) in builder.PACKAGES.items():
        prefix = "" if directory == "." else directory + "/"
        text = project.format(distribution)
        if directory == ".":
            text += 'readme = "README.md"\n[tool.setuptools.packages.find]\ninclude = ["instinctflash*"]\n[tool.setuptools.package-data]\n"instinctflash.distill.evidence" = ["*.json"]\n'
        put(root, prefix + "pyproject.toml", text)
        put(root, prefix + module + "/__init__.py")
    for name in builder.ROOT_FILES - {"pyproject.toml"}:
        put(root, name, "current explicitly selected document")
    for name in builder.ADAPTER_LICENSE_FILES:
        put(root, name, (root / "LICENSE").read_bytes())
    for name in builder.TOOL_FILES:
        put(root, name, "# selected public tool\n")
    for name in builder.CONFIG_FILES:
        put(root, "benchmarks/vla/config/" + name, "{}")
    for name in builder.REGRESSION_FILES:
        put(root, "benchmarks/regression/" + name)
    put(root, "benchmarks/__init__.py")
    put(root, "benchmarks/vla/__init__.py")
    put(root, builder.WORKER_SOURCE, "value = 'reviewed worker'\n")
    worker = root / builder.WORKER
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.symlink_to("../../" + builder.WORKER_SOURCE)
    monkeypatch.setattr(builder, "WORKER_SHA", builder.sha((root / builder.WORKER_SOURCE).read_bytes()))
    monkeypatch.setattr(builder, "SELECTED_DATA", {})
    monkeypatch.setattr(builder, "PUBLIC_VENDOR_FILES", {})
    monkeypatch.setattr(builder, "FULL_TEST_FIXTURES", {})
    controls = {"scripts/public_deploy.py": "# selected control\n",
                **{canonical: "{}" for canonical in builder.PROFILE_MIRRORS.values()}}
    for relative, content in controls.items():
        put(root, relative, content)
    for mirror,canonical in builder.PROFILE_MIRRORS.items():
        put(root, mirror, controls[canonical])
    monkeypatch.setattr(builder, "SELECTED_CONTROLS", {name: builder.sha(content.encode()) for name, content in controls.items()})

    def git(repo, *args):
        if args == ("rev-parse", "HEAD"):
            return b"fixture-head\n"
        if args == ("ls-files", "--deleted", "-z"):
            return b""
        return b"\0".join(str(p.relative_to(root)).encode() for p in root.rglob("*") if p.is_file() or p.is_symlink()) + b"\0"
    monkeypatch.setattr(builder, "git", git)
    return root


def test_explicit_distribution_excludes_held_native_ops_and_bulk_documents(repository, tmp_path):
    excluded = ["instinctflash/train/secret.py", "instinctflash/distill/pipeline.py", "instinctflash/native/kernel.cu",
                "instinctflash/native/kernel.so", "benchmarks/regression/deploy_thor.py", "benchmarks/regression/systemd/worker.service",
                "eval/study/readme.md", "eval/large.npz", "examples/pi05_vla/private_experiment.py", "unselected.md"]
    for path in excluded:
        put(repository, path, "private or unselected")
    result = builder.create_stage(repository, tmp_path / "stage")
    source = Path(result["source"])
    assert not any((source / name).exists() for name in excluded)
    assert (source / "README.md").read_text() == "current explicitly selected document"
    assert all(not p.is_symlink() for p in source.rglob("*"))
    assert len(result["packages"]) == 7 and result["native_binaries"] == []
    assert result["publication_ready"] is result["pipeline_validated"] is False


def test_materializes_only_exact_reviewed_worker_and_preserves_source(repository, tmp_path):
    result = builder.create_stage(repository, tmp_path / "stage")
    original = repository / builder.WORKER
    staged = Path(result["source"]) / builder.WORKER
    assert original.is_symlink() and not staged.is_symlink()
    assert original.read_bytes() == staged.read_bytes()
    assert result["files"][builder.WORKER]["source_sha256"] == builder.WORKER_SHA
    assert result["files"][builder.WORKER]["resolved_source_path"] == builder.WORKER_SOURCE


def test_worker_changed_bytes_fail_before_creating_output(repository, tmp_path):
    (repository / builder.WORKER_SOURCE).write_text("changed = True\n")
    with pytest.raises(ValueError, match="worker source/hash"):
        builder.create_stage(repository, tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


@pytest.mark.parametrize("relative,target", [
    ("instinctflash/runtime/escape.py", "instinctflash/train/secret.py"),
    ("instinctflash/runtime/escape.py", "instinctflash/native/secret.py"),
    ("README.md", "eval/study.md"),
])
def test_unreviewed_and_held_symlink_targets_are_rejected(repository, tmp_path, relative, target):
    destination = put(repository, target, "value = 1\n")
    link = repository / relative
    link.unlink(missing_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(destination)
    with pytest.raises(ValueError, match="held|unreviewed"):
        builder.create_stage(repository, tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_parent_symlink_outside_checkout_is_rejected_without_reading(repository, tmp_path):
    external = tmp_path / "external"
    put(external, "escape.py", b"\xffnot to read")
    (repository / "instinctflash/outside").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="inside checkout"):
        builder.source_bytes(repository.resolve(), "instinctflash/outside/escape.py")


@pytest.mark.parametrize("where", ["inside", "existing", "dangling_symlink", "parent_symlink_inside"])
def test_output_must_be_new_and_outside(repository, tmp_path, where):
    output = tmp_path / "stage"
    if where == "inside":
        output = repository / "stage"
    elif where == "existing":
        output.mkdir()
    elif where == "dangling_symlink":
        output.symlink_to(tmp_path / "missing")
    else:
        (tmp_path / "alias").symlink_to(repository, target_is_directory=True)
        output = tmp_path / "alias/stage"
    with pytest.raises(ValueError, match="existing|outside"):
        builder.create_stage(repository, output)


def test_staged_pyproject_excludes_train_data_without_changing_source_metadata(repository, tmp_path):
    before = (repository / "pyproject.toml").read_bytes()
    result = builder.create_stage(repository, tmp_path / "stage")
    config = builder.tomllib.loads((Path(result["source"]) / "pyproject.toml").read_text())
    assert config["tool"]["setuptools"]["packages"]["find"]["exclude"] == ["instinctflash.train*", "instinctflash.distill*"]
    assert config["tool"]["setuptools"]["package-data"] == {"benchmarks.vla": ["config/*.json"],
                                                            "benchmarks.regression": ["fixtures/deployment_profiles.json", "fixtures/deployment_profiles_rtx4090.json", "fixtures/deployment_profiles_rtx5090.json"]}
    assert (repository / "pyproject.toml").read_bytes() == before
    assert result["files"]["pyproject.toml"]["source_sha256"] == builder.sha(before)


def test_metadata_readme_is_an_explicit_document_selection(repository, tmp_path):
    folder = "examples/pi05_vla"
    path = repository / folder / "pyproject.toml"
    path.write_text(path.read_text() + 'readme = "README.md"\n')
    put(repository, folder + "/README.md", "current adapter guide")
    put(repository, folder + "/unselected.md", "do not copy")
    result = builder.create_stage(repository, tmp_path / "stage")
    assert folder + "/README.md" in result["files"]
    assert folder + "/unselected.md" not in result["files"]


def test_selected_fixture_is_hash_bound_and_mismatch_fails(repository, tmp_path, monkeypatch):
    relative = "benchmarks/regression/fixtures/recorded_inputs_v1.npz"
    put(repository, relative, b"synthetic packaging-only fixture")
    monkeypatch.setattr(builder, "SELECTED_DATA", {relative: builder.sha((repository / relative).read_bytes())})
    result = builder.create_stage(repository, tmp_path / "stage")
    assert result["selected_fixture"] == relative
    (repository / relative).write_bytes(b"changed")
    with pytest.raises(ValueError, match="data hash"):
        builder.create_stage(repository, tmp_path / "changed_stage")


def test_only_explicit_hash_bound_deployment_controls_are_selected(repository, tmp_path):
    put(repository, "release/private_control.json", "{}")
    put(repository, "scripts/private_deploy.py", "# unselected\n")
    result = builder.create_stage(repository, tmp_path / "stage")
    assert set(builder.SELECTED_CONTROLS) <= result["files"].keys()
    assert "release/private_control.json" not in result["files"]
    assert "scripts/private_deploy.py" not in result["files"]
    (repository / "release/deployment_profiles.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="control path/hash"):
        builder.create_stage(repository, tmp_path / "changed_stage")


def test_deployment_control_alias_is_not_an_additional_source_exception(repository, tmp_path):
    original = repository / "release/deployment_profiles.json"
    content = original.read_bytes()
    original.unlink()
    target = put(repository, "release/unreviewed_alias.json", content)
    original.symlink_to(target)
    with pytest.raises(ValueError, match="control path/hash"):
        builder.create_stage(repository, tmp_path / "stage")


def populate_full_source(repository):
    for name in builder.FULL_FILES:
        put(repository, name, "# explicitly selected public build/test source\n" if name.endswith(".py") else "source fixture\n")
    put(repository, "serving/pyproject.toml", '[build-system]\nrequires=["setuptools>=64"]\nbuild-backend="setuptools.build_meta"\n[project]\nname="flash-rt"\nversion="0.1.0"\nreadme="README.rst"\n[tool.setuptools.package-data]\n"flash_rt.configs"=["*.yaml"]\n')
    put(repository, "serving/README.rst", "Explicit serving guide\n")
    put(repository, "serving/flash_rt/__init__.py")
    put(repository, "serving/flash_rt/configs/model.yaml", "model: fixture\n")
    put(repository, "examples/cosmos3_sde1/pyproject.toml", '[build-system]\nrequires=["setuptools>=64"]\nbuild-backend="setuptools.build_meta"\n[project]\nname="instinctflash-cosmos3-sde1"\nversion="0.1.0"\n')
    put(repository, "examples/cosmos3_sde1/cosmos3_sde1/__init__.py")
    put(repository, "examples/cosmos3_sde1/instinct_compress/__init__.py")


def test_full_source_scope_includes_current_internal_code_and_vendor_notices(repository, tmp_path):
    populate_full_source(repository)
    included = {
        "instinctflash/train/trainer.py": "value = 1\n",
        "instinctflash/distill/evidence/frontier.json": "{}",
        "instinctflash/native/bf16/kernel.cu": "// source\n",
        "serving/csrc/attention/flash_attn_2_src/flash_attn/kernel.h": "// vendored source\n",
        "serving/csrc/attention/flash_attn_2_src/flash_attn/LICENSE": "original license\n",
        "serving/training/_vendor/openpi_pi0_pytorch/VENDOR.md": "original provenance\n",
        "serving/training/_vendor/openpi_pi0_pytorch/LICENSE.openpi": "original upstream license\n",
    }
    excluded = ["serving/third_party/cutlass/include/large.h", "serving/flash_rt/kernel.so",
                "serving/flash_rt/model.safetensors", "serving/flash_rt/.env", "serving/training/_runs/host_job.py"]
    for name, content in included.items():
        put(repository, name, content)
    for name in excluded:
        put(repository, name, "excluded bytes")
    before = (repository / "pyproject.toml").read_bytes()
    result = builder.create_stage(repository, tmp_path / "full", scope="full")
    assert set(included) <= result["files"].keys()
    assert not set(excluded) & result["files"].keys()
    assert result["source_scope"] == "full" and len(result["packages"]) == 9
    assert (Path(result["source"]) / "pyproject.toml").read_bytes() == before
    assert result["files"][builder.PROFILE_MIRROR]["canonical_source_path"] == "release/deployment_profiles.json"
    assert result["files"]["serving/flash_rt/configs/model.yaml"]["wheel_required"] is True
    assert result["files"]["examples/cosmos3_sde1/instinct_compress/__init__.py"]["wheel_required"] is True
    assert result["publication_ready"] is result["pipeline_validated"] is False


def test_full_source_scope_still_rejects_alias_to_excluded_downloaded_tree(repository, tmp_path):
    populate_full_source(repository)
    target = put(repository, "serving/third_party/private.py", "value = 1\n")
    (repository / "serving/flash_rt/alias.py").symlink_to(target)
    with pytest.raises(ValueError, match="held symlink"):
        builder.create_stage(repository, tmp_path / "full", scope="full")


def test_package_data_globs_do_not_cross_directory_boundaries():
    assert builder.package_data_matches("README.md", "*.md")
    assert not builder.package_data_matches("bf16/README.md", "*.md")
    assert builder.package_data_matches("bf16/kernel.cu", "bf16/*.cu")


def test_full_vendor_bundle_is_explicit_hash_bound_not_a_directory_copy(repository, tmp_path, monkeypatch):
    populate_full_source(repository)
    relative = "release/vendor/model/pins.json"
    content = b'{"revision":"pinned"}'
    put(repository, relative, content)
    put(repository, "release/vendor/private.json", "not selected")
    monkeypatch.setattr(builder, "PUBLIC_VENDOR_FILES", {relative: builder.sha(content)})
    result = builder.create_stage(repository, tmp_path / "full", scope="full")
    assert relative in result["files"] and "release/vendor/private.json" not in result["files"]
    (repository / relative).write_text("changed")
    with pytest.raises(ValueError, match="control path/hash"):
        builder.create_stage(repository, tmp_path / "changed", scope="full")


def test_full_source_includes_only_pinned_test_metadata(repository, tmp_path, monkeypatch):
    populate_full_source(repository)
    relative = "eval/study/task_inventory.json"
    content = b'{"task_count":120}'
    put(repository, relative, content)
    put(repository, "eval/study/unselected_images.npz", b"not selected")
    monkeypatch.setattr(builder, "FULL_TEST_FIXTURES", {relative: builder.sha(content)})
    result = builder.create_stage(repository, tmp_path / "full", scope="full")
    assert (Path(result["source"]) / relative).read_bytes() == content
    assert "eval/study/unselected_images.npz" not in result["files"]
    core = builder.create_stage(repository, tmp_path / "core", scope="core")
    assert relative not in core["files"]


def test_full_source_omits_git_tracked_files_deleted_from_worktree(repository, tmp_path, monkeypatch):
    populate_full_source(repository)
    original_git = builder.git
    deleted = b"tests/test_retired_pointer.py\0"

    def git(repo, *args):
        if args == ("ls-files", "--deleted", "-z"):
            return deleted
        value = original_git(repo, *args)
        return value + deleted if args[0] == "ls-files" else value

    monkeypatch.setattr(builder, "git", git)
    result = builder.create_stage(repository, tmp_path / "full", scope="full")
    assert "tests/test_retired_pointer.py" not in result["files"]
    assert "tests/test_public_release_builder.py" in result["files"]


@pytest.mark.parametrize("change", ["bytes", "alias"])
def test_full_test_metadata_cannot_drift_or_alias(repository, tmp_path, monkeypatch, change):
    populate_full_source(repository)
    relative = "eval/study/task_inventory.json"
    content = b'{"task_count":120}'
    original = put(repository, relative, content)
    monkeypatch.setattr(builder, "FULL_TEST_FIXTURES", {relative: builder.sha(content)})
    if change == "bytes":
        original.write_bytes(b'{"task_count":119}')
    else:
        original.unlink()
        original.symlink_to(put(repository, "eval/study/unreviewed_alias.json", content))
    with pytest.raises(ValueError, match="control path/hash|held symlink"):
        builder.create_stage(repository, tmp_path / "full", scope="full")
    assert not (tmp_path / "full").exists()


def test_wheel_inspection_rejects_native_and_unselected_sources(tmp_path):
    for extra in ("instinctflash/native/secret.so", "instinctflash/train/secret.py", "../escape.py"):
        wheel = tmp_path / (str(len(list(tmp_path.iterdir()))) + ".whl")
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("instinctflash/__init__.py", b"")
            archive.writestr("instinctflash-0.1.dist-info/METADATA", "Name: instinctflash\nVersion: 0.1\n")
            archive.writestr(extra, b"")
        with pytest.raises(ValueError):
            builder.inspect_wheel(wheel, "instinctflash", ".", "instinctflash", {"instinctflash/__init__.py": {"sha256": builder.sha(b"")}})


def test_wheel_inspection_requires_selected_benchmark_data(tmp_path):
    wheel = tmp_path / "core.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("instinctflash/__init__.py", b"")
        archive.writestr("instinctflash-0.1.dist-info/METADATA", "Name: instinctflash\nVersion: 0.1\n")
    files = {name: {"sha256": builder.sha(b"")} for name in ("instinctflash/__init__.py", "benchmarks/vla/config/registry.json")}
    with pytest.raises(ValueError, match="dropped"):
        builder.inspect_wheel(wheel, "instinctflash", ".", "instinctflash", files)


def test_packaged_profile_mirror_drift_fails_before_staging(repository, tmp_path):
    put(repository, builder.PROFILE_MIRROR, '{"drift":true}')
    with pytest.raises(ValueError, match="mirror drifted"):
        builder.create_stage(repository, tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_experimental_wheel_must_include_the_reviewed_helper_namespace(tmp_path):
    wheel = tmp_path / "experimental.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("cosmos3_sde1/__init__.py", b"")
        archive.writestr("instinctflash_cosmos3_sde1-0.1.dist-info/METADATA", "Name: instinctflash-cosmos3-sde1\nVersion: 0.1\n")
    files = {"examples/cosmos3_sde1/" + name: {"sha256": builder.sha(b""), "wheel_required": True}
             for name in ("cosmos3_sde1/__init__.py", "instinct_compress/__init__.py")}
    with pytest.raises(ValueError, match="dropped"):
        builder.inspect_wheel(wheel, "instinctflash-cosmos3-sde1", "examples/cosmos3_sde1", "cosmos3_sde1", files, scope="full")


@pytest.mark.parametrize("mirror", list(builder.PROFILE_MIRRORS))
def test_each_profile_mirror_has_its_own_canonical_binding(repository, tmp_path, mirror):
    put(repository, mirror, '{"other_target":true}')
    with pytest.raises(ValueError, match="mirror drifted"):
        builder.create_stage(repository, tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_full_staged_rtx_closure_plans_all_eight_without_installs(repository, tmp_path, monkeypatch):
    original = Path(__file__).resolve().parents[1]
    populate_full_source(repository)
    for relative in {*CURRENT_CONTROLS, *CURRENT_VENDOR_FILES, *builder.PROFILE_MIRRORS,
                     "benchmarks/regression/reproduce.py", "benchmarks/regression/hardware.py"}:
        put(repository, relative, (original / relative).read_bytes())
    monkeypatch.setattr(builder, "SELECTED_CONTROLS", CURRENT_CONTROLS)
    monkeypatch.setattr(builder, "PUBLIC_VENDOR_FILES", CURRENT_VENDOR_FILES)
    for relative in ("release/vendor/rtx4090/private.json", "release/rtx4090/unrelated.md",
                     "release/rtx4090/results/private.log", "scripts/private_wheelhouse.py",
                     "release/vendor/rtx5090/private.json", "release/rtx5090/results/private.log"):
        put(repository, relative, "unselected input")
    result = builder.create_stage(repository, tmp_path / "full", scope="full")
    stage = Path(result["source"])
    assert all(relative not in result["files"] for relative in (
        "release/vendor/rtx4090/private.json", "release/rtx4090/unrelated.md",
        "release/rtx4090/results/private.log", "scripts/private_wheelhouse.py",
        "release/vendor/rtx5090/private.json", "release/rtx5090/results/private.log"))
    for relative in ("release/rtx4090/qualification.json", "release/rtx4090/results/results.json",
                     "release/rtx4090/results/reproduce_manifest.json", "release/rtx4090/results/render_results.py",
                     "release/rtx5090/qualification.json", "release/rtx5090/results/results.json",
                     "release/rtx5090/results/reproduce_manifest.json", "release/rtx5090/results/render_results.py"):
        assert builder.sha((stage / relative).read_bytes()) == CURRENT_CONTROLS[relative]
    assert "scripts/qualify_sm89_fp8.py" in result["files"]
    assert "scripts/qualify_sm120_fp8.py" in result["files"]
    assert result["files"]["scripts/reproduce_va_2v4a.py"]["sha256"] == CURRENT_CONTROLS["scripts/reproduce_va_2v4a.py"]
    assert "benchmarks/regression/hardware.py" in result["files"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", UV_OFFLINE="1")
    environment.pop("PYTHONPATH", None)
    for family in ("pi05", "va", "vla4", "vla2", "groot", "edge", "nano", "dreamzero"):
        command = [sys.executable, str(stage / "scripts/bootstrap_vendor.py"), "plan", family, "--target", "rtx4090"]
        completed = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=15, check=True)
        profile = json.loads(completed.stdout)
        assert profile["deployment_target"] == "rtx4090" and profile["target"]["machine"] == "x86_64"
        for reference in profile.get("python_metadata_overlays", []):
            path = stage / "release/vendor" / reference["path"]
            assert builder.sha(path.read_bytes()) == reference["sha256"]
        completed = subprocess.run([sys.executable, str(stage / "scripts/prepare_auxiliary_assets.py"), "plan", family],
                                   capture_output=True, text=True, env=environment, timeout=15, check=True)
        assert json.loads(completed.stdout)["family"] == family
    for target in ("jetson_thor", "rtx4090", "rtx5090"):
        completed = subprocess.run([sys.executable, str(stage / "scripts/public_deploy.py"), "plan", "all", "--target", target],
                                   capture_output=True, text=True, env=environment, timeout=15, check=True)
        assert json.loads(completed.stdout)["ok"] is True
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from benchmarks.regression.reproduce import make_plan; "
        "models=('pi05','va','vla4','vla2','groot','edge','nano','dreamzero'); "
        "assert all(make_plan(m,'native',target='rtx4090')['target']['capability']==[8,9] for m in models); "
        "assert all(make_plan(m,'native',target='rtx5090')['target']['capability']==[12,0] for m in models); "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(stage)], env=environment, timeout=15, check=True)


@pytest.mark.parametrize("omitted", ["benchmarks/regression/hardware.py",
                                    "benchmarks/regression/fixtures/deployment_profiles_rtx4090.json",
                                    "benchmarks/regression/fixtures/deployment_profiles_rtx5090.json"])
def test_core_wheel_cannot_drop_rtx_dispatch_or_profile(tmp_path, omitted):
    wheel = tmp_path / "core.whl"
    members = {"instinctflash/__init__.py": b"", "benchmarks/regression/hardware.py": b"# explicit targets\n",
               "benchmarks/regression/fixtures/deployment_profiles.json": b"{}",
               "benchmarks/regression/fixtures/deployment_profiles_rtx4090.json": b'{"target":"rtx4090"}',
               "benchmarks/regression/fixtures/deployment_profiles_rtx5090.json": b'{"target":"rtx5090"}'}
    files = {name: {"sha256": builder.sha(content), "wheel_required": True} for name, content in members.items()}
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in members.items():
            if name != omitted:
                archive.writestr(name, content)
        archive.writestr("instinctflash-0.1.dist-info/METADATA", "Name: instinctflash\nVersion: 0.1\n")
    with pytest.raises(ValueError, match="dropped selected"):
        builder.inspect_wheel(wheel, "instinctflash", ".", "instinctflash", files)

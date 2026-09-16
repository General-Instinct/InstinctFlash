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


def args(tmp_path):
    return SimpleNamespace(root=tmp_path / "run", env_dir=None, vendor_dir=None,
                           cache_dir=None, checkout=ROOT, timeout=10, python=sys.executable,
                           uv="uv", repaired_wheel=None, repair_receipt=None, package_wheel_dir=None)


@pytest.mark.parametrize("family", bootstrap.FAMILIES)
def test_all_profiles_have_exact_source_and_inputs(family):
    p = bootstrap.load_profile(family)
    assert p["qualification"]["GPU_verified"] is False
    assert p["target"]["machine"] == "aarch64"
    if family != "pi05":
        assert len(p["source"]["source"]["revision"]) == 40
        assert set(p["packaging_patch"]["before"]) <= {"pyproject.toml", "setup.py"}
        assert p["packaging_patch"]["before"].keys() == p["packaging_patch"]["after"].keys()


@pytest.mark.parametrize("family", bootstrap.FAMILIES)
def test_rtx4090_profiles_preserve_vendor_source_and_bind_x86_wheels(family):
    thor = bootstrap.load_profile(family)
    rtx = bootstrap.load_profile(family, target="rtx4090")
    assert rtx["deployment_target"] == "rtx4090"
    assert rtx["target"]["machine"] == "x86_64"
    assert rtx["wheel_metadata_repair"] is None
    assert rtx["qualification"]["GPU_verified"] is False
    assert rtx["source_manifest"] == thor["source_manifest"]
    if family in {"edge", "nano"}:
        assert rtx["packaging_patch"]["path"] == "rtx4090/cosmos/inference_packaging.patch"
        assert rtx["packaging_patch"]["before"] == thor["packaging_patch"]["before"]
    else:
        assert rtx["packaging_patch"] == thor["packaging_patch"]
    for name in ("torch", "torchvision"):
        assert "x86_64.whl#sha256=" in rtx["public_wheel_overrides"][name]
    assert all("aarch64" not in url for url in rtx["public_wheel_overrides"].values())
    if family in {"edge", "nano"}:
        assert ".gb300" not in rtx["public_wheel_overrides"]["natten"]
        assert "x86_64" in rtx["public_wheel_overrides"]["triton"]


def test_rtx4090_plan_is_offline_and_requires_explicit_target(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/bootstrap_vendor.py"), "plan", "pi05",
                        "--target", "rtx4090", "--root", str(tmp_path / "absent")],
                       capture_output=True, text=True, check=True, env={"PATH": os.environ["PATH"]})
    assert json.loads(r.stdout)["target"]["machine"] == "x86_64"
    assert not (tmp_path / "absent").exists()
    assert bootstrap.load_profile("pi05")["target"]["machine"] == "aarch64"


def test_rtx4090_dependency_command_never_installs_thor_wheel_repair(tmp_path, monkeypatch):
    b = bootstrap.Bootstrap(args(tmp_path), bootstrap.load_profile("pi05", target="rtx4090"))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: '["Linux","x86_64","3.12"]')
    monkeypatch.setattr(b, "prepare_source", lambda: None)
    monkeypatch.setattr(b, "repaired_wheel", lambda: pytest.fail("Thor repair must not run on x86_64"))
    monkeypatch.setattr(b, "auxiliary_python_wheels", lambda: [])
    commands = []

    class StopBeforeInstall(Exception):
        pass

    def command(label, argv, **kwargs):
        commands.append((label, argv))
        if label == "dependency_resolution":
            raise StopBeforeInstall

    monkeypatch.setattr(b, "command", command)
    with pytest.raises(StopBeforeInstall):
        b.install()
    argv = commands[-1][1]
    assert commands[-1][0] == "dependency_resolution"
    assert "None" not in argv
    assert not any("aarch64" in item for item in argv)
    assert any(item.startswith("torch @ ") and "x86_64" in item for item in argv)
    assert "--no-deps" not in argv


def test_plan_does_not_require_uv_torch_network_or_create_destination(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/bootstrap_vendor.py"), "plan", "nano",
                        "--root", str(tmp_path / "absent")], capture_output=True, text=True, check=True,
                       env={"PATH": os.environ["PATH"]})
    assert json.loads(r.stdout)["family"] == "nano"
    assert not (tmp_path / "absent").exists()


def test_clean_environment_drops_foreign_stack_and_legacy_policy_flags(monkeypatch):
    for k in ["IFL_RECOMPUTE", "VIRTUAL_ENV", "PYTHONPATH", "PIP_INDEX_URL", "UV_INDEX", "LOAD_TRT_ENGINE",
              "DYNAMIC_CACHE_SCHEDULE", "NUM_DIT_STEPS", "LINGBOT_ROOT", "UV_OFFLINE", "UV_CONSTRAINT"]:
        monkeypatch.setenv(k, "unwanted")
    e = bootstrap.clean_environment(None)
    assert e["CUDA_VISIBLE_DEVICES"] == ""
    assert e["UV_CONCURRENT_DOWNLOADS"] == "1"
    assert e["GIT_LFS_SKIP_SMUDGE"] == "1"
    assert all(k not in e for k in ["PYTHONPATH", "LOAD_TRT_ENGINE", "DYNAMIC_CACHE_SCHEDULE", "IFL_RECOMPUTE", "LINGBOT_ROOT", "PIP_INDEX_URL"])
    assert "UV_OFFLINE" not in e and "UV_CONSTRAINT" not in e


def test_dreamzero_runtime_activation_disables_only_update_check(tmp_path):
    values = bootstrap.activation_environment(bootstrap.load_profile("dreamzero"), tmp_path)
    assert values == {"DREAMZERO_ROOT": str(tmp_path), "NO_ALBUMENTATIONS_UPDATE": "1"}
    assert "NO_ALBUMENTATIONS_UPDATE" not in bootstrap.activation_environment(bootstrap.load_profile("va"), tmp_path)


def test_thor_compiler_activation_binds_selected_binary_and_both_triton_paths(tmp_path, monkeypatch):
    compiler = tmp_path / "CUDA toolkit" / "ptxas"
    compiler.parent.mkdir()
    compiler.write_bytes(b"selected assembler")
    compiler.chmod(0o700)
    commands = []

    def assemble(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 30
        assert "--gpu-name=sm_110a" in command
        Path(command[-1]).write_bytes(b"assembled cubin")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", assemble)
    output = tmp_path / "compiler-receipt"
    result = bootstrap.prepare_ptxas(compiler, output)
    assert result["status"] == "passed" and result["GPU_used"] is False
    assert result["sha256"] == bootstrap.sha(compiler)
    assert len(commands) == 1 and commands[0][0] == str(compiler)
    assert result["environment"] == {
        "TRITON_PTXAS_PATH": str(compiler), "TRITON_PTXAS_BLACKWELL_PATH": str(compiler)}
    assert "export TRITON_PTXAS_BLACKWELL_PATH='" in (output / "run.env").read_text()


@pytest.mark.parametrize("returncode", [0, 1])
def test_thor_compiler_rejects_unsupported_target_or_absent_binary(tmp_path, monkeypatch, returncode):
    compiler = tmp_path / "ptxas"
    compiler.write_bytes(b"old assembler")
    compiler.chmod(0o700)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, returncode, b"", b"unsupported sm_110a"))
    output = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="cannot assemble sm_110a"):
        bootstrap.prepare_ptxas(compiler, output)
    assert json.loads((output / "receipt.json").read_text())["status"] == "failed"
    assert not (output / "run.env").exists()


def test_thor_compiler_rejects_nonexecutable_before_creating_output(tmp_path):
    compiler = tmp_path / "not-executable"
    compiler.write_bytes(b"data")
    with pytest.raises(ValueError, match="executable"):
        bootstrap.prepare_ptxas(compiler, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_rtx4090_compiler_uses_ada_target_and_omits_blackwell_override(tmp_path, monkeypatch):
    compiler = tmp_path / "ptxas"
    compiler.write_bytes(b"Ada-capable assembler")
    compiler.chmod(0o700)

    def assemble(command, **kwargs):
        assert "--gpu-name=sm_89" in command
        assert ".target sm_89" in Path(command[2]).read_text()
        Path(command[-1]).write_bytes(b"sm89 cubin")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", assemble)
    result = bootstrap.prepare_ptxas(compiler, tmp_path / "probe", target="rtx4090")
    assert result["target"] == "sm_89" and result["GPU_used"] is False
    assert result["environment"] == {"TRITON_PTXAS_PATH": str(compiler)}


def test_dreamzero_checks_complete_native_entrypoint_and_pinned_cli_dependency():
    profile = bootstrap.load_profile("dreamzero")
    requirements = (ROOT / "release/vendor" / profile["requirements"]["path"]).read_text().splitlines()
    assert "tyro==1.0.16" in requirements
    assert profile["mandatory_native_host_imports"] == {"tyro": "1.0.16"}
    catalog = json.loads((ROOT / "release/deployment_profiles.json").read_text())
    row = next(r for r in catalog["models"] if r["id"] == "dreamzero")
    assert {"module": "eval_utils.serve_dreamzero_wan22", "attributes": [
        "DreamZeroWan225BPolicy", "_get_expected_video_resolution", "_maybe_init_distributed",
    ]} in row["dependency_imports"]


def test_vla4_mandatory_native_ipdb_import_has_exact_install_and_doctor_pin():
    profile = bootstrap.load_profile("vla4")
    requirements = (ROOT / "release/vendor" / profile["requirements"]["path"]).read_text().splitlines()
    assert "ipdb==0.13.13" in requirements
    assert profile["public_wheel_overrides"]["ipdb"].endswith(
        "#sha256=45529994741c4ab6d2388bfa5d7b725c2cf7fe9deffabdb8a6113aa5ed449ed4")
    catalog = json.loads((ROOT / "release/deployment_profiles.json").read_text())
    vla4 = next(row for row in catalog["models"] if row["id"] == "vla4")
    assert {"module": "ipdb", "attributes": []} in vla4["dependency_imports"]


def test_vla2_native_data_import_dependencies_are_pinned_before_gpu_import():
    profile = bootstrap.load_profile("vla2")
    requirements = (ROOT / "release/vendor" / profile["requirements"]["path"]).read_text().splitlines()
    assert {"torchdata==0.11.0", "pydantic==2.13.4"} <= set(requirements)
    assert profile["mandatory_native_host_imports"] == {"torchdata": "0.11.0", "pydantic": "2.13.4"}


@pytest.mark.parametrize("family", ["edge", "nano"])
def test_cosmos_native_loader_import_dependencies_are_pinned(family):
    profile = bootstrap.load_profile(family)
    requirements = (ROOT / "release/vendor" / profile["requirements"]["path"]).read_text().splitlines()
    assert {"boto3==1.43.80", "pandas==3.0.5"} <= set(requirements)
    catalog = json.loads((ROOT / "release/deployment_profiles.json").read_text())
    row = next(r for r in catalog["models"] if r["id"] == family)
    assert {"module": "cosmos_framework.scripts.action_policy_server_robolab",
            "attributes": ["RobolabPolicyService", "RobolabServerArgs"]} in row["dependency_imports"]
    assert "cosmos_framework.scripts.action_policy_server_robolab" not in row["source_only_modules"]
    assert "uv==0.12.5" in requirements
    assert profile["native_tool"]["path"] == "cosmos/hf_tool.json"
    assert bootstrap.sha(ROOT / "release/vendor/cosmos/hf_tool.json") == profile["native_tool"]["sha256"]


@pytest.mark.parametrize("family", ["edge", "nano"])
def test_cosmos_dependency_command_pins_complete_pytorch_triton_build(tmp_path, monkeypatch, family):
    # Same version/ABI on PyPI identifies a different compiler build. Exercise
    # the actual install command construction, stopping before any installation.
    expected = ("https://download.pytorch.org/whl/"
                "triton-3.6.0-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
                "#sha256=58d57d6796b0004076315433526fe9d4af42044d430afdee1e6cd42a76bd6d09")
    profile = bootstrap.load_profile(family)
    assert profile["public_wheel_overrides"]["triton"] == expected
    assert profile["target"]["python_minor"] == "3.13"
    assert profile["compiler_wheel_origin"]["sha256"] == expected.split("#sha256=")[1]
    b = bootstrap.Bootstrap(args(tmp_path), profile)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: '["Linux","aarch64","3.13"]')
    monkeypatch.setattr(b, "prepare_source", lambda: None)
    monkeypatch.setattr(b, "repaired_wheel", lambda: tmp_path / "bounded-repair.whl")
    monkeypatch.setattr(b, "auxiliary_python_wheels", lambda: [])
    commands = []

    class StopBeforeInstall(Exception):
        pass

    def command(label, argv, **kwargs):
        commands.append((label, argv))
        if label == "dependency_resolution":
            raise StopBeforeInstall
        assert label == "venv"

    monkeypatch.setattr(b, "command", command)
    with pytest.raises(StopBeforeInstall):
        b.install()
    assert [label for label, _ in commands] == ["venv", "dependency_resolution"]
    argv = commands[-1][1]
    assert argv.count("triton @ " + expected) == 1
    assert argv[argv.index("--only-binary") + 1] == ":all:"
    assert "--no-deps" not in argv
    constraints = Path(argv[argv.index("-c") + 1]).read_text().splitlines()
    assert "triton==3.6.0" in constraints
    assert all(f"{name} @ {url}" in argv for name, url in profile["public_wheel_overrides"].items())
    assert not b.env_dir.exists()  # No wheel/native payload was installed or replaced.


def test_cosmos_cp313_compiler_pin_does_not_replace_other_family_builds():
    for family in set(bootstrap.FAMILIES) - {"edge", "nano"}:
        profile = bootstrap.load_profile(family)
        assert profile["target"]["python_minor"] == "3.12"
        assert "triton" not in profile["public_wheel_overrides"]


@pytest.mark.parametrize("existing", ["root", "env", "vendor"])
def test_existing_user_state_refused_without_mutation(tmp_path, existing):
    a = args(tmp_path)
    b = bootstrap.Bootstrap(a, bootstrap.load_profile("va"))
    p = {"root": b.root, "env": b.env_dir, "vendor": b.vendor}[existing]
    p.mkdir(parents=True)
    (p / "keep").write_text("original")
    with pytest.raises(ValueError, match="existing destination"):
        b.prepare_roots()
    assert (p / "keep").read_text() == "original"
    assert not (b.root / "receipts").exists()


def test_interpreter_mismatch_fails_before_fetch_or_install_and_preserves_receipt(tmp_path, monkeypatch):
    a = args(tmp_path)
    b = bootstrap.Bootstrap(a, bootstrap.load_profile("va"))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: '["Linux","x86_64","3.12"]')
    monkeypatch.setattr(b, "command", lambda *a, **kw: pytest.fail("unexpected install"))
    with pytest.raises(ValueError, match="selected target"):
        b.install()
    assert json.loads((b.root / "failure.json").read_text())["automatic_retry"] is False
    assert not b.env_dir.exists()


def test_exact_commit_and_both_patch_results_on_cpu_git_fixture(tmp_path):
    # Uses actual public patch files against their captured original metadata,
    # and git's parser, rather than mocking patch application.
    data = ROOT / "eval/public_release_2026-09-15/vendor_provenance_sources/upstream_packaging_v1.json"
    if not data.is_file():
        pytest.skip("local immutable upstream packaging capture is not part of public release")
    captured = json.loads(data.read_text())
    for family in ["va", "vla4", "vla2", "groot", "dreamzero", "edge"]:
        p = bootstrap.load_profile(family)
        root = tmp_path / family
        root.mkdir()
        source = "cosmos" if family == "edge" else family
        patch = p["packaging_patch"]
        for name, expected in patch["before"].items():
            (root / name).write_text(captured[source][name])
            assert bootstrap.sha(root / name) == expected
        subprocess.run(["git", "apply", "--check", str(ROOT / "release/vendor" / patch["path"])], cwd=root, check=True)
        subprocess.run(["git", "apply", str(ROOT / "release/vendor" / patch["path"])], cwd=root, check=True)
        for name, expected in patch["after"].items():
            assert bootstrap.sha(root / name) == expected


def test_failed_command_has_exact_log_and_stops(tmp_path):
    b = bootstrap.Bootstrap(args(tmp_path), bootstrap.load_profile("va"))
    b.prepare_roots()
    with pytest.raises(RuntimeError, match="failed"):
        b.command("failure", [sys.executable, "-c", "print('dependency conflict');raise SystemExit(2)"])
    assert len(b.commands) == 1 and b.commands[0]["returncode"] == 2
    log = Path(b.commands[0]["log"])
    assert b.commands[0]["log_sha256"] == bootstrap.sha(log)
    assert "dependency conflict" in log.read_text()


def test_packaging_manifest_cannot_touch_model_source(tmp_path):
    import shutil

    shutil.copytree(ROOT / "release/vendor", tmp_path / "release/vendor")
    source = tmp_path / "release/vendor/va/bootstrap.json"
    profile = json.loads(source.read_text())
    profile["packaging_patch"]["before"]["wan_va/model.py"] = "0" * 64
    source.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="must not change runtime source"):
        bootstrap.load_profile("va", tmp_path)


def test_missing_repair_receipt_cannot_bypass_public_hash_gate():
    with pytest.raises(SystemExit):
        bootstrap.main(["install", "va", "--root", "/unused", "--repaired-wheel", "/untrusted.whl"])


def metadata_fixture(tmp_path):
    import csv
    import hashlib
    import io
    import zipfile

    files = {"example/model.py": b"def action(x): return x\n",
             "example-1.dist-info/LICENSE": b"upstream license\n",
             "example-1.dist-info/METADATA": b"Metadata-Version: 2.1\nName: example\nVersion: 1\nRequires-Dist: training-only\nProvides-Extra: hardware\n\nOriginal description.\n"}
    record = "example-1.dist-info/RECORD"
    out = io.StringIO(newline="")
    csv.writer(out, lineterminator="\n").writerows(
        [[n, bootstrap.repair_vendor_wheel.digest(v), str(len(v))] for n, v in files.items()] + [[record, "", ""]])
    files[record] = out.getvalue().encode()
    original = tmp_path / "original.whl"
    with zipfile.ZipFile(original, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    rule = {"sha256": bootstrap.sha(original), "bytes": original.stat().st_size,
            "metadata_member": "example-1.dist-info/METADATA", "record_member": record,
            "metadata_sha256": hashlib.sha256(files["example-1.dist-info/METADATA"]).hexdigest(),
            "historical_imported_source_sha256": {"example/model.py": hashlib.sha256(files["example/model.py"]).hexdigest()},
            "inference_dependencies": ["numpy==1.26.4"], "scope": "test inference subset"}
    return original, rule, files


def test_inference_overlay_preserves_all_source_license_description_and_valid_record(tmp_path):
    import csv
    import io
    import zipfile

    original, rule, before = metadata_fixture(tmp_path)
    output = tmp_path / "repacked.whl"
    receipt = bootstrap.prepare_inference_metadata(original, output, rule)
    assert receipt["all_implementation_and_notice_bytes_preserved"] is True
    with zipfile.ZipFile(output) as z:
        assert z.read("example/model.py") == before["example/model.py"]
        assert z.read("example-1.dist-info/LICENSE") == before["example-1.dist-info/LICENSE"]
        meta = z.read(rule["metadata_member"])
        assert b"Requires-Dist: numpy==1.26.4" in meta and b"training-only" not in meta
        assert meta.split(b"\n\n", 1)[1] == before[rule["metadata_member"]].split(b"\n\n", 1)[1]
        for name, digest, size in csv.reader(io.StringIO(z.read(rule["record_member"]).decode())):
            if name != rule["record_member"]:
                assert digest == bootstrap.repair_vendor_wheel.digest(z.read(name))
                assert size == str(len(z.read(name)))


@pytest.mark.parametrize("tamper", ["artifact", "source", "metadata", "dependency"])
def test_inference_overlay_rejects_unbound_inputs_and_metadata_injection(tmp_path, tamper):
    original, rule, _ = metadata_fixture(tmp_path)
    if tamper == "artifact":
        rule["sha256"] = "0" * 64
    elif tamper == "source":
        rule["historical_imported_source_sha256"]["example/model.py"] = "0" * 64
    elif tamper == "metadata":
        rule["metadata_sha256"] = "0" * 64
    else:
        rule["inference_dependencies"] = ["numpy\nRequires-Dist: injected"]
    with pytest.raises(ValueError):
        bootstrap.prepare_inference_metadata(original, tmp_path / "repacked.whl", rule)
    assert not (tmp_path / "repacked.whl").exists()

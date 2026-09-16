"""5090 planning and installation boundaries; no device or model is constructed."""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.regression import hardware, reproduce

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"target_5090_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = load_script("bootstrap_vendor")
deploy = load_script("public_deploy")
builder = load_script("build_public_release")


@pytest.mark.parametrize("family", bootstrap.FAMILIES)
def test_5090_preserves_exact_original_dependency_contracts(family):
    previous = bootstrap.load_profile(family, target="rtx4090")
    current = bootstrap.load_profile(family, target="rtx5090")
    restored = copy.deepcopy(current)
    restored["deployment_target"] = "rtx4090"
    if family in {"edge", "nano"}:
        for key in ("requirements", "constraints", "packaging_patch"):
            before = previous[key]
            after = current[key]
            assert (ROOT / "release/vendor" / before["path"]).read_bytes() == (
                ROOT / "release/vendor" / after["path"]).read_bytes()
            restored[key]["path"] = before["path"]
    assert restored == previous
    assert current["target"]["machine"] == "x86_64"
    assert current["qualification"]["GPU_verified"] is False
    assert current["qualification"]["evidence"] is None


@pytest.mark.parametrize("family", bootstrap.FAMILIES)
def test_all_5090_modes_keep_full_pair_and_original_policy(family):
    catalog = deploy.load_profiles(target="rtx5090")
    row = next(row for row in catalog["models"] if row["id"] == family)
    assert row["bootstrap"]["qualification"] is None
    assert "pending" in row["bootstrap"]["status"]
    for name, mode in row["execution_modes"].items():
        assert "recorded_e2e_qualification" not in mode
        assert "evidence_cell" not in mode
        assert "pending" in mode["evidence_kind"]
        assert mode["gpu_qualified_by_plan"] is mode["task_quality_certified"] is False
        previous = reproduce.make_plan(family, name, target="rtx4090")
        current = reproduce.make_plan(family, name, target="rtx5090")
        assert current["target"] == current["matrix"]["target"] == {
            "name": "rtx5090", "capability": [12, 0]}
        assert current["matrix"]["cells"] == previous["matrix"]["cells"]
        assert current["matrix"]["protocol"] == previous["matrix"]["protocol"]
        assert current["matrix"]["expected_main_cells"] == 3
        assert current["task_quality_validated"] is current["publication_ready"] is False
        planned = deploy.make_plan(row, name)
        assert planned["commands"]["doctor_before_weights"][-2:] == ["--target", "rtx5090"]


def test_5090_catalog_is_pending_for_all_23_modes_and_mirrored_into_wheel():
    catalog = deploy.load_profiles(target="rtx5090")
    assert sum(len(row["execution_modes"]) for row in catalog["models"]) == 23
    assert "recorded_e2e_qualification" not in catalog
    raw = (ROOT / "release/rtx5090/deployment_profiles.json").read_bytes()
    assert b"rtx4090" not in raw and b"capacity_excluded" not in raw
    assert raw == reproduce.profiles_path("rtx5090").read_bytes()
    assert builder.SELECTED_CONTROLS["release/rtx5090/deployment_profiles.json"] == builder.sha(raw)
    staged = builder.tomllib.loads(builder.staged_pyproject((ROOT / "pyproject.toml").read_bytes()).decode())
    assert "fixtures/deployment_profiles_rtx5090.json" in staged["tool"]["setuptools"]["package-data"]["benchmarks.regression"]
    controls = {**builder.SELECTED_CONTROLS, **builder.PUBLIC_VENDOR_FILES}
    assert all(str(path.relative_to(ROOT)) in builder.PUBLIC_VENDOR_FILES
               for path in (ROOT / "release/vendor/rtx5090").rglob("*") if path.is_file())
    for name, digest in controls.items():
        if name.startswith(("release/rtx5090/", "release/vendor/rtx5090/")) or name in {
                "scripts/bootstrap_vendor.py", "scripts/public_deploy.py", "scripts/prepare_native_tools.py"}:
            assert builder.sha((ROOT / name).read_bytes()) == digest


@pytest.mark.parametrize("other_target", ["jetson_thor", "rtx4090"])
def test_5090_rejects_other_target_catalogs_and_row_relabeling(tmp_path, other_target):
    path = deploy.PROFILE_PATHS[other_target]
    with pytest.raises(ValueError, match="schema or target"):
        deploy.load_profiles(path, target="rtx5090")
    with pytest.raises(ValueError, match="catalog target"):
        reproduce.make_plan("pi05", target="rtx5090", catalog_path=path)
    catalog = deploy.load_profiles(target="rtx5090")
    catalog["models"][0]["deployment_target"] = other_target
    tampered = tmp_path / "mixed.json"
    tampered.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="model deployment target"):
        deploy.load_profiles(tampered, target="rtx5090")


@pytest.mark.parametrize("name,capability", [
    ("NVIDIA GeForce RTX 4090", [8, 9]),
    ("NVIDIA GeForce RTX 4090", [12, 0]),
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", [12, 0]),
    ("NVIDIA GeForce RTX 5090", [12.0, 0]),
])
def test_5090_requires_exact_model_and_integer_capability(name, capability):
    target = hardware.target_record("rtx5090")
    assert not hardware.device_matches(target, capability, name)
    with pytest.raises(ValueError, match="actual GPU"):
        hardware.validate_device_receipt({"target": target, "name": name,
            "capability": capability, "status": "passed", "uuid": "fixture-only",
            "total_memory_bytes": 32 << 30}, target)


def test_5090_device_binding_does_not_change_default_thor_target():
    assert hardware.device_matches(hardware.target_record("rtx5090"), [12, 0], "NVIDIA GeForce RTX 5090")
    assert hardware.bound_target(None) == {"name": "jetson_thor", "capability": [11, 0]}
    with pytest.raises(ValueError, match="target/capability"):
        hardware.bound_target({"name": "rtx5090", "capability": [8, 9]})


@pytest.mark.parametrize("exit_code", [0, 1])
def test_5090_assembler_request_binds_sm120_and_both_triton_paths(tmp_path, monkeypatch, exit_code):
    compiler = tmp_path / "ptxas"
    compiler.write_bytes(b"test assembler path; never executed")
    compiler.chmod(0o700)

    def assemble(command, **kwargs):
        assert command[0] == str(compiler) and command[1] == "--gpu-name=sm_120"
        source = Path(command[2]).read_text()
        assert source.startswith(".version 9.0\n.target sm_120\n")
        assert kwargs["timeout"] == 30
        if exit_code == 0:
            Path(command[-1]).write_bytes(b"test-only cubin")
        return SimpleNamespace(returncode=exit_code, stdout=b"", stderr=b"test result")

    monkeypatch.setattr(subprocess, "run", assemble)
    output = tmp_path / "probe"
    if exit_code:
        with pytest.raises(RuntimeError, match="cannot assemble sm_120"):
            bootstrap.prepare_ptxas(compiler, output, target="rtx5090")
    else:
        bootstrap.prepare_ptxas(compiler, output, target="rtx5090")
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == ("failed" if exit_code else "passed")
    assert receipt["GPU_used"] is False and receipt["deployment_target"] == "rtx5090"
    assert receipt["environment"] == {"TRITON_PTXAS_PATH": str(compiler),
                                      "TRITON_PTXAS_BLACKWELL_PATH": str(compiler)}


def test_5090_cli_plan_stays_cpu_only_without_creating_install_destination(tmp_path):
    output = tmp_path / "not-created"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/bootstrap_vendor.py"),
        "plan", "pi05", "--target", "rtx5090", "--root", str(output)],
        capture_output=True, text=True, timeout=15, check=True)
    assert json.loads(result.stdout)["deployment_target"] == "rtx5090"
    assert not output.exists()

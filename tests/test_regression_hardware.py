"""CPU tests of target binding and owned-process deadlines; no GPU is initialized."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.regression import hardware, reproduce, user_e2e


@pytest.fixture
def target_catalog(tmp_path, monkeypatch):
    catalog = json.loads(reproduce.profiles_path().read_text())
    # Synthetic metadata fixture only, never a claim that these routes run on RTX.
    catalog["target"] = "rtx4090"
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(catalog))
    original = reproduce.profiles_path
    monkeypatch.setattr(reproduce, "profiles_path", lambda target="jetson_thor":
                        path if target == "rtx4090" else original(target))
    return path


def test_target_catalog_and_all_model_protocols_are_bound(target_catalog):
    for family in reproduce.ACTION_SHAPES:
        thor = reproduce.make_plan(family)
        rtx = reproduce.make_plan(family, target="rtx4090")
        assert rtx["target"] == rtx["matrix"]["target"] == {"name": "rtx4090", "capability": [8, 9]}
        assert thor["target"] == {"name": "jetson_thor", "capability": [11, 0]}
        assert rtx["matrix"]["protocol"] == thor["matrix"]["protocol"]
        assert rtx["matrix"]["cells"] == thor["matrix"]["cells"]
        assert rtx["task_quality_validated"] is rtx["publication_ready"] is False


def test_target_cannot_relabel_another_catalog():
    with pytest.raises(ValueError, match="catalog target"):
        reproduce.make_plan("pi05", target="rtx4090", catalog_path=reproduce.profiles_path())


def test_explicit_target_cli_plan(target_catalog, capsys):
    assert reproduce.main(["plan", "--model", "pi05", "--target", "rtx4090"]) == 0
    assert json.loads(capsys.readouterr().out)["target"]["capability"] == [8, 9]


def test_target_prepare_roundtrip_and_cross_target_tamper(target_catalog, tmp_path, monkeypatch):
    def download(**kwargs):
        path = tmp_path / "snapshot" / kwargs["revision"]
        path.mkdir(parents=True)
        return path
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    root = tmp_path / "prepared"
    reproduce.prepare("va", "fp8", root, target="rtx4090", extra_modes=["2v4a-fp8"])
    plan, receipt = reproduce.validate_bundle(root)
    assert plan["target"] == receipt["target"]
    assert len(plan["matrix"]["cells"]) == 4
    receipt["target"] = hardware.target_record("jetson_thor")
    (root / "preparation.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="preparation target differs"):
        reproduce.validate_bundle(root)


def test_original_targetless_thor_bundle_remains_readable(tmp_path, monkeypatch):
    def download(**kwargs):
        path = tmp_path / "snapshot" / kwargs["revision"]
        path.mkdir(parents=True)
        return path
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    root = tmp_path / "prepared"
    reproduce.prepare("pi05", "native", root)
    plan, receipt = reproduce.validate_bundle(root)
    plan.pop("target")
    plan["matrix"].pop("target")
    plan["limitations"][2] = "Runtime captures target Jetson Thor SM110 using the unchanged capture contract."
    (root / "plan.json").write_bytes(reproduce.encoded(plan))
    (root / "matrix.json").write_bytes(reproduce.encoded(plan["matrix"]))
    receipt.pop("target")
    receipt.update(plan_sha256=reproduce.sha(root / "plan.json"),
                   matrix_sha256=reproduce.sha(root / "matrix.json"))
    (root / "preparation.json").write_bytes(reproduce.encoded(receipt))
    loaded, _ = reproduce.validate_bundle(root)
    assert "target" not in loaded and hardware.bound_target(loaded.get("target"))["name"] == "jetson_thor"


@pytest.mark.parametrize("value", ["rtx5090", {"name": "rtx4090", "capability": [12, 0]},
                                   {"name": "rtx4090", "capability": [8, 9.0]}])
def test_wrong_or_untyped_targets_are_rejected(value):
    with pytest.raises(ValueError):
        hardware.bound_target(value)


@pytest.mark.parametrize("capability,name,passed", [
    ((8, 9), "NVIDIA GeForce RTX 4090", True),
    ((12, 0), "NVIDIA GeForce RTX 5090", False),
    ((8, 9), "NVIDIA L40S", False),
    ((11, 0), "NVIDIA Thor", False),
])
def test_actual_device_guard_uses_capability_and_model_name(capability, name, passed):
    cuda = SimpleNamespace(get_device_properties=lambda index: SimpleNamespace(
        name=name, uuid="unit-test-device", total_memory=24 << 30),
        get_device_capability=lambda index: capability)
    target = hardware.target_record("rtx4090")
    observed = hardware.probe_device(target, SimpleNamespace(cuda=cuda))
    assert observed["status"] == ("passed" if passed else "failed")
    if passed:
        hardware.validate_device_receipt(observed, target)
    else:
        with pytest.raises(ValueError, match="actual GPU"):
            hardware.validate_device_receipt(observed, target)


def test_gpu_process_lookup_resolves_path_and_filters_selected_uuid(monkeypatch):
    monkeypatch.setattr(user_e2e.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    calls = []
    def query(command, **kwargs):
        calls.append((command, kwargs))
        return f"GPU-selected, {os.getpid()}\nGPU-selected, 98765\nGPU-other, 54321\n"
    monkeypatch.setattr(user_e2e.subprocess, "check_output", query)
    torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda index:
                            SimpleNamespace(uuid="GPU-selected")))
    assert user_e2e.gpu_competitors(torch) == [98765]
    assert calls[0][0][0] == "/usr/bin/nvidia-smi" and calls[0][1]["timeout"] == 15
    monkeypatch.setattr(user_e2e.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="required on PATH"):
        user_e2e.gpu_competitors(torch)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, 86401])
def test_invalid_deadlines_cannot_disable_the_bound(value):
    with pytest.raises(ValueError, match="cell timeout"):
        reproduce.run("unused", "unused", cell_timeout=value)


def test_real_cpu_process_group_is_stopped_at_deadline(tmp_path):
    code = ("import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
            "print(child.pid,flush=True); time.sleep(60)")
    logfile = tmp_path / "process.log"
    with logfile.open("w") as log:
        result = reproduce.capture_process([sys.executable, "-I", "-c", code], cwd=tmp_path,
                                           env=os.environ.copy(), log=log, timeout=1)
    assert result["timed_out"] is True and result["exit_code"] != 0
    assert result["elapsed_seconds"] < 5
    child_pid = int(logfile.read_text().strip())
    state_file = Path(f"/proc/{child_pid}/stat")
    if state_file.exists():
        # A terminated grandchild can await adoption/reaping briefly on Linux.
        assert state_file.read_text().split(") ", 1)[1][0] == "Z"


def test_timed_out_run_preserves_failure_and_never_retries(tmp_path, monkeypatch):
    plan = reproduce.make_plan("pi05")
    root = tmp_path / "prepared"
    root.mkdir()
    for name in ("plan.json", "matrix.json", "profiles.json", "preparation.json", "inputs/recorded_inputs_v1.npz"):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("unit-test prepared bytes")
    monkeypatch.setattr(reproduce, "validate_bundle", lambda *args: (plan, {}))
    monkeypatch.setattr(reproduce, "require_installed", lambda: None)
    calls = []
    def timeout(command, **kwargs):
        calls.append(command)
        kwargs["log"].write("partial child output retained\n")
        return {"exit_code": 0, "timed_out": True, "timeout_seconds": 2}
    monkeypatch.setattr(reproduce, "capture_process", timeout)
    monkeypatch.setattr(reproduce, "report", lambda *args: {"status": "incomplete"})
    result = reproduce.run(root, tmp_path / "run", cell_timeout=2)
    assert len(calls) == 1 and result["status"] == "failed_or_incomplete"
    assert result["attempts"][0]["timed_out"] is True
    assert "deadline exceeded" in result["attempts"][0]["error"]
    persisted = json.loads((tmp_path / "run/run.json").read_text())
    assert persisted == result
    assert (tmp_path / "run/logs/pi05-eager_native.log").read_text() == "partial child output retained\n"

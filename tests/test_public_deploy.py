"""Release deployment planning and bounded CPU dependency checks; no native execution."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_deploy", ROOT / "scripts/public_deploy.py")
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


@pytest.fixture
def catalog():
    return deploy.load_profiles()


def model(catalog, name):
    return next(row for row in catalog["models"] if row["id"] == name)


def test_rtx4090_catalog_is_distinct_and_all_commands_keep_the_target(catalog):
    rtx = deploy.load_profiles(target="rtx4090")
    assert rtx["target"] == "rtx4090"
    assert len(rtx["models"]) == 8
    assert (ROOT / "release/rtx4090/deployment_profiles.json").read_bytes() == (
        ROOT / "benchmarks/regression/fixtures/deployment_profiles_rtx4090.json").read_bytes()
    for row in rtx["models"]:
        assert row["checkpoint"] == model(catalog, row["id"])["checkpoint"]
        plan = deploy.make_plan(row)
        assert plan["target"] == "rtx4090"
        assert plan["commands"]["doctor_before_weights"][-2:] == ["--target", "rtx4090"]
        assert "rtx4090" in row["bootstrap"]["command"]
        for mode in row["execution_modes"].values():
            assert mode["evidence_kind"] == "pending_rtx4090_measurement"
            assert mode["gpu_qualified_by_plan"] is False
    with pytest.raises(ValueError, match="schema or target"):
        deploy.load_profiles(ROOT / "release/deployment_profiles.json", target="rtx4090")


def test_rtx4090_cli_plan_binds_correct_catalog_hash():
    run = subprocess.run([sys.executable, str(ROOT / "scripts/public_deploy.py"),
                          "plan", "all", "--target", "rtx4090"], capture_output=True, text=True, check=True)
    receipt = json.loads(run.stdout)
    assert receipt["ok"] and receipt["target"] == "rtx4090"
    assert receipt["profiles_sha256"] == hashlib.sha256(
        (ROOT / "release/rtx4090/deployment_profiles.json").read_bytes()).hexdigest()
    assert len(receipt["results"]) == 8


def test_eight_models_six_adapter_packages_seven_backbones(catalog):
    assert len(catalog["models"]) == 8
    assert len({p["backbone"] for p in catalog["models"]}) == 7
    assert len({p["adapter"]["package_directory"] for p in catalog["models"]
                if p["adapter"]["package_directory"]}) == 6
    for profile in catalog["models"]:
        assert (ROOT / profile["adapter"]["source"]).is_file()
        if profile["id"] == "pi05":
            assert profile["upstream"]["revision"] is None
            assert profile["upstream"]["package_pin"] == {"distribution": "lerobot", "version": "0.6.1"}
        else:
            assert len(profile["upstream"]["revision"]) == 40
            assert (ROOT / profile["upstream"]["source_manifest"]).is_file()
        if profile["id"] in {"edge", "nano"}:
            assert profile["bootstrap"]["status"] == "compiler_wheel_origin_corrected_installation_pending"
            assert profile["bootstrap"]["prior_recipe_status"] == "fresh_thor_package_checks_passed"
            assert profile["bootstrap"]["compiler_wheel"]["sha256"] == (
                "58d57d6796b0004076315433526fe9d4af42044d430afdee1e6cd42a76bd6d09"
            )
        else:
            assert profile["bootstrap"]["status"] == "fresh_thor_package_checks_passed"


def test_dreamzero_empty_legacy_trt_variable_is_still_an_explicit_request(catalog, monkeypatch):
    monkeypatch.setenv("LOAD_TRT_ENGINE", "")
    result = deploy.doctor(model(catalog, "dreamzero"), "fp8",
                           probe_runner=lambda *a: {"checks": [deploy._check("CPU", True)]})
    check = next(c for c in result["checks"] if c["name"] == "DreamZero_FP8_TensorRT_disabled")
    assert check["status"] == "failed"


def test_cosmos_historical_torch_is_actual_210_not_suggested_213(catalog):
    for family in ("edge", "nano"):
        text = json.dumps(model(catalog, family))
        assert "Torch 2.10.0+cu130" in text
        assert "Torch 2.13/" not in text


def test_checkpoint_and_recipe_values_are_from_actual_matrix(catalog):
    source = ROOT / catalog["checkpoint_revision_source"]["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == catalog["checkpoint_revision_source"]["sha256"]
    cells = {c["id"]: c for c in json.loads(source.read_text())["cells"]}
    for profile in catalog["models"]:
        for name, mode in profile["execution_modes"].items():
            cell = cells[mode["evidence_cell"]]
            assert profile["checkpoint"] == {k: cell[k] for k in ("model_id", "revision")}
            assert mode["effective_schedule"] == cell["effective_schedule"]
            for key, value in cell["expected_runtime_kwargs"].items():
                assert mode["runtime_kwargs"][key] == value, (profile["id"], name, key)
            for key, value in cell["expected_optimizer_environment"].items():
                if key == "IFL_BF16_KERNEL_LIBRARY":
                    assert any(req["name"] == key for req in mode["native_requirements"])
                else:
                    assert mode["environment"][key] == value
    assert model(catalog, "pi05")["checkpoint"]["model_id"] == "lerobot/pi05_libero_finetuned_v044"


def test_native_defaults_and_opt_in_loss_are_independent(catalog):
    for profile in catalog["models"]:
        default = deploy.make_plan(profile)["selected_execution"]
        assert default["runtime_kwargs"]["precision"] == "native"
        assert default["runtime_kwargs"]["tier_ceiling"] == "bitexact"
        assert default["runtime_kwargs"]["step_cache"] == "checkpoint"
        assert not default["schedule_changed"]
        assert "nfe" not in default["runtime_kwargs"]
    numeric = deploy.make_plan(model(catalog, "edge"), "numeric")["selected_execution"]
    assert numeric["runtime_kwargs"]["precision"] == "native"
    assert numeric["effective_schedule"]["steps"] == 4
    assert numeric["effective_schedule"]["guidance"] == 3
    assert not numeric["schedule_changed"]
    fewstep = deploy.make_plan(model(catalog, "va"), "2v4a-fp8")["selected_execution"]
    assert fewstep["runtime_kwargs"]["nfe"] == {"video": 2, "action": 4}
    assert fewstep["runtime_kwargs"]["tier_ceiling"] == "behavioral"
    dynamic = deploy.make_plan(model(catalog, "dreamzero"), "dynamic-native")["selected_execution"]
    assert dynamic["runtime_kwargs"]["step_cache"] == "dynamic"
    assert dynamic["effective_schedule"]["solver_updates"] == 16


def test_portable_catalog_and_native_dependency_scope(catalog):
    text = json.dumps(catalog)
    assert "/home/" not in text and "/workspace/" not in text
    edge = deploy.make_plan(model(catalog, "edge"), "numeric")["selected_execution"]
    nano = deploy.make_plan(model(catalog, "nano"), "numeric")["selected_execution"]
    assert edge["native_requirements"][0]["name"] == "IFL_BF16_KERNEL_LIBRARY"
    assert "IFL_BF16_KERNEL_LIBRARY" not in edge["environment"]
    assert nano["native_requirements"] == []
    for name in ("vla4", "vla2"):
        requirements = deploy.make_plan(model(catalog, name), "fp8")["selected_execution"]["native_requirements"]
        assert any(r["name"] == "flash_rt.flash_rt_fa2" for r in requirements)


def test_planned_cli_configs_match_current_public_parser(catalog, tmp_path):
    from instinctflash.cli import ServeConfig
    from instinctflash.cli_config import parse_config

    for profile in catalog["models"]:
        for execution in profile["execution_modes"]:
            plan = deploy.make_plan(profile, execution)
            path = tmp_path / "serve.json"
            path.write_text(json.dumps(plan["serve_config_template"]))
            parsed = parse_config(ServeConfig, [f"--config_path={path}"])
            assert parsed.serve.model == "<resolved_checkpoint_path>"
            assert parsed.runtime.precision == plan["selected_execution"]["runtime_kwargs"]["precision"]
            code = plan["commands"]["prepare_pinned_checkpoint_after_doctor"][-1]
            compile(code, "<planned-weight-preparation>", "exec")
            assert profile["checkpoint"]["revision"] in code
            assert "from instinctflash.descriptors.package import from_pretrained" in code
            assert profile["checkpoint"]["model_id"] not in " ".join(plan["commands"]["serve_after_saving_config"])


@pytest.mark.parametrize("mutation", [
    "duplicate", "unpinned", "fp8_bitexact", "silent_schedule", "silent_dynamic", "native_fp8",
])
def test_invalid_catalog_rejected(catalog, tmp_path, mutation):
    rows = catalog["models"]
    mode = rows[0]["execution_modes"]["native"]
    if mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "unpinned":
        rows[0]["checkpoint"]["revision"] = "main"
    elif mutation == "fp8_bitexact":
        mode["runtime_kwargs"]["precision"] = "fp8"
    elif mutation == "silent_schedule":
        mode["runtime_kwargs"]["nfe"] = {"video": 2, "action": 4}
    elif mutation == "silent_dynamic":
        mode["runtime_kwargs"]["step_cache"] = "dynamic"
    else:
        mode["runtime_kwargs"].update(precision="fp8", tier_ceiling="numeric")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError):
        deploy.load_profiles(path)


def test_plan_stays_cpu_metadata_only_from_outside_checkout(tmp_path):
    script = ROOT / "scripts/public_deploy.py"
    result = subprocess.run([sys.executable, "-I", "-B", str(script), "plan", "all"],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    output = json.loads(result.stdout)
    assert output["ok"] and len(output["results"]) == 8
    for row in output["results"]:
        assert row["selected_execution"]["gpu_qualified_by_plan"] is False


def test_unsupported_selection_fails_before_any_probe(monkeypatch, capsys):
    monkeypatch.setattr(deploy, "doctor", lambda *a, **kw: pytest.fail("unexpected import probe"))
    assert deploy.main(["doctor", "all", "--execution", "2v4a-fp8"]) == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_successful_cpu_probe_never_claims_gpu_or_fresh_install(catalog):
    def fake(payload, timeout):
        assert payload["backbone"] == "pi05"
        assert any(p["module"] == "lerobot.policies.factory" for p in payload["dependencies"])
        return {"checks": [deploy._check("test_CPU_import", True)]}
    report = deploy.doctor(model(catalog, "pi05"), probe_runner=fake)
    assert report["ok"] and report["status"] == "cpu_preflight_passed"
    assert report["bootstrap_status"] == "fresh_thor_package_checks_passed"
    assert not report["GPU_verified"] and not report["weights_downloaded"]
    assert not report["model_constructed"] and not report["task_quality_certified"]


def test_missing_explicit_root_and_marker_are_actionable(catalog, monkeypatch, tmp_path):
    def fake(*args):
        return {"checks": [deploy._check("test_CPU_import", True)]}
    monkeypatch.delenv("LINGBOT_VLA_ROOT", raising=False)
    report = deploy.doctor(model(catalog, "vla4"), probe_runner=fake)
    assert not report["ok"]
    assert any(c["name"] == "source_root:LINGBOT_VLA_ROOT" and c["status"] == "failed" for c in report["checks"])
    monkeypatch.setenv("LINGBOT_VLA_ROOT", str(tmp_path))
    report = deploy.doctor(model(catalog, "vla4"), probe_runner=fake)
    assert any(c["name"] == "source_file:deploy/lingbot_vla_policy.py" and c["status"] == "failed" for c in report["checks"])


def test_vla2_deferred_import_requires_exact_sources_and_remains_deferred(catalog, tmp_path, monkeypatch):
    profile = copy.deepcopy(model(catalog, "vla2"))
    marker = tmp_path / "deploy/lingbot_vla_v2_policy.py"
    marker.parent.mkdir()
    marker.write_text("raise RuntimeError('must not execute during CPU source qualification')\n")
    profile["deferred_native_imports"][0]["source_sha256"] = {
        "deploy/lingbot_vla_v2_policy.py": hashlib.sha256(marker.read_bytes()).hexdigest()}
    monkeypatch.setenv("LINGBOT_VLA_V2_ROOT", str(tmp_path))

    def fake_probe(payload, timeout):
        modules = {d["module"] for d in payload["dependencies"]}
        assert {"torchdata", "pydantic"} <= modules
        assert "deploy.lingbot_vla_v2_policy" not in modules
        assert "deploy.lingbot_vla_v2_policy" in payload["source_only_modules"]
        return {"checks": [deploy._check("mandatory_dependencies", True)]}

    report = deploy.doctor(profile, probe_runner=fake_probe)
    assert report["ok"] and report["status"] == "cpu_preflight_passed_with_deferred_gpu_import"
    assert report["deferred_checks"][0]["status"] == "deferred"
    assert report["deferred_checks"][0]["full_import_passed"] is False
    assert report["deferred_checks"][0]["source_binding_verified"] is True
    marker.write_text("changed source\n")
    rejected = deploy.doctor(profile, probe_runner=fake_probe)
    assert not rejected["ok"]
    assert not rejected["deferred_checks"][0]["source_binding_verified"]


def payload(tmp_path, dependencies):
    return {"source_roots": [str(tmp_path)], "dependencies": dependencies,
            "source_only_modules": [], "backbone": "unknown_test_adapter",
            "adapter_entry_point": "missing:Adapter", "native_requirements": [], "distributions": []}


def dependency(module, *attributes):
    return {"module": module, "attributes": list(attributes)}


def test_actual_import_probe_checks_apis_discards_vendor_output(tmp_path):
    (tmp_path / "test_vendor.py").write_text("print('DO_NOT_PRINT_VENDOR_TOKEN')\nanswer = 42\n")
    result = deploy.run_probe(payload(tmp_path, [dependency("test_vendor", "answer"),
                                               dependency("test_vendor", "missing")]), 10)
    found = [check for check in result["checks"] if check["name"] == "import:test_vendor"]
    assert [c["status"] for c in found] == ["passed", "failed"]
    assert found[1]["error_type"] == "AttributeError"
    assert "DO_NOT_PRINT_VENDOR_TOKEN" not in json.dumps(result)
    assert not (tmp_path / "__pycache__").exists()


def test_missing_transitive_deployment_dependency_cannot_pass_import_probe(tmp_path):
    (tmp_path / "complete_deployment.py").write_text("import missing_native_dataset_dependency\n")
    result = deploy.run_probe(payload(tmp_path, [dependency("complete_deployment")]), 10)
    check = next(c for c in result["checks"] if c["name"] == "import:complete_deployment")
    assert check["status"] == "failed" and check["error_type"] == "ModuleNotFoundError"


@pytest.mark.parametrize("body,event", [
    ("import socket\nsocket.create_connection(('127.0.0.1', 9))", "socket.getaddrinfo"),
    ("import subprocess\nsubprocess.run(['python', '-c', 'raise RuntimeError()'])", "subprocess.Popen"),
    ("open(__file__ + '.safetensors', 'rb')", "open"),
    ("import subprocess\nsubprocess.run(['uname', '-p'])", "subprocess.Popen"),
    ("import socket\ns=socket.socket(socket.AF_INET6);s.bind(('::1',0))", "socket.bind"),
])
def test_actual_worker_blocks_network_process_and_weight_access(tmp_path, body, event):
    (tmp_path / "blocked_vendor.py").write_text(body)
    result = deploy.run_probe(payload(tmp_path, [dependency("blocked_vendor")]), 10)
    assert any(c["name"] == "import:blocked_vendor" and c["status"] == "failed" for c in result["checks"])
    guard = next(c for c in result["checks"] if c["name"] == "offline_cpu_operation_guard")
    assert guard["status"] == "failed" and event in guard["blocked_operations"]


def test_actual_stdlib_processor_probe_is_allowed_but_remains_cpu_only(tmp_path):
    (tmp_path / "cpu_platform.py").write_text(
        "import platform\nplatform._uname_cache=None\nvalue=platform.processor()\n"
    )
    result = deploy.run_probe(payload(tmp_path, [dependency("cpu_platform", "value")]), 10)
    check = next(c for c in result["checks"] if c["name"] == "import:cpu_platform")
    assert check["status"] == "passed"
    guard = next(c for c in result["checks"] if c["name"] == "offline_cpu_operation_guard")
    assert guard["status"] == "passed"
    assert "stdlib_platform_uname_processor_probe" in guard["allowed_cpu_import_plumbing"]


def test_urllib3_exact_loopback_capability_probe_is_allowed(tmp_path):
    package = tmp_path / "urllib3/util"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "connection.py").write_text(
        "import socket\ndef _has_ipv6():\n"
        " with socket.socket(socket.AF_INET6,socket.SOCK_STREAM) as s:\n"
        "  s.bind(('::1',0))\n return True\nvalue=_has_ipv6()\n")
    result = deploy.run_probe(payload(tmp_path, [dependency("urllib3.util.connection", "value")]), 10)
    guard = next(c for c in result["checks"] if c["name"] == "offline_cpu_operation_guard")
    assert guard["status"] == "passed"
    assert guard["allowed_cpu_import_plumbing"] == ["urllib3_ipv6_loopback_capability_probe"]


def test_vendor_cannot_hide_caught_network_attempt(tmp_path):
    (tmp_path / "catching_vendor.py").write_text(
        "import socket\ntry:\n socket.getaddrinfo('example.com',443)\nexcept RuntimeError:\n pass\n"
    )
    result = deploy.run_probe(payload(tmp_path, [dependency("catching_vendor")]), 10)
    assert result["checks"][0]["status"] == "passed"
    guard = next(c for c in result["checks"] if c["name"] == "offline_cpu_operation_guard")
    assert guard["status"] == "failed"


def test_actual_worker_masks_cuda_and_blocks_initialization(tmp_path):
    (tmp_path / "torch.py").write_text("from types import SimpleNamespace\ncuda=SimpleNamespace()\n")
    (tmp_path / "gpu_vendor.py").write_text(
        "import os, torch\nassert os.environ['CUDA_VISIBLE_DEVICES']==''\n"
        "assert torch.cuda.is_available() is False\ntorch.cuda.init()\n"
    )
    result = deploy.run_probe(payload(tmp_path, [dependency("torch"), dependency("gpu_vendor")]), 10)
    guard = next(c for c in result["checks"] if c["name"] == "offline_cpu_operation_guard")
    assert "torch.cuda.initialization" in guard["blocked_operations"]
    assert guard["status"] == "failed"


def test_source_location_does_not_execute_initializer(tmp_path, monkeypatch):
    package = tmp_path / "source_only_package"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('must not execute')\n")
    (package / "policy.py").write_text("raise RuntimeError('must not execute')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = deploy._find_without_import("source_only_package.policy")
    assert Path(spec.origin) == package / "policy.py"
    assert "source_only_package" not in sys.modules


def test_actual_probe_timeout_is_a_failure(tmp_path):
    (tmp_path / "slow_vendor.py").write_text("import time\ntime.sleep(10)\n")
    result = deploy.run_probe(payload(tmp_path, [dependency("slow_vendor")]), 0.2)
    assert result["checks"] == [{"name": "probe_process", "status": "failed", "reason": "timeout",
                                "timeout_seconds": 0.2}]


def test_probe_environment_isolated_without_mutating_parent(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    def fake(argv, **kwargs):
        assert argv[1:3] == ["-I", "-B"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["NO_ALBUMENTATIONS_UPDATE"] == "1"
        assert kwargs["timeout"] == 3
        return SimpleNamespace(returncode=0, stdout='{"checks":[{"name":"sample","status":"passed"}]}')
    monkeypatch.setattr(deploy.subprocess, "run", fake)
    assert deploy.run_probe({}, 3)["checks"][0]["status"] == "passed"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5"


def test_cuda_guard_preserves_distinct_callable_identity_for_dynamo_import_maps(tmp_path):
    (tmp_path / "torch.py").write_text(
        "class Cuda:\n"
        "    def _lazy_init(self): pass\n"
        "    def init(self): pass\n"
        "    def set_device(self): pass\n"
        "    def current_device(self): pass\n"
        "    def get_device_capability(self): pass\n"
        "    def get_device_properties(self): pass\n"
        "cuda = Cuda()\n")
    (tmp_path / "dispatch_map.py").write_text(
        "import torch, os\n"
        "names = ('_lazy_init','init','set_device','current_device','get_device_capability','get_device_properties')\n"
        "assert len({getattr(torch.cuda, name) for name in names}) == 6\n"
        "assert all(getattr(torch.cuda, name).__name__ == name for name in names)\n"
        "assert os.environ['NO_ALBUMENTATIONS_UPDATE'] == '1'\n")
    result = deploy.run_probe(payload(tmp_path, [dependency("torch"), dependency("dispatch_map")]), 10)
    assert next(c for c in result["checks"] if c["name"] == "import:dispatch_map")["status"] == "passed"
    assert next(c for c in result["checks"] if c["name"] == "offline_cpu_operation_guard")["status"] == "passed"


def test_relative_native_and_upstream_paths_keep_callers_meaning(catalog, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vendor = tmp_path / "vendor"
    (vendor / "deploy").mkdir(parents=True)
    (vendor / "deploy/lingbot_vla_policy.py").write_text("# native source\n")
    monkeypatch.setenv("LINGBOT_VLA_ROOT", "vendor")
    captured = {}
    def fake(payload, timeout):
        captured.update(payload)
        return {"checks": [deploy._check("CPU", True)]}
    assert deploy.doctor(model(catalog, "vla4"), probe_runner=fake)["ok"]
    assert captured["root_environment"] == {"LINGBOT_VLA_ROOT": str(vendor)}
    monkeypatch.setenv("IFL_TEST_LIBRARY", "build/libnative.so")
    captured["native_requirements"] = [{"kind": "file_env", "name": "IFL_TEST_LIBRARY"}]
    def process(argv, **kwargs):
        assert kwargs["env"]["LINGBOT_VLA_ROOT"] == str(vendor)
        assert kwargs["env"]["IFL_TEST_LIBRARY"] == str(tmp_path / "build/libnative.so")
        return SimpleNamespace(returncode=0, stdout='{"checks":[{"name":"CPU","status":"passed"}]}')
    monkeypatch.setattr(deploy.subprocess, "run", process)
    deploy.run_probe(captured, 3)


def test_native_check_hashes_without_loading_library(tmp_path, monkeypatch):
    path = tmp_path / "libnative.so"
    requirement = {"kind": "file_env", "name": "IFL_TEST_NATIVE_LIBRARY", "remedy": "build it"}
    monkeypatch.setenv("IFL_TEST_NATIVE_LIBRARY", str(path))
    assert deploy._native_file_check(requirement)["status"] == "failed"
    machine = {"aarch64": 183, "arm64": 183, "x86_64": 62, "AMD64": 62}[platform.machine()]
    header = b"\x7fELF\x02\x01" + b"\x00" * 12 + struct.pack("<H", machine)
    path.write_bytes(header + b"not a loadable library")
    result = deploy._native_file_check(requirement)
    assert result["status"] == "passed"
    assert "untested" in result["scope"]
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(header[:18] + struct.pack("<H", 62 if machine != 62 else 183))
    assert deploy._native_file_check(requirement)["status"] == "failed"


def test_wrong_python_extension_abi_rejected_without_import(tmp_path, monkeypatch):
    package = tmp_path / "test_native_package"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('do not import')")
    (package / "kernels.cpython-999-unknown-linux-gnu.so").write_bytes(b"not loadable")
    monkeypatch.syspath_prepend(str(tmp_path))
    result = deploy._native_file_check({"kind": "extension", "name": "test_native_package.kernels", "remedy": "rebuild"})
    assert result["status"] == "failed"
    assert "test_native_package" not in sys.modules


def test_dreamzero_fp8_conflicting_tensorrt_is_rejected(catalog, monkeypatch):
    monkeypatch.setenv("LOAD_TRT_ENGINE", "enabled")
    report = deploy.doctor(model(catalog, "dreamzero"), "fp8",
                           probe_runner=lambda *a: {"checks": [deploy._check("CPU", True)]})
    assert any(c["name"] == "DreamZero_FP8_TensorRT_disabled" and c["status"] == "failed" for c in report["checks"])


def test_doctor_failure_exit_code_and_no_sensitive_exception_text(monkeypatch, capsys):
    monkeypatch.setattr(deploy, "doctor", lambda *a, **kw: {"ok": False, "status": "blocked"})
    assert deploy.main(["doctor", "pi05"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=7, stdout="SECRET_STDOUT", stderr="SECRET_STDERR"))
    assert "SECRET" not in json.dumps(deploy.run_probe({}, 1))


def test_all_rtx4090_models_expose_fp8_without_changing_the_native_schedule():
    from benchmarks.regression.reproduce import make_plan

    catalog = deploy.load_profiles(target="rtx4090")
    for profile in catalog["models"]:
        native = profile["execution_modes"]["native"]
        fp8 = profile["execution_modes"]["fp8"]
        assert fp8["runtime_kwargs"]["precision"] == "fp8"
        assert fp8["runtime_kwargs"]["tier_ceiling"] == "numeric"
        assert fp8["effective_schedule"] == native["effective_schedule"]
        assert fp8["schedule_changed"] is False
        assert fp8["gpu_qualified_by_plan"] is False
        # Exercise the installed runner's selection validation too, rather than
        # allowing a catalog-only setting that the user cannot actually prepare.
        assert make_plan(profile["id"], "fp8", target="rtx4090")
        if profile["id"] in ("edge", "nano"):
            environment = profile["execution_modes"]["numeric"]["environment"]
            assert not {"IFL_COSMOS3_GEN_REGIONS", "IFL_COSMOS3_SPLIT_PREFILL",
                        "IFL_BF16_LINEAR_RELU2"}.intersection(environment)

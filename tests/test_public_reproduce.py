"""Portable reproduction planning and orchestration without model execution."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.regression import reproduce


CATALOG = json.loads(reproduce.profiles_path().read_text())
MODES = [(profile["id"], mode) for profile in CATALOG["models"] for mode in profile["execution_modes"]]


def test_packaged_profile_mirror_matches_canonical_source():
    root = Path(__file__).resolve().parents[1]
    canonical = root / "release/deployment_profiles.json"
    packaged = root / "benchmarks/regression/fixtures/deployment_profiles.json"
    assert packaged.read_bytes() == canonical.read_bytes(), "refresh the checked packaged profile mirror"
    assert reproduce.profiles_path().resolve() == packaged.resolve()


@pytest.mark.parametrize("model,mode", MODES)
def test_explicit_profiles_preserve_policy_and_separate_operating_points(model, mode):
    plan = reproduce.make_plan(model, mode)
    profile = next(p for p in CATALOG["models"] if p["id"] == model)
    selected = profile["execution_modes"][mode]
    cells = plan["matrix"]["cells"]
    assert [cell["arm"] for cell in cells[:3]] == list(reproduce.MAIN_ARMS)
    assert cells[0]["expected_runtime_kwargs"] == cells[0]["expected_optimizer_environment"] == {}
    assert cells[1]["expected_runtime_kwargs"] == {"device": "cuda:0"}
    assert cells[1]["expected_optimizer_environment"] == {}
    assert all(cell["action_shape"] == reproduce.ACTION_SHAPES[model] for cell in cells)
    assert all(cell["revision"] == profile["checkpoint"]["revision"] for cell in cells)
    if selected["schedule_changed"]:
        assert len(cells) == 4
        assert cells[2]["expected_runtime_kwargs"] == profile["execution_modes"]["native"]["runtime_kwargs"]
        assert cells[3]["arm"] == "operating_point" and cells[3]["experimental"] is True
        assert cells[3]["cross_policy_baseline"] == cells[0]["id"]
        assert cells[3]["comparison_group"] != cells[0]["comparison_group"]
        candidate = cells[3]
    else:
        assert len(cells) == 3
        candidate = cells[2]
    assert candidate["expected_runtime_kwargs"] == selected["runtime_kwargs"]
    assert candidate["expected_optimizer_environment"] == selected["environment"]
    assert candidate["effective_schedule"] == selected["effective_schedule"]
    assert plan["publication_ready"] is plan["task_quality_validated"] is False
    assert "/home/" not in json.dumps(plan) and "/workspace/" not in json.dumps(plan)


def test_plan_defaults_native_and_hub_alias_is_identical():
    plan = reproduce.make_plan("nano")
    assert plan == reproduce.make_plan(plan["checkpoint"]["model_id"], "native")
    assert plan["execution_mode"] == "native"
    assert plan["matrix"]["cells"][2]["expected_runtime_kwargs"]["precision"] == "native"


def test_va_extra_mode_shares_three_main_arms_and_preserves_both_recipes():
    standard = reproduce.make_plan("va", "fp8")
    changed = reproduce.make_plan("va", "2v4a-fp8")
    combined = reproduce.make_plan("va", "fp8", extra_modes=["2v4a-fp8"])
    assert combined["matrix"]["cells"][:3] == standard["matrix"]["cells"]
    assert combined["matrix"]["cells"][3] == changed["matrix"]["cells"][-1]
    assert combined["matrix"]["expected_main_cells"] == 3
    assert combined["matrix"]["expected_operating_point_cells"] == 1
    assert combined["extra_execution_modes"] == ["2v4a-fp8"]
    assert combined["matrix"]["cells"][3]["cross_policy_baseline"] == "va-eager_native"
    assert combined["matrix"]["cells"][3]["experimental"] is True


@pytest.mark.parametrize("mode,extras", [("fp8", ["fp8"]), ("native", ["fp8"]),
                                        ("fp8", ["2v4a-fp8", "2v4a-fp8"]),
                                        ("2v4a-native", ["2v4a-fp8"]), ("fp8", ["unknown"])])
def test_invalid_duplicate_or_same_policy_extra_modes(mode, extras):
    with pytest.raises(ValueError):
        reproduce.make_plan("va", mode, extra_modes=extras)


def test_extra_mode_prepare_regenerates_exact_bound_plan(tmp_path, monkeypatch):
    def download(**kwargs):
        result = tmp_path / "snapshots" / kwargs["revision"]
        result.mkdir(parents=True)
        return result
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    root = tmp_path / "prepared"
    reproduce.prepare("va", "fp8", root, extra_modes=["2v4a-fp8"])
    plan, _ = reproduce.validate_bundle(root)
    assert len(plan["matrix"]["cells"]) == 4
    assert plan["extra_execution_modes"] == ["2v4a-fp8"]


def test_extra_mode_cli_plan(capsys):
    assert reproduce.main(["plan", "--model", "va", "--mode", "fp8", "--extra-mode", "2v4a-fp8"]) == 0
    assert len(json.loads(capsys.readouterr().out)["matrix"]["cells"]) == 4


def test_plan_is_read_only_and_has_no_model_imports(tmp_path):
    source = Path(reproduce.__file__).resolve()
    script = '''
import importlib.abc, pathlib, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'PIL', 'huggingface_hub', 'transformers'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
def audit(event, args):
    if event.startswith('socket.') or event in {'subprocess.Popen','os.mkdir','os.rename','os.remove'}:
        raise AssertionError(event)
    if event == 'open' and isinstance(args[1], str) and any(c in args[1] for c in 'wax+'):
        raise AssertionError(event)
sys.addaudithook(audit)
namespace = runpy.run_path(sys.argv[1])
assert namespace['make_plan']('edge')['execution_mode'] == 'native'
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(source)], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("model,mode", [("unknown", "native"), ("edge", "fp8"), ("nano", "numeric-auto")])
def test_unknown_model_or_implicit_mode_is_rejected(model, mode):
    with pytest.raises(ValueError, match="unknown"):
        reproduce.make_plan(model, mode)


def test_only_explicit_required_library_option_is_bound(tmp_path):
    library = tmp_path / "kernels.so"
    library.write_bytes(b"metadata-only synthetic library")
    plan = reproduce.make_plan("edge", "numeric", libraries={"IFL_BF16_KERNEL_LIBRARY": library})
    assert plan["unresolved_library_options"] == []
    assert plan["library_bindings"]["IFL_BF16_KERNEL_LIBRARY"]["sha256"] == reproduce.sha(library)
    assert plan["matrix"]["cells"][2]["expected_optimizer_environment"]["IFL_BF16_KERNEL_LIBRARY"] == str(library)
    with pytest.raises(ValueError, match="not required"):
        reproduce.make_plan("nano", "native", libraries={"IFL_BF16_KERNEL_LIBRARY": library})
    with pytest.raises(ValueError, match="unique"):
        reproduce.libraries_from_args(["X=a", "X=b"])


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        snapshot = tmp_path / "cache/snapshots" / kwargs["revision"]
        snapshot.mkdir(parents=True, exist_ok=True)
        return str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    root = tmp_path / "prepared"
    reproduce.prepare("nano", "numeric", root, cache_dir=tmp_path / "cache", local_files_only=True)
    return root, calls


def test_prepare_pins_checkpoint_and_copies_exact_safe_input(prepared):
    root, calls = prepared
    plan, receipt = reproduce.validate_bundle(root)
    assert len(calls) == 1
    assert calls[0]["revision"] == plan["checkpoint"]["revision"] and calls[0]["local_files_only"] is True
    assert calls[0]["repo_id"] == plan["checkpoint"]["model_id"]
    assert receipt["status"] == "prepared_primary_checkpoint"
    assert reproduce.sha(root / "inputs/recorded_inputs_v1.npz") == reproduce.FIXTURE_SHA
    with pytest.raises(ValueError, match="already exists"):
        reproduce.prepare("nano", "numeric", root)
    assert len(calls) == 1


def test_failed_prepare_keeps_error_and_predeclared_matrix(tmp_path, monkeypatch):
    def failed(**kwargs):
        raise RuntimeError("synthetic unavailable checkpoint")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=failed))
    root = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="unavailable"):
        reproduce.prepare("edge", "native", root)
    receipt = json.loads((root / "preparation.json").read_text())
    assert receipt["status"] == "failed" and "unavailable" in receipt["error"]
    assert (root / "matrix.json").is_file()


def test_prepare_accepts_pinned_snapshot_reference_symlink_to_model_store(tmp_path, monkeypatch):
    physical = tmp_path / "original-model-store"
    physical.mkdir()
    def download(**kwargs):
        reference = tmp_path / "snapshots" / kwargs["revision"]
        reference.parent.mkdir()
        reference.symlink_to(physical, target_is_directory=True)
        return reference
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    root = tmp_path / "prepared"
    receipt = reproduce.prepare("va", "fp8", root, extra_modes=["2v4a-fp8"])
    assert receipt["checkpoint_snapshot"] == str(physical)
    assert Path(receipt["checkpoint_snapshot_reference"]).name == receipt["checkpoint"]["revision"]
    assert receipt["checkpoint_snapshot_is_linked"] is True
    assert "file bytes" in receipt["checkpoint_identity_scope"]
    reproduce.validate_bundle(root)


def test_prepare_rejects_wrong_returned_reference_even_when_target_name_matches_revision(tmp_path, monkeypatch):
    def download(**kwargs):
        physical = tmp_path / "store" / kwargs["revision"]
        physical.mkdir(parents=True)
        wrong_reference = tmp_path / "snapshots" / ("0" * 40)
        wrong_reference.parent.mkdir()
        wrong_reference.symlink_to(physical, target_is_directory=True)
        return wrong_reference
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    with pytest.raises(ValueError, match="pinned snapshot reference"):
        reproduce.prepare("va", "fp8", tmp_path / "failed")
    receipt = json.loads((tmp_path / "failed/preparation.json").read_text())
    assert receipt["status"] == "failed"


def test_changed_prepared_matrix_or_fixture_is_rejected(prepared):
    root, _ = prepared
    (root / "matrix.json").write_text("{}")
    with pytest.raises(ValueError, match="prepared artifact changed"):
        reproduce.validate_bundle(root)


def test_all_inherited_optimizer_options_are_cleared(prepared):
    root, _ = prepared
    plan, receipt = reproduce.validate_bundle(root)
    inherited = {"PATH": "/bin", "NANO_ROOT": "/vendor", "IFL_SECRET": "1", "IFL_BF16_KERNEL_LIBRARY": "/stale",
                 "PYTHONPATH": "/checkout", "PYTHONHOME": "/wrong", "HF_HUB_OFFLINE": "0",
                 "DYNAMIC_CACHE_SCHEDULE": "1", "NUM_DIT_STEPS": "2", "LOAD_TRT_ENGINE": "1", "ENABLE_TENSORRT": "1"}
    for cell in plan["matrix"]["cells"]:
        env = reproduce.child_environment(plan, cell, receipt, inherited)
        assert {k: v for k, v in env.items() if k.startswith("IFL_")} == cell["expected_optimizer_environment"]
        assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
        assert "DYNAMIC_CACHE_SCHEDULE" not in env and "NUM_DIT_STEPS" not in env
        assert "LOAD_TRT_ENGINE" not in env and env["ENABLE_TENSORRT"] == "false"
        assert env["NANO_ROOT"] == "/vendor" and env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        assert env["HF_HUB_CACHE"] == receipt["cache_dir"]
    assert inherited["IFL_SECRET"] == "1"


@pytest.mark.parametrize("failure_index", [None, 0, 2])
def test_run_uses_fresh_children_retains_failure_without_retry(prepared, tmp_path, monkeypatch, failure_index):
    root, _ = prepared
    calls = []
    def capture(command, **kwargs):
        index = len(calls)
        calls.append((command, kwargs))
        kwargs["stdout"].write("synthetic capture output\n")
        return SimpleNamespace(returncode=7 if index == failure_index else 0)
    monkeypatch.setattr(reproduce, "require_installed", lambda: None)
    monkeypatch.setattr(reproduce.subprocess, "run", capture)
    # Existing scientific validator has independent tests; this isolates orchestration.
    monkeypatch.setattr(reproduce, "report", lambda root, output: {"status": "passed"})
    result = reproduce.run(root, tmp_path / "run")
    expected = 3 if failure_index is None else failure_index + 1
    assert len(calls) == expected and len(result["attempts"]) == expected
    assert result["status"] == ("passed" if failure_index is None else "failed_or_incomplete")
    for command, kwargs in calls:
        assert command[:5] == [sys.executable, "-I", "-B", "-m", "benchmarks.regression.user_e2e"]
        assert kwargs["check"] is False and kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert (tmp_path / "run/run.json").is_file()
    assert len(list((tmp_path / "run/logs").glob("*.log"))) == expected
    with pytest.raises(ValueError, match="already exists"):
        reproduce.run(root, tmp_path / "run")
    assert len(calls) == expected


def test_missing_receipts_are_incomplete_in_strict_existing_report(prepared, tmp_path):
    root, _ = prepared
    result = reproduce.report(root, tmp_path / "report")
    assert result["status"] == "incomplete" and result["counts"]["incomplete_cells"] == 3
    assert result["task_quality_validated"] is False
    assert (tmp_path / "report/report.csv").is_file()
    with pytest.raises(ValueError, match="already exists"):
        reproduce.report(root, tmp_path / "report")


def test_cpu_report_does_not_require_original_accelerator_file(tmp_path, monkeypatch):
    library = tmp_path / "kernels.so"
    library.write_bytes(b"synthetic file to bind only")
    def download(**kwargs):
        path = tmp_path / "snapshots" / kwargs["revision"]
        path.mkdir(parents=True)
        return str(path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    root = tmp_path / "prepared"
    reproduce.prepare("edge", "numeric", root, libraries={"IFL_BF16_KERNEL_LIBRARY": library})
    library.unlink()
    result = reproduce.report(root, tmp_path / "report")
    assert result["status"] == "incomplete"
    with pytest.raises(FileNotFoundError):
        reproduce.validate_bundle(root)

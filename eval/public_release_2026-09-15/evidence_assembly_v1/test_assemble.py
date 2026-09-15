"""CPU adversarial checks against preserved local evidence; no GPU imports."""
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).with_name("assemble.py")
spec = importlib.util.spec_from_file_location("evidence_assembly", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
QUALIFICATION = m.STUDY / "qualification"
STAGE = Path("/home/ubuntu/ifl-public-full-stage-20260915-v5")


def stage_module(name):
    relative = f"benchmarks/regression/{name}.py"
    manifest = m.read(STAGE / "manifest.json")
    return m.load_module(STAGE / "source" / relative, manifest["files"][relative]["sha256"], f"test_public_{name}")


def mutate_read(monkeypatch, target, change):
    original = m.read

    def altered(path):
        result = original(path)
        if Path(path) == target:
            change(result)
        return result
    monkeypatch.setattr(m, "read", altered)


def test_missing_family_stays_pending(tmp_path):
    result = m.guarded(m.validate_main, tmp_path / "missing", "nano", None, STAGE / "source", tmp_path)
    assert result["status"] == "pending"


def test_changed_source_rejected_before_execution(tmp_path):
    source, sentinel = tmp_path / "bad.py", tmp_path / "executed"
    source.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n")
    with pytest.raises(ValueError, match="before import"):
        m.load_module(source, "0" * 64, "changed_source")
    assert not sentinel.exists()


def test_actual_historical_same_arm_parity_is_not_native_fp8_equivalence():
    result = m.validate_historical(QUALIFICATION / "pi05", m.REPO / "eval/user_e2e_2026-09-14")
    assert result["status"] == "passed" and len(result["arrays"]) == 6
    report = m.read(QUALIFICATION / "pi05/run/validated_report/report.json")
    selected = next(row for row in report["comparisons"] if row["candidate"] == "pi05-runtime_selected")
    assert selected["actions"]["exact_bytes"] is False
    assert selected["actions"]["max_abs"] > 0


def test_duplicate_historical_claim_is_rejected(monkeypatch):
    target = QUALIFICATION / "pi05/historical_action_comparison_v1.json"
    mutate_read(monkeypatch, target, lambda rows: rows.append(rows[0]))
    with pytest.raises(ValueError, match="duplicate historical"):
        m.validate_historical(target.parent, m.REPO / "eval/user_e2e_2026-09-14")


def test_claimed_historical_hash_tampering_is_rejected(monkeypatch):
    target = QUALIFICATION / "va/historical_action_comparison_v1.json"
    mutate_read(monkeypatch, target, lambda rows: rows[0].update(new_file_sha256="0" * 64))
    with pytest.raises(ValueError, match="historical archive hash"):
        m.validate_historical(target.parent, m.REPO / "eval/user_e2e_2026-09-14")


def test_historical_redundant_speed_field_tampering_is_rejected(monkeypatch):
    target = QUALIFICATION / "groot/historical_action_comparison_v1.json"
    mutate_read(monkeypatch, target, lambda value: value["comparisons"][0].update(ratio=100))
    with pytest.raises(ValueError, match="timing/native-equivalence"):
        m.validate_historical(target.parent, m.REPO / "eval/user_e2e_2026-09-14")


def test_completed_ws_receipt_with_native_signal_exit_is_rejected(monkeypatch):
    directory = QUALIFICATION / "pi05/serving_fp8_v2"
    mutate_read(monkeypatch, directory / "receipt.json", lambda value: value.update(server_exit_code=-15))
    with pytest.raises(ValueError, match="close successfully"):
        m.validate_serving(directory, QUALIFICATION / "pi05/run", stage_module("user_e2e"))


def test_ws_wrong_request_is_rejected(monkeypatch):
    directory = QUALIFICATION / "pi05/serving_fp8_v2"
    mutate_read(monkeypatch, directory / "receipt.json", lambda value: value["calls"][2].update(request_sha256="0" * 64))
    with pytest.raises(ValueError, match="request bytes"):
        m.validate_serving(directory, QUALIFICATION / "pi05/run", stage_module("user_e2e"))


def test_native_completed_report_with_nonzero_process_is_rejected(monkeypatch, tmp_path):
    run = QUALIFICATION / "pi05/run"
    mutate_read(monkeypatch, run / "run.json", lambda value: value["attempts"][0].update(exit_code=1))
    with pytest.raises(ValueError, match="native process failed"):
        m.validate_main(run, "pi05", stage_module("user_report"), STAGE / "source", tmp_path)


def test_published_latency_tampering_is_rejected(monkeypatch, tmp_path):
    run = QUALIFICATION / "pi05/run"
    mutate_read(monkeypatch, run / "validated_report/report.json", lambda value: value["cells"][0]["primary"].update(p50_ms=1))
    with pytest.raises(ValueError, match="metric/contract"):
        m.validate_main(run, "pi05", stage_module("user_report"), STAGE / "source", tmp_path)


def test_foreign_capture_with_timeout_is_rejected(monkeypatch):
    directory = QUALIFICATION / "foreign/lerobot-pi05"
    mutate_read(monkeypatch, directory / "completion.json", lambda value: value.update(timed_out=True))
    with pytest.raises(ValueError, match="failed or timed out"):
        m.validate_foreign("lerobot-pi05", directory, m.STUDY / "framework_compare_v1/prepared_assets_v2/lerobot-pi05",
                           stage_module("framework_compare"), STAGE / "source/benchmarks/regression/fixtures/recorded_inputs_v1.npz")


def test_not_qualified_is_distinct_from_unsupported_or_pending():
    helper = stage_module("framework_compare")
    catalog = helper.catalog()
    results = {cell["id"]: {"status": "pending"} for cell in catalog["cells"]}
    assert m.foreign_status(catalog, results, "vllm-omni", "groot") == "not_qualified"
    assert m.foreign_status(catalog, results, "vllm-omni", "vla4") == "unsupported"
    assert m.foreign_status(catalog, results, "vllm-omni", "edge") == "pending"

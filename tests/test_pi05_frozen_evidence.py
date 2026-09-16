"""CPU-only audit of the real-GPU frozen execution evidence."""
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.vla.pi05_sm120_libero import paired_statistics

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "examples/pi05_vla/sm120_frozen_results.json"


def test_frozen_evidence_binds_sources_binary_and_three_real_process_restarts():
    result = json.loads(PATH.read_text())
    assert result["status"] == "PASS" and result["schema_version"] == 1
    for name, expected in result["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((ROOT / "scripts/summarize_pi05_frozen.py").read_bytes()).hexdigest() == result["exporter_source_sha256"]
    identity = result["execution_identity"]
    assert identity["contract"]["computed"] == 50
    assert identity["contract"]["returned"] == identity["contract"]["nfe"] == 10
    assert identity["binary"]["kernels"] == result["environment"]["binary_sha256"]["flash_rt_kernels"]
    assert identity["binary"]["fa2"] == result["environment"]["binary_sha256"]["flash_rt_fa2"]
    assert result["state_format"]["scale_count"] == 250
    assert result["state_format"]["token_lengths"] == list(range(128, 201))
    replay = result["numeric_replay"]
    assert replay["status"] == "PASS"
    assert replay["state_sha256"] == result["state_format"]["catalog_sha256"]
    for mode, receipts in replay["receipts"].items():
        assert len(receipts) == 3
        assert [r["repetition"] for r in receipts] == [0, 1, 2]
        assert all(r["mode"] == mode and r["cases"] == 8 for r in receipts)
        summary = replay["summary"][mode]
        assert summary["bit_exact_actions"] == (len({r["actions_sha256"] for r in receipts}) == 1)
        assert summary["identical_scales"] == (len({r["scale_sha256"] for r in receipts}) == 1)
        assert summary["speedup"] == replay["native"]["infer_p50_ms"] / statistics.median(r["infer_p50_ms"] for r in receipts)
        assert all(r["noise_sha256"] == replay["native"]["noise_sha256"] for r in receipts)
    both = replay["summary"]["both"]
    assert both["bit_exact_actions"] and both["scales_match_recording"] and both["algorithms_match_recording"]
    assert both["minimum_native_cosine"] >= .98 and both["speedup"] >= 1.05
    assert both["peak_allocated_ratio"] <= .75
    assert "/workspace/" not in PATH.read_text()


def test_two_complete_task5_runs_have_identical_actions_noise_and_outcomes():
    result = json.loads(PATH.read_text())["repeated_task5"]
    assert result["status"] == "PASS" and result["bit_exact_trajectories"]
    assert result["episodes_per_repeat"] == 50 and not result["mismatch_seeds"]
    pairs = result["pairs"]
    assert [r["seed"] for r in pairs] == list(range(40100, 40150))
    for row in pairs:
        assert type(row["success"]) is bool and type(row["repeat_success"]) is bool
        assert row["success"] == row["repeat_success"]
        assert row["steps"] == row["repeat_steps"]
        assert row["action_digest"] == row["repeat_action_digest"]
        assert row["noise_sha256"] == row["repeat_noise_sha256"]
        assert len(row["noise_sha256"]) == (row["steps"] + 9) // 10
    assert result["successes"] == [sum(r[k] for r in pairs) for k in ("success", "repeat_success")]
    assert result["full_campaign_repeat"]["status"] == "PASS"
    assert result["full_campaign_repeat"]["bit_exact_trajectories"]


def test_frozen_500_pair_certificate_is_fresh_complete_and_recomputable():
    result = json.loads(PATH.read_text())
    qualification = result["qualification"]
    assert qualification["status"] == "PASS"
    assert qualification["protocol"]["protocol"] == "pi05-sm120-frozen-matched-v1"
    assert qualification["protocol"]["frozen_catalog_sha256"] == result["state_format"]["catalog_sha256"]
    assert len(qualification["task_states"]) == 10
    assert qualification["task_states"]["5"]["state_sha256"] == result["state_format"]["catalog_sha256"]
    pairs = qualification["pairs"]
    assert len(pairs) == 500
    assert {(p["task_id"], p["seed"]) for p in pairs} == {(t, s) for t in range(10) for s in range(40100, 40150)}
    assert all(type(p["native"]) is bool and type(p["fp8"]) is bool for p in pairs)
    assert paired_statistics(pairs) == qualification["summary"]
    assert qualification["summary"]["noninferior"] and not qualification["summary"]["collapsed_tasks"]
    assert qualification["pairing"]["matched_initial_states_and_observations"] == 500
    assert qualification["pairing"]["matched_noise_calls"] >= 500
    repeated = {p["seed"]: p for p in result["repeated_task5"]["pairs"]}
    for pair in pairs:
        if pair["task_id"] == 5:
            assert pair["fp8_action_digest"] == repeated[pair["seed"]]["action_digest"]

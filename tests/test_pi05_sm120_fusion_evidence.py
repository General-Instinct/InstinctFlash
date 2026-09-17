"""CPU evidence arithmetic/provenance only; this never substitutes for GPU execution."""
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.vla.pi05_sm120_fusion import source_hashes

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "examples/pi05_vla/sm120_fusion_results.json"


def test_fusion_sources_and_prior_qualification_are_bound_without_overwriting():
    result = json.loads(PATH.read_text())
    assert result["schema_version"] == 1 and result["status"] == "PASS"
    assert result["source_sha256"] == source_hashes()
    assert result["exporter_source_sha256"] == hashlib.sha256((ROOT / "scripts/summarize_pi05_sm120_fusion.py").read_bytes()).hexdigest()
    assert result["base_evidence_sha256"] == hashlib.sha256((ROOT / "examples/pi05_vla/sm120_frozen_results.json").read_bytes()).hexdigest()
    assert "/workspace/" not in PATH.read_text()
    assert result["gpu_operator_tests"]["passed"] == 26
    assert result["gpu_operator_tests"]["source_sha256"] == hashlib.sha256((ROOT / "tests/test_pi05_sm120_fusion_cuda.py").read_bytes()).hexdigest()
    assert all(row["error_summary"] == "0 errors" for row in result["sanitizer"].values())


def test_abba_has_separate_fusion_and_upload_contributions_with_raw_timings():
    evidence = json.loads(PATH.read_text())
    for mode in ("all8", "hoist", "all8-hoist"):
        result = evidence["ablation"][mode]
        rows = result["rows"]
        assert [r["arm"] for r in rows] == ["baseline", mode, mode, "baseline"]
        assert all(r["source_sha256"] == source_hashes() and not r["profiled"] for r in rows)
        assert len({r["actions_sha256"] for r in rows}) == 1
        for row in rows:
            assert row["all_action_bytes_equal"] and row["all_noise_equal"] and row["cases"] == 8
            assert len(row["timings_ms"]) == row["iterations"] and row["iterations"] >= 64
            assert all(t > 0 for t in row["timings_ms"])
            assert row["p50_ms"] == statistics.median(row["timings_ms"])
            assert row["base_identity"]["contract"] == {"computed": 50, "returned": 10, "nfe": 10, "views": 2, "layout": "nk"}
            if row["fusion"]:
                assert row["fusion"]["extension_sha256"] == evidence["extension_sha256"]
                assert row["uninstall_exact"]
                assert row["fusion"]["hoist_scales"] == mode.endswith("hoist")
                assert (row["skipped_scale_installs"] > 0) == mode.endswith("hoist")
        base = statistics.mean(rows[i]["p50_ms"] for i in (0, 3))
        candidate = statistics.mean(rows[i]["p50_ms"] for i in (1, 2))
        assert result["baseline_ms"] == base and result["candidate_ms"] == candidate
        assert result["speedup"] == base / candidate and result["saved_ms"] == base - candidate
        assert result["performance_improved"] == (base > candidate * 1.005 and max(result["arm_spread_percent"].values()) <= 1.0)
        assert result["performance_improved"]
    for previous in evidence["additional_timing_runs"]:
        assert not previous["performance_improved"]
        assert all(row["source_sha256"] == source_hashes() for row in previous["rows"])
        assert max(previous["arm_spread_percent"].values()) > 1.0


def test_microbenchmarks_keep_resets_separate_from_end_to_end_claims():
    evidence = json.loads(PATH.read_text())
    result = evidence["microbenchmarks"]
    assert "identical reset" in result["protocol"]
    assert result["binary_sha256"]["fusion"] == evidence["extension_sha256"]
    assert {r["name"] for r in result["rows"]} == {"qkv", "ffn4", "ffn8"}
    for row in result["rows"]:
        assert row["bit_exact"]
        medians = {arm: statistics.median(values) for arm, values in row["timings_us"].items()}
        assert medians == row["median_us_including_reset"]
        assert row["speedup"] == medians["baseline"] / medians["fusion"]


def test_new_trajectory_screen_matches_historical_actions_but_is_not_new_500_pair_gate():
    result = json.loads(PATH.read_text())["trajectory_replay"]
    baseline = json.loads((ROOT / "examples/pi05_vla/sm120_frozen_results.json").read_text())
    published = {(p["task_id"], p["seed"]): p for p in baseline["qualification"]["pairs"]}
    assert result["status"] == "PASS" and result["kind"] == "SCREEN"
    assert not result["native_arm_rerun"] and result["episodes_per_task"] == 3
    pairs = result["pairs"]
    assert len(pairs) == 30
    assert {(p["task"], p["seed"]) for p in pairs} == {(t, s) for t in range(10) for s in range(40100, 40103)}
    for pair in pairs:
        prior = published[pair["task"], pair["seed"]]
        assert not pair["mismatch_fields"]
        assert pair["action_digest"] == pair["reference_action_digest"] == prior["fp8_action_digest"]
        assert pair["success"] == prior["fp8"] and pair["steps"] == prior["fp8_steps"]
        assert len(pair["noise_sha256"]) == (pair["steps"] + 9) // 10

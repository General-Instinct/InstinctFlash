"""Offline integrity gates for the real-model Pi0.5 SM120 FP8 qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "examples/pi05_vla/sm120_fp8_results.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sm120_fp8_result_is_pinned_pass_and_source_bound():
    result = json.loads(RESULT.read_text())
    assert result["status"] == "PASS"
    assert result["hardware"] == "RTX 5090 / SM120"
    assert result["model"]["repo"] == "lerobot/pi05_libero_finetuned_v044"
    assert result["model"]["revision"] == "8e174154ef5f6c60a8da12ae99c303d8963138c1"
    assert result["model"]["model_sha256"] == "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"
    assert result["dataset"]["repo"] == "lerobot/libero_spatial_image"
    assert result["dataset"]["revision"] == "d86c0b94922572b3b657e1d1a3d01f0952ddeb46"
    assert result["dataset"]["file_sha256"] == "cc4681188f4c5eeec4253d51ddbaa5ad90ad01bcd7ed4c23fc767a1e0c14686b"
    assert result["protocol"]["process_isolation"] is True
    assert result["protocol"]["warmup_replays"] == 2
    assert result["protocol"]["timing_replays"] == 9
    assert result["protocol"]["action_operating_point"] == (
        "checkpoint chunk 50, executed horizon 10, action dim 7"
    )

    for relative, expected in result["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected, f"qualified source changed: {relative}"


def test_sm120_fp8_numeric_and_performance_thresholds_hold():
    result = json.loads(RESULT.read_text())
    thresholds = result["protocol"]["thresholds"]
    assert result["arms"]["native"]["fp8_weight_count"] == 0
    assert result["arms"]["fp8"]["fp8_weight_count"] == 253
    assert result["arms"]["fp8"]["fp8_layout"] == "nk"
    assert result["arms"]["fp8"]["capability"] == [12, 0]
    assert result["arms"]["native"]["binary_sha256"] == result["arms"]["fp8"]["binary_sha256"]
    assert set(result["arms"]["fp8"]["binary_sha256"]) == {"flash_rt_kernels", "flash_rt_fa2"}
    assert result["arms"]["native"]["released_bf16_bytes"] == 0
    assert result["arms"]["fp8"]["released_bf16_key_count"] == thresholds["expected_released_bf16_keys"]
    assert result["arms"]["fp8"]["released_bf16_bytes"] >= thresholds["min_released_bf16_bytes"]
    assert result["summary"]["fp8_to_native_resident_ratio"] <= thresholds["max_fp8_resident_ratio"]
    assert result["summary"]["fp8_to_native_peak_ratio"] <= thresholds["max_fp8_peak_ratio"]
    assert result["protocol"]["calibration_seed"] == 5090120
    assert result["summary"]["resident_memory_reduction_bytes"] > 0
    assert result["arms"]["native"]["action_shape"] == [3, 10, 7]
    assert result["arms"]["fp8"]["action_shape"] == [3, 10, 7]
    for arm in result["arms"].values():
        assert arm["computed_action_chunk"] == 50
        assert arm["executed_action_horizon"] == 10
        assert arm["state_tokenized"] is True
        assert arm["state_changes_tokens"] is True
        assert arm["missing_state_refused"] is True
        assert arm["graph_lifecycle"]["profiles"] == 12
        assert arm["graph_lifecycle"]["evicted_and_destroyed"] >= 4
        assert arm["graph_lifecycle"]["all_closed"] is True
        assert arm["normalization"] == "checkpoint MEAN_STD"
    assert len(result["comparisons"]) == 3
    for comparison in result["comparisons"]:
        assert comparison["noise_bitwise_equal"] is True
        assert comparison["action_cosine"] >= thresholds["min_action_cosine"]
    assert result["summary"]["min_action_cosine"] >= thresholds["min_action_cosine"]
    assert result["summary"]["speedup"] >= thresholds["min_speedup"]
    assert result["failures"] == []


def test_base_runtime_reproduction_accepts_a_pinned_local_checkpoint():
    source = (ROOT / "examples/pi05_vla/run_pi05_end_to_end.py").read_text()
    assert 'os.environ.get("IFL_PI05_BASE", "lerobot/pi05_base")' in source
    for key in (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
        "observation.state",
    ):
        assert key in source


def test_sm120_actions_match_the_fully_loaded_official_lerobot_reference():
    result = json.loads((ROOT / "examples/pi05_vla/sm120_checkpoint_reference_results.json").read_text())
    assert result["status"] == "PASS"
    assert result["loaded_weights"]["checked_tensors"] == 812
    assert result["loaded_weights"]["checkpoint_sha256"] == json.loads(RESULT.read_text())["model"]["model_sha256"]
    assert result["packages"]["lerobot"] == "0.4.4"
    assert result["packages"]["transformers"] == "4.53.2"
    assert len(result["comparisons"]) == 3
    assert all(case["action_cosine"] >= .98 for case in result["comparisons"])
    assert all(case["token_count"] > 100 for case in result["comparisons"])
    for relative, expected in result["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected, f"reference source changed: {relative}"


def test_historical_sm120_libero_screen_keeps_its_original_source_identity():
    path = ROOT / "examples/pi05_vla/sm120_libero_screen_results.json"
    result = json.loads(path.read_text())
    assert result["status"] == "SCREEN"
    assert result["hardware"]["capability"] == [12, 0]
    assert result["model"]["revision"] == "8e174154ef5f6c60a8da12ae99c303d8963138c1"
    assert result["model"]["model_sha256"] == (
        "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"
    )
    assert result["simulator"]["assets_revision"] == (
        "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
    )
    assert result["protocol"]["task_ids"] == list(range(10))
    assert result["protocol"]["trials_per_task"] == 3
    assert result["protocol"]["action_horizon"] == 10
    assert len(result["tasks"]) == 10
    assert sum(task["successes"] for task in result["tasks"]) == 21
    assert sum(task["num_trials"] for task in result["tasks"]) == 30
    assert result["summary"] == {"successes": 21, "episodes": 30, "success_rate": 0.7}
    # This is the immutable 30-episode observation from 08771e1, not evidence for
    # the current implementation. Current-source integrity is checked above and
    # by the matched campaign; rewriting old screen hashes would invent a rerun.
    assert result["source_sha256"]["serving/flash_rt/frontends/torch/pi05_rtx.py"] == (
        "f259ecb3f27cce579093f8c033e7efa266a6b714d03b5e9772366fb1d5faf210")
    assert result["source_sha256"]["serving/flash_rt/models/pi05/pipeline_rtx.py"] == (
        "c8f6b0a8fadb188ab8878f0ba2b451ae97d2836935c69f5bd28a7d46d4014e0d")
    assert "/workspace/" not in path.read_text()


def test_full_matched_sm120_qualification_is_complete_and_recomputable():
    from benchmarks.vla.pi05_sm120_libero import paired_statistics
    path = ROOT / "examples/pi05_vla/sm120_libero_matched_results.json"
    result = json.loads(path.read_text())
    assert result["status"] == "PASS"
    protocol = result["protocol"]
    assert protocol["computed_action_chunk"] == 50
    assert protocol["action_horizon"] == protocol["replan_steps"] == protocol["nfe"] == 10
    assert protocol["max_steps"] == 280 and protocol["wait_steps"] == 10
    assert protocol["noninferiority_margin"] == .05
    assert protocol["selected"]["percentile"] in (99.0, 99.9, 100.0)
    pairs = result["pairs"]
    assert len(pairs) == 500
    assert {(p["task_id"], p["seed"]) for p in pairs} == {
        (task, seed) for task in range(10) for seed in range(40100, 40150)}
    assert all(type(p["native"]) is bool and type(p["fp8"]) is bool for p in pairs)
    assert paired_statistics(pairs) == result["summary"]
    assert result["summary"]["noninferior"] and not result["summary"]["collapsed_tasks"]
    assert result["pairing"]["matched_initial_states_and_observations"] == 500
    assert result["pairing"]["matched_noise_calls"] >= 500
    for row in result["calibration"]["tasks"].values():
        assert len(row["calibration_positions"]) == len(row["holdout_positions"]) == 8
        assert not set(row["calibration_positions"]) & set(row["holdout_positions"])
    for relative, expected in result["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected, f"matched source changed: {relative}"
    assert _sha256(ROOT / "scripts/summarize_pi05_sm120.py") == result["exporter_source_sha256"]
    assert "/workspace/" not in path.read_text()

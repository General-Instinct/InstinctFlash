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
    assert result["protocol"]["action_operating_point"] == (
        "FlashRT pi05 horizon 10, action dim 7"
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
    assert result["arms"]["native"]["action_shape"] == [3, 10, 7]
    assert result["arms"]["fp8"]["action_shape"] == [3, 10, 7]
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

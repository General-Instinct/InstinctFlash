"""CPU checks of the pairing and finite-sample decision rules; no GPU claims."""
import json
from pathlib import Path

import pytest

from benchmarks.vla.pi05_sm120_libero import paired_statistics, report, source_hashes
from benchmarks.vla.util import sha256_file


def test_identical_small_screen_cannot_certify_noninferiority():
    pairs = [{"task_id": 7, "native": True, "fp8": True} for _ in range(3)]
    result = paired_statistics(pairs)
    assert result["difference"] == 0
    assert result["noninferior"] is False
    assert result["paired_difference_ci95"][0] < 0


def test_paired_statistic_detects_direction_and_discordances():
    pairs = [{"task_id": t, "native": True, "fp8": i < 40}
             for t in range(10) for i in range(50)]
    result = paired_statistics(pairs)
    assert result["native_only_success"] == 100
    assert result["fp8_only_success"] == 0
    assert result["difference"] == pytest.approx(-.2)
    assert not result["noninferior"]
    swapped = [{**p, "native": p["fp8"], "fp8": p["native"]} for p in pairs]
    assert paired_statistics(swapped)["difference"] == pytest.approx(.2)


def test_two_wins_do_not_have_a_degenerate_perfect_confidence_interval():
    result = paired_statistics([{"task_id": 7, "native": False, "fp8": True}] * 2)
    assert result["difference"] == 1
    assert result["one_sided_lower95"] < 0
    assert not result["noninferior"]


@pytest.mark.parametrize("mutation", ["seed", "scene", "noise", "plan", "count"])
def test_report_refuses_unmatched_or_incomplete_receipts(tmp_path, mutation):
    (tmp_path / "plan.json").write_text(json.dumps({"source_sha256": source_hashes()}))
    episode = {"seed": 40100, "scene": {"init_state_index": 0},
               "noise_sha256": ["abc"], "success": True}
    native = {"episodes": [episode], "plan_sha256": sha256_file(tmp_path / "plan.json")}
    fp8 = json.loads(json.dumps(native))
    if mutation == "seed":
        fp8["episodes"][0]["seed"] += 1
    elif mutation == "scene":
        fp8["episodes"][0]["scene"]["init_state_index"] = 1
    elif mutation == "noise":
        fp8["episodes"][0]["noise_sha256"] = ["changed"]
    elif mutation == "plan":
        fp8["plan_sha256"] = "changed"
    else:
        fp8["episodes"] = []
    (tmp_path / "native-7.json").write_text(json.dumps(native))
    (tmp_path / "fp8-7.json").write_text(json.dumps(fp8))
    with pytest.raises(ValueError):
        report(tmp_path, [7], 1)


@pytest.mark.parametrize("horizon", [0, 5, 50, True, 10.0])
def test_public_pi05_refuses_uncertified_horizon_before_model_load(monkeypatch, horizon):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "serving"))
    from flash_rt.api import load_model
    with pytest.raises(ValueError, match="action_horizon=10"):
        load_model("does-not-exist", action_horizon=horizon)


def test_public_api_preserves_explicit_calibration_and_forwards_state(monkeypatch):
    import sys
    from types import SimpleNamespace
    import numpy as np
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "serving"))
    from flash_rt import api, hardware
    calls = []

    class Frontend:
        requires_state = True
        calibrated = False

        def __init__(self, checkpoint, num_views=2, hardware=None, use_fp8=True, action_horizon=10):
            calls.append(("load", action_horizon))

        def set_prompt(self, prompt, state=None):
            calls.append(("prompt", prompt))

        def calibrate(self, observations, **kwargs):
            calls.append(("calibrate", len(observations), kwargs["percentile"]))
            self.calibrated = True

        calibrate_with_real_data = calibrate

        def infer(self, observation):
            calls.append(("infer", observation["state"].copy()))
            return {"actions": np.zeros((10, 7), dtype=np.float32)}

    monkeypatch.setattr(hardware, "resolve_pipeline_class", lambda *a: Frontend)
    monkeypatch.setitem(sys.modules, "flash_rt.frontends.torch.pi05_checkpoint",
                        SimpleNamespace(Pi05CheckpointFrontend=Frontend))
    model = api.load_model("unused", hardware="rtx_sm120", action_horizon=10)
    model.calibrate([{}, {}, {}], prompt="pick up the bowl", percentile=100)
    state = np.arange(8, dtype=np.float32)
    assert model.predict([np.zeros((224, 224, 3), dtype=np.uint8)] * 2, state=state).shape == (10, 7)
    assert calls[:3] == [("load", 10), ("prompt", "pick up the bowl"), ("calibrate", 3, 100)]
    assert len(calls) == 4
    np.testing.assert_array_equal(calls[-1][1], state)

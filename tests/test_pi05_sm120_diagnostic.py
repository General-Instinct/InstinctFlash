"""CPU tests for the task diagnostic's pairing and non-invasive recording."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.vla import pi05_sm120_diagnostic as diagnostic


def episode(steps=20):
    return {"seed": 40100, "success": True, "steps": steps,
            "scene": {"seed": 40100, "init_state_index": 0, "initial_observation_sha256": "camera"},
            "noise_sha256": ["a" * 64] * ((steps + 9) // 10)}


@pytest.mark.parametrize("mutation", ["seed", "scene", "noise", "empty", "short", "invalid", "steps", "success"])
def test_diagnostic_rejects_unpaired_or_incomplete_rows(mutation):
    left, right = episode(), episode()
    if mutation == "seed":
        right["seed"] += 1
    elif mutation == "scene":
        right["scene"]["initial_observation_sha256"] = "different"
    elif mutation == "noise":
        right["noise_sha256"][0] = "b" * 64
    elif mutation == "empty":
        right["noise_sha256"] = []
    elif mutation == "short":
        right["noise_sha256"].pop()
    elif mutation == "invalid":
        right["noise_sha256"][0] = "g" * 64
    elif mutation == "steps":
        right["steps"] = 281
    else:
        right["success"] = 1
    with pytest.raises(ValueError):
        diagnostic.validate_pair(left, right)


def test_diagnostic_checks_common_noise_when_one_arm_finishes_earlier():
    left, right = episode(11), episode(23)
    right["noise_sha256"][-1] = "b" * 64
    assert diagnostic.validate_pair(left, right) == 2


def test_trace_environment_does_not_add_resets_steps_or_change_actions(monkeypatch, tmp_path):
    import sys
    calls, frames = [], []
    writer = SimpleNamespace(append_data=lambda x: frames.append(x.copy()), close=lambda: None)
    v2 = SimpleNamespace(get_writer=lambda *a, **k: writer)
    monkeypatch.setitem(sys.modules, "imageio", SimpleNamespace(v2=v2))
    monkeypatch.setitem(sys.modules, "imageio.v2", v2)
    pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(1, 4, 4, 3)
    obs = {"state": np.zeros((1, 8), np.float32), "pixels": {"image": pixels, "image2": pixels}}
    env = SimpleNamespace(sim=SimpleNamespace(
        data=SimpleNamespace(qpos=np.zeros(9), body_xpos=np.zeros((2, 3))),
        model=SimpleNamespace(body_names=["world", "bowl"])),
        parsed_problem={"goal_state": [["on", "bowl", "plate"]]}, _eval_predicate=lambda g: False)

    class Vec:
        envs = [SimpleNamespace(unwrapped=SimpleNamespace(inner=SimpleNamespace(env=env)))]

        def reset(self, **kwargs):
            calls.append(("reset", kwargs))
            return obs, {}

        def step(self, action):
            calls.append(("step", action.copy()))
            return obs, np.array([0]), np.array([False]), np.array([False]), {}

    recorded = diagnostic.TraceEnvironment(Vec(), tmp_path / "video.mp4")
    assert recorded.reset(seed=[40100])[0] is obs
    action = np.arange(7, dtype=np.float32)[None]
    assert recorded.step(action)[0] is obs
    assert len(calls) == 2 and calls[0] == ("reset", {"seed": [40100]})
    np.testing.assert_array_equal(calls[1][1], action)
    np.testing.assert_array_equal(recorded.actions[0], action[0])
    np.testing.assert_array_equal(frames[0][:, :4], pixels[0, ::-1, ::-1])
    assert recorded.goals == [[False], [False]]
    assert len(recorded.states) == 2


def test_recorded_inputs_are_copied_and_inference_runs_once():
    calls = []
    noise = SimpleNamespace(float=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(
        numpy=lambda: np.ones((50, 32), np.float32))))

    class Runtime:
        _noise_buf = noise

        def infer(self, observation):
            calls.append(copy.deepcopy(observation))
            return {"actions": np.ones((10, 7), np.float32)}

    runtime = diagnostic.TraceRuntime(Runtime())
    inputs = {"image": np.zeros((4, 4, 3), np.uint8),
              "wrist_image": np.zeros((4, 4, 3), np.uint8), "state": np.zeros(8, np.float32)}
    result = runtime.infer(inputs)
    inputs["state"][:] = 1
    result["actions"][:] = 2
    assert len(calls) == 1
    assert not runtime.inputs[0]["state"].any()
    assert (runtime.actions[0] == 1).all()
    assert runtime._noise_buf is noise


def trace_fixture():
    row = episode(11)
    row["goal_predicates"] = [["On", "bowl", "plate"]]
    arrays = {"actions": np.zeros((11, 7), np.float32), "states": np.zeros((12, 8), np.float32),
              "input_state": np.zeros((2, 8), np.float32), "chunks": np.zeros((2, 10, 7), np.float32),
              "noises": np.ones((2, 50, 32), np.float32), "goal_satisfied": np.zeros((12, 1), bool)}
    arrays["goal_satisfied"][-1] = True
    row["noise_sha256"] = [hashlib.sha256(np.full((50, 32), 0x3f80, dtype="<u2").tobytes()).hexdigest()] * 2
    row["action_digest"] = diagnostic.matched.action_digest(arrays["actions"].astype(np.float64).ravel())
    return row, arrays


def test_replay_arrays_match_receipt_and_noise_without_importing_torch():
    row, arrays = trace_fixture()
    diagnostic.validate_trace(row, arrays)


@pytest.mark.parametrize("mutation", ["actions", "queue", "state", "noise", "noise_dtype", "goal", "count", "nan"])
def test_replay_arrays_reject_inconsistent_or_tampered_data(mutation):
    row, arrays = trace_fixture()
    if mutation == "actions":
        arrays["actions"][0, 0] = 1
    elif mutation == "queue":
        arrays["chunks"][0, 0, 0] = 1
    elif mutation == "state":
        arrays["input_state"][0, 0] = 1
    elif mutation == "noise":
        arrays["noises"][0, 0, 0] = 2
    elif mutation == "noise_dtype":
        arrays["noises"][0, 0, 0] = 1.00001
    elif mutation == "goal":
        arrays["goal_satisfied"][-1] = False
    elif mutation == "count":
        arrays["states"] = arrays["states"][:-1]
    else:
        arrays["chunks"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        diagnostic.validate_trace(row, arrays)


def test_geometry_does_not_mistake_near_plate_for_success():
    from scripts.analyze_pi05_diagnostic import geometry
    row, arrays = trace_fixture()
    row["success"] = False
    arrays["body_names"] = np.array(["bowl_main", "plate_main"])
    positions = np.zeros((12, 2, 3))
    positions[:, 0, 0] = .04
    positions[3, 0, 2] = .1
    arrays["body_positions"] = positions
    arrays["goal_satisfied"][:] = False
    result = geometry(row, arrays)
    assert result["label"] == "lifted_and_near_target_but_goal_false"
    assert result["final_xy_violates_libero_on"] is True
    assert result["never_within_libero_on_xy"] is True
    assert result["success"] is False


def test_initial_numeric_comparison_refuses_different_noise_or_camera():
    from scripts.analyze_pi05_diagnostic import initial_action_comparison
    data = {"image": np.ones((1, 4, 4, 3)), "wrist_image": np.ones((1, 4, 4, 3)),
            "input_state": np.zeros((1, 8)), "noises": np.ones((1, 50, 32)),
            "chunks": np.ones((1, 10, 7))}
    assert initial_action_comparison(data, data)["cosine"] == pytest.approx(1)
    for name in ("image", "wrist_image", "input_state", "noises"):
        changed = copy.deepcopy(data)
        changed[name].flat[0] += 1
        with pytest.raises(ValueError):
            initial_action_comparison(data, changed)


def test_published_task_diagnosis_is_complete_source_bound_and_not_a_certificate():
    from instinctflash.verify.certify import _tango_paired_score_bounds
    root = Path(__file__).resolve().parents[1]
    path = root / "examples/pi05_vla/sm120_task5_diagnostic_results.json"
    result = json.loads(path.read_text())
    baseline = json.loads((root / "examples/pi05_vla/sm120_libero_matched_results.json").read_text())
    assert result["status"] == "DIAGNOSTIC"
    assert result["task"] == 5 and result["episodes_per_arm"] == 50
    assert result["baseline_sha256"]["plan.json"] == baseline["execution_plan_sha256"]
    assert result["official_loaded_weights"]["checked_tensors"] == 812
    assert result["official_loaded_weights"]["checkpoint_sha256"] == diagnostic.matched.MODEL_SHA256
    assert result["arm_order"] == list(diagnostic.ARMS)
    pairs = result["pairs"]
    assert [p["seed"] for p in pairs] == list(range(40100, 40150))
    for arm in diagnostic.ARMS:
        assert result["successes"][arm] == sum(p["arms"][arm]["success"] for p in pairs)
        for pair in pairs:
            row = pair["arms"][arm]
            assert type(row["success"]) is bool
            assert len(row["noise_sha256"]) == (row["steps"] + 9) // 10
            assert len(row["artifacts"]) == 2
    for left, right in itertools.combinations(diagnostic.ARMS, 2):
        count = 0
        outcomes = []
        for pair in pairs:
            a, b = pair["arms"][left], pair["arms"][right]
            common = min(len(a["noise_sha256"]), len(b["noise_sha256"]))
            assert a["noise_sha256"][:common] == b["noise_sha256"][:common]
            count += common
            outcomes.append((a["success"], b["success"]))
        assert count == result["matched_noise_calls"][f"{left}:{right}"]
        comparison = result["comparisons"][f"{right}_minus_{left}"]
        assert comparison["difference"] == sum(int(b) - int(a) for a, b in outcomes) / 50
        assert comparison["central_ci95"] == list(_tango_paired_score_bounds(outcomes, z=1.959963984540054))
    for relative, expected in result["source_sha256"].items():
        assert diagnostic.sha256_file(root / relative) == expected
    assert diagnostic.sha256_file(root / "scripts/analyze_pi05_diagnostic.py") == result["analysis"]["analysis_source_sha256"]
    assert "/workspace/" not in path.read_text()

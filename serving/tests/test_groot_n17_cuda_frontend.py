from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from flash_rt.api import load_model
from flash_rt.frontends.torch.groot_n17_cuda import (
    GrootN17TorchFrontendCuda,
    normalise_observation,
)


class _Modality:
    def __init__(self, keys, horizon):
        self.modality_keys = tuple(keys)
        self.delta_indices = tuple(range(horizon))


class _FakePolicy:
    def __init__(self):
        self.modality_configs = {
            "video": _Modality(("exterior_image_1_left", "wrist_image_left"), 2),
            "state": _Modality(("eef_9d", "gripper_position", "joint_position"), 1),
            "language": _Modality(("annotation.language.language_instruction",), 1),
            "action": _Modality(("eef_9d", "gripper_position", "joint_position"), 40),
        }
        self.embodiment_tag = SimpleNamespace(
            value="oxe_droid_relative_eef_relative_joint",
            name="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        )
        self.observation = None

    def get_action(self, observation):
        self.observation = observation
        return {
            "eef_9d": np.zeros((1, 40, 9), np.float32),
            "gripper_position": np.zeros((1, 40, 1), np.float32),
            "joint_position": np.zeros((1, 40, 7), np.float32),
        }, {}


def test_normaliser_maps_two_images_and_flat_state_to_upstream_contract():
    policy = _FakePolicy()
    image = np.zeros((180, 320, 3), np.uint8)
    observation = normalise_observation(
        policy,
        {"images": [image, image], "state": np.zeros(17, np.float32)},
        prompt="pick up the object",
        state_dims={"eef_9d": 9, "gripper_position": 1, "joint_position": 7},
    )
    assert observation["video"]["wrist_image_left"].shape == (1, 2, 180, 320, 3)
    assert observation["state"]["eef_9d"].shape == (1, 1, 9)
    assert observation["language"]["annotation.language.language_instruction"] == [
        ["pick up the object"]
    ]


def test_frontend_returns_concatenated_and_split_actions():
    frontend = GrootN17TorchFrontendCuda.__new__(GrootN17TorchFrontendCuda)
    frontend._policy = _FakePolicy()
    frontend._prompt = "pick up the object"
    frontend._state_dims = {"eef_9d": 9, "gripper_position": 1, "joint_position": 7}
    frontend.action_horizon = 16
    frontend._last_action_dict = None
    frontend._graph_driver = None
    image = np.zeros((180, 320, 3), np.uint8)
    result = frontend.infer({"images": [image, image], "state": np.zeros(17, np.float32)})
    assert result["actions"].shape == (16, 17)
    assert result["action_dict"]["joint_position"].shape == (16, 7)


def test_frontend_reports_graph_capture_counters():
    frontend = GrootN17TorchFrontendCuda.__new__(GrootN17TorchFrontendCuda)
    frontend._graph_driver = SimpleNamespace(captured=True, captures=2, replays=12)
    frontend.action_horizon = 40
    frontend.embodiment_tag = "oxe_droid_relative_eef_relative_joint"
    stats = frontend.backend_stats
    assert stats["backend"] == "upstream_bf16_cuda_graph"
    assert stats["captured"] is True
    assert stats["graph_captures"] == 2
    assert stats["graph_replays"] == 12


def test_public_loader_routes_n17_options_to_sm80_baseline():
    captured = {}

    class FakeFrontend:
        def __init__(self, checkpoint, num_views=2, embodiment_tag=None,
                     action_horizon=None, source_root=None, use_cuda_graph=True):
            captured.update(locals())

        def set_prompt(self, prompt):
            pass

        def infer(self, observation):
            split = {"joint_position": np.zeros((40, 7), np.float32)}
            return {"actions": np.zeros((40, 17), np.float32), "action_dict": split}

    with patch("flash_rt.hardware.detect_arch", return_value="cuda_sm80"), patch(
        "flash_rt.hardware.resolve_pipeline_class", return_value=FakeFrontend
    ):
        model = load_model(
            "/checkpoint",
            config="groot_n17",
            framework="torch",
            embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
            action_horizon=16,
            source_root="/source",
        )
        image = np.zeros((180, 320, 3), np.uint8)
        action = model.predict([image, image], prompt="pick", state=np.zeros(17, np.float32))
    assert action.shape == (40, 17)
    assert model.action_dict["joint_position"].shape == (40, 7)
    assert captured["embodiment_tag"] == "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    assert captured["action_horizon"] == 16
    assert captured["source_root"] == "/source"
    assert captured["use_cuda_graph"] is True


def test_hardware_table_exposes_n17_only_on_validated_cuda_baselines():
    from flash_rt.hardware import _PIPELINE_MAP

    assert ("groot_n17", "torch", "cuda_sm80") in _PIPELINE_MAP
    assert ("groot_n17", "torch", "cuda_sm90") in _PIPELINE_MAP
    assert ("groot_n17", "torch", "thor") not in _PIPELINE_MAP


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
                failures += 1
            else:
                print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)

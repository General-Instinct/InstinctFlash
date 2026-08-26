from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "examples" / "groot_n17"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from instinctflash.adapters.base import GuidanceMode
from instinctflash.descriptors.known import lookup
from instinctflash.descriptors.package import _declared_view, validate_package
from groot_n17_iwm.adapter import (
    GR00TN17Adapter,
    _GR00TN17Loop,
    _configure_preprocessing_threads,
    _env_flag,
    _resolve_model_path,
)
from groot_n17_iwm.backbone_fastpath import install_backbone_fastpath
from groot_n17_iwm.static_capture import install_static_capture


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
        self.embodiment_tag = SimpleNamespace(value="oxe_droid_relative_eef_relative_joint")
        self.observation = None

    def reset(self):
        pass

    def get_action(self, observation):
        self.observation = observation
        return {
            "eef_9d": np.ones((1, 40, 9), dtype=np.float32),
            "gripper_position": np.ones((1, 40, 1), dtype=np.float32) * 2,
            "joint_position": np.ones((1, 40, 7), dtype=np.float32) * 3,
        }, {"fake": True}


def _write_statistics(path: Path):
    stats = {
        "oxe_droid_relative_eef_relative_joint": {
            "state": {
                "eef_9d": {"mean": [0.0] * 9},
                "gripper_position": {"mean": [0.0]},
                "joint_position": {"mean": [0.0] * 7},
            }
        }
    }
    (path / "statistics.json").write_text(json.dumps(stats))


def test_example_surface_stays_product_shaped():
    # The exactness gates and their committed measurement are part of the surface ON PURPOSE:
    # the published fastpath numbers must keep their in-repo proof (verify_fastpaths.py +
    # fastpath_results.json, plus the per-fastpath verify scripts). A cleanup that removes a
    # gate while keeping the number it proved fails here.
    root_files = {path.name for path in PLUGIN_ROOT.iterdir() if path.is_file()}
    assert root_files == {
        "README.md",
        "benchmark_runtime.py",
        "config.json",
        "fastpath_results.json",
        "instinctflash.json",
        "profile_runtime.py",
        "pyproject.toml",
        "static_capture.py",
        "verify_backbone_fastpath.py",
        "verify_fast_decode.py",
        "verify_fastpaths.py",
        "verify_static_capture.py",
    }


def test_spec_declares_n17_control_cycle_without_persistent_kv():
    spec = GR00TN17Adapter().spec()
    assert spec.model_id == "nvidia/GR00T-N1.7-3B"
    assert spec.streams == ()
    assert spec.phase("backbone").nfe == 1
    assert spec.phase("action").nfe == 4
    assert spec.phase("action").truncatable is True
    assert spec.guidance["action"].mode is GuidanceMode.NONE
    assert spec.shapes_static_across_cycles()[0] is True
    assert [field.shape for field in spec.observation.fields] == [
        (2, 180, 320, 3), (2, 180, 320, 3), (1, 9), (1, 1), (1, 7),
    ]


def test_pointer_package_and_known_hub_declaration_validate():
    package = ROOT / "examples" / "groot_n17"
    report = validate_package(package)
    assert report.ok, report.explain()
    known = lookup("nvidia/GR00T-N1.7-3B")
    assert known["execution"]["backbone"] == "groot_n17"
    assert known["execution"]["nfe"]["action"] == 4


def test_known_hub_release_gets_a_declared_view_without_copying_weights():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        (snapshot / "config.json").write_text('{"model_type":"Gr00tN1d7"}')
        (snapshot / "model.safetensors.index.json").write_text("{}")
        cache = root / "cache"
        previous = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = str(cache)
        try:
            view = _declared_view(
                snapshot,
                "nvidia/GR00T-N1.7-3B",
                lookup("nvidia/GR00T-N1.7-3B"),
            )
            assert (view / "model.safetensors.index.json").is_symlink()
            assert validate_package(view).ok
        finally:
            if previous is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = previous


def test_runtime_loop_accepts_compact_inputs_and_preserves_action_modalities():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        _write_statistics(path)
        policy = _FakePolicy()
        loop = _GR00TN17Loop(policy, model_path=path, action_nfe=4)
        loop.reset(prompt="pick up the object")
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        result = loop.predict({"images": [image, image], "state": np.zeros(17, np.float32)})
        assert result["action"].shape == (40, 17)
        assert set(result["actions"]) == {"eef_9d", "gripper_position", "joint_position"}
        assert policy.observation["video"]["exterior_image_1_left"].shape == (
            1, 2, 180, 320, 3,
        )
        assert policy.observation["state"]["joint_position"].shape == (1, 1, 7)
        language = policy.observation["language"]["annotation.language.language_instruction"]
        assert language == [["pick up the object"]]


def test_model_path_prefers_an_explicit_local_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        checkpoint_dir = root / "weights"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors.index.json").write_text("{}")
        (checkpoint_dir / "processor_config.json").write_text("{}")
        package = root / "package"
        package.mkdir()
        checkpoint = SimpleNamespace(
            path=str(package),
            model_id="nvidia/GR00T-N1.7-3B",
            execution=SimpleNamespace(extra={"base_weights": "nvidia/GR00T-N1.7-3B"}),
        )
        previous = os.environ.get("GR00T_N17_CHECKPOINT")
        os.environ["GR00T_N17_CHECKPOINT"] = str(checkpoint_dir)
        try:
            assert _resolve_model_path(checkpoint) == checkpoint_dir.resolve()
        finally:
            if previous is None:
                os.environ.pop("GR00T_N17_CHECKPOINT", None)
            else:
                os.environ["GR00T_N17_CHECKPOINT"] = previous


def test_static_capture_installer_is_idempotent():
    dit = SimpleNamespace(forward=lambda **kwargs: kwargs["hidden_states"])
    model = SimpleNamespace(action_head=SimpleNamespace(model=dit))
    first = install_static_capture(model)
    second = install_static_capture(model)
    assert first is second
    assert dit.forward is first


def test_backbone_fastpath_installer_is_idempotent_and_restorable():
    def original_forward(*args, **kwargs):
        return args, kwargs

    visual = SimpleNamespace(
        fast_pos_embed_interpolate=original_forward,
        rot_pos_emb=original_forward,
        forward=original_forward,
    )
    lm_head = SimpleNamespace(forward=original_forward)
    base = SimpleNamespace(
        get_rope_index=original_forward,
        get_image_features=original_forward,
    )
    backbone = SimpleNamespace(
        model=SimpleNamespace(model=base, visual=visual, lm_head=lm_head)
    )
    model = SimpleNamespace(backbone=backbone)
    first = install_backbone_fastpath(model)
    second = install_backbone_fastpath(model)
    assert first is second
    assert lm_head.forward(np.ones((2, 3, 4))).shape == (2, 3, 0)
    first.close()
    assert visual.forward is original_forward
    assert lm_head.forward is original_forward


def test_runtime_installs_planned_capture_only_when_opted_in():
    driver = SimpleNamespace(captured=False, captures=0, replays=0)
    policy = SimpleNamespace(model=object())
    plan = SimpleNamespace(results=[SimpleNamespace(name="graph_capture", applies=True)])
    previous = os.environ.get(GR00TN17Adapter.STATIC_CAPTURE_ENV)
    try:
        os.environ[GR00TN17Adapter.STATIC_CAPTURE_ENV] = "1"
        with patch("torch.cuda.is_available", return_value=True), patch(
            "groot_n17_iwm.static_capture.install_static_capture", return_value=driver
        ):
            installed = GR00TN17Adapter.install(policy, plan, device="cuda:0")
        assert installed == ["graph_capture"]
        assert policy._instinctflash_static_capture is driver
    finally:
        if previous is None:
            os.environ.pop(GR00TN17Adapter.STATIC_CAPTURE_ENV, None)
        else:
            os.environ[GR00TN17Adapter.STATIC_CAPTURE_ENV] = previous


def test_static_capture_flag_is_strict():
    name = "IFL_TEST_GROOT_CAPTURE"
    previous = os.environ.get(name)
    try:
        os.environ.pop(name, None)
        assert _env_flag(name, default=False) is False
        os.environ[name] = "yes"
        assert _env_flag(name, default=False) is True
        os.environ[name] = "off"
        assert _env_flag(name, default=True) is False
        os.environ[name] = "maybe"
        try:
            _env_flag(name, default=False)
        except RuntimeError as error:
            assert name in str(error)
        else:
            raise AssertionError("invalid capture flag must fail closed")
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def test_preprocessing_threads_support_auto_disable_explicit_and_restore():
    # The pin is process-global, so it is env-opt-in only and MUST restore on close: the handle
    # carries the previous torch/cv2 values and puts them back.
    fake_cv2 = SimpleNamespace(setNumThreads=Mock(), getNumThreads=Mock(return_value=208))
    with patch.dict(sys.modules, {"cv2": fake_cv2}), patch(
        "torch.set_num_threads"
    ) as set_torch, patch("torch.get_num_threads", return_value=104):
        assert _configure_preprocessing_threads(None) is None
        assert _configure_preprocessing_threads("0") is None
        pin = _configure_preprocessing_threads("7")
        assert pin.target == 7
        set_torch.assert_called_once_with(7)
        fake_cv2.setNumThreads.assert_called_once_with(7)
        pin.restore()
        assert set_torch.call_args_list[-1].args == (104,)
        assert fake_cv2.setNumThreads.call_args_list[-1].args == (208,)
        assert pin.restored
        pin.restore()  # idempotent
        assert len(set_torch.call_args_list) == 2

    fake_cv2.setNumThreads.reset_mock()
    with patch.dict(sys.modules, {"cv2": fake_cv2}), patch(
        "os.cpu_count", return_value=240
    ), patch("torch.set_num_threads") as set_torch, patch(
        "torch.get_num_threads", return_value=240
    ):
        # `auto` caps at min(16, cores): a fixed 16 on a <16-core host would RAISE threads.
        assert _configure_preprocessing_threads("auto").target == 16
        set_torch.assert_called_once_with(16)
        fake_cv2.setNumThreads.assert_called_once_with(16)


def test_loop_close_restores_the_thread_pin():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        _write_statistics(path)
        pin = SimpleNamespace(target=7, restored=False,
                              restore=Mock(side_effect=lambda: None))
        loop = _GR00TN17Loop(_FakePolicy(), model_path=path, action_nfe=4, cpu_threads=pin)
        assert loop.backend_stats["cpu_threads"] == 7
        loop.close()
        pin.restore.assert_called_once()


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))

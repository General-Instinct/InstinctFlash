from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "examples" / "lingbot_vla"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from instinctflash.adapters.base import GuidanceMode, KVLifetime
from instinctflash.descriptors.known import lookup
from instinctflash.descriptors.package import _declared_view, validate_package
from lingbot_vla_iwm.adapter import (
    LingBotVLA4BAdapter,
    _LingBotVLA4BLoop,
    _resolve_model_path,
    _resolve_norm_stats,
)

MODEL_ID = "robbyant/lingbot-vla-4b-posttrain-robotwin"


def test_example_surface_stays_product_shaped():
    # verify_static_capture.py + static_capture_results.json are LOAD-BEARING: they back the
    # README H100 row (670.9 -> 185.2 ms, 3.62x, six 0.0 gates), and reproduce_h100.sh is the
    # committed re-measurement protocol. This pinned set exists so a cleanup can never silently
    # delete or replace them.
    root_files = {path.name for path in PLUGIN_ROOT.iterdir() if path.is_file()}
    assert root_files == {
        "README.md",
        "profile_infer.py",
        "pyproject.toml",
        "reproduce_h100.sh",
        "static_capture_results.json",
        "verify_static_capture.py",
    }, root_files
    assert os.access(PLUGIN_ROOT / "reproduce_h100.sh", os.X_OK)


def test_published_h100_artifact_is_the_six_case_bitexact_protocol():
    import json

    doc = json.loads((PLUGIN_ROOT / "static_capture_results.json").read_text())
    assert doc["gates"]["static_eager_vs_stock"] == 0.0
    assert len(doc["gates"]["cases"]) == 6
    assert all(case["max_abs_d"] == 0.0 for case in doc["gates"]["cases"])
    assert doc["ours_ms_p50_inprocess"] < doc["stock_ms_p50_inprocess"]


def test_spec_declares_the_published_control_cycle():
    spec = LingBotVLA4BAdapter().spec()
    assert spec.model_id == MODEL_ID
    assert spec.streams[0].lifetime is KVLifetime.CHUNK
    assert spec.phase("prefix").nfe == 1
    assert spec.phase("action").nfe == 10
    assert spec.phase("action").truncatable is True
    assert spec.guidance["action"].mode is GuidanceMode.NONE
    # chunk-lifetime prefix -> static shapes: the property the capture arm rests on
    assert spec.shapes_static_across_cycles()[0] is True
    assert spec.observation.batched is False
    assert [field.shape for field in spec.observation.fields] == [
        (480, 640, 3), (480, 640, 3), (480, 640, 3), (14,),
    ]


def test_known_hub_release_declares_backbone_and_serving_config():
    doc = lookup(MODEL_ID)
    assert doc is not None
    ex = doc["execution"]
    assert ex["backbone"] == "lingbot_vla"
    assert ex["nfe"] == {"prefix": 1, "action": 10}
    assert ex["robot"] == "robotwin"
    assert ex["use_length"] == 25
    # the norm stats are a serving-config fact: without them the reply is in a normalised
    # space nobody can execute, so the declaration must carry the pointer
    assert ex["norm_stats"].endswith("robotwin_50.json")
    assert ex["tokenizer_repo"] == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_known_release_gets_a_declared_view_without_copying_weights():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        # the real snapshot is flat: config.json + model.safetensors + lingbotvla_cli.yaml
        (snapshot / "config.json").write_text('{"type":"pi0","chunk_size":50}')
        (snapshot / "model.safetensors").write_text("")
        (snapshot / "lingbotvla_cli.yaml").write_text("model: {}\n")
        previous = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = str(root / "cache")
        try:
            view = _declared_view(snapshot, MODEL_ID, lookup(MODEL_ID))
            report = validate_package(view)
            assert report.ok, report.explain()
            assert report.declaration.backbone == "lingbot_vla"
        finally:
            if previous is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = previous


def test_model_path_and_norm_stats_resolve_declared_layouts_and_fail_loud():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        weights = root / "weights"
        weights.mkdir()
        (weights / "model.safetensors").write_text("")
        (weights / "lingbotvla_cli.yaml").write_text("model: {}\n")
        package = root / "package"
        package.mkdir()
        checkpoint = SimpleNamespace(
            path=str(package), model_id=MODEL_ID,
            execution=SimpleNamespace(
                extra={"base_weights": str(weights),
                       "norm_stats": "assets/norm_stats/robotwin_50.json"}),
        )
        assert _resolve_model_path(checkpoint) == weights

        source_root = root / "checkout"
        norm = source_root / "assets" / "norm_stats"
        norm.mkdir(parents=True)
        (norm / "robotwin_50.json").write_text("{}")
        resolved = _resolve_norm_stats(checkpoint, source_root)
        assert resolved == (norm / "robotwin_50.json").resolve()

        # a missing declared file is refused with the fix in the message, never guessed around
        checkpoint.execution.extra["norm_stats"] = "assets/norm_stats/other_robot.json"
        try:
            _resolve_norm_stats(checkpoint, source_root)
        except RuntimeError as e:
            assert "norm_stats" in str(e)
        else:
            raise AssertionError("a missing norm-stats file must be refused")


def test_runtime_loop_maps_prompt_resets_under_project_cwd_and_returns_numpy():
    import torch

    class FakeServer:
        def __init__(self):
            self.reset_to = None
            self.observation = None

        def reset(self, robot):
            self.reset_to = robot
            self.reset_cwd = Path.cwd()

        def infer(self, observation):
            self.observation = observation
            return {"action": np.zeros((25, 14), dtype=np.float32)}

    with tempfile.TemporaryDirectory() as td:
        server = FakeServer()
        loop = _LingBotVLA4BLoop(server, Path(td).resolve(), robot="robotwin")
        assert server.reset_to == "robotwin"
        assert server.reset_cwd == Path(td).resolve()      # upstream reset needs its own cwd
        loop.reset(prompt="pick up the block")
        result = loop.predict({
            "observation.state": torch.zeros(14),          # tensors are converted, not rejected
            "observation.images.cam_high": np.zeros((480, 640, 3), np.uint8),
        })
        assert server.observation["prompt"] == server.observation["task"] == "pick up the block"
        assert isinstance(server.observation["observation.state"], np.ndarray)
        assert result["action"].shape == (25, 14)
        assert loop.graph_stats["captured"] is False


def test_install_reads_the_plan_instead_of_decorating_it():
    adapter = LingBotVLA4BAdapter()
    server = SimpleNamespace(vla=SimpleNamespace(model=object()))
    declined = SimpleNamespace(results=[SimpleNamespace(name="graph_capture", applies=False)])
    assert adapter.install(server, declined, device="cuda:0") is None
    previous = os.environ.get("IFL_VLA4B_BACKEND")
    try:
        os.environ["IFL_VLA4B_BACKEND"] = "eager"
        applied = SimpleNamespace(results=[
            SimpleNamespace(name="graph_capture", applies=True, params={})])
        assert adapter.install(server, applied, device="cuda:0") is None
        # the eager selector is RECORDED on the plan's capture entry, like the kill-switch
        assert any("IFL_VLA4B_BACKEND=eager" in line
                   for line in applied.results[0].params["decision"])
        os.environ["IFL_VLA4B_BACKEND"] = "sometimes"
        try:
            adapter.install(server, applied, device="cuda:0")
        except RuntimeError as e:
            assert "IFL_VLA4B_BACKEND" in str(e)
        else:
            raise AssertionError("an invalid backend flag must fail closed")
    finally:
        if previous is None:
            os.environ.pop("IFL_VLA4B_BACKEND", None)
        else:
            os.environ["IFL_VLA4B_BACKEND"] = previous


def test_gpu_smoke_one_predict_finite_actions():
    """Load -> one predict -> finite actions, through the public Runtime API.

    Opt-in (IFL_GPU_SMOKE=1): loads ~17 GB onto whatever GPU is visible; run it serialized on
    one idle device from the upstream venv:

        IFL_GPU_SMOKE=1 CUDA_VISIBLE_DEVICES=7 \
          /home/ubuntu/lingbot-vla-repo/.venv/bin/python tests/test_lingbot_vla4b_adapter.py

    Every missing precondition is a SKIP, never a failure, so public CI stays green.
    """
    if os.environ.get("IFL_GPU_SMOKE") != "1":
        print("SKIP: set IFL_GPU_SMOKE=1 (and CUDA_VISIBLE_DEVICES to one idle GPU)")
        return
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: no torch here ({e})")
        return
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return
    adapter = LingBotVLA4BAdapter()
    ok, why = adapter.can_host_in_process()
    if not ok:
        print(f"SKIP: {why}")
        return
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, local_files_only=True)
    except Exception:  # noqa: BLE001
        print(f"SKIP: {MODEL_ID} is not in the local HF cache")
        return

    from instinctflash import Runtime
    from instinctflash.runtime.loader import available_models, register

    if "lingbot_vla" not in available_models():
        register("lingbot_vla", LingBotVLA4BAdapter)
    runtime = Runtime.from_pretrained(MODEL_ID, placement="in_process")
    try:
        obs = runtime.observation.example()
        with runtime.episode(prompt="pick up the block and place it in the tray") as ep:
            out = ep.predict(obs)
        action = np.asarray(out["action"])
        assert action.shape == (25, 14), action.shape
        assert np.isfinite(action).all(), "non-finite action from smoke"
        print(f"smoke OK: {MODEL_ID} -> action {action.shape}, finite")
    finally:
        runtime.close()


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))

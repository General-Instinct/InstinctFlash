from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "examples" / "dreamzero"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from instinctflash.adapters.base import GuidanceMode, KVLifetime
from instinctflash.descriptors.known import lookup
from instinctflash.descriptors.package import _declared_view, validate_package
from dreamzero_iwm.adapter import (
    DYNAMIC_CACHE_ENV,
    DreamZeroAdapter,
    _DreamZeroLoop,
    _resolve_model_path,
)

MODEL_ID = "GEAR-Dreams/DreamZero-DROID"


def test_example_surface_stays_product_shaped():
    # The research artifacts (cfg_batch + its gates, step_cache provenance) are LOAD-BEARING
    # records of measured findings, and reproduce_h100.sh is the committed protocol behind the
    # README row. Pinned so a cleanup can never silently delete or replace them.
    root_files = {path.name for path in PLUGIN_ROOT.iterdir() if path.is_file()}
    assert root_files == {
        "README.md",
        "cfg_batch.py",
        "diag_batch.py",
        "measure_dreamzero.py",
        "pyproject.toml",
        "reproduce_h100.sh",
        "step_cache.py",
        "verify_cfg_batch.py",
    }, root_files
    assert os.access(PLUGIN_ROOT / "reproduce_h100.sh", os.X_OK)


def test_spec_declares_a_window_stream_and_cfg():
    spec = DreamZeroAdapter().spec()
    assert spec.model_id == MODEL_ID
    assert spec.streams[0].lifetime is KVLifetime.WINDOW
    assert spec.streams[0].tokens_per_frame == 50           # upstream's frame_seqlen
    assert spec.phase("video_action").nfe == 16
    assert spec.phase("kv_commit").commit_steps == frozenset({0})
    assert spec.guidance["video_action"].mode is GuidanceMode.CFG
    assert spec.guidance["video_action"].scale == 5.0
    # the KV cache outlives a control cycle, so whole-cycle capture must NOT look profitable
    static, why = spec.shapes_static_across_cycles()
    assert static is False, why
    assert spec.observation.conditioning == ("prompt",)


def test_known_release_declares_dynamic_cache_off_by_default():
    doc = lookup(MODEL_ID)
    assert doc is not None
    ex = doc["execution"]
    assert ex["backbone"] == "dreamzero"
    assert ex["nfe"] == {"video_action": 16}
    assert ex["guidance"] == {"video_action": "cfg"}
    assert ex["embodiment_tag"] == "oxe_droid"
    # SCREEN-tier option: declared, visible, and OFF. A declaration turning it on would be a
    # different operating point that must bring its own closed-loop certificate.
    assert ex["dynamic_cache_schedule"] is False


def test_known_release_gets_a_declared_view_without_copying_weights():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        (snapshot / "config.json").write_text('{"model_type":"dreamzero"}')
        (snapshot / "model.safetensors.index.json").write_text('{"weight_map":{}}')
        previous = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = str(root / "cache")
        try:
            view = _declared_view(snapshot, MODEL_ID, lookup(MODEL_ID))
            report = validate_package(view)
            assert report.ok, report.explain()
            assert report.declaration.backbone == "dreamzero"
        finally:
            if previous is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = previous


def test_nfe_override_is_refused_with_the_screen_tier_explanation():
    # DreamZero's step count is not a latency flag: fewer computed steps go through upstream's
    # own SCREEN-tier knobs. Asking the runtime for nfe video_action=8 must refuse, not comply.
    adapter = DreamZeroAdapter()
    ckpt = SimpleNamespace(
        path="x",
        execution=SimpleNamespace(model_id=MODEL_ID, nfe={"video_action": 16}, extra={}),
    )
    try:
        adapter.build_in_process(ckpt, plan=SimpleNamespace(results=[]),
                                 nfe={"video_action": 8})
    except RuntimeError as e:
        message = str(e)
        if "SCREEN" not in message:
            # The adapter's CUDA gate fires before the nfe refusal, and it raises RuntimeError
            # too — so the CUDA-less fallback must live HERE, not in a later except clause a
            # RuntimeError can never reach (the bug this branch replaces: on a box with CUDA
            # masked, the gate's message failed the SCREEN assertion instead of skipping).
            assert "CUDA" in message, message
        else:
            assert "SCREEN" in message and "DYNAMIC_CACHE_SCHEDULE" in message
    except Exception as e:  # noqa: BLE001 - only acceptable on a CUDA-less test box
        assert "CUDA" in str(e)
    else:
        raise AssertionError("a reduced NFE must be refused as SCREEN-tier, not served")


def test_runtime_loop_threads_prompt_and_session_and_returns_numpy():
    class FakeWrapper:
        def __init__(self):
            self.obs = None
            self.resets = 0

        def reset(self, reset_info):
            self.resets += 1

        def infer(self, obs):
            self.obs = obs
            return np.zeros((24, 8), dtype=np.float32)

    wrapper = FakeWrapper()
    loop = _DreamZeroLoop(wrapper, dynamic_cache=False)
    loop.reset(prompt="pick up the banana")
    frames = np.zeros((4, 160, 320, 3), np.uint8)
    out = loop.predict({"observation/exterior_image_0_left": frames})
    assert wrapper.resets == 1
    assert wrapper.obs["prompt"] == "pick up the banana"
    session_a = wrapper.obs["session_id"]
    assert out["action"].shape == (24, 8)
    # a new episode is a new session id, so upstream's own boundary logic also fires
    loop.reset(prompt="open the drawer")
    loop.predict({"observation/exterior_image_0_left": frames})
    assert wrapper.obs["session_id"] != session_a
    assert loop.backend_stats["dynamic_cache_schedule"] is False
    # a promptless episode is refused, not silently served with an empty instruction
    bare = _DreamZeroLoop(FakeWrapper(), dynamic_cache=False)
    try:
        bare.predict({"observation/exterior_image_0_left": frames})
    except ValueError as e:
        assert "prompt" in str(e)
    else:
        raise AssertionError("a promptless request must be refused")


def test_model_path_resolves_local_layouts_and_fails_loud():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        good = root / "ckpt"
        good.mkdir()
        (good / "config.json").write_text("{}")
        (good / "model.safetensors.index.json").write_text('{"weight_map":{}}')
        ckpt = SimpleNamespace(path=str(good), model_id=MODEL_ID,
                               execution=SimpleNamespace(extra={}))
        assert _resolve_model_path(ckpt) == good.resolve()
        empty = root / "empty"
        empty.mkdir()
        bad = SimpleNamespace(path=str(empty), model_id=MODEL_ID,
                              execution=SimpleNamespace(extra={"base_weights": str(empty)}))
        try:
            _resolve_model_path(bad)
        except RuntimeError as e:
            assert "ensure_file" in str(e) or "checkpoint" in str(e)
        else:
            raise AssertionError("an empty checkpoint dir must be refused")


def test_gpu_smoke_one_predict_finite_actions():
    """Load -> one predict -> finite actions, through the public Runtime API.

    Opt-in (IFL_GPU_SMOKE=1): loads ~77 GB of weights onto whatever GPU is visible (it fits
    alone on an 80 GB device); run it serialized on one idle GPU from the GEAR-Dreams venv:

        IFL_GPU_SMOKE=1 CUDA_VISIBLE_DEVICES=7 \
          /home/ubuntu/dreamzero-repo/.venv/bin/python tests/test_dreamzero_adapter.py

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
    adapter = DreamZeroAdapter()
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
    assert os.environ.get(DYNAMIC_CACHE_ENV, "false").lower() != "true", (
        "run the smoke with the SCREEN-tier flag off; it must exercise the default arm")

    from instinctflash import Runtime
    from instinctflash.runtime.loader import available_models, register

    if "dreamzero" not in available_models():
        register("dreamzero", DreamZeroAdapter)
    runtime = Runtime.from_pretrained(MODEL_ID, placement="in_process")
    try:
        obs = runtime.observation.example()
        with runtime.episode(prompt="pick up the banana and place it in the bowl") as ep:
            out = ep.predict(obs)
        action = np.asarray(out["action"])
        assert action.ndim == 2 and action.shape[-1] == 8, action.shape
        assert np.isfinite(action).all(), "non-finite action from smoke"
        print(f"smoke OK: {MODEL_ID} -> action {action.shape}, finite")
    finally:
        runtime.close()


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))

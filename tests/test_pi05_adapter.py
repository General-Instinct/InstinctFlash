from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "examples" / "pi05_vla"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from instinctflash.adapters.base import GuidanceMode, KVLifetime
from instinctflash.descriptors.known import lookup
from instinctflash.descriptors.package import _declared_view, validate_package
from pi05_iwm.adapter import Pi05Adapter, _Pi05Loop

#: Both pi05 releases the runtime serves by bare Hub id. v044 is the bf16-stored LIBERO
#: fine-tune the README H100 row (206.7 -> 72.8 ms) was measured on.
KNOWN_IDS = ("lerobot/pi05_base", "lerobot/pi05_libero_finetuned_v044")


def test_example_surface_stays_product_shaped():
    # The measurement artifacts are LOAD-BEARING: verify_static_capture.py +
    # static_capture_results.json back the README pi05 H100 row, and reproduce_h100.sh is the
    # committed protocol for re-measuring it. This pinned set exists so a cleanup can never
    # silently delete or replace them.
    root_files = {path.name for path in PLUGIN_ROOT.iterdir() if path.is_file()}
    assert root_files == {
        "README.md",
        "instinctwm.json",
        "measure_chunk_cost.py",
        "pyproject.toml",
        "reproduce_h100.py",
        "reproduce_h100.sh",
        "run_pi05_end_to_end.py",
        "static_capture_results.json",
        "verify_capture_equivalence.py",
        "verify_static_capture.py",
    }, root_files
    assert os.access(PLUGIN_ROOT / "reproduce_h100.sh", os.X_OK)


def test_spec_declares_the_published_control_cycle():
    spec = Pi05Adapter().spec()
    assert spec.model_id == "lerobot/pi05_base"
    assert spec.streams[0].name == "prefix"
    assert spec.streams[0].lifetime is KVLifetime.CHUNK
    assert spec.phase("prefix").nfe == 1
    assert spec.phase("action").nfe == 10
    assert spec.phase("action").truncatable is True
    assert spec.guidance["action"].mode is GuidanceMode.NONE
    # The property the whole capture arm rests on: chunk-lifetime prefix -> static shapes.
    assert spec.shapes_static_across_cycles()[0] is True
    assert [field.shape for field in spec.observation.fields] == [
        (3, 224, 224), (3, 224, 224), (3, 224, 224), (32,),
    ]
    assert spec.observation.conditioning == ("prompt",)


def test_known_hub_releases_declare_the_pi05_backbone():
    for repo in KNOWN_IDS:
        doc = lookup(repo)
        assert doc is not None, f"{repo} missing from KNOWN_DECLARATIONS"
        ex = doc["execution"]
        assert ex["backbone"] == "pi05"
        assert ex["servable"] is True
        assert ex["nfe"] == {"prefix": 1, "action": 10}
        # build_in_process refuses to load without this pointer; the declaration must carry it.
        assert ex["base_weights"] == repo
        assert ex["param_bytes"] > 0


def test_known_releases_get_a_declared_view_without_copying_weights():
    for repo in KNOWN_IDS:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            # the real snapshots are flat: config.json + model.safetensors + processor jsons
            (snapshot / "config.json").write_text('{"type":"pi05"}')
            (snapshot / "model.safetensors").write_text("")
            previous = os.environ.get("XDG_CACHE_HOME")
            os.environ["XDG_CACHE_HOME"] = str(root / "cache")
            try:
                view = _declared_view(snapshot, repo, lookup(repo))
                assert (view / "model.safetensors").is_symlink()
                report = validate_package(view)
                assert report.ok, report.explain()
                assert report.declaration.backbone == "pi05"
            finally:
                if previous is None:
                    os.environ.pop("XDG_CACHE_HOME", None)
                else:
                    os.environ["XDG_CACHE_HOME"] = previous


def test_observation_contract_is_declared_per_checkpoint_and_fails_loud():
    from types import SimpleNamespace

    adapter = Pi05Adapter()
    # v044 declares its own geometry; the contract must come from the declaration, not the base.
    doc = lookup("lerobot/pi05_libero_finetuned_v044")["execution"]
    ckpt = SimpleNamespace(path="x", execution=SimpleNamespace(
        model_id=doc["model_id"], extra={"obs_features": doc["obs_features"]}))
    obs, source = adapter.observation_contract(ckpt)
    assert "declared" in source
    shapes = {f.key: f.shape for f in obs.fields}
    assert shapes["observation.state"] == (8,)
    assert shapes["observation.images.image"] == (3, 256, 256)
    # the base release may fall back to the adapter's static declaration
    base = SimpleNamespace(path="x", execution=SimpleNamespace(
        model_id="lerobot/pi05_base", extra={}))
    obs, source = adapter.observation_contract(base)
    assert {f.key for f in obs.fields} == {
        "observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb", "observation.state"}
    # any other undeclared checkpoint is refused with the fix in the message
    other = SimpleNamespace(path="x", execution=SimpleNamespace(
        model_id="someone/pi05-finetune", extra={}))
    try:
        adapter.observation_contract(other)
    except RuntimeError as e:
        assert "obs_features" in str(e)
    else:
        raise AssertionError("an undeclared fine-tune geometry must be refused, not guessed")


def test_runtime_loop_maps_prompt_to_lerobot_task_and_returns_numpy():
    import torch

    class FakePolicy:
        def __init__(self):
            self.batch = None
            self.resets = 0

        def reset(self):
            self.resets += 1

        def select_action(self, batch):
            self.batch = batch
            return torch.zeros(1, 32)

    policy = FakePolicy()
    loop = _Pi05Loop(policy, pre=lambda b: b, post=lambda a: a, device="cpu")
    loop.reset(prompt="put the exhaust fans back")
    obs = {
        "observation.images.base_0_rgb": np.zeros((1, 3, 224, 224), np.float32),
        "observation.state": np.zeros((1, 32), np.float32),
        "ignored.key": np.zeros(3, np.float32),
    }
    result = loop.predict(obs)
    assert policy.resets == 1
    # LeRobot names the instruction `task`; the declaration calls it `prompt`.
    assert policy.batch["task"] == "put the exhaust fans back"
    assert "ignored.key" not in policy.batch
    assert isinstance(result["action"], np.ndarray)
    assert result["action"].shape == (32,)
    # a per-call prompt overrides the episode prompt
    loop.predict(dict(obs, prompt="stack the bowls"))
    assert policy.batch["task"] == "stack the bowls"


def test_gpu_smoke_one_predict_finite_actions():
    """Load -> one predict -> finite actions, through the public Runtime API.

    Opt-in (IFL_GPU_SMOKE=1) because it loads gigabytes onto whatever GPU is visible: on a
    shared box run it serialized on one idle device, e.g.

        IFL_GPU_SMOKE=1 CUDA_VISIBLE_DEVICES=7 <pi05-venv>/bin/python tests/test_pi05_adapter.py

    Every missing precondition is a SKIP, never a failure, so public CI stays green.
    """
    if os.environ.get("IFL_GPU_SMOKE") != "1":
        print("SKIP: set IFL_GPU_SMOKE=1 (and CUDA_VISIBLE_DEVICES to one idle GPU)")
        return
    try:
        import lerobot  # noqa: F401
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: pi05 model stack not importable here ({type(e).__name__}: {e})")
        return
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return
    repo = os.environ.get("IFL_PI05_SMOKE_CKPT", "lerobot/pi05_libero_finetuned_v044")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
    except Exception:  # noqa: BLE001
        print(f"SKIP: {repo} is not in the local HF cache")
        return

    from instinctflash import Runtime
    from instinctflash.runtime.loader import available_models, register

    if "pi05" not in available_models():
        register("pi05", Pi05Adapter)
    runtime = Runtime.from_pretrained(repo, placement="in_process")
    try:
        obs = runtime.observation.example()
        with runtime.episode(prompt="pick up the black bowl and place it on the plate") as ep:
            out = ep.predict(obs)
        action = np.asarray(out["action"])
        assert action.size > 0 and np.isfinite(action).all(), "non-finite action from smoke"
        print(f"smoke OK: {repo} -> action {action.shape}, finite")
    finally:
        runtime.close()


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))

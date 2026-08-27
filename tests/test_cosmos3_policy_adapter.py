from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "examples" / "cosmos3_policy"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from instinctflash.adapters.base import GuidanceMode
from instinctflash.descriptors.known import lookup
from instinctflash.descriptors.package import _declared_view, validate_package
from cosmos3_iwm.adapter import (
    REQUIRED_SERVING_KEYS,
    Cosmos3PolicyAdapter,
    _Cosmos3PolicyLoop,
    _encode_png_b64,
    _resolve_model_path,
)

EDGE = "nvidia/Cosmos3-Edge-Policy-DROID"
NANO = "nvidia/Cosmos3-Nano-Policy-DROID"


def test_example_surface_stays_product_shaped():
    # The measure clients and the stock launcher are LOAD-BEARING: they are the byte-identical
    # protocol behind the README rows, and reproduce_h100.sh drives them. Pinned so a cleanup
    # can never silently delete or replace them.
    root_files = {path.name for path in PLUGIN_ROOT.iterdir() if path.is_file()}
    assert root_files == {
        "README.md",
        "launch_robolab_stock.py",
        "measure_openpi_ws.py",
        "measure_predict.py",
        "pyproject.toml",
        "reproduce_h100.sh",
    }, root_files
    assert os.access(PLUGIN_ROOT / "reproduce_h100.sh", os.X_OK)


def test_spec_declares_a_stateless_four_step_policy():
    spec = Cosmos3PolicyAdapter().spec()
    assert spec.model_id == EDGE
    assert spec.streams == ()                       # no KV survives a request, by upstream design
    assert spec.phase("prefix").nfe == 1
    assert spec.phase("action").nfe == 4
    assert spec.phase("action").truncatable is True
    # guidance 1.0 at the published operating point: no negative branch
    assert spec.guidance["action"].mode is GuidanceMode.NONE
    assert spec.shapes_static_across_cycles()[0] is True
    assert spec.observation.conditioning == ("prompt",)


def test_both_known_releases_share_the_adapter_and_the_measured_serving_config():
    edge, nano = lookup(EDGE), lookup(NANO)
    assert edge is not None and nano is not None
    for doc in (edge, nano):
        ex = doc["execution"]
        assert ex["backbone"] == "cosmos3_policy"
        assert ex["nfe"] == {"prefix": 1, "action": 4}
        # the canonical policy request of the published rows, declared not guessed
        assert ex["domain_name"] == "droid_lerobot"
        assert ex["action_dim"] == 8
        assert ex["action_chunk_size"] == 16
        assert (ex["image_height"], ex["image_width"]) == (540, 640)
        for key in REQUIRED_SERVING_KEYS:
            assert key in ex, f"declaration missing {key}"
    assert nano["execution"]["param_bytes"] > edge["execution"]["param_bytes"]


def test_known_releases_get_a_declared_view_without_copying_weights():
    for repo in (EDGE, NANO):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text('{"model_type":"cosmos3"}')
            (snapshot / "model.safetensors.index.json").write_text('{"weight_map":{}}')
            previous = os.environ.get("XDG_CACHE_HOME")
            os.environ["XDG_CACHE_HOME"] = str(root / "cache")
            try:
                view = _declared_view(snapshot, repo, lookup(repo))
                report = validate_package(view)
                assert report.ok, report.explain()
                assert report.declaration.backbone == "cosmos3_policy"
            finally:
                if previous is None:
                    os.environ.pop("XDG_CACHE_HOME", None)
                else:
                    os.environ["XDG_CACHE_HOME"] = previous


def test_observation_contract_comes_from_the_declaration_and_fails_loud():
    adapter = Cosmos3PolicyAdapter()
    doc = lookup(EDGE)["execution"]
    extra = {k: doc[k] for k in REQUIRED_SERVING_KEYS}
    ckpt = SimpleNamespace(path="x", execution=SimpleNamespace(model_id=EDGE, extra=extra))
    obs, source = adapter.observation_contract(ckpt)
    assert "declared" in source
    shapes = {f.key: f.shape for f in obs.fields}
    assert shapes["image"] == (540, 640, 3)
    assert shapes["state"] == (8,)
    # a declaration missing serving-config keys is refused with the keys named
    bare = SimpleNamespace(path="x", execution=SimpleNamespace(model_id="x/y", extra={}))
    try:
        adapter.observation_contract(bare)
    except RuntimeError as e:
        assert "domain_name" in str(e) and "action_dim" in str(e)
    else:
        raise AssertionError("a missing serving config must be refused, not defaulted")


def test_runtime_loop_builds_the_canonical_request_and_returns_numpy():
    class FakeService:
        def __init__(self):
            self.req = None
            self.episodes = 0

        def notify_next_episode(self, payload=None):
            self.episodes += 1
            return {"status": "ok"}

        def predict(self, req):
            self.req = req
            return {"action": [[0.0] * 8] * 16, "timing": {"total_ms": 1.0}}

    service = FakeService()
    loop = _Cosmos3PolicyLoop(service)
    loop.reset(prompt="pick up the banana and place it in the bowl")
    assert service.episodes == 1
    image = np.zeros((540, 640, 3), np.uint8)
    out = loop.predict({"image": image, "state": np.zeros(8, np.float32)})
    assert out["action"].shape == (16, 8)
    assert isinstance(service.req["image"], str)          # PNG base64, the measured channel
    assert service.req["prompt"].startswith("pick up")
    assert service.req["state"] == [0.0] * 8
    # a per-call prompt overrides; a missing prompt is refused
    loop.predict({"image": image, "state": [0.0] * 8, "prompt": "open the drawer"})
    assert service.req["prompt"] == "open the drawer"
    loop2 = _Cosmos3PolicyLoop(FakeService())
    try:
        loop2.predict({"image": image, "state": [0.0] * 8})
    except ValueError as e:
        assert "prompt" in str(e)
    else:
        raise AssertionError("a promptless request must be refused")


def test_png_encoding_is_lossless_and_strict():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(8, 6, 3), dtype=np.uint8)
    encoded = _encode_png_b64(image)

    import base64
    import io

    from PIL import Image

    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))))
    assert (decoded == image).all()
    try:
        _encode_png_b64(image.astype(np.float32))
    except ValueError as e:
        assert "uint8" in str(e)
    else:
        raise AssertionError("non-uint8 pixels must be refused, not silently cast")


def test_model_path_resolves_local_layouts_and_fails_loud():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        good = root / "ckpt"
        good.mkdir()
        (good / "config.json").write_text("{}")
        (good / "model.safetensors.index.json").write_text('{"weight_map":{}}')
        ckpt = SimpleNamespace(path=str(good), model_id=EDGE,
                               execution=SimpleNamespace(extra={}))
        assert _resolve_model_path(ckpt) == good.resolve()
        empty = root / "empty"
        empty.mkdir()
        bad = SimpleNamespace(path=str(empty), model_id=EDGE,
                              execution=SimpleNamespace(extra={"base_weights": str(empty)}))
        try:
            _resolve_model_path(bad)
        except RuntimeError as e:
            assert "checkpoint" in str(e)
        else:
            raise AssertionError("an empty checkpoint dir must be refused")


def test_gpu_smoke_one_predict_finite_actions():
    """Load -> one predict -> finite actions, through the public Runtime API.

    Opt-in (IFL_GPU_SMOKE=1): loads gigabytes onto whatever GPU is visible; run it serialized
    on one idle device from the patched cosmos-framework venv:

        IFL_GPU_SMOKE=1 CUDA_VISIBLE_DEVICES=7 \
          /home/ubuntu/cosmos-framework/.venv/bin/python tests/test_cosmos3_policy_adapter.py

    IFL_COSMOS3_SMOKE_CKPT selects Edge (default) or the Nano repo id. Every missing
    precondition is a SKIP, never a failure, so public CI stays green.
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
    adapter = Cosmos3PolicyAdapter()
    ok, why = adapter.can_host_in_process()
    if not ok:
        print(f"SKIP: {why}")
        return
    repo = os.environ.get("IFL_COSMOS3_SMOKE_CKPT", EDGE)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
    except Exception:  # noqa: BLE001
        print(f"SKIP: {repo} is not in the local HF cache")
        return

    from instinctflash import Runtime
    from instinctflash.runtime.loader import available_models, register

    if "cosmos3_policy" not in available_models():
        register("cosmos3_policy", Cosmos3PolicyAdapter)
    runtime = Runtime.from_pretrained(repo, placement="in_process")
    try:
        obs = runtime.observation.example()
        with runtime.episode(prompt="pick up the banana and place it in the bowl") as ep:
            out = ep.predict(obs)
        action = np.asarray(out["action"])
        assert action.shape == (16, 8), action.shape
        assert np.isfinite(action).all(), "non-finite action from smoke"
        print(f"smoke OK: {repo} -> action {action.shape}, finite")
    finally:
        runtime.close()


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))

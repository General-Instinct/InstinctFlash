"""CPU controller contracts and parity with pinned upstream method ASTs."""
from __future__ import annotations

import ast
import builtins
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from instinctflash.runtime.step_cache import StepCacheConfig, StepCacheController


@pytest.fixture(autouse=True)
def inference_context():
    with torch.no_grad():
        yield


def prediction(value=1.0, *, dtype=torch.float32):
    return {"video": torch.full((1, 2, 4), value, dtype=dtype),
            "action": torch.ones((1, 3, 2), dtype=dtype)}


def run_generation(controller, predictions):
    controller.begin_generation(len(predictions))
    for index, outputs in enumerate(predictions):
        if controller.decide(index):
            controller.record_prediction(outputs)
        else:
            controller.reuse()
    return controller.end_generation()


def test_config_is_frozen_serializable_and_torch_lazy(monkeypatch):
    original = builtins.__import__

    def forbid_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Config/module import attempted to import Torch")
        return original(name, *args, **kwargs)

    path = Path(__file__).parents[1] / "instinctflash/runtime/step_cache.py"
    namespace = {"__name__": "instinctflash.runtime.step_cache"}
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", forbid_torch)
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        config = namespace["StepCacheConfig"](thresholds=[0.95, 0.93], skip_counts=[4, 2])
        assert config.thresholds == (0.95, 0.93)
        assert config.to_dict()["skip_counts"] == [4, 2]
        namespace["StepCacheController"](config).report()
    with pytest.raises(FrozenInstanceError):
        config.max_bytes = 1
    controller = StepCacheController()
    with pytest.raises(AttributeError):
        controller.config = StepCacheConfig(thresholds=(1.0,), skip_counts=(1,))


@pytest.mark.parametrize("kwargs", [
    {"profile": "unknown"}, {"thresholds": ()}, {"thresholds": (0.95,)},
    {"thresholds": (0.9, 0.95)}, {"thresholds": (0.95, 0.95)},
    {"thresholds": (float("nan"), 0.9)}, {"thresholds": (float("inf"), 0.9)},
    {"thresholds": (1.1, 0.9)}, {"thresholds": (True, 0.9)},
    {"skip_counts": (0, 2)}, {"skip_counts": (True, 2)},
    {"skip_counts": (1025, 2)}, {"skip_counts": (4.0, 2)}, {"max_bytes": 0}, {"max_bytes": True},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        StepCacheConfig(**kwargs)


def test_exact_countdown_and_generation_reset():
    controller = StepCacheController()
    for _ in range(2):
        result = run_generation(controller, [prediction() for _ in range(16)])
        assert [i for i, value in enumerate(result["compute_mask"]) if value] == [0, 1, 6, 11]
        assert (result["computed_steps"], result["reused_steps"]) == (4, 12)
        assert result["trace"][-1]["countdown_after"] == 1
        assert all(row["consumed"] for row in result["trace"])
        assert controller.report()["history_entries"] == controller.report()["cache_bytes"] == 0
    report = controller.report()
    assert report["aggregate"]["generations_completed"] == 2
    assert report["aggregate"]["computed_steps"] == 8
    report["last_generation"]["compute_mask"].clear()
    assert len(controller.report()["last_generation"]["compute_mask"]) == 16


def test_unstable_and_strict_threshold_never_reuse():
    controller = StepCacheController()
    result = run_generation(controller, [prediction((-1.0) ** i) for i in range(16)])
    assert result["computed_steps"] == 16
    controller = StepCacheController(StepCacheConfig(thresholds=(1.0,), skip_counts=(4,)))
    outputs = prediction()
    outputs["video"].zero_()
    outputs["video"][..., 0] = 1
    result = run_generation(controller, [outputs for _ in range(16)])
    assert result["computed_steps"] == 16
    result = run_generation(controller, [prediction(0.0) for _ in range(16)])
    assert result["computed_steps"] == 16


def test_outputs_are_owned_after_input_alias_and_reuse_mutation():
    controller = StepCacheController()
    controller.begin_generation(4)
    inputs = prediction()
    inputs["action"] = inputs["video"]  # Caller aliasing cannot alias retained keys.
    for index in (0, 1):
        inputs["video"].fill_(index + 1)
        assert controller.decide(index)
        controller.record_prediction(inputs)
        inputs["video"].fill_(-100)
    assert not controller.decide(2)
    reused = controller.reuse()
    assert torch.equal(reused["video"], torch.full_like(reused["video"], 2))
    reused["video"].fill_(-9)
    assert torch.all(reused["action"] == 2)
    assert not controller.decide(3)
    again = controller.reuse()
    assert torch.all(again["video"] == 2)
    controller.end_generation()
    assert torch.all(again["video"] == 2)  # Returned ownership outlives controller history.


def test_abort_releases_history_and_allows_clean_restart():
    controller = StepCacheController()
    controller.begin_generation(16)
    assert controller.decide(0)
    controller.record_prediction(prediction())
    owned = weakref.ref(controller._history[0]["video"])
    assert controller.decide(1)
    controller.abort_generation()
    assert owned() is None
    report = controller.report()
    assert report["last_generation"]["status"] == "aborted"
    assert report["last_generation"]["pending_decision"] == "compute"
    assert report["aggregate"]["generations_aborted"] == 1
    assert report["cache_bytes"] == report["history_entries"] == 0
    assert run_generation(controller, [prediction()] * 3)["computed_steps"] == 2
    controller.begin_generation(1)
    controller.close()
    controller.close()
    assert controller.report()["closed"]
    with pytest.raises(RuntimeError, match="closed"):
        controller.begin_generation(1)


def test_call_order_and_pending_decisions():
    controller = StepCacheController()
    with pytest.raises(RuntimeError, match="No active"):
        controller.decide(0)
    controller.begin_generation(3)
    with pytest.raises(RuntimeError, match="active generation"):
        controller.begin_generation(3)
    with pytest.raises(ValueError, match="sequential"):
        controller.decide(1)
    with pytest.raises(RuntimeError, match="pending compute"):
        controller.record_prediction(prediction())
    assert controller.decide(0)
    with pytest.raises(RuntimeError, match="consumed"):
        controller.decide(1)
    with pytest.raises(RuntimeError, match="consumed"):
        controller.end_generation()
    controller.record_prediction(prediction())
    with pytest.raises(RuntimeError, match="scheduler slots"):
        controller.end_generation()
    with pytest.raises(ValueError, match="sequential"):
        controller.decide(0)
    assert controller.decide(1)
    controller.record_prediction(prediction())
    assert not controller.decide(2)
    with pytest.raises(RuntimeError, match="consumed"):
        controller.end_generation()
    controller.reuse()
    with pytest.raises(RuntimeError, match="pending reuse"):
        controller.reuse()
    with pytest.raises(ValueError, match="sequential"):
        controller.decide(3)
    controller.end_generation()


@pytest.mark.parametrize("total", [0, -1, True, 1.5, 4097])
def test_total_steps_are_bounded(total):
    with pytest.raises(ValueError):
        StepCacheController().begin_generation(total)


@pytest.mark.parametrize("change", ["shape", "dtype", "keys", "signal", "nan", "inf", "requires_grad"])
def test_invalid_predictions_do_not_replace_history(change):
    controller = StepCacheController()
    controller.begin_generation(3)
    assert controller.decide(0)
    controller.record_prediction(prediction())
    assert controller.decide(1)
    outputs, signal = prediction(), "video"
    if change == "shape":
        outputs["video"] = torch.ones((1, 3, 4))
    elif change == "dtype":
        outputs["video"] = outputs["video"].double()
    elif change == "keys":
        outputs.pop("action")
    elif change == "signal":
        signal = "action"
    elif change in ("nan", "inf"):
        outputs["action"][0, 0, 0] = float(change)
    else:
        outputs["video"].requires_grad_()
    with pytest.raises(ValueError):
        controller.record_prediction(outputs, signal_key=signal)
    assert controller.report()["history_entries"] == 1
    assert controller.report()["pending_decision"] == "compute"
    controller.abort_generation()


def test_budget_bounds_owned_history_and_cleans_on_end():
    size = sum(t.numel() * t.element_size() for t in prediction().values())
    controller = StepCacheController(StepCacheConfig(max_bytes=2 * size))
    result = run_generation(controller, [prediction((-1.0) ** i) for i in range(32)])
    assert result["peak_cache_bytes"] == 2 * size
    assert controller.report()["cache_bytes"] == 0
    controller = StepCacheController(StepCacheConfig(max_bytes=2 * size - 1))
    controller.begin_generation(2)
    assert controller.decide(0)
    with pytest.raises(ValueError, match="max_bytes"):
        controller.record_prediction(prediction())
    assert controller.report()["history_entries"] == 0
    controller.abort_generation()
    controller = StepCacheController(StepCacheConfig(max_bytes=size))
    assert run_generation(controller, [prediction()])["peak_cache_bytes"] == size


def test_grad_and_capture_refuse_before_changing_state(monkeypatch):
    controller = StepCacheController()
    with torch.enable_grad(), pytest.raises(RuntimeError, match="no_grad"):
        controller.begin_generation(2)
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_initialized", lambda: True)
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="capture"):
            controller.begin_generation(2)
    assert not controller.report()["active"]
    controller.begin_generation(1)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="no_grad"):
        controller.decide(0)
    with torch.enable_grad():
        controller.abort_generation()  # Cleanup works even after leaving inference mode.


def test_cuda_stream_contract_without_initializing_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    controller = StepCacheController()
    device, stream_id = torch.device("cuda:0"), [11]
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "device", lambda _: nullcontext())
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        patch.setattr(torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=stream_id[0]))
        controller._stream = controller._context(device)
        assert controller._stream == (0, 11)
        assert controller._context(device) == (0, 11)
        stream_id[0] = 12
        with pytest.raises(RuntimeError, match="same CUDA stream"):
            controller._context(device)
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="capture"):
            controller._context(device)
    assert not torch.cuda.is_initialized()


def _pinned_method(path, expected_hash, name):
    if not path.exists():
        pytest.skip("Pinned external source absent; no download or model import")
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_hash
    tree = ast.parse(data.decode())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def pinned_methods():
    native = _pinned_method(
        Path("/home/ubuntu/dreamzero-repo/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py"),
        "7193cd73423472aa252bee73bd80e0d673c89d773ec852e90f50154729b50845", "should_run_model")
    omni = _pinned_method(
        Path("/home/ubuntu/work_clones/vllm-omni-benchmark-20260913/vllm_omni/diffusion/cache/stepcache/state.py"),
        "32b00daaf6cd6e4ac73cf9c6f40eb3c69c46386f5fc0759d6cd6f23569b75a87", "should_run_step")
    return native, omni


@pytest.mark.parametrize("threshold", [0.93, 0.95])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_fp32_threshold_boundaries_match_upstream(monkeypatch, threshold, direction):
    native, omni = pinned_methods()
    value = torch.tensor(threshold, dtype=torch.float32)
    if direction:
        value = torch.nextafter(value, torch.tensor(float("inf") * direction))
    monkeypatch.setattr(torch.nn.functional, "cosine_similarity", lambda *args, **kwargs: value.reshape(1))
    controller = StepCacheController()
    controller.begin_generation(3)
    history = []
    for index in range(2):
        assert controller.decide(index)
        outputs = prediction()
        controller.record_prediction(outputs)
        history.append((index, outputs["video"], outputs["action"]))
    left = SimpleNamespace(dynamic_cache_schedule=True, skip_countdown=0)
    right = SimpleNamespace(config=SimpleNamespace(enabled=True, min_history_steps=2,
                                                   sim_thresholds=(0.95, 0.93), skip_countdowns=(4, 2)), skip_countdown=0)
    expected = bool(native(left, 2, 2, history))
    assert bool(omni(right, [(v,) for _, v, _ in history])) == expected
    assert controller.decide(2) == expected
    assert controller.report()["current_generation"]["trace"][-1]["countdown_after"] == left.skip_countdown == right.skip_countdown
    if expected:
        controller.record_prediction(prediction())
    else:
        controller.reuse()
    controller.end_generation()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pinned_native_and_omni_ast_parity(dtype):
    native, omni = pinned_methods()
    controller = StepCacheController()
    for case in range(160):
        rng = torch.Generator(device="cpu").manual_seed(20260914 + case)
        left = SimpleNamespace(dynamic_cache_schedule=True, skip_countdown=0)
        right = SimpleNamespace(config=SimpleNamespace(enabled=True, min_history_steps=2,
                                                       sim_thresholds=(0.95, 0.93), skip_countdowns=(4, 2)), skip_countdown=0)
        history = []
        video = torch.randn((1 + case % 3, 2, 4), generator=rng).to(dtype)
        controller.begin_generation(16)
        for index in range(16):
            expected = bool(native(left, index, index, history))
            assert expected == bool(omni(right, [(v,) for _, v, _ in history]))
            assert controller.decide(index) == expected
            current = controller.report()["current_generation"]
            assert current["trace"][-1]["countdown_after"] == left.skip_countdown == right.skip_countdown
            if expected:
                video = (video.float() + torch.randn(video.shape, generator=rng) * (0.03, 0.35, 1.0)[case % 3]).to(dtype)
                outputs = {"video": video.clone(), "action": torch.randn((video.shape[0], 3, 2), generator=rng).to(dtype)}
                controller.record_prediction(outputs)
                history.append((index, outputs["video"].clone(), outputs["action"].clone()))
                history[:] = history[-2:]
            else:
                outputs = controller.reuse()
                assert torch.equal(outputs["video"], history[-1][1])
                assert torch.equal(outputs["action"], history[-1][2])
        result = controller.end_generation()
        assert result["computed_steps"] + result["reused_steps"] == 16

"""Complete native KV transport and failure ownership, without large weights."""
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/dreamzero"))
from dreamzero_iwm import residency


class Block(torch.nn.Module):
    def forward(self, x, *, kv_cache):
        return x + kv_cache.sum().to(x.dtype), torch.cat((kv_cache, kv_cache + 1), dim=2)


def test_cpu_storage_retains_complete_history_and_native_commit(monkeypatch):
    # CPU transport exercises hook/ownership semantics; this is not a GPU gate.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    original = Block().eval()
    owned = Block().eval()
    storage = residency.CPUKVStorage([owned], device="cpu")
    cache = torch.arange(24, dtype=torch.bfloat16).reshape(2, 1, 3, 2, 2)
    expected_cache = cache.clone()
    with torch.inference_mode():
        for cycle in range(3):
            x = torch.tensor([cycle], dtype=torch.bfloat16)
            expected, updated = original(x, kv_cache=expected_cache)
            actual, returned = owned(x, kv_cache=cache)
            assert torch.equal(actual, expected)
            assert torch.equal(returned.view(torch.uint8), updated.view(torch.uint8))
            cache, expected_cache = returned.clone(), updated.clone()
    assert cache.shape[2] == 24
    assert storage.report()["forwards"] == 3
    assert storage.report()["complete_history"] is True
    storage.close()
    storage.close()
    assert not owned._forward_hooks and not owned._forward_pre_hooks


@pytest.mark.parametrize("failure", ["forward", "invalid_cache", "grad", "capture"])
def test_failed_native_call_releases_active_state(monkeypatch, failure):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: failure == "capture")

    class FailBlock(Block):
        def forward(self, x, *, kv_cache):
            if failure == "forward":
                raise ArithmeticError("original native failure")
            return super().forward(x, kv_cache=kv_cache)

    block = FailBlock().eval()
    storage = residency.CPUKVStorage([block], device="cpu")
    cache = torch.ones((2, 1, 3, 2, 2), dtype=torch.bfloat16)
    if failure == "invalid_cache":
        cache = cache.float()
    error = ArithmeticError if failure == "forward" else ValueError if failure == "invalid_cache" else RuntimeError
    with torch.set_grad_enabled(failure == "grad"), pytest.raises(error):
        block(torch.ones(1), kv_cache=cache)
    assert not storage.active
    storage.close()
    assert not block._forward_hooks and not block._forward_pre_hooks


def test_construction_scope_cleans_initialized_owner_when_later_processor_fails():
    owner = SimpleNamespace(initialized=True, close=Mock())
    with pytest.raises(LookupError, match="processor"):
        with residency.construction_scope() as owners:
            owners.append(owner)
            raise LookupError("processor missing")
    owner.close.assert_called_once()
    assert residency._CONSTRUCTION.get() is None


def test_construction_scope_rejects_missing_post_initialize():
    owner = SimpleNamespace(initialized=False, close=Mock())
    with pytest.raises(RuntimeError, match="initialized"):
        with residency.construction_scope() as owners:
            owners.append(owner)
    owner.close.assert_called_once()


def test_loop_keeps_residency_owned_for_entire_inference_call():
    from dreamzero_iwm.adapter import _DreamZeroLoop

    entered, release = threading.Event(), threading.Event()

    def infer(observation):
        entered.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test inference release")
        return [[1., 2.]]

    owner = SimpleNamespace(close=Mock())
    loop = _DreamZeroLoop(SimpleNamespace(infer=infer, reset=Mock()),
                         dynamic_cache=False, residency=owner)
    failures = []

    def predict():
        try:
            loop.predict({'prompt': 'test'})
        except Exception as error:
            failures.append(error)

    worker = threading.Thread(target=predict)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        # This is between native blocks, with no layer-level active marker.
        with pytest.raises(RuntimeError, match="serial"):
            loop.close()
        with pytest.raises(RuntimeError, match="serial"):
            loop.reset(prompt='new episode')
        owner.close.assert_not_called()
    finally:
        release.set()
        worker.join(timeout=10)
    assert not worker.is_alive() and not failures
    loop.close()
    owner.close.assert_called_once()


@pytest.mark.parametrize("capability,memory,expected", [
    ((8, 9), 24 << 30, True), ((8, 9), 48 << 30, False),
    ((11, 0), 128 << 30, False), ((12, 0), 32 << 30, False),
])
def test_residency_is_specific_to_small_sm89_devices(monkeypatch, capability, memory, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=memory))
    assert residency.use_sm89_residency() is expected


@pytest.mark.parametrize("capability,memory,expected", [
    ((8, 9), 24 << 30, True), ((12, 0), 32 << 30, True),
    ((12, 0), 48 << 30, False), ((11, 0), 128 << 30, False),
    ((9, 0), 80 << 30, False),
])
def test_desktop_residency_requires_explicit_capability_and_small_device(monkeypatch, capability, memory, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=memory))
    assert residency.use_desktop_residency() is expected


@pytest.mark.parametrize("builder", ["build_sm89_fp8", "build_sm120_fp8"])
def test_fp8_owned_builder_requires_permission_before_loading(monkeypatch, builder):
    from dreamzero_iwm.adapter import DreamZeroAdapter

    from instinctflash.planners.planner import Plan, Tier

    monkeypatch.delenv("LOAD_TRT_ENGINE", raising=False)
    monkeypatch.setenv("ENABLE_TENSORRT", "false")
    adapter = DreamZeroAdapter()
    build = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr(adapter, "_build_native", build)
    with pytest.raises(ValueError, match="explicit executor"):
        getattr(adapter, builder)(None, Plan("dreamzero", [], tier_ceiling=Tier.NUMERIC))
    build.assert_not_called()


@pytest.mark.skipif(os.environ.get("IFL_TEST_CUDA_RESIDENCY") != "1",
                    reason="explicitly opt in on a reserved CUDA device")
def test_cuda_cache_transport_matches_original_full_history():
    original, streamed = Block().cuda().eval(), Block().cuda().eval()
    storage = residency.CPUKVStorage([streamed])
    native_cache = torch.arange(24, dtype=torch.bfloat16, device="cuda").reshape(2, 1, 3, 2, 2)
    cpu_cache = native_cache.cpu()
    try:
        with torch.inference_mode():
            for cycle in range(4):
                x = torch.tensor([cycle], dtype=torch.bfloat16, device="cuda")
                expected, native_cache = original(x, kv_cache=native_cache)
                actual, cpu_cache = streamed(x, kv_cache=cpu_cache)
                assert cpu_cache.device.type == "cpu"
                assert torch.equal(actual, expected)
                assert torch.equal(cpu_cache.view(torch.uint8), native_cache.cpu().view(torch.uint8))
                native_cache, cpu_cache = native_cache.clone(), cpu_cache.clone()
    finally:
        storage.close()

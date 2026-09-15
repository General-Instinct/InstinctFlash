"""Lifecycle and fail-closed gates for Cosmos3 persistent prompt K/V."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "cosmos3_policy"
sys.path.insert(0, str(PLUGIN))

from cosmos3_iwm.adapter import _Cosmos3PolicyLoop
from cosmos3_iwm.persistent_text_kv import (
    PersistentTextKV,
    install_persistent_text_kv,
)


class FakeLayerCache:
    def __init__(self):
        self.is_initialized = False


class FakeModel:
    input_caption_key = "ai_caption"
    config = SimpleNamespace(
        joint_attn_implementation="two_way",
        video_temporal_causal=False,
        sound_gen=False,
    )
    parallel_dims = None

    def __init__(self):
        self.net = SimpleNamespace(
            language_model=SimpleNamespace(model=SimpleNamespace(layers=[0, 1, 2]))
        )
        self.make_calls = 0
        self.fail_next = False
        self.skip_cache = False

    def _can_reuse_inference_text_kv(self, *args, **kwargs):
        return True

    def _make_inference_text_kv_cache(self, net=None):
        self.make_calls += 1
        return [FakeLayerCache() for _ in range(3)]

    def generate_samples_from_batch(self, data_batch, **kwargs):
        if self.skip_cache:
            return {"caption": tuple(data_batch[self.input_caption_key])}
        caches = self._make_inference_text_kv_cache()
        caches[0].is_initialized = True
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("synthetic mid-request failure")
        for cache in caches:
            cache.is_initialized = True
        return {"caption": tuple(data_batch[self.input_caption_key])}


def service(guidance=1.0):
    return SimpleNamespace(model=FakeModel(), cfg=SimpleNamespace(guidance=guidance))


def request(prompt):
    return {"ai_caption": [prompt]}


def test_same_prompt_hits_and_changed_prompt_misses():
    target = service()
    cache = install_persistent_text_kv(target, max_prompts=4)
    assert target.model.generate_samples_from_batch(request("a"), guidance=1.0)
    assert target.model.generate_samples_from_batch(request("a"), guidance=1.0)
    assert target.model.generate_samples_from_batch(request("b"), guidance=1.0)
    assert cache.stats == {"misses": 2, "hits": 1}
    assert target.model.make_calls == 2
    assert list(cache.caches) == [("a",), ("b",)]


def test_failed_request_never_publishes_partial_cache():
    target = service()
    cache = install_persistent_text_kv(target)
    target.model.fail_next = True
    try:
        target.model.generate_samples_from_batch(request("a"), guidance=1.0)
    except RuntimeError as error:
        assert "synthetic" in str(error)
    else:
        raise AssertionError("synthetic request should fail")
    assert not cache.caches
    target.model.generate_samples_from_batch(request("a"), guidance=1.0)
    assert cache.stats["misses"] == 1
    assert target.model.make_calls == 2


def test_lru_is_bounded_and_recent_hit_survives():
    target = service()
    cache = install_persistent_text_kv(target, max_prompts=2)
    for prompt in ("a", "b", "a", "c"):
        target.model.generate_samples_from_batch(request(prompt), guidance=1.0)
    assert list(cache.caches) == [("a",), ("c",)]
    assert cache.stats["evictions"] == 1


def test_noncertified_modes_bypass_without_persisting():
    target = service()
    cache = install_persistent_text_kv(target)
    target.model.generate_samples_from_batch(request("a"), guidance=2.0)
    target.model.generate_samples_from_batch(
        request("a"), guidance=1.0, upsample_task="i2v"
    )
    assert not cache.caches
    assert cache.stats["bypass"] == 2


def test_upstream_predicate_nonuse_returns_original_result():
    target = service()
    cache = install_persistent_text_kv(target)
    target.model.skip_cache = True
    result = target.model.generate_samples_from_batch(request("a"), guidance=1.0)
    assert result == {"caption": ("a",)}
    assert not cache.caches
    assert cache.stats["predicate_bypass"] == 1


def test_install_guards_and_duplicate_install():
    try:
        install_persistent_text_kv(service(guidance=2.0))
    except RuntimeError as error:
        assert "guidance=1.0" in str(error)
    else:
        raise AssertionError("noncertified guidance installed")
    target = service()
    install_persistent_text_kv(target)
    try:
        install_persistent_text_kv(target)
    except RuntimeError as error:
        assert "already installed" in str(error)
    else:
        raise AssertionError("duplicate install succeeded")


def test_constructor_rejects_incompatible_attention_and_parallelism():
    model = FakeModel()
    model.config = SimpleNamespace(
        joint_attn_implementation="three_way",
        video_temporal_causal=False,
        sound_gen=False,
    )
    try:
        PersistentTextKV(model)
    except RuntimeError as error:
        assert "two_way" in str(error)
    else:
        raise AssertionError("three_way attention installed")


def test_official_robolab_loop_maps_the_declared_droid_state():
    import numpy as np

    class OfficialService:
        _ifl_official_robolab = True

        def __init__(self):
            import threading
            self._lock = threading.Lock()
            self.cfg = SimpleNamespace(seed=0)
            self.request = None

        def infer(self, request):
            self.request = request
            return {"action": np.zeros((16, 8), dtype=np.float32)}

    service = OfficialService()
    loop = _Cosmos3PolicyLoop(service)
    loop.reset(prompt="pick")
    state = np.arange(8, dtype=np.float32)
    output = loop.predict(
        {"image": np.zeros((540, 640, 3), dtype=np.uint8), "state": state}
    )
    assert output["action"].shape == (16, 8)
    assert service.request["observation/joint_position"].shape == (7,)
    assert service.request["observation/gripper_position"].shape == (1,)
    assert np.array_equal(service.request["observation/joint_position"], state[:7])
    assert service.request["observation/gripper_position"][0] == state[7]


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))

"""Weight-free regressions for pi0.5's replay-safe static denoiser."""
from __future__ import annotations

import contextlib
import io
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "pi05_vla"
sys.path.insert(0, str(PLUGIN))

from pi05_iwm.static_capture import (  # noqa: E402
    _StaticKV,
    StaticDenoiser,
    install_static_capture,
)
from pi05_iwm.surface import Pi05Surface  # noqa: E402


def _fake_lerobot_modules():
    modules = {name: types.ModuleType(name) for name in (
        "lerobot", "lerobot.policies", "lerobot.policies.pi05",
        "lerobot.policies.pi05.modeling_pi05",
    )}
    modeling = modules["lerobot.policies.pi05.modeling_pi05"]
    modeling.make_att_2d_masks = lambda pad, _att: pad[:, None, :].expand(
        pad.shape[0], pad.shape[1], pad.shape[1]
    )
    modeling.prepare_attention_masks_4d = lambda mask: mask[:, None, :, :]
    return modules


def _kv(value):
    key = torch.full((1, 1, 4, 2), value)
    return [(key, -key, None)]


def test_prompt_mask_refresh_keeps_graph_addresses():
    with patch.dict(sys.modules, _fake_lerobot_modules()):
        den = StaticDenoiser(SimpleNamespace(config=SimpleNamespace(chunk_size=2)), step_tables=False)
        den._begin_chunk(torch.tensor([[True, True, True, True]]), _kv(1.0))
        mask, positions, cache_positions = den._const
        pointers = tuple(t.data_ptr() for t in den._const)
        old_mask, old_positions = mask.clone(), positions.clone()
        den._begin_chunk(torch.tensor([[True, True, False, False]]), _kv(2.0))
        assert tuple(t.data_ptr() for t in den._const) == pointers
        assert not torch.equal(mask, old_mask)
        assert torch.equal(old_positions, torch.tensor([[4, 5]]))
        assert torch.equal(positions, torch.tensor([[2, 3]]))
        assert torch.equal(cache_positions, torch.tensor([4, 5]))
        assert torch.equal(den._kv.k[0][:, :, :4], torch.full((1, 1, 4, 2), 2.0))


def test_install_is_instance_scoped_and_idempotent():
    class Model:
        pass
    first_model, second_model = Model(), Model()
    first = install_static_capture(first_model, step_tables=False)
    assert install_static_capture(first_model, step_tables=True) is first
    second = install_static_capture(second_model, step_tables=False)
    assert second is not first
    assert first_model.denoise_step.__self__ is first_model
    assert second_model.denoise_step.__self__ is second_model


def test_surface_hoist_does_not_patch_other_model_instances():
    class Model:
        def denoise_step(self, **_kwargs):
            return None
        def embed_suffix(self, *_args):
            return None
    def model():
        item = Model()
        item.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(model=SimpleNamespace(
                language_model=SimpleNamespace(config=SimpleNamespace()))),
            gemma_expert=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace())),
        )
        return item
    one, two = model(), model()
    original = Model.embed_suffix
    Pi05Surface(one).hoist_loop_constants()
    assert Model.embed_suffix is original
    assert "embed_suffix" in one.__dict__
    assert "embed_suffix" not in two.__dict__


class _UpstreamLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _UpstreamCache:
    """Transformers-main-shaped cache: storage is exposed through ``layers``."""

    def __init__(self, entries):
        self.layers = [_UpstreamLayer(k, v) for k, v in entries]


def _expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_full_chunk_and_prefix_graph_mode_contract():
    model = SimpleNamespace(config=SimpleNamespace(chunk_size=2))
    _expect_raises(
        ValueError, StaticDenoiser, model,
        step_tables=False, full_chunk=False, prefix_graph=True)
    baked = StaticDenoiser(
        model, step_tables=True, full_chunk=True, prefix_graph=False)
    assert baked._step_tables and baked._full_chunk

    class Installable:
        config = SimpleNamespace(
            chunk_size=2, max_action_dim=3, num_inference_steps=10)

        def sample_actions(self, *_args, **_kwargs):
            return "upstream"

    target = Installable()
    with patch.dict(os.environ, {
        "IFL_PI05_FULL_CHUNK_GRAPH": "1",
        "IFL_PI05_PREFIX_GRAPH": "1",
    }):
        driver = install_static_capture(target)
    assert driver._full_chunk
    assert driver._prefix_graph_enabled
    assert not driver._step_tables
    _expect_raises(
        RuntimeError, install_static_capture, target,
        step_tables=False, full_chunk=False, prefix_graph=False)

    baked_target = Installable()
    with patch.dict(os.environ, {
        "IFL_PI05_FULL_CHUNK_GRAPH": "1",
        "IFL_PI05_PREFIX_GRAPH": "0",
        "IFL_PI05_FULL_STEP_TABLES": "1",
    }):
        baked_driver = install_static_capture(baked_target)
    assert baked_driver._full_chunk and baked_driver._step_tables
    assert not baked_driver._prefix_graph_enabled
    _expect_raises(
        RuntimeError, install_static_capture, baked_target,
        step_tables=False, full_chunk=True, prefix_graph=False)


def test_full_chunk_wrapper_falls_back_for_dynamic_schedule_and_rtc():
    class Model:
        config = SimpleNamespace(
            chunk_size=2, max_action_dim=3, num_inference_steps=10)

        def __init__(self):
            self.calls = []
            self.rtc = False

        def _rtc_enabled(self):
            return self.rtc

        def sample_actions(self, *_args, num_steps=None, **kwargs):
            self.calls.append((num_steps, kwargs))
            return num_steps, kwargs

    model = Model()
    install_static_capture(
        model, step_tables=False, full_chunk=True, prefix_graph=True)
    inputs = ([], [], torch.zeros(1, 2, dtype=torch.long),
              torch.ones(1, 2, dtype=torch.bool))

    assert model.sample_actions(*inputs, num_steps=4)[0] == 4
    # Zero is an explicit dynamic schedule, not the configured default.
    assert model.sample_actions(*inputs, num_steps=0)[0] == 0
    model.rtc = True
    _, kwargs = model.sample_actions(*inputs, num_steps=10, inference_delay=1)
    assert kwargs["inference_delay"] == 1
    assert [steps for steps, _ in model.calls] == [4, 0, 10]


def test_baked_full_step_tables_bind_each_step_then_restore_modules():
    first_real = torch.nn.Linear(1, 1)
    second_real = torch.nn.Linear(1, 1)
    first_norm = SimpleNamespace(dense=first_real)
    second_norm = SimpleNamespace(dense=second_real)
    driver = StaticDenoiser(
        SimpleNamespace(config=SimpleNamespace(chunk_size=2)),
        step_tables=True,
        full_chunk=True,
    )
    driver._denses = [(first_norm, first_real), (second_norm, second_real)]
    driver._chunk_times = [torch.tensor([1.0]), torch.tensor([0.5])]
    constants = [
        (torch.tensor([[10.0]]), [torch.tensor([[11.0]]), torch.tensor([[12.0]])]),
        (torch.tensor([[20.0]]), [torch.tensor([[21.0]]), torch.tensor([[22.0]])]),
    ]
    driver._table = {
        round(float(timestep[0]), 9): value
        for timestep, value in zip(driver._chunk_times, constants, strict=True)
    }
    saved_cond = torch.tensor([[30.0]])
    saved_dense = [torch.tensor([[31.0]]), torch.tensor([[32.0]])]
    driver._adarms_buf, driver._dense_bufs = saved_cond, saved_dense

    schedule, saved = driver._enter_baked_full_step_tables()
    assert driver._tabled_active
    assert first_norm.dense.buf is constants[0][1][0]
    assert second_norm.dense.buf is constants[0][1][1]
    driver._select_baked_full_step(schedule[1])
    assert driver._adarms_buf is constants[1][0]
    assert first_norm.dense.buf is constants[1][1][0]
    assert second_norm.dense.buf is constants[1][1][1]

    driver._exit_baked_full_step_tables(saved)
    assert not driver._tabled_active
    assert first_norm.dense is first_real and second_norm.dense is second_real
    assert driver._adarms_buf is saved_cond and driver._dense_bufs is saved_dense


def test_full_sample_self_check_reports_exact_unseen_inputs():
    callback = []
    model = SimpleNamespace(config=SimpleNamespace(chunk_size=2))
    driver = StaticDenoiser(
        model,
        step_tables=False,
        orig_denoise=object(),
        on_self_check=callback.append,
        self_check_inputs=2,
        full_chunk=True,
        prefix_graph=True,
    )
    driver._original_sample_actions = object()
    driver._eager_sample_actions = lambda *_args: _args[6] * 2
    driver.run_prefix = lambda *_args: (None, None)
    driver.run_full_chunk = lambda _ppm, _kv, noise, _steps: noise * 2
    inputs = (
        [torch.arange(4.0).reshape(1, 1, 2, 2)],
        [torch.ones(1, dtype=torch.bool)],
        torch.tensor([[2, 3, 4]]),
        torch.tensor([[True, True, False]]),
        None,
        None,
    )
    with patch.object(torch.cuda, "synchronize", lambda: None):
        assert driver._full_self_check(
            *inputs, torch.ones(1, 2, 3), 10, {})
    assert driver._full_self_checked
    assert driver.self_check["scope"] == "sample_actions"
    assert driver.self_check["max_abs_delta"] == 0
    assert [case["prefix"] for case in driver.self_check["cases"]] == [
        "captured-chunk", "refilled"]
    assert callback[0]["bitexact"] is True


def test_full_rejection_releases_both_graphs_and_restores_upstream():
    class Model:
        config = SimpleNamespace(
            chunk_size=2, max_action_dim=3, num_inference_steps=10)

        def denoise_step(self, **_kwargs):
            return "eager-step"

        def sample_actions(self, *_args, **_kwargs):
            return "eager-sample"

    model = Model()
    upstream_sample = model.sample_actions
    driver = install_static_capture(
        model, step_tables=False, full_chunk=True, prefix_graph=True)
    driver._graph = driver._prefix_graph = driver._chunk_graph = object()
    driver._out = driver._prefix_kv = driver._chunk_final = object()
    with contextlib.redirect_stderr(io.StringIO()):
        driver._release_and_fall_back(0.25)
    assert driver.rejected
    assert driver._graph is driver._prefix_graph is driver._chunk_graph is None
    assert driver._out is driver._prefix_kv is driver._chunk_final is None
    assert model.sample_actions.__func__ is upstream_sample.__func__
    assert model.sample_actions([], [], None, None) == "eager-sample"


def test_static_kv_matches_transformers_main_cache_protocol():
    prefix = _UpstreamCache([
        (torch.full((1, 1, 4, 2), 1.0), torch.full((1, 1, 4, 2), -1.0)),
        (torch.full((1, 1, 4, 2), 2.0), torch.full((1, 1, 4, 2), -2.0)),
    ])
    cache = _StaticKV(prefix, suffix_len=2)
    pointers = tuple(t.data_ptr() for pair in zip(cache.k, cache.v) for t in pair)

    assert len(cache) == 2
    assert cache.get_seq_length() == 4
    assert cache.get_max_cache_shape() == 6
    assert cache.get_mask_sizes(query_length=2, layer_idx=0) == (6, 0)
    assert cache.max_batch_size == 1
    assert cache.max_cache_len == 6
    assert cache.is_compileable and cache.is_initialized
    assert cache.is_sliding == [False, False]

    suffix_k = torch.full((1, 1, 2, 2), 7.0)
    suffix_v = -suffix_k
    K, V = cache.update(
        suffix_k, suffix_v, 0,
        cache_position=torch.tensor([4, 5]), arbitrary_future_kwarg=True)
    assert torch.equal(K[:, :, :4], torch.full((1, 1, 4, 2), 1.0))
    assert torch.equal(V[:, :, :4], torch.full((1, 1, 4, 2), -1.0))
    assert torch.equal(K[:, :, 4:], suffix_k)
    assert torch.equal(V[:, :, 4:], suffix_v)

    refill = _UpstreamCache([
        (torch.full((1, 1, 4, 2), 3.0), torch.full((1, 1, 4, 2), -3.0)),
        (torch.full((1, 1, 4, 2), 4.0), torch.full((1, 1, 4, 2), -4.0)),
    ])
    cache.refill(refill)
    assert tuple(t.data_ptr() for pair in zip(cache.k, cache.v) for t in pair) == pointers
    assert torch.equal(cache.k[0][:, :, :4], torch.full((1, 1, 4, 2), 3.0))
    assert len(list(cache)) == 2


def test_static_kv_refuses_shape_and_layer_drift():
    prefix = _UpstreamCache([
        (torch.zeros(1, 1, 4, 2), torch.zeros(1, 1, 4, 2)),
    ])
    cache = _StaticKV(prefix, suffix_len=2)

    _expect_raises(ValueError, cache.refill, _UpstreamCache([]))
    _expect_raises(
        ValueError, cache.refill,
        _UpstreamCache([(torch.zeros(1, 1, 5, 2), torch.zeros(1, 1, 5, 2))]))
    _expect_raises(
        ValueError, cache.update,
        torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 3, 2), 0)
    _expect_raises(
        IndexError, cache.update,
        torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), 1)


if __name__ == "__main__":
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))


def test_full_capture_exception_restores_eager_with_same_noise():
    import pytest
    class Model:
        config = SimpleNamespace(num_inference_steps=2, chunk_size=2, max_action_dim=3)
        def denoise_step(self, *args, **kwargs):
            return None
        def sample_actions(self, images, img_masks, tokens, masks, noise=None, **kwargs):
            return noise
    model = Model()
    original = model.sample_actions
    driver = install_static_capture(model, step_tables=False, full_chunk=True)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic capture failure")
    driver.run_prefix = fail
    noise = torch.randn(1, 2, 3)
    with pytest.warns(RuntimeWarning, match="synthetic capture failure"):
        output = model.sample_actions([], [], torch.ones(1, 2), torch.ones(1, 2), noise=noise)
    assert output is noise
    assert driver.rejected
    assert model.sample_actions == original


def test_full_wrapper_preserves_legacy_positional_noise():
    class Legacy:
        config = SimpleNamespace(chunk_size=2, max_action_dim=3, num_inference_steps=10)
        def denoise_step(self, **kwargs):
            return None
        def sample_actions(self, images, img_masks, tokens, masks, noise=None,
                           num_steps=None, **kwargs):
            return noise
    model = Legacy()
    driver = install_static_capture(model, full_chunk=True, step_tables=False)
    # RTC arguments deliberately select the fallback, before touching CUDA.
    noise = torch.ones(1, 2, 3)
    assert model.sample_actions([], [], torch.ones(1, 2), None, noise,
                                inference_delay=1) is noise


def test_forward_prefix_accepts_legacy_four_argument_embedding():
    class Legacy:
        config = SimpleNamespace(chunk_size=2)
        def embed_prefix(self, images, masks, tokens, token_mask):
            return torch.ones(1, 2, 3), token_mask, torch.zeros_like(token_mask)
    model = Legacy()
    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace()))),
        forward=lambda **kwargs: (None, "cache"))
    driver = StaticDenoiser(model, step_tables=False)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with patch.dict(sys.modules, _fake_lerobot_modules()):
        actual_mask, cache = driver.run_prefix([], [], torch.ones(1, 2), mask)
    assert actual_mask is mask and cache == "cache"


def test_full_wrapper_does_not_drop_new_upstream_controls():
    class Future:
        config = SimpleNamespace(chunk_size=2, max_action_dim=3, num_inference_steps=10)
        def denoise_step(self, **kwargs):
            return None
        def sample_actions(self, images, img_masks, tokens, masks, noise=None,
                           num_steps=None, new_control=None, **options):
            return new_control, options
    model = Future()
    install_static_capture(model, full_chunk=True, step_tables=False)
    assert model.sample_actions([], [], None, None, new_control='keep') == ('keep', {})
    assert model.sample_actions([], [], None, None, another_control='keep') == (None, {'another_control': 'keep'})


def test_legacy_fallback_without_variadic_kwargs_remains_callable():
    class Legacy:
        config = SimpleNamespace(chunk_size=2, max_action_dim=3, num_inference_steps=10)
        def denoise_step(self, **kwargs):
            return None
        def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=None):
            return noise
    model = Legacy()
    driver = install_static_capture(model, full_chunk=True, step_tables=False)
    driver.rejected = True
    noise = torch.ones(1, 2, 3)
    assert model.sample_actions([], [], None, None, noise=noise) is noise

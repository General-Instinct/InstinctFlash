"""Native and FP8 builders share an owned checkpoint view and native processors."""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/dreamzero'))
from dreamzero_iwm.adapter import _build_owned_native_loop, SHIPPED_DIT_MASK
from instinctflash.runtime import dreamzero_checkpoint as loading


@pytest.fixture
def setup(tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text(json.dumps({'action_head_cfg': {'config': {
        'train_architecture': 'full', 'diffusion_model_cfg': {'_target_': loading.DIT_TARGET}}}}))
    module = ModuleType(loading.DIT_TARGET.rsplit('.', 1)[0])
    module.CausalWanModel = object
    monkeypatch.setitem(sys.modules, module.__name__, module)
    view = SimpleNamespace(name=str(tmp_path / 'owned-view'), cleanup=Mock())
    prepare = Mock(return_value=(view, {'verified_dit_tensors': 1317, 'expected_shapes': {'q': [2, 2]}}))
    monkeypatch.setattr(loading, 'prepare_full_checkpoint', prepare)
    head = SimpleNamespace(num_inference_steps=16, cfg_scale=5.,
                           dynamic_cache_schedule=False, dit_step_mask=SHIPPED_DIT_MASK)
    policy = SimpleNamespace(trained_model=SimpleNamespace(action_head=head))
    return tmp_path, view, prepare, policy


def test_owned_view_lives_until_close_and_receipt_cannot_be_mutated(setup):
    root, view, prepare, policy = setup
    factory = Mock(return_value=policy)
    wrapper = Mock(return_value=SimpleNamespace())
    loop = _build_owned_native_loop(root, factory, wrapper)
    factory.assert_called_once_with(Path(view.name))
    wrapper.assert_called_once_with(policy)
    view.cleanup.assert_not_called()
    assert loop.declaration()['precision'] == 'native'
    assert loop.backend_stats['loading']['initialization_rng_equivalent'] is False
    assert 'expected_shapes' not in loop.backend_stats['loading']
    loop.backend_stats['loading']['verified_dit_tensors'] = 0
    assert loop.backend_stats['loading']['verified_dit_tensors'] == 1317
    loop.close()
    loop.close()
    view.cleanup.assert_called_once()


@pytest.mark.parametrize('stage', ['policy', 'wrapper'])
def test_construction_error_releases_view(setup, stage):
    root, view, prepare, policy = setup
    factory = Mock(return_value=policy)
    wrapper = Mock(return_value=SimpleNamespace())
    (factory if stage == 'policy' else wrapper).side_effect = RuntimeError('build failed')
    with pytest.raises(RuntimeError, match='build failed'):
        _build_owned_native_loop(root, factory, wrapper)
    view.cleanup.assert_called_once()


def test_lora_retains_original_native_loading(setup):
    root, view, prepare, policy = setup
    config = json.loads((root / 'config.json').read_text())
    config['action_head_cfg']['config']['train_architecture'] = 'lora'
    (root / 'config.json').write_text(json.dumps(config))
    factory = Mock(return_value=policy)
    loop = _build_owned_native_loop(root, factory, lambda _: SimpleNamespace())
    prepare.assert_not_called()
    factory.assert_called_once_with(root)
    assert loop.backend_stats['loading'] is None
    loop.close()
    view.cleanup.assert_not_called()


def test_resolved_default_lora_view_preserves_original_and_freezes_schedule(tmp_path):
    from instinctflash.runtime.step_cache_policy import ResolvedStepCache

    config = {'action_head_cfg': {
        '_target_': 'groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead',
        'config': {'train_architecture': 'lora', 'lora_rank': 4}}}
    original = json.dumps(config).encode()
    (tmp_path / 'config.json').write_bytes(original)
    (tmp_path / 'model.safetensors').write_bytes(b'fixture-weight-bytes')
    observed = {}

    def load_owned(path):
        observed['path'] = path
        observed['config'] = json.loads((path / 'config.json').read_text())
        assert (path / 'model.safetensors').resolve() == tmp_path / 'model.safetensors'
        assert (path / 'model.safetensors').read_bytes() == b'fixture-weight-bytes'
        # Simulate a legacy constructor that saw different ambient flags. The
        # owned schedule must still be bound before the serving wrapper exists.
        head = SimpleNamespace(num_inference_steps=16, cfg_scale=5.,
                               dynamic_cache_schedule=True, dit_step_mask=[False] * 16)
        return SimpleNamespace(trained_model=SimpleNamespace(action_head=head))

    def wrap(policy):
        head = policy.trained_model.action_head
        assert head.dynamic_cache_schedule is False
        assert tuple(head.dit_step_mask) == SHIPPED_DIT_MASK
        return SimpleNamespace()

    loop = _build_owned_native_loop(tmp_path, load_owned, wrap,
        step_cache=ResolvedStepCache(False, 8, None, 'runtime.step_cache=checkpoint'))
    view = observed['path']
    assert view != tmp_path and view.is_dir()
    owned = observed['config']['action_head_cfg']
    assert owned['_target_'] == 'dreamzero_iwm.schedule.build_head'
    assert owned['config'] == config['action_head_cfg']['config']
    assert owned['ifl_dynamic_cache_schedule'] is False and owned['ifl_fixed_dit_steps'] == 8
    assert (tmp_path / 'config.json').read_bytes() == original
    assert loop.backend_stats['loading'] is None
    loop.close()
    assert not view.exists()
    assert (tmp_path / 'config.json').read_bytes() == original
    assert (tmp_path / 'model.safetensors').read_bytes() == b'fixture-weight-bytes'

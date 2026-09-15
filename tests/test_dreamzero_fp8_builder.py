"""Builder gates preserve the actual native scheduler before FP8 mutation."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/dreamzero'))
from dreamzero_iwm.adapter import DreamZeroAdapter, _DreamZeroLoop
from instinctflash.runtime import dreamzero_fp8

MASK = [True, True, True, False, False, False, True, False,
        False, False, True, False, False, True, True, True]


@pytest.fixture(autouse=True)
def native_pytorch_environment(monkeypatch):
    monkeypatch.delenv('LOAD_TRT_ENGINE', raising=False)
    monkeypatch.setenv('ENABLE_TENSORRT', 'false')
    monkeypatch.delenv('NUM_DIT_STEPS', raising=False)
    monkeypatch.delenv('DYNAMIC_CACHE_SCHEDULE', raising=False)


def checkpoint():
    return SimpleNamespace(execution=SimpleNamespace(backbone='dreamzero', extra={}))


def make_loop(mask=MASK, dynamic=False):
    head = SimpleNamespace(num_inference_steps=16, dit_step_mask=list(mask))
    wrapper = SimpleNamespace(_policy=SimpleNamespace(trained_model=SimpleNamespace(action_head=head)))
    return _DreamZeroLoop(wrapper, dynamic_cache=dynamic), head


@pytest.mark.parametrize('enabled,path', [('true', None), ('TRUE', None),
                                         ('false', 'engine.plan'), ('false', '')])
def test_tensorrt_environment_refused_before_native_loading(monkeypatch, enabled, path):
    from unittest.mock import Mock

    monkeypatch.setenv('ENABLE_TENSORRT', enabled)
    if path is not None:
        monkeypatch.setenv('LOAD_TRT_ENGINE', path)
    adapter = DreamZeroAdapter()
    build = Mock(side_effect=AssertionError('native model must not be loaded'))
    pack = Mock(side_effect=AssertionError('projections must not be packed'))
    monkeypatch.setattr(adapter, 'build_in_process', build)
    monkeypatch.setattr(dreamzero_fp8, 'ThorFP8Linear', pack)
    with pytest.raises(ValueError, match='native PyTorch path'):
        adapter.build_fp8(None)
    build.assert_not_called()
    pack.assert_not_called()


@pytest.mark.parametrize('engine', [object(), False])
def test_loaded_tensorrt_engine_refused_before_projection_access(monkeypatch, engine):
    from unittest.mock import Mock

    # Match native's `is not None` branch, even for a false-valued engine.
    pack = Mock(side_effect=AssertionError('projections must not be packed'))
    monkeypatch.setattr(dreamzero_fp8, 'ThorFP8Linear', pack)
    head = SimpleNamespace(cpu_offload=False, trt_engine=engine)
    with pytest.raises(ValueError, match='loaded TensorRT engine'):
        dreamzero_fp8.install_dreamzero_fp8(head)
    pack.assert_not_called()


def test_loaded_tensorrt_refusal_closes_builder_loop(monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr('torch.cuda.get_device_capability', lambda: (11, 0))
    loop, head = make_loop()
    head.trt_engine = object()
    adapter = DreamZeroAdapter()
    monkeypatch.setattr(adapter, 'build_in_process', lambda *a, **kw: loop)
    pack = Mock(side_effect=AssertionError('projections must not be packed'))
    monkeypatch.setattr(dreamzero_fp8, 'ThorFP8Linear', pack)
    with pytest.raises(ValueError, match='loaded TensorRT engine'):
        adapter.build_fp8(checkpoint())
    assert loop._wrapper is None
    pack.assert_not_called()


@pytest.mark.parametrize('fixed_steps', [5, 6, 7])
@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('selection_source', ['environment', 'resolved'])
def test_unsupported_fixed_mask_refused_before_loading(monkeypatch, fixed_steps, dynamic,
                                                     selection_source):
    from unittest.mock import Mock
    from instinctflash.planners.planner import Plan, Tier
    from instinctflash.runtime.step_cache_policy import ResolvedStepCache

    plan = Plan('dreamzero', [], tier_ceiling=Tier.BEHAVIORAL)
    selection = ResolvedStepCache(dynamic, fixed_steps,
        'dreamzero_velocity_v1' if dynamic else None, 'test')
    if selection_source == 'environment':
        monkeypatch.setenv('NUM_DIT_STEPS', str(fixed_steps))
        monkeypatch.setenv('DYNAMIC_CACHE_SCHEDULE', str(dynamic).lower())
        selection = None
    adapter = DreamZeroAdapter()
    build = Mock(side_effect=AssertionError('native model must not be loaded'))
    pack = Mock(side_effect=AssertionError('projections must not be packed'))
    monkeypatch.setattr(adapter, 'build_in_process', build)
    monkeypatch.setattr(dreamzero_fp8, 'ThorFP8Linear', pack)
    with pytest.raises(ValueError, match='fixed 8-of-16'):
        adapter.build_fp8(checkpoint(), plan=plan, step_cache=selection)
    build.assert_not_called()
    pack.assert_not_called()


def test_builder_installs_receipt_without_changing_mask(monkeypatch):
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (11, 0))
    loop, head = make_loop()
    adapter = DreamZeroAdapter()
    monkeypatch.setattr(adapter, 'build_in_process', lambda *a, **kw: loop)
    receipt = {'recipe': 'test', 'projections': [{'path': 'q'}]}
    seen = []
    monkeypatch.setattr(dreamzero_fp8, 'install_dreamzero_fp8', lambda h, *, include_ffn: (seen.append(h), receipt)[1] if include_ffn else None)
    assert adapter.build_fp8(checkpoint()) is loop
    assert seen == [head] and head.dit_step_mask == MASK
    assert loop.backend_stats['precision'] == 'fp8'
    assert loop.backend_stats['fp8_recipe'] == receipt


@pytest.mark.parametrize('dynamic,mask', [(True, MASK), (False, [True]*16),
    (False, list(reversed(MASK)))])
def test_schedule_refused_before_packing_and_loop_closed(monkeypatch, dynamic, mask):
    loop, _ = make_loop(mask, dynamic)
    adapter = DreamZeroAdapter()
    monkeypatch.setattr(adapter, 'build_in_process', lambda *a, **kw: loop)
    monkeypatch.setattr(dreamzero_fp8, 'install_dreamzero_fp8', lambda _: pytest.fail('packed changed schedule'))
    with pytest.raises(ValueError, match='behavioral' if dynamic else 'fixed 8-of-16'):
        adapter.build_fp8(checkpoint())
    assert loop._wrapper is None


def test_explicit_dynamic_fp8_requires_installed_controller_and_behavioral_plan(monkeypatch):
    from unittest.mock import Mock
    from instinctflash.planners.planner import Plan, Tier

    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (11, 0))
    adapter = DreamZeroAdapter()
    plan = Plan('dreamzero', [], tier_ceiling=Tier.BEHAVIORAL)
    loop, head = make_loop(dynamic=True)
    hook = SimpleNamespace(close=Mock())
    loop._step_cache_hook = hook
    monkeypatch.setattr(adapter, 'build_in_process', lambda *a, **kw: loop)
    receipt = {'recipe': 'test', 'projections': [{'path': 'q'}]}
    monkeypatch.setattr(dreamzero_fp8, 'install_dreamzero_fp8', lambda *a, **kw: receipt)
    assert adapter.build_fp8(checkpoint(), plan=plan) is loop
    assert head.dit_step_mask == MASK
    assert loop._fp8_recipe == receipt
    loop.close()
    hook.close.assert_called_once()

    uninstalled, _ = make_loop(dynamic=True)
    monkeypatch.setattr(adapter, 'build_in_process', lambda *a, **kw: uninstalled)
    with pytest.raises(ValueError, match='installed shared controller'):
        adapter.build_fp8(checkpoint(), plan=plan)
    assert uninstalled._wrapper is None


def test_packing_failure_closes_loop(monkeypatch):
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (11, 0))
    loop, _ = make_loop()
    adapter = DreamZeroAdapter()
    monkeypatch.setattr(adapter, 'build_in_process', lambda *a, **kw: loop)
    def fail(_, **kwargs):
        raise ValueError('unsupported projection')
    monkeypatch.setattr(dreamzero_fp8, 'install_dreamzero_fp8', fail)
    with pytest.raises(ValueError, match='unsupported projection'):
        adapter.build_fp8(checkpoint())
    assert loop._wrapper is None


def test_loop_refuses_unhandled_history_and_closed_calls():
    loop, _ = make_loop()
    with pytest.raises(ValueError, match='executed-action'):
        loop.predict({}, executed_action=[1])
    loop.close()
    loop.close()
    with pytest.raises(RuntimeError, match='closed'):
        loop.predict({})
    with pytest.raises(RuntimeError, match='closed'):
        loop.reset()


@pytest.mark.parametrize('native_backend', [False, True])
def test_feedback_refused_before_inference_for_both_dispatch_paths(monkeypatch, native_backend):
    import numpy as np
    from unittest.mock import Mock
    from instinctflash.runtime.execution import InProcessBackend
    loop, _ = make_loop()
    infer = Mock(return_value=np.zeros((24, 8), dtype=np.float32))
    loop._wrapper.infer = infer
    target = loop
    if native_backend:
        target = object.__new__(InProcessBackend)
        monkeypatch.setattr(target, '_ensure', lambda: loop)
    with pytest.raises(ValueError, match='executed-action'):
        target.predict({'prompt': 'pick up the object'}, executed_action=[1])
    infer.assert_not_called()
    result = target.predict({'prompt': 'pick up the object'})
    assert result['action'].shape == (24, 8)
    infer.assert_called_once()


def test_native_receipt_tracks_actual_mask_grid_and_guidance():
    from dreamzero_iwm.adapter import _head_declaration
    head = SimpleNamespace(num_inference_steps=16, num_inference_timesteps=4,
                           cfg_scale=5.0, dynamic_cache_schedule=False,
                           dit_step_mask=MASK)
    loop = _DreamZeroLoop(SimpleNamespace(), dynamic_cache=False,
                         build_declaration=_head_declaration(head))
    assert loop.declaration()['steps'] == {'video_action': 16, 'kv_commit': 1}
    assert loop.declaration()['guidance']['video_action'] == ('cfg', 5.0)
    assert loop.backend_stats['tier'] == 'upstream shipped mask'
    # Caller edits must not change the future serving declaration.
    loop.backend_stats['build']['dit_step_mask'][0] = False
    assert loop.declaration()['dit_step_mask'] == MASK
    head.dit_step_mask = [True]*5 + [False]*11
    reduced = _DreamZeroLoop(SimpleNamespace(), dynamic_cache=False,
                            build_declaration=_head_declaration(head))
    assert reduced.backend_stats['tier'] == 'SCREEN'
    assert sum(reduced.declaration()['dit_step_mask']) == 5
    # Precision and schedule evidence are independent.
    reduced._fp8_recipe = {'recipe': 'test'}
    assert reduced.declaration()['precision'] == 'fp8'
    assert reduced.backend_stats['tier'] == 'SCREEN'

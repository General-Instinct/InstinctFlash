"""Public FP8 dispatch must verify the built DreamZero policy before serving."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from instinctflash.runtime.engine_backend import EngineBackend, engine_available
from instinctflash.runtime.execution import choose_backend
from instinctflash.planners.planner import Plan, PassResult, Tier


@pytest.mark.parametrize('steps,commit,cfg,packed,precision', [
    (16, 1, 5., True, 'fp8'), (8, 1, 5., True, 'fp8'),
    (16, 2, 5., True, 'fp8'), (16, 1, 1., True, 'fp8'),
    (16, 1, 5., False, 'fp8'), (16, 1, 5., True, 'native')])
def test_built_schedule_precision_and_cleanup(steps, commit, cfg, packed, precision):
    loop = SimpleNamespace(
        declaration=lambda: {'precision': precision, 'frontend': 'test',
                             'steps': {'video_action': steps, 'kv_commit': commit},
                             'guidance': {'video_action': ('cfg', cfg)}, 'evidence': 'test'},
        backend_stats={'fp8_recipe': {'projections': ['q'] if packed else []}},
        close=Mock(), predict=Mock(), reset=Mock())
    adapter = SimpleNamespace(build_fp8=Mock(return_value=loop))
    checkpoint = SimpleNamespace(execution=SimpleNamespace(backbone='dreamzero'))
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.get_device_capability', return_value=(11, 0)), \
         patch('instinctflash.runtime.engine_backend.requested_operating_point',
               return_value=({'video_action': 16, 'kv_commit': 1},
                             {'video_action': ('cfg', 5.)}, 'test')):
        if (steps, commit, cfg, packed, precision) == (16, 1, 5., True, 'fp8'):
            backend = EngineBackend(adapter, checkpoint, None)
            obs = {'camera': 'actual caller data'}
            backend.predict(obs)
            loop.predict.assert_called_once_with(obs, executed_action=None)
            backend.reset(prompt='new episode')
            loop.reset.assert_called_once_with(prompt='new episode')
            backend.close()
        else:
            with pytest.raises(RuntimeError):
                EngineBackend(adapter, checkpoint, None)
            loop.predict.assert_not_called()
        loop.close.assert_called_once()


def test_pytorch_projection_route_does_not_require_flash_rt():
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.get_device_capability', return_value=(11, 0)), \
         patch.dict('sys.modules', {'flash_rt': None}):
        assert engine_available('dreamzero')[0]


def test_precision_dispatch_uses_family_device_probe():
    checkpoint = SimpleNamespace(execution=SimpleNamespace(backbone='dreamzero'))
    plan = Plan('test', [PassResult('engine_offload', True, Tier.NUMERIC,
                                   reason='test', params={'backend': 'engine'})])
    with patch('instinctflash.runtime.engine_backend.engine_available',
               return_value=(True, 'test')) as available, \
         patch('instinctflash.runtime.engine_backend.EngineBackend') as backend:
        actual, _ = choose_backend('auto', object(), checkpoint, plan, precision='fp8')
    available.assert_called_once_with('dreamzero')
    assert actual is backend.return_value


def test_published_checkpoint_declaration_resolves_actual_scheduler_and_cfg(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/dreamzero'))
    from dreamzero_iwm.adapter import DreamZeroAdapter
    from instinctflash.descriptors.known import lookup
    from instinctflash.descriptors.package import _declared_view, validate_package
    from instinctflash.runtime.engine_backend import requested_operating_point

    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    # Configuration geometry gate only; this does not pretend to contain weights.
    (snapshot / 'config.json').write_text(json.dumps({'action_head_cfg': {'config': {
        'diffusion_model_cfg': {'frame_seqlen': 880, 'dim': 5120, 'num_layers': 40}}}}))
    (snapshot / 'model.safetensors.index.json').write_text('{"weight_map":{}}')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    model = 'GEAR-Dreams/DreamZero-DROID'
    view = _declared_view(snapshot, model, lookup(model))
    report = validate_package(view)
    assert report.ok, report.explain()
    checkpoint = SimpleNamespace(path=view, execution=report.declaration)
    adapter = DreamZeroAdapter()
    steps, guidance, _ = requested_operating_point(adapter, checkpoint)
    assert steps == {'video_action': 16, 'kv_commit': 1}
    assert guidance == {'video_action': ('cfg', 5.0)}
    assert adapter.spec_for_checkpoint(checkpoint).streams[0].tokens_per_frame == 880

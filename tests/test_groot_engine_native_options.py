"""FP8 keeps native processing optimizations and their lifecycle ownership."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from instinctflash.runtime.groot_engine import _GrootEngineLoop


@pytest.mark.parametrize('override', [False, True])
def test_native_options_forwarding_and_cleanup(override):
    class Head:
        def get_action(self):
            return 'native'
    head = Head()
    original = Mock() if override else None
    if override:
        head.get_action = original
    native = SimpleNamespace(_policy=SimpleNamespace(model=SimpleNamespace(action_head=head)),
        _state_dims={'arm': 7}, reset=Mock(), predict=Mock(return_value={'action': 'decoded'}), close=Mock(),
        backend_stats={'fast_decode': True, 'backbone_fastpath': True, 'backbone_cache_hits': 3,
                       'cpu_threads': 4, 'precision': 'bfloat16'})
    generator = SimpleNamespace(get_action=Mock())
    frontend = SimpleNamespace(_dit_graphs={'signature': object()})
    runner = SimpleNamespace(graph=object(), replays=5)
    loop = _GrootEngineLoop(native, frontend, runner, generator, 'cuda:0')
    assert loop._state_dims is native._state_dims
    assert head.get_action is generator.get_action
    loop.reset(prompt='new task')
    native.reset.assert_called_once_with(prompt='new task')
    obs = {'images': ['current frame'], 'state': {'arm': [0]*7}}
    with patch('torch.cuda.device', return_value=nullcontext()):
        assert loop.predict(obs) == {'action': 'decoded'}
    native.predict.assert_called_once_with(obs)
    stats = loop.backend_stats
    assert stats['fast_decode'] and stats['backbone_fastpath'] and stats['captured']
    assert stats['cpu_threads'] == 4 and stats['precision'] == 'fp8'
    assert native.backend_stats['precision'] == 'bfloat16'
    # The native Runtime ignores action feedback when the policy has no commit
    # hook; changing precision must preserve that stateless contract.
    with patch('torch.cuda.device', return_value=nullcontext()):
        assert loop.predict(obs, executed_action=[1]) == {'action': 'decoded'}
    assert native.predict.call_count == 2
    native.predict.assert_called_with(obs)
    loop.close()
    loop.close()
    native.close.assert_called_once()
    if override:
        assert head.get_action is original
    else:
        assert 'get_action' not in head.__dict__ and head.get_action() == 'native'
    assert loop._frontend is loop._runner is loop._generator is None
    native._policy = None
    assert loop.backend_stats['text_graphs'] == 0
    with pytest.raises(RuntimeError, match='closed'):
        loop.predict(obs)
    with pytest.raises(RuntimeError, match='closed'):
        loop.reset()


@pytest.mark.parametrize('fail', [False, True])
def test_builder_reuses_native_construction_and_closes_it_on_failure(fail, monkeypatch):
    import sys
    from pathlib import Path
    from types import ModuleType
    from instinctflash.runtime import groot_engine
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/groot_n17'))
    from groot_n17_iwm.adapter import GR00TN17Adapter
    head = SimpleNamespace()
    native = SimpleNamespace(_policy=SimpleNamespace(
        model=SimpleNamespace(action_head=head, backbone=object()),
        modality_configs={'video': SimpleNamespace(modality_keys=['left', 'wrist'])},
        embodiment_tag=SimpleNamespace(value='droid')), _model_path=Path('/weights'), close=Mock())
    build = Mock(return_value=native)
    monkeypatch.setattr(GR00TN17Adapter, 'build_in_process', build)
    frontend = Mock(side_effect=RuntimeError('engine failed')) if fail else Mock(return_value=object())
    module = ModuleType('flash_rt.frontends.torch.groot_n17_thor')
    module.GrootN17TorchFrontendThor = frontend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runner_module = ModuleType('flash_rt.models.groot_n17.vlsa_runner')
    runner_module.GrootN17VlsaRunner = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    monkeypatch.setattr(groot_engine, 'GrootActionGenerator', Mock(return_value=SimpleNamespace(get_action=Mock())))
    from instinctflash.runtime import groot_fp8
    monkeypatch.setattr(groot_fp8, "install_groot_backbone_fp8", Mock(return_value={"recipe": "test"}))
    checkpoint = object()
    with patch('torch.cuda.device', return_value=nullcontext()):
        if fail:
            with pytest.raises(RuntimeError, match='engine failed'):
                groot_engine.build_groot_engine_loop(checkpoint, device='cuda:0')
        else:
            loop = groot_engine.build_groot_engine_loop(checkpoint, device='cuda:0')
            native.close.assert_not_called()
            loop.close()
    build.assert_called_once_with(checkpoint, None, device='cuda:0', nfe={'action': 4})
    native.close.assert_called_once()

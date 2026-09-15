"""CUDA graph inputs must distinguish semantic metadata and aliasing."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/cosmos3_policy'))
from cosmos3_iwm.thor_graphs import _signature, _tree


def test_graph_key_tracks_scalar_metadata():
    assert _signature({'split_lens': [4, 8]}) != _signature({'split_lens': [5, 7]})
    assert _signature({'mask': True}) != _signature({'mask': 1})


def test_unknown_objects_are_not_silently_captured():
    with pytest.raises(TypeError):
        _signature(object())
    with pytest.raises(TypeError):
        _tree(object(), lambda x: x)


def test_splitinfo_metadata_is_copied_and_keyed():
    cls = type('SplitInfo', (), {'__module__': 'cosmos_framework.model.generator.mot.attention'})
    value = cls(); value.split_lens = [4, 8]; value.sample_lens = [12]
    clone = _tree(value, lambda x: x)
    assert _signature(clone) == _signature(value)
    clone.split_lens[0] = 5
    assert value.split_lens == [4, 8]
    assert _signature(clone) != _signature(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_thor_graph_replay_and_weight_replacement():
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip('Thor qualification')
    from cosmos3_iwm import exact_pointwise
    from benchmarks.regression.verify_cosmos_kernels import verify_graphs
    verify_graphs(exact_pointwise)


@pytest.mark.parametrize('capability,precision,nano,disabled,expected,elide', [
    ((11, 0), 'native', False, False, ['pointwise', 'graphs'], False),
    ((11, 0), 'native', True, False, ['pointwise', 'graphs'], True),
    ((11, 0), 'fp8', True, False, [], False),
    ((9, 0), 'native', True, False, [], False),
    ((12, 0), 'native', True, False, [], True),
    ((11, 0), 'native', True, True, [], False),
])
def test_adapter_native_defaults_keep_precision_and_hardware_separate(
    monkeypatch, capability, precision, nano, disabled, expected, elide
):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from cosmos3_iwm import adapter, nano_action_only
    from instinctflash.runtime import cosmos_droid, engine_backend
    import os
    for key in list(os.environ):
        if key.startswith('IFL_COSMOS3_'):
            monkeypatch.delenv(key)
    if disabled:
        for key in ('EXACT_POINTWISE', 'LAYER_GRAPHS', 'NANO_ACTION_ONLY'):
            monkeypatch.setenv('IFL_COSMOS3_' + key, '0')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: capability)
    installed, construction = [], []
    for name, label in [('exact_pointwise', 'pointwise'), ('thor_graphs', 'graphs')]:
        monkeypatch.setitem(sys.modules, 'cosmos3_iwm.' + name,
                            SimpleNamespace(install=lambda service, label=label: installed.append(label)))
    monkeypatch.setattr(nano_action_only, 'nano_action_only_construction',
                        lambda enabled: (construction.append(enabled), nullcontext())[1])
    monkeypatch.setattr(nano_action_only, 'verify_nano_action_only_model', lambda model: 1234)
    monkeypatch.setattr(adapter, '_resolve_model_path', lambda checkpoint: Path('/unused'))
    monkeypatch.setattr(engine_backend, 'requested_operating_point',
                        lambda *args: ({'prefix': 1, 'action': 4}, {'action': ('cfg', 3.)}, None))
    service = SimpleNamespace(model=object())
    monkeypatch.setattr(cosmos_droid, 'build_droid_service', lambda *args, **kwargs: (service, None))
    model_id = adapter.NANO_MODEL_ID if nano else adapter.MODEL_ID
    extra = dict(domain_name='droid_lerobot', action_dim=8, action_chunk_size=32,
                 conditioning_fps=15, image_height=540, image_width=640, format_prompt_as_json=False)
    checkpoint = SimpleNamespace(model_id=model_id, execution=SimpleNamespace(model_id=model_id, extra=extra))
    adapter.Cosmos3PolicyAdapter()._build_droid(checkpoint, device='cuda', nfe=None, precision=precision)
    assert installed == expected and construction == [elide]
    assert service._ifl_native_optimizations['elided_lm_head_bytes'] == (1234 if elide else 0)

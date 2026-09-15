"""Fail-closed admission and lifecycle for the optional native conditioning cache."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/cosmos3_policy'))
from cosmos3_iwm import conditioning_cache as module
from instinctflash.runtime.cosmos_droid import CosmosDROIDLoop


def service():
    model = SimpleNamespace(config=SimpleNamespace(joint_attn_implementation='two_way',
        video_temporal_causal=False, sound_gen=False), parallel_dims=None,
        generate_samples_from_batch=lambda: 'original')
    return SimpleNamespace(model=model, cfg=SimpleNamespace(num_steps=4, guidance=3.0))


def test_unsupported_device_keeps_original_forward(monkeypatch):
    target = service()
    original = target.model.generate_samples_from_batch
    monkeypatch.setattr(module.torch.cuda, 'get_device_capability', lambda: (9, 0))
    assert module.install(target) is None
    assert target.model.generate_samples_from_batch is original
    assert target._ifl_conditioning_cache_status == {'admitted': False, 'reason': 'Thor only'}


def test_unknown_source_keeps_original_forward(monkeypatch, tmp_path):
    target = service()
    original = target.model.generate_samples_from_batch
    monkeypatch.setattr(module.torch.cuda, 'get_device_capability', lambda: (11, 0))
    monkeypatch.setitem(sys.modules, 'cosmos_framework', SimpleNamespace(__file__=str(tmp_path/'__init__.py')))
    monkeypatch.setattr(module, 'SOURCE_HASHES', {'changed.py': '0'*64})
    (tmp_path/'changed.py').write_text('unknown implementation')
    assert module.install(target) is None
    assert 'Unqualified Cosmos source' in target._ifl_conditioning_cache_status['reason']
    assert target.model.generate_samples_from_batch is original


@pytest.mark.parametrize('steps,guidance', [(2, 3.0), (4, 1.0)])
def test_changed_schedule_is_not_admitted(monkeypatch, steps, guidance):
    target = service()
    target.cfg.num_steps, target.cfg.guidance = steps, guidance
    monkeypatch.setattr(module.torch.cuda, 'get_device_capability', lambda: (11, 0))
    assert module.install(target) is None
    assert not target._ifl_conditioning_cache_status['admitted']


def test_duplicate_install_retains_one_owner():
    target = service()
    owner = object()
    target._ifl_conditioning_cache = owner
    assert module.install(target) is owner


def test_invalid_bound_rejected_before_cuda():
    with pytest.raises(ValueError, match='positive'):
        module.ConditioningCache(None, max_slots=0)


def test_loop_close_releases_cache_and_is_idempotent():
    calls = []
    loop = CosmosDROIDLoop.__new__(CosmosDROIDLoop)
    loop._service = SimpleNamespace(_ifl_conditioning_cache=SimpleNamespace(close=lambda: calls.append('closed')))
    loop.close()
    loop.close()
    assert calls == ['closed'] and loop._service is None


def test_regression_rejects_silent_cache_fallback():
    from benchmarks.regression.run_cosmos import require_conditioning_evidence
    valid = {'backend_stats': {'conditioning_cache_status': {'admitted': True},
        'conditioning_cache': {'disabled': False, 'rejected': [], 'verified_tensors': 9,
                               'graph_stats': [{'replays': 2, 'rejected': []}]}}}
    require_conditioning_evidence(valid)
    valid['backend_stats']['conditioning_cache_status']['admitted'] = False
    with pytest.raises(ValueError, match='not admitted'):
        require_conditioning_evidence(valid)
    with pytest.raises(ValueError, match='not admitted'):
        require_conditioning_evidence({'backend_stats': {}})


def test_graph_pool_lifetime_tracks_cache_slot(monkeypatch):
    handles = []

    def new_handle():
        handle = object()
        handles.append(handle)
        return handle

    class Graph:
        def __init__(self, original, stats, name, *, pool):
            self.original, self.pool = original, pool

        def __call__(self, value):
            return self.original(value)

    monkeypatch.setattr(module.torch.cuda, 'graph_pool_handle', new_handle)
    monkeypatch.setattr(module, 'LayerGraph', Graph)
    cache = module.ConditioningCache.__new__(module.ConditioningCache)
    cache.disabled = cache.filling = cache.validating = False
    cache.active = dict(graphs={}, graph_stats={})
    first = cache.active
    layer0 = cache.wrap_layer(lambda x: x + 1, lambda x: -1, 0)
    layer1 = cache.wrap_layer(lambda x: x * 2, lambda x: -1, 1)
    assert layer0(2) == 3 and layer1(2) == 4
    assert len(handles) == 1
    assert first['graphs'][0].pool is first['graphs'][1].pool
    # Eviction or an aborted transaction can destroy all old graphs. Keep the
    # retired handle visible to ensure it is never selected for the new slot.
    retired = first['pool']
    first['graphs'].clear()
    cache.active = dict(graphs={}, graph_stats={})
    assert layer0(3) == 4
    assert cache.active['graphs'][0].pool is not retired and len(handles) == 2
    cache.active = None
    assert layer0(3) == -1 and len(handles) == 2


def test_retired_graph_failure_cannot_be_hidden_by_a_healthy_slot():
    from collections import OrderedDict
    from benchmarks.regression.run_cosmos import require_conditioning_evidence
    cache = module.ConditioningCache.__new__(module.ConditioningCache)
    cache.retired_graph_stats = dict(captures=0, checks=0, replays=0,
                                    rejection_count=0, rejection_samples=[])
    errors = [dict(module=str(i), error='capture failure') for i in range(20)]
    cache.slots = OrderedDict(old=dict(graph_stats={0: dict(
        captures=1, checks=2, replays=3, rejected=errors)}))
    cache.clear_slots()
    assert not cache.slots
    assert cache.retired_graph_stats['rejection_count'] == 20
    assert len(cache.retired_graph_stats['rejection_samples']) == 8
    cache.clear_slots()
    assert cache.retired_graph_stats['rejection_count'] == 20
    report = {'backend_stats': {'conditioning_cache_status': {'admitted': True},
        'conditioning_cache': {'disabled': False, 'rejected': [], 'verified_tensors': 9,
            'graph_stats': [{'replays': 10, 'rejected': []}],
            'retired_graph_stats': cache.retired_graph_stats}}}
    with pytest.raises(ValueError, match='without rejection'):
        require_conditioning_evidence(report)


def test_successful_retired_replays_remain_evidence_after_close():
    from benchmarks.regression.run_cosmos import require_conditioning_evidence
    require_conditioning_evidence({'backend_stats': {
        'conditioning_cache_status': {'admitted': True},
        'conditioning_cache': {'disabled': False, 'rejected': [], 'verified_tensors': 9,
            'graph_stats': [], 'retired_graph_stats': {
                'replays': 12, 'rejection_count': 0, 'rejection_samples': []}}}})

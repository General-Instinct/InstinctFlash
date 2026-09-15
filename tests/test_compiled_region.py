from types import SimpleNamespace

import pytest
import torch

from instinctflash.planners.planner import Tier
from instinctflash.runtime.compiled_region import CompiledRegion


def make(fn, **kwargs):
    return CompiledRegion(fn, plan=SimpleNamespace(tier_ceiling=Tier.NUMERIC),
                          name='test', **kwargs)


def test_bitexact_rejected_before_compiler_called():
    with pytest.raises(ValueError, match='numeric'):
        CompiledRegion(lambda x: x, name='x', plan=SimpleNamespace(tier_ceiling=Tier.BITEXACT),
                       compiler=lambda *a, **k: pytest.fail('compiler called'))


def test_current_tensor_values_are_not_cached_and_outputs_survive_next_call():
    def compiler(fn, **options):
        assert options == dict(fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
        return fn
    region = make(lambda x, metadata: x + metadata['offset'], compiler=compiler)
    with torch.no_grad():
        first = region(torch.ones(2), metadata={'offset': torch.tensor(2.)})
        second = region(torch.ones(2), metadata={'offset': torch.tensor(7.)})
    assert torch.equal(first, torch.full((2,), 3.))
    assert torch.equal(second, torch.full((2,), 8.))
    assert region.report()['input_signatures'] == 1
    assert not region.report()['quality_certified']


def test_lazy_failure_is_not_retried_eagerly():
    calls = []
    def fail(x):
        calls.append(1)
        raise ValueError('lazy compile failure')
    region = make(lambda x: pytest.fail('eager fallback'), compiler=lambda *a, **k: fail)
    with torch.no_grad(), pytest.raises(ValueError, match='lazy'):
        region(torch.ones(1))
    with torch.no_grad(), pytest.raises(RuntimeError, match='failed'):
        region(torch.ones(1))
    assert calls == [1] and region.report()['calls'] == 0


def test_signature_bound_rejects_new_shapes_but_allows_known_shapes():
    region = make(lambda x: x + 1, compiler=lambda f, **k: f, max_signatures=1)
    with torch.no_grad():
        region(torch.ones(2))
        with pytest.raises(RuntimeError, match='bound'):
            region(torch.ones(3))
        assert region(torch.zeros(2)).tolist() == [1, 1]


def test_cache_object_is_rejected_before_execution():
    region = make(lambda x: pytest.fail('executed'), compiler=lambda f, **k: f)
    with torch.no_grad(), pytest.raises(TypeError, match='cache objects'):
        region({'cache': SimpleNamespace(value=torch.ones(1))})


def test_training_and_closed_calls_rejected():
    region = make(lambda x: x + 1, compiler=lambda f, **k: f)
    with torch.enable_grad(), pytest.raises(RuntimeError, match='no_grad'):
        region(torch.ones(1))
    region.close()
    region.close()
    with torch.no_grad(), pytest.raises(RuntimeError, match='closed'):
        region(torch.ones(1))


def test_real_dynamo_region_handles_changed_tensor_metadata():
    region = make(lambda x, offset: x * 2 + offset,
                  compiler=lambda f, **k: torch.compile(f, backend='eager', fullgraph=k['fullgraph']))
    with torch.no_grad():
        assert torch.equal(region(torch.arange(3.), torch.tensor(1.)), torch.tensor([1., 3., 5.]))
        assert torch.equal(region(torch.arange(3.), torch.tensor(4.)), torch.tensor([4., 6., 8.]))

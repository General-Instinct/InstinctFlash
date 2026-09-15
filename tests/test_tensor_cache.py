import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from instinctflash.runtime.tensor_cache import TensorResultCache

torch = pytest.importorskip("torch")


def test_ownership_eviction_and_failure():
    cache = TensorResultCache(max_bytes=16, max_entries=1)
    with torch.no_grad():
        first = cache.get_or_compute('a', lambda: torch.arange(4.))
        first.zero_()
        hit = cache.get_or_compute('a', lambda: pytest.fail('recomputed hit'))
        assert torch.equal(hit, torch.arange(4.))
        hit.zero_()
        assert torch.equal(cache.get_or_compute('a', lambda: None), torch.arange(4.))
        cache.get_or_compute('b', lambda: torch.ones(4))
        assert cache.report()['evictions'] == 1
        cache.get_or_compute('large', lambda: torch.ones(5))
        assert cache.report()['bytes'] == 16
        def fail():
            raise ValueError('cold error')
        with pytest.raises(ValueError, match='cold error'):
            cache.get_or_compute('failed', fail)
        assert cache.report()['entries'] == 1
        cache.close()
        with pytest.raises(RuntimeError, match='closed'):
            cache.get_or_compute('b', lambda: None)


def test_concurrent_single_compute():
    cache = TensorResultCache()
    calls = []
    def run(_):
        with torch.no_grad():
            def compute():
                calls.append(1)
                return torch.ones(8)
            return cache.get_or_compute('shared', compute)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(12)))
    assert len(calls) == 1
    assert len({r.data_ptr() for r in results}) == 12
    assert cache.report()['hits'] == 11


def test_rejects_grad_and_invalid_bounds():
    with pytest.raises(RuntimeError, match='inference'):
        TensorResultCache().get_or_compute(0, lambda: torch.ones(1))
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            TensorResultCache(max_bytes=value)


@pytest.mark.skipif(os.environ.get('IFL_TEST_CUDA') != '1' or not torch.cuda.is_available(),
                    reason='Set IFL_TEST_CUDA=1 on the reserved test GPU')
def test_cuda_cross_stream_eviction():
    cache = TensorResultCache(max_entries=1)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.no_grad(), torch.cuda.stream(producer):
        reference = torch.arange(65536, device='cuda', dtype=torch.float32)
        cache.get_or_compute('a', lambda: reference * 2)
    with torch.no_grad(), torch.cuda.stream(consumer):
        hit = cache.get_or_compute('a', lambda: None)
    with torch.no_grad(), torch.cuda.stream(producer):
        cache.get_or_compute('b', lambda: torch.zeros_like(reference))
        churn = [torch.ones_like(reference) for _ in range(8)]
    torch.cuda.synchronize()
    assert torch.equal(hit, reference * 2)
    assert len(churn) == 8


def test_noncontiguous_result_preserves_native_layout_without_caching():
    cache = TensorResultCache()
    with torch.no_grad():
        result = cache.get_or_compute('slice', lambda: torch.arange(8.)[::2])
        assert result.stride() == (2,)
        assert cache.report()['entries'] == 0

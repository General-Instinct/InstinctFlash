import pytest

from examples.cosmos3_policy.cosmos3_iwm.timestep_cache import NativeTimestepCache

torch = pytest.importorskip("torch")


class TimestepEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frequency_embedding_size = 1
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.SiLU(), torch.nn.Linear(4, 4))

    def forward(self, t):
        return self.mlp(t[:, None])


def test_input_weights_and_close():
    module = TimestepEmbedder().eval()
    cache = NativeTimestepCache(module)
    module.forward = cache
    with torch.no_grad():
        t = torch.tensor([1., 2.])
        expected = cache.original(t)
        module(t).zero_()
        assert torch.equal(module(t), expected)
        t[1] = 3.
        assert torch.equal(module(t), cache.original(t))
        module.mlp[0].weight.add_(1)
        assert torch.equal(module(t), cache.original(t))
    assert cache.report()['hits'] == 1
    assert cache.report()['invalidations'] == 1
    cache.close()
    assert module.forward == cache.original


def test_training_and_hooks_are_not_skipped():
    module = TimestepEmbedder().eval()
    cache = NativeTimestepCache(module)
    with torch.no_grad():
        t = torch.tensor([1.])
        cache(t)
        calls = []
        hook = module.mlp[0].register_forward_hook(lambda *args: calls.append(1))
        cache(t)
        assert calls == [1]
        hook.remove()
    module.train()
    t.requires_grad_()
    cache(t).sum().backward()
    assert t.grad is not None
    assert cache.report()['fallbacks'] == 2


def test_autocast_and_checkpoint_reload_invalidate():
    module = TimestepEmbedder().eval()
    cache = NativeTimestepCache(module)
    t = torch.tensor([1., 2.])
    with torch.no_grad():
        cache(t)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = cache(t)
            assert result.dtype == torch.bfloat16
            assert torch.equal(result, cache.original(t))
        assert torch.equal(cache(t), cache.original(t))
        replacement = TimestepEmbedder().state_dict()
        module.load_state_dict(replacement)
        assert torch.equal(cache(t), cache.original(t))
    assert cache.report()['invalidations'] == 3


@pytest.mark.parametrize('width,budget', [(2048, 128), (4096, 256)])
def test_model_capacity_and_idempotent_install(width, budget):
    from types import SimpleNamespace
    from examples.cosmos3_policy.cosmos3_iwm.timestep_cache import install
    module = TimestepEmbedder().eval()
    module.mlp[-1] = torch.nn.Linear(4, width)
    service = SimpleNamespace(model=SimpleNamespace(net=SimpleNamespace(time_embedder=module)))
    cache = install(service)
    assert cache is install(service)
    assert cache.report()['max_bytes'] == budget * 1024 * 1024
    cache.close()


def test_fp8_rejected_before_loading(monkeypatch):
    from examples.cosmos3_policy.cosmos3_iwm.adapter import Cosmos3PolicyAdapter
    monkeypatch.setenv('IFL_COSMOS3_TIMESTEP_CACHE', '1')
    with pytest.raises(ValueError, match='requires native precision'):
        Cosmos3PolicyAdapter()._build_droid(None, device=None, nfe=None, precision='fp8')

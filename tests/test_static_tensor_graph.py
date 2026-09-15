import pytest
import torch
from instinctflash.runtime.static_tensor_graph import StaticTensorGraph


def test_cpu_keeps_reference_and_never_attempts_capture():
    graph = StaticTensorGraph(lambda x: pytest.fail('candidate called'), reference=lambda x: x + 1)
    assert torch.equal(graph(torch.ones(3)), torch.full((3,), 2.))
    assert not graph.stats['captured']
    graph.close()
    graph.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph admission')
def test_graph_refreshes_inputs_shapes_and_releases_storage():
    graph = StaticTensorGraph(lambda x: (x * 2, [x + 1]))
    for shape in (3, 3, 7):
        value = torch.randn(shape, device='cuda')
        out = graph(value)
        assert torch.equal(out[0], value * 2)
        assert torch.equal(out[1][0], value + 1)
        assert graph.stats['captured'] and graph.stats['self_check']['passed']
    graph.close()
    assert graph.graph is graph.input is graph.output is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph admission')
def test_bad_candidate_falls_back_and_stays_disabled():
    graph = StaticTensorGraph(lambda x: x + 2, reference=lambda x: x + 1)
    value = torch.ones(4, device='cuda')
    assert torch.equal(graph(value), value + 1)
    assert graph.disabled and not graph.stats['self_check']['passed']
    value.fill_(4)
    assert torch.equal(graph(value), value + 1)


def test_close_releases_bound_model_weights():
    import weakref
    model = torch.nn.Linear(16, 16)
    reference = weakref.ref(model)
    graph = StaticTensorGraph(model.forward)
    del model
    assert reference() is not None
    graph.close()
    assert reference() is None
    assert graph.graph is graph.input is graph.output is graph.forward is graph.reference is None
    with pytest.raises(RuntimeError, match="closed"):
        graph(torch.zeros(16))

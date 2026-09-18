"""Real raw CUDA graph ownership and replay; no model weights or native kernels."""
from collections import Counter
import ctypes
import gc
import importlib.util
from pathlib import Path
import weakref
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
if not torch.cuda.is_available():
    pytest.skip('CUDA required', allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]


class CountedRuntime:
    """Forward to the real runtime; count successful native ownership operations."""
    def __init__(self, runtime):
        self.runtime = runtime
        self.counts = Counter()

    def __getattr__(self, name):
        function = getattr(self.runtime, name)
        def call(*args):
            status = function(*args)
            if status == 0:
                self.counts[name] += 1
            return status
        return call


@pytest.fixture
def graphs():
    spec = importlib.util.spec_from_file_location(
        '_real_lifecycle_graph', ROOT / 'serving/flash_rt/core/cuda_graph.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = CountedRuntime(module._cudart)
    module._cudart = runtime
    yield module.CUDAGraph, runtime
    torch.cuda.synchronize()
    gc.collect()
    assert runtime.counts['cudaStreamEndCapture'] == runtime.counts['cudaGraphDestroy']
    assert runtime.counts['cudaGraphInstantiate'] == runtime.counts['cudaGraphExecDestroy']
    assert runtime.counts['cudaStreamCreate'] == runtime.counts['cudaStreamDestroy']


def capture_copy(graph, runtime, stream, source, destination):
    graph.begin_capture(stream)
    status = runtime.cudaMemcpyAsync(ctypes.c_void_p(destination.data_ptr()),
                                    ctypes.c_void_p(source.data_ptr()),
                                    ctypes.c_size_t(source.numel() * source.element_size()),
                                    3, stream)
    assert status == 0
    graph.end_capture(stream)


@pytest.mark.parametrize('borrowed', [False, True])
@pytest.mark.parametrize('finalize', [False, True])
def test_repeated_capture_replay_and_release(graphs, borrowed, finalize):
    cls, runtime = graphs
    owner = torch.cuda.Stream() if borrowed else None
    source = torch.arange(1024, device='cuda', dtype=torch.int32)
    destination = torch.empty_like(source)
    torch.cuda.synchronize()
    for iteration in range(50):
        graph = cls()
        stream = ctypes.c_void_p(owner.cuda_stream) if owner else graph.create_stream()
        source.fill_(iteration + 1)
        destination.zero_()
        torch.cuda.synchronize()
        capture_copy(graph, runtime, stream, source, destination)
        graph.replay(stream)
        # Deliberately release before synchronizing queued work. Source/destination
        # stay alive; CUDA must complete the launch before freeing graph resources.
        if finalize:
            reference = weakref.ref(graph)
            del graph
            gc.collect()
            assert reference() is None
        else:
            graph.close()
            before = runtime.counts.copy()
            graph.close()
            assert runtime.counts == before
            with pytest.raises(RuntimeError, match='closed'):
                graph.replay(stream)
        torch.cuda.synchronize()
        torch.testing.assert_close(destination, source, rtol=0, atol=0)
        if owner is not None:
            with torch.cuda.stream(owner):
                destination.add_(1)
            owner.synchronize()
            torch.testing.assert_close(destination, source + 1, rtol=0, atol=0)
    assert runtime.counts['cudaGraphExecDestroy'] == 50


@pytest.mark.parametrize('borrowed', [False, True])
def test_python_failure_ends_real_capture(graphs, borrowed):
    cls, runtime = graphs
    owner = torch.cuda.Stream() if borrowed else None
    tensor = torch.zeros(128, device='cuda', dtype=torch.int32)
    torch.cuda.synchronize()
    with pytest.raises(ValueError, match='capture body'):
        with cls() as graph:
            stream = ctypes.c_void_p(owner.cuda_stream) if owner else graph.create_stream()
            graph.begin_capture(stream)
            status = runtime.cudaMemsetAsync(ctypes.c_void_p(tensor.data_ptr()), 0,
                                            ctypes.c_size_t(tensor.numel() * 4), stream)
            assert status == 0
            raise ValueError('capture body')
    assert runtime.counts['cudaGraphInstantiate'] == 0
    if owner is not None:
        with torch.cuda.stream(owner):
            tensor.fill_(7)
        owner.synchronize()
        assert tensor.min().item() == 7


def test_capture_again_requires_a_new_wrapper(graphs):
    cls, runtime = graphs
    tensor = torch.ones(128, device='cuda')
    other = torch.zeros_like(tensor)
    torch.cuda.synchronize()
    with cls() as graph:
        stream = graph.create_stream()
        capture_copy(graph, runtime, stream, tensor, other)
        with pytest.raises(RuntimeError, match='another capture'):
            graph.begin_capture(stream)
        graph.replay(stream)
        graph.sync(stream)
        torch.testing.assert_close(tensor, other, rtol=0, atol=0)


@pytest.mark.parametrize('family', ['pi0', 'pi05'])
@pytest.mark.parametrize('borrowed', [False, True])
def test_pipeline_recapture_with_real_cuda_copy(graphs, family, borrowed):
    # Extract only the actual pipeline capture method; no model constructor/math.
    cls, runtime = graphs
    p = SimpleNamespace(use_fp8=False, autotune_gemms=lambda: None,
                        _cudart=runtime, _graph=None, _graph_stream=None)
    import ast
    path = ROOT / f'serving/flash_rt/models/{family}/pipeline_rtx.py'
    tree = ast.parse(path.read_text())
    container = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in container.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'record_infer_graph')
    import logging
    namespace = dict(CUDAGraph=cls, ctypes=ctypes, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    source = torch.arange(128, device='cuda', dtype=torch.int32)
    destination = torch.zeros_like(source)
    owner = torch.cuda.Stream() if borrowed else None
    def copy(stream):
        assert runtime.cudaMemcpyAsync(ctypes.c_void_p(destination.data_ptr()),
                                       ctypes.c_void_p(source.data_ptr()),
                                       ctypes.c_size_t(source.numel() * 4), 3,
                                       ctypes.c_void_p(stream)) == 0
    p.run_pipeline = copy
    for _ in range(10):
        torch.cuda.synchronize()
        old = p._graph
        namespace['record_infer_graph'](p, owner.cuda_stream if owner else None)
        if old is not None:
            assert not old.captured
        destination.zero_()
        torch.cuda.synchronize()
        p._graph.replay(p._graph_stream)
        p._graph.sync(p._graph_stream)
        torch.testing.assert_close(source, destination, rtol=0, atol=0)
    p._graph.close()
    if owner:
        with torch.cuda.stream(owner):
            destination.fill_(42)
        owner.synchronize()
        assert destination.min().item() == 42

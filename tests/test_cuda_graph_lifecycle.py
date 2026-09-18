"""Native ownership contracts with an instrumented CUDA runtime; no GPU needed."""
import ast
import ctypes
import gc
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

ROOT = Path(__file__).resolve().parents[1]


def value(arg):
    return arg.value if hasattr(arg, 'value') else arg


class Runtime:
    def __init__(self):
        self.calls = []
        self.current = 0
        self.next_handle = 100
        self.live = {'stream': {17: 0}, 'graph': {}, 'exec': {}}
        self.captures = set()
        self.fail = {}

    def allocate(self, kind, output):
        self.next_handle += 1
        self.live[kind][self.next_handle] = self.current
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = self.next_handle

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, tuple(value(a) for a in args), self.current))
            status = self.fail.pop(name, 0)
            if status:
                if name == 'cudaStreamEndCapture' and status in (901, 904):
                    self.captures.remove(value(args[0]))
                return status
            if name == 'cudaGetDevice':
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_int))[0] = self.current
            elif name == 'cudaSetDevice':
                self.current = value(args[0])
            elif name == 'cudaStreamCreate':
                self.allocate('stream', args[0])
            elif name == 'cudaStreamBeginCapture':
                assert value(args[0]) not in self.captures
                self.captures.add(value(args[0]))
            elif name == 'cudaStreamEndCapture':
                self.captures.remove(value(args[0]))
                self.allocate('graph', args[1])
            elif name == 'cudaStreamIsCapturing':
                ctypes.cast(args[1], ctypes.POINTER(ctypes.c_int))[0] = int(
                    value(args[0]) in self.captures)
            elif name == 'cudaGraphInstantiate':
                assert value(args[1]) in self.live['graph']
                self.allocate('exec', args[0])
            elif name in ('cudaGraphDestroy', 'cudaGraphExecDestroy', 'cudaStreamDestroy'):
                kind = {'cudaGraphDestroy': 'graph', 'cudaGraphExecDestroy': 'exec',
                        'cudaStreamDestroy': 'stream'}[name]
                pointer = value(args[0])
                assert self.live[kind][pointer] == self.current
                if kind == 'stream':
                    assert pointer not in self.captures
                del self.live[kind][pointer]
            elif name == 'cudaGraphLaunch':
                assert value(args[0]) in self.live['exec']
            elif name != 'cudaStreamSynchronize':
                raise AssertionError(f'Unexpected CUDA call: {name}')
            return 0
        return call

    def balanced(self):
        assert self.live == {'stream': {17: 0}, 'graph': {}, 'exec': {}}
        assert not self.captures


@pytest.fixture
def runtime(monkeypatch):
    cuda = Runtime()
    monkeypatch.setattr(ctypes, 'CDLL', lambda name: cuda)
    path = ROOT / 'serving/flash_rt/core/cuda_graph.py'
    spec = importlib.util.spec_from_file_location('_lifecycle_cuda_graph', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(cuda=cuda, graph=module.CUDAGraph)


def captured(h, borrowed=False):
    graph = h.graph()
    stream = ctypes.c_void_p(17) if borrowed else graph.create_stream()
    graph.begin_capture(stream)
    graph.end_capture(stream)
    return graph, stream


@pytest.mark.parametrize('borrowed', [False, True])
def test_close_releases_only_owned_resources_once(runtime, borrowed):
    h = runtime
    for _ in range(10):
        graph, stream = captured(h, borrowed)
        graph.replay(stream)
        graph.close()
        calls = len(h.cuda.calls)
        graph.close()
        assert len(h.cuda.calls) == calls
        assert not graph.captured
        with pytest.raises(RuntimeError, match='closed'):
            graph.replay(stream)
        h.cuda.balanced()


def test_unused_close_does_not_initialize_cuda(runtime):
    graph = runtime.graph()
    graph.close()
    assert not runtime.cuda.calls


def test_multiple_owned_streams_are_released(runtime):
    graph = runtime.graph()
    graph.create_stream()
    graph.create_stream()
    graph.close()
    runtime.cuda.balanced()


@pytest.mark.parametrize('borrowed', [False, True])
def test_finalizer_releases_abandoned_graph(runtime, borrowed):
    graph, _ = captured(runtime, borrowed)
    reference = weakref.ref(graph)
    del graph
    gc.collect()
    assert reference() is None
    runtime.cuda.balanced()


@pytest.mark.parametrize('borrowed', [False, True])
def test_exception_ends_capture_and_preserves_original_error(runtime, borrowed):
    with pytest.raises(ValueError, match='kernel failed'):
        with runtime.graph() as graph:
            stream = ctypes.c_void_p(17) if borrowed else graph.create_stream()
            graph.begin_capture(stream)
            raise ValueError('kernel failed')
    runtime.cuda.balanced()
    assert not any(n == 'cudaGraphInstantiate' for n, _, _ in runtime.cuda.calls)


def test_context_manager_closes_successful_capture(runtime):
    with runtime.graph() as graph:
        stream = graph.create_stream()
        graph.begin_capture(stream)
        graph.end_capture(stream)
        graph.replay(stream)
    runtime.cuda.balanced()


@pytest.mark.parametrize('failure', ['cudaStreamCreate', 'cudaStreamBeginCapture',
                                    'cudaStreamEndCapture', 'cudaGraphInstantiate'])
def test_capture_failure_releases_allocations(runtime, failure):
    runtime.cuda.fail[failure] = 901 if failure == 'cudaStreamEndCapture' else 2
    with pytest.raises(RuntimeError, match=failure):
        with runtime.graph() as graph:
            stream = graph.create_stream()
            graph.begin_capture(stream)
            graph.end_capture(stream)
    runtime.cuda.balanced()


def test_instantiate_failure_cleans_up_even_without_with(runtime):
    graph = runtime.graph()
    stream = graph.create_stream()
    graph.begin_capture(stream)
    runtime.cuda.fail['cudaGraphInstantiate'] = 2
    with pytest.raises(RuntimeError, match='cudaGraphInstantiate'):
        graph.end_capture(stream)
    runtime.cuda.balanced()
    assert not graph.captured


def test_invalidated_abandoned_capture_is_ended(runtime):
    graph = runtime.graph()
    stream = graph.create_stream()
    graph.begin_capture(stream)
    runtime.cuda.fail['cudaStreamEndCapture'] = 901
    graph.close()
    runtime.cuda.balanced()


def test_other_end_error_that_terminated_capture_does_not_leak_stream(runtime):
    graph = runtime.graph()
    stream = graph.create_stream()
    graph.begin_capture(stream)
    runtime.cuda.fail['cudaStreamEndCapture'] = 904  # unjoined capture
    with pytest.raises(RuntimeError, match='cudaStreamEndCapture'):
        graph.end_capture(stream)
    runtime.cuda.balanced()


def test_close_failure_can_retry_without_destroying_active_stream(runtime):
    graph = runtime.graph()
    stream = graph.create_stream()
    graph.begin_capture(stream)
    runtime.cuda.fail['cudaStreamEndCapture'] = 2
    with pytest.raises(RuntimeError, match='cudaStreamEndCapture'):
        graph.close()
    assert stream.value in runtime.cuda.captures
    assert stream.value in runtime.cuda.live['stream']
    graph.close()
    runtime.cuda.balanced()


@pytest.mark.parametrize('failure', ['cudaGraphDestroy', 'cudaGraphExecDestroy', 'cudaStreamDestroy'])
def test_failed_release_does_not_skip_other_cleanup(runtime, failure):
    graph, _ = captured(runtime)
    runtime.cuda.fail[failure] = 2
    with pytest.raises(RuntimeError, match=failure):
        graph.close()
    assert sum(len(v) for v in runtime.cuda.live.values()) == 2  # borrowed + failed handle
    graph.close()
    runtime.cuda.balanced()


def test_cleanup_restores_callers_current_device(runtime):
    graph, _ = captured(runtime)
    runtime.cuda.current = 1
    graph.close()
    assert runtime.cuda.current == 1
    runtime.cuda.balanced()


def test_cleanup_restores_device_after_error(runtime):
    graph, _ = captured(runtime)
    runtime.cuda.current = 1
    runtime.cuda.fail['cudaGraphDestroy'] = 2
    with pytest.raises(RuntimeError, match='cudaGraphDestroy'):
        graph.close()
    assert runtime.cuda.current == 1
    graph.close()
    runtime.cuda.balanced()


def test_another_capture_cannot_overwrite_live_handles(runtime):
    graph, stream = captured(runtime)
    with pytest.raises(RuntimeError, match='another capture'):
        graph.begin_capture(stream)
    graph.replay(stream)
    graph.close()
    runtime.cuda.balanced()


def test_wrong_end_stream_does_not_lose_capture(runtime):
    graph = runtime.graph()
    stream = graph.create_stream()
    graph.begin_capture(stream)
    with pytest.raises(RuntimeError, match='stream that began'):
        graph.end_capture(ctypes.c_void_p(17))
    graph.close()
    runtime.cuda.balanced()


def pipeline(h, family, fail_at=None):
    path = ROOT / f'serving/flash_rt/models/{family}/pipeline_rtx.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'record_infer_graph')
    namespace = {'CUDAGraph': h.graph, 'ctypes': ctypes, 'logger': logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    calls = 0
    def run(**kw):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise ValueError('pipeline failed')
    p = SimpleNamespace(use_fp8=False, autotune_gemms=lambda: None,
                        run_pipeline=run, _cudart=h.cuda, _graph=None, _graph_stream=None)
    return p, lambda borrowed: namespace['record_infer_graph'](p, 17 if borrowed else None)


@pytest.mark.parametrize('family', ['pi0', 'pi05'])
@pytest.mark.parametrize('borrowed', [False, True])
def test_actual_pipeline_recapture_releases_previous_graph(runtime, family, borrowed):
    p, record = pipeline(runtime, family)
    for _ in range(10):
        old = p._graph
        record(borrowed)
        if old is not None:
            assert not old.captured  # explicit close even when another reference survives
        assert len(runtime.cuda.live['graph']) == len(runtime.cuda.live['exec']) == 1
    p._graph.close()
    runtime.cuda.balanced()


@pytest.mark.parametrize('family', ['pi0', 'pi05'])
@pytest.mark.parametrize('fail_at', [1, 4])  # warmup, active capture
def test_pipeline_failure_closes_temporary_graph(runtime, family, fail_at):
    p, record = pipeline(runtime, family, fail_at)
    with pytest.raises(ValueError, match='pipeline failed'):
        record(False)
    assert p._graph is p._graph_stream is None
    runtime.cuda.balanced()

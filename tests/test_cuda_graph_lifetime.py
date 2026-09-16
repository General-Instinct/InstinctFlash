"""Raw CUDA graph handles must be released when dynamic profiles are replaced."""
import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_graph_close_releases_both_handles_once_and_refuses_replay(monkeypatch):
    calls = []
    runtime = SimpleNamespace(
        cudaGraphExecDestroy=lambda p: calls.append(("exec", p.value)) or 0,
        cudaGraphDestroy=lambda p: calls.append(("graph", p.value)) or 0,
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda name: runtime)
    path = Path(__file__).resolve().parents[1] / "serving/flash_rt/core/cuda_graph.py"
    spec = importlib.util.spec_from_file_location("isolated_cuda_graph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    graph = module.CUDAGraph()
    graph._graph_exec, graph._graph = ctypes.c_void_p(12), ctypes.c_void_p(34)
    graph._captured = True
    graph.close()
    graph.close()
    assert calls == [("exec", 12), ("graph", 34)]
    assert not graph.captured
    with pytest.raises(RuntimeError, match="No graph"):
        graph.replay(ctypes.c_void_p(56))

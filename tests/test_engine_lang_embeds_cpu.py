"""CPU lifetime regressions for real RTX pipeline and buffer/graph methods.

CUDA allocation, copies and graph replay are simulated. Model constructors and
math are bypassed; these tests do not establish actual CUDA ordering or outputs.
"""
import ctypes
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVING = ROOT / "serving"


def _address(value):
    return value.value if hasattr(value, "value") else int(value)


class CudaCalls:
    """Virtual device addresses; captured copies retain integers, not buffers."""

    def __init__(self):
        self.memory, self.graphs, self.functions = {}, {}, {}
        self.capture = None
        self.next_handle = 0x100000
        self.freed = []

    def handle(self):
        self.next_handle += 0x100000
        return self.next_handle

    def region(self, pointer, count):
        for base, data in self.memory.items():
            if base <= pointer and pointer + count <= base + len(data):
                return memoryview(data)[pointer - base:pointer - base + count]
        raise AssertionError(f"Copy references released/unallocated address {pointer:#x}")

    def copy(self, dst, src, count, kind):
        # Host inputs are real NumPy arrays; device addresses are never dereferenced.
        data = ctypes.string_at(src, count) if kind == 1 else bytes(self.region(src, count))
        assert kind in (1, 3), "Only H2D and D2D are needed by these tests"
        self.region(dst, count)[:] = data

    def __getattr__(self, name):
        if name not in self.functions:
            def call(*args):
                if name == "cudaGetDevice":
                    ctypes.cast(args[0], ctypes.POINTER(ctypes.c_int))[0] = 0
                elif name in ("cudaGraphDestroy", "cudaGraphExecDestroy"):
                    self.graphs.pop(_address(args[0]))
                elif name in ("cudaMalloc", "cudaMallocManaged"):
                    pointer = self.handle()
                    self.memory[pointer] = bytearray(_address(args[1]))
                    ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = pointer
                elif name == "cudaFree":
                    pointer = _address(args[0])
                    self.memory.pop(pointer)
                    self.freed.append(pointer)
                elif name in ("cudaMemcpy", "cudaMemcpyAsync"):
                    command = tuple(_address(arg) for arg in args[:4])
                    if self.capture is not None and name == "cudaMemcpyAsync":
                        self.capture.append(command)
                    else:
                        self.copy(*command)
                elif name == "cudaStreamBeginCapture":
                    self.capture = []
                elif name == "cudaStreamEndCapture":
                    pointer = self.handle()
                    self.graphs[pointer] = self.capture
                    self.capture = None
                    ctypes.cast(args[1], ctypes.POINTER(ctypes.c_void_p))[0] = pointer
                elif name == "cudaGraphInstantiate":
                    pointer = self.handle()
                    self.graphs[pointer] = list(self.graphs[_address(args[1])])
                    ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = pointer
                elif name == "cudaGraphLaunch":
                    for command in self.graphs[_address(args[0])]:
                        self.copy(*command)
                elif name not in ("cudaDeviceSynchronize", "cudaStreamSynchronize"):
                    raise AssertionError(f"Unexpected CUDA call: {name}")
                return 0
            self.functions[name] = call
        return self.functions[name]


@pytest.fixture(params=["pi0", "pi05"])
def pipeline(request, monkeypatch):
    cuda = CudaCalls()
    monkeypatch.syspath_prepend(str(SERVING))
    monkeypatch.setattr(ctypes, "CDLL", lambda name: cuda)
    # Only an unused dtype constant is needed while importing the Pi0.5 control.
    dtype_module = ModuleType("ml_dtypes")
    dtype_module.bfloat16 = np.uint16
    monkeypatch.setitem(sys.modules, "ml_dtypes", dtype_module)

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, SERVING / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    buffers = load("flash_rt.core.cuda_buffer", "flash_rt/core/cuda_buffer.py")
    load("flash_rt.core.cuda_graph", "flash_rt/core/cuda_graph.py")
    family = request.param
    module = load(f"_lang_embeds_{family}", f"flash_rt/models/{family}/pipeline_rtx.py")
    cls = module.Pi0Pipeline if family == "pi0" else module.Pi05Pipeline
    pipe = object.__new__(cls)
    pipe.max_prompt_len, pipe.vision_seq = 8, 1
    pipe._cudart = cuda
    pipe.use_fp8 = False
    pipe._graph = pipe._graph_stream = None
    pipe.bufs = {
        "encoder_x": buffers.CudaBuffer(9 * module.ENC_D * 2, managed=False),
        "diffusion_noise": buffers.CudaBuffer(64, managed=False),
    }
    # Retain real run_pipeline/capture/forward; replace only the model math.
    for name in ("_state_project", "vision_encoder", "transformer_encoder",
                 "transformer_decoder", "autotune_gemms"):
        setattr(pipe, name, lambda *args, **kwargs: None)
    pipe._set_decoder_rope_for_prompt = lambda length: setattr(pipe, "rope_length", length)
    return SimpleNamespace(pipe=pipe, cuda=cuda, width=module.ENC_D)


def embeds(h, fill, length=2, strided=False):
    values = np.full((length, h.width * (2 if strided else 1)), fill, dtype=np.uint16)
    return values[:, ::2] if strided else values


def assert_encoder(h, expected):
    pointer = h.pipe.bufs["encoder_x"].ptr.value + h.width * 2
    actual = bytes(h.cuda.region(pointer, expected.nbytes))
    assert actual == expected.tobytes()
    assert h.pipe._current_prompt_len == h.pipe.rope_length == len(expected)


def test_initial_prompt_copies_current_bytes(pipeline):
    h = pipeline
    expected = embeds(h, 1)
    h.pipe.set_language_embeds(expected)
    assert_encoder(h, expected)
    assert h.pipe._graph is None


@pytest.mark.parametrize("strided", [False, True])
def test_same_length_updates_preserve_captured_source(pipeline, strided):
    h = pipeline
    h.pipe.set_language_embeds(embeds(h, 1))
    h.pipe.record_infer_graph(external_stream_int=17)
    pointer, graph = h.pipe._lang_embeds_buf.ptr.value, h.pipe._graph
    allocations = len(h.cuda.memory)
    for value in (2, 3, 4):
        expected = embeds(h, value, strided=strided)
        h.pipe.set_language_embeds(expected)
        assert h.pipe._lang_embeds_buf.ptr.value == pointer
        assert pointer not in h.cuda.freed
        assert len(h.cuda.memory) == allocations
        assert h.pipe._graph is graph
        # Erase the immediate copy so only graph replay can produce the result.
        h.cuda.region(h.pipe.bufs["encoder_x"].ptr.value, 9 * h.width * 2)[:] = bytes(9 * h.width * 2)
        h.pipe.forward()
        assert_encoder(h, expected)


@pytest.mark.parametrize("old_length,new_length", [(2, 4), (4, 2)])
def test_size_change_invalidates_before_replay(pipeline, old_length, new_length):
    h = pipeline
    h.pipe.set_language_embeds(embeds(h, 1, old_length))
    h.pipe.record_infer_graph(external_stream_int=17)
    pointer = h.pipe._lang_embeds_buf.ptr.value
    expected = embeds(h, 2, new_length)
    h.pipe.set_language_embeds(expected)
    assert h.pipe._graph is None
    assert h.pipe._lang_embeds_buf.ptr.value != pointer
    assert pointer in h.cuda.freed
    h.pipe.forward()  # Existing eager fallback reads the replacement buffer.
    assert_encoder(h, expected)
    h.pipe.record_infer_graph(external_stream_int=17)
    h.pipe.forward()
    assert_encoder(h, expected)


def test_resize_without_a_graph(pipeline):
    h = pipeline
    h.pipe.set_language_embeds(embeds(h, 1))
    expected = embeds(h, 2, 3)
    h.pipe.set_language_embeds(expected)
    assert h.pipe._graph is None
    assert_encoder(h, expected)


@pytest.mark.parametrize("shape", [(9, 2048), (2, 2047)])
def test_invalid_shape_preserves_active_buffer_and_graph(pipeline, shape):
    h = pipeline
    expected = embeds(h, 1)
    h.pipe.set_language_embeds(expected)
    h.pipe.record_infer_graph(external_stream_int=17)
    pointer, graph = h.pipe._lang_embeds_buf.ptr.value, h.pipe._graph
    with pytest.raises(AssertionError):
        h.pipe.set_language_embeds(np.zeros(shape, dtype=np.uint16))
    assert h.pipe._lang_embeds_buf.ptr.value == pointer
    assert h.pipe._graph is graph
    h.pipe.forward()
    assert_encoder(h, expected)

"""RTX Pi0/Pi0.5 prompt buffers must remain valid across captured copies.

Same-length prompt swaps reuse the pipeline, so the graph's recorded source
address must keep referring to current embeddings. These GPU tests exercise real
allocation, upload and copy-only graph replay without model weights or kernels.
"""

import ctypes
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required: the contract under test is device-pointer stability",
                allow_module_level=True)

# the engine ships as source under serving/; the buffer semantics under test are plain
# cudaMalloc/cudaMemcpy and import fine on any CUDA machine (the SM-specific kernels do not
# load at import time)
import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / "serving"))
pytest.importorskip("ml_dtypes", reason="engine source imports ml_dtypes")

from flash_rt.core.cuda_buffer import CudaBuffer, _cudart  # noqa: E402
from flash_rt.models.pi0.pipeline_rtx import Pi0Pipeline  # noqa: E402
from flash_rt.models.pi05 import pipeline_rtx  # noqa: E402

ENC_D = pipeline_rtx.ENC_D


@pytest.fixture(params=[Pi0Pipeline, pipeline_rtx.Pi05Pipeline], ids=["pi0", "pi05"])
def pipeline_type(request):
    return request.param


class _Stub:
    """Just enough of either pipeline for set_language_embeds to run for real."""

    max_prompt_len = 48

    def __init__(self):
        self._graph = object()                    # pretend a graph was recorded

    def _set_decoder_rope_for_prompt(self, prompt_len):
        self.rope_len = prompt_len

    def _copy_lang_embeds_to_encoder_x(self, stream: int = 0):
        pass                                       # the captured copy; irrelevant to the contract


def _readback(buf: CudaBuffer) -> np.ndarray:
    out = np.empty(buf.nbytes, dtype=np.uint8)
    status = _cudart.cudaMemcpy(ctypes.c_void_p(out.ctypes.data), buf.ptr, buf.nbytes, 2)
    assert status == 0, f"D2H copy failed with CUDA status {status}"
    return out


def _embeds(fill: int, n: int = 10) -> np.ndarray:
    return np.full((n, ENC_D), fill, dtype=np.uint16)


def test_same_length_swap_keeps_pointer_and_updates_bytes(pipeline_type):
    p = _Stub()
    set_lang = types.MethodType(pipeline_type.set_language_embeds, p)

    set_lang(_embeds(1))
    first_ptr = p._lang_embeds_buf.ptr.value

    set_lang(_embeds(2))                           # same length: the frontend fast path
    assert p._lang_embeds_buf.ptr.value == first_ptr, (
        "same-length prompt swap reallocated the buffer: the pointer baked into any captured "
        "graph now dangles — this is the exact bug the fix removed")
    got = _readback(p._lang_embeds_buf).view(np.uint16)
    assert int(got[0]) == 2, "pointer kept but bytes not updated: replay would serve prompt #1"
    assert p._graph is not None                    # graph stays valid for same-length swaps


def test_length_change_drops_recorded_graph(pipeline_type):
    p = _Stub()
    set_lang = types.MethodType(pipeline_type.set_language_embeds, p)

    set_lang(_embeds(1, n=10))
    set_lang(_embeds(3, n=20))                     # different length: shapes changed anyway
    assert p._graph is None, (
        "a graph captured for the old length/pointer survived a size change; replaying it "
        "would read a freed buffer at the old baked address")


@pytest.mark.parametrize("lengths", [(10, 10, 10), (10, 20, 5)])
def test_captured_copy_reads_current_prompt(pipeline_type, lengths):
    # Only encoder storage and the prompt buffer are allocated. PyTorch owns the
    # copy-only graphs so this test does not depend on raw graph cleanup changes.
    p = object.__new__(pipeline_type)
    p.max_prompt_len, p.vision_seq = 48, 1
    p._cudart = _cudart
    p._graph = None
    p._set_decoder_rope_for_prompt = lambda length: None
    p.bufs = {"encoder_x": CudaBuffer((p.max_prompt_len + 1) * ENC_D * 2, managed=False)}
    stream = torch.cuda.Stream()
    previous_length = previous_pointer = None

    for fill, length in enumerate(lengths, start=1):
        expected = _embeds(fill, length)
        graph = p._graph
        p.set_language_embeds(expected)
        torch.cuda.synchronize()
        if length == previous_length:
            # Fail before replay on old code, avoiding a deliberate invalid read.
            assert p._lang_embeds_buf.ptr.value == previous_pointer
            assert p._graph is graph
        else:
            assert p._graph is None
            # Warm up the exact copy before capturing it on a non-default stream.
            p._copy_lang_embeds_to_encoder_x(stream.cuda_stream)
            stream.synchronize()
            p._graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                p._graph.capture_begin()
                p._copy_lang_embeds_to_encoder_x(stream.cuda_stream)
                p._graph.capture_end()

        # Clear the immediate upload's result: only replay can restore the bytes.
        p.bufs["encoder_x"].zero_()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            p._graph.replay()
        stream.synchronize()
        actual = _readback(p.bufs["encoder_x"]).view(np.uint16).reshape(-1, ENC_D)
        np.testing.assert_array_equal(actual[1:1 + length], expected)
        previous_length, previous_pointer = length, p._lang_embeds_buf.ptr.value

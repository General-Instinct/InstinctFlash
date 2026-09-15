"""The engine's language-embed buffer must be pointer-stable across same-length prompt swaps.

WHY THIS TEST EXISTS. `Pi05Pipeline.record_infer_graph` captures `run_pipeline`, which issues a
D2D memcpy whose SOURCE is `self._lang_embeds_buf.ptr` — that pointer value is baked into the
graph. The rtx frontend's `set_prompt` fast path skips pipeline rebuild when the new prompt has
the same token length, then calls `set_language_embeds` again. The original implementation
allocated a fresh `CudaBuffer` there, freeing the old one while the captured graph still read
it: every subsequent replay sourced freed memory. This test pins the fixed contract at the
method level (the kernels themselves need SM89/SM110 and cannot run here; the buffer semantics
are architecture-independent cudaMalloc/cudaMemcpy and can).
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
from flash_rt.models.pi05 import pipeline_rtx  # noqa: E402

ENC_D = pipeline_rtx.ENC_D


class _Stub:
    """Just enough of Pi05Pipeline for set_language_embeds to run for real."""

    max_prompt_len = 48

    def __init__(self):
        self._graph = object()                    # pretend a graph was recorded

    def _set_decoder_rope_for_prompt(self, prompt_len):
        self.rope_len = prompt_len

    def _copy_lang_embeds_to_encoder_x(self, stream: int = 0):
        pass                                       # the captured copy; irrelevant to the contract


def _readback(buf: CudaBuffer) -> np.ndarray:
    out = np.empty(buf.nbytes, dtype=np.uint8)
    _cudart.cudaMemcpy(ctypes.c_void_p(out.ctypes.data), buf.ptr, buf.nbytes, 2)  # D2H
    return out


def _embeds(fill: int, n: int = 10) -> np.ndarray:
    return np.full((n, ENC_D), fill, dtype=np.uint16)


def test_same_length_swap_keeps_pointer_and_updates_bytes():
    p = _Stub()
    set_lang = types.MethodType(pipeline_rtx.Pi05Pipeline.set_language_embeds, p)

    set_lang(_embeds(1))
    first_ptr = p._lang_embeds_buf.ptr.value

    set_lang(_embeds(2))                           # same length: the frontend fast path
    assert p._lang_embeds_buf.ptr.value == first_ptr, (
        "same-length prompt swap reallocated the buffer: the pointer baked into any captured "
        "graph now dangles — this is the exact bug the fix removed")
    got = _readback(p._lang_embeds_buf).view(np.uint16)
    assert int(got[0]) == 2, "pointer kept but bytes not updated: replay would serve prompt #1"
    assert p._graph is not None                    # graph stays valid for same-length swaps


def test_length_change_drops_recorded_graph():
    p = _Stub()
    set_lang = types.MethodType(pipeline_rtx.Pi05Pipeline.set_language_embeds, p)

    set_lang(_embeds(1, n=10))
    set_lang(_embeds(3, n=20))                     # different length: shapes changed anyway
    assert p._graph is None, (
        "a graph captured for the old length/pointer survived a size change; replaying it "
        "would read a freed buffer at the old baked address")

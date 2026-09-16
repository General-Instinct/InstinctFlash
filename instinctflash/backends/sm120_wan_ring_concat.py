"""P009-A6: bitexact SM120 K/V ring-wrap concatenation.

P003 presents a wrapped live interval in ascending physical-slot order by concatenating two
pool slices. A6 copies both K and V in one vectorized kernel into one shared scratch arena.
No floating-point operation changes; the same BF16 words are written in the same logical order.
"""

from __future__ import annotations

import ctypes
import os
from collections import Counter
from pathlib import Path

import torch

from instinctflash.backends.sm120_wan_stage2 import (
    CERTIFIED_CUDA_VERSION,
    CERTIFIED_TORCH_VERSION,
    REQUIRED_ALIGNMENT,
)

LIBRARY_ENV = "IFL_SM120_RING_CONCAT_LIBRARY"
LIBRARY_NAME = "libinstinctflash_sm120_wan_ring_concat.so"
ABI_VERSION = 1
CERTIFIED_BATCH = 2
CERTIFIED_TOTAL = 9792
CERTIFIED_HEADS = 24
CERTIFIED_HEAD_DIM = 128
CERTIFIED_INNER = CERTIFIED_HEADS * CERTIFIED_HEAD_DIM
CERTIFIED_MODULES = 30


def library_candidates() -> tuple[Path, ...]:
    configured = os.environ.get(LIBRARY_ENV)
    packaged = Path(__file__).resolve().parents[1] / "native" / LIBRARY_NAME
    return tuple([Path(configured)] if configured else []) + (packaged,)


def _library_abi(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        library = ctypes.CDLL(str(path))
        version = library.instinctflash_sm120_wan_ring_concat_abi_version
        version.argtypes = []
        version.restype = ctypes.c_int
        return int(version())
    except (OSError, AttributeError):
        return None


def available() -> bool:
    return any(_library_abi(path) == ABI_VERSION for path in library_candidates())


def resolve_library() -> Path:
    for path in library_candidates():
        if _library_abi(path) == ABI_VERSION:
            return path
    searched = ", ".join(str(path) for path in library_candidates())
    raise RuntimeError(
        f"SM120 Wan ring-concat ABI v{ABI_VERSION} unavailable; searched {searched}. "
        f"Build instinctflash/native/CMakeLists.txt or set {LIBRARY_ENV}."
    )


class SM120WanRingConcatKernels:
    """One shared, persistent K/V scratch arena and the independent A6 ABI."""

    def __init__(self, library: str | Path | None = None):
        if not torch.cuda.is_available():
            raise RuntimeError("P009-A6 requires CUDA")
        index = torch.cuda.current_device()
        if torch.cuda.get_device_capability(index) != (12, 0):
            raise RuntimeError("P009-A6 is certified only on SM120")
        if (
            torch.__version__ != CERTIFIED_TORCH_VERSION
            or torch.version.cuda != CERTIFIED_CUDA_VERSION
        ):
            raise RuntimeError(
                f"P009-A6 requires torch={CERTIFIED_TORCH_VERSION}, "
                f"CUDA={CERTIFIED_CUDA_VERSION}; got torch={torch.__version__}, "
                f"CUDA={torch.version.cuda}"
            )

        path = Path(library) if library is not None else resolve_library()
        abi = _library_abi(path)
        if abi != ABI_VERSION:
            raise RuntimeError(f"expected A6 ABI v{ABI_VERSION}, got {abi} from {path}")
        self.path = path
        self.device = torch.device("cuda", index)
        self.library = ctypes.CDLL(str(path))
        u64 = ctypes.c_uint64
        self._run = self.library.wan_ring_concat_bf16
        self._run.argtypes = [
            u64,
            u64,
            u64,
            u64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            u64,
        ]
        self._run.restype = ctypes.c_int

        shape = (
            CERTIFIED_BATCH,
            CERTIFIED_TOTAL,
            CERTIFIED_HEADS,
            CERTIFIED_HEAD_DIM,
        )
        self.key_scratch = torch.empty(
            shape,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.value_scratch = torch.empty_like(self.key_scratch)
        self._stream = None
        self.calls = Counter()

    @staticmethod
    def _check_tensor(name: str, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA")
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{name} must be torch.bfloat16, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % REQUIRED_ALIGNMENT:
            raise ValueError(f"{name} must be {REQUIRED_ALIGNMENT}-byte aligned")

    def _validate(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        start: int,
        count: int,
        total: int,
    ) -> None:
        self._check_tensor("key", key)
        self._check_tensor("value", value)
        expected = (
            CERTIFIED_BATCH,
            CERTIFIED_TOTAL,
            CERTIFIED_HEADS,
            CERTIFIED_HEAD_DIM,
        )
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(
                f"A6 requires K/V shape {expected}, got "
                f"{tuple(key.shape)} and {tuple(value.shape)}"
            )
        if key.device != self.device or value.device != self.device:
            raise ValueError(f"A6 K/V must be on {self.device}")
        if key.data_ptr() == value.data_ptr():
            raise ValueError("A6 key and value must not alias")
        if int(total) != CERTIFIED_TOTAL:
            raise ValueError(f"A6 requires total={CERTIFIED_TOTAL}, got {total}")
        start = int(start)
        count = int(count)
        if (
            start < 0
            or start >= CERTIFIED_TOTAL
            or count <= 0
            or count >= CERTIFIED_TOTAL
            or start + count <= CERTIFIED_TOTAL
            or start + count > 2 * CERTIFIED_TOTAL
        ):
            raise ValueError(
                f"A6 requires a wrapped live interval, got start={start}, "
                f"count={count}, total={total}"
            )

    def _stream_for(self, tensor: torch.Tensor) -> int:
        stream = torch.cuda.current_stream(tensor.device).cuda_stream
        if self._stream is None:
            self._stream = stream
        elif self._stream != stream:
            raise RuntimeError("P009-A6 requires one CUDA stream per Runtime")
        return stream

    def concat(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        start: int,
        count: int,
        total: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(
            key,
            value,
            start=start,
            count=count,
            total=total,
        )
        status = self._run(
            key.data_ptr(),
            value.data_ptr(),
            self.key_scratch.data_ptr(),
            self.value_scratch.data_ptr(),
            CERTIFIED_BATCH,
            CERTIFIED_TOTAL,
            CERTIFIED_INNER,
            int(start),
            int(count),
            self._stream_for(key),
        )
        if status:
            raise RuntimeError(f"A6 CUDA launch failed with error {status}")
        count = int(count)
        words = CERTIFIED_BATCH * count * CERTIFIED_INNER
        shape = (
            CERTIFIED_BATCH,
            count,
            CERTIFIED_HEADS,
            CERTIFIED_HEAD_DIM,
        )
        key_out = self.key_scratch.view(-1)[:words].view(shape)
        value_out = self.value_scratch.view(-1)[:words].view(shape)
        self.calls[count] += 1
        return key_out, value_out


def install_wan_ring_concat(
    transformer,
    kernels: SM120WanRingConcatKernels | None = None,
) -> SM120WanRingConcatKernels:
    if getattr(transformer, "_ifl_wan_gemm_kernels", None) is None:
        raise RuntimeError("P009-A6 requires the installed A1-A5 chain")
    blocks = list(transformer.blocks)
    if len(blocks) != CERTIFIED_MODULES or not all(
        getattr(block.attn1, "_ifl_wan_qk_rope_installed", False) for block in blocks
    ):
        raise RuntimeError("P009-A6 requires all 30 A4 self-attention sites")

    kernels = kernels or SM120WanRingConcatKernels()
    for index, block in enumerate(blocks):
        attention = block.attn1
        if getattr(attention, "_ifl_wan_ring_concat_installed", False):
            raise RuntimeError(f"P009-A6 duplicate install at block {index}")
        attention._iwm_ring_concat = kernels.concat
        attention._ifl_wan_ring_concat_installed = True

    transformer._ifl_wan_ring_concat_kernels = kernels
    transformer._ifl_wan_ring_concat_sites = tuple(
        f"blocks.{index}.attn1" for index in range(len(blocks))
    )
    return kernels


__all__ = [
    "ABI_VERSION",
    "CERTIFIED_BATCH",
    "CERTIFIED_HEADS",
    "CERTIFIED_HEAD_DIM",
    "CERTIFIED_INNER",
    "CERTIFIED_MODULES",
    "CERTIFIED_TOTAL",
    "LIBRARY_ENV",
    "LIBRARY_NAME",
    "SM120WanRingConcatKernels",
    "available",
    "install_wan_ring_concat",
    "resolve_library",
]

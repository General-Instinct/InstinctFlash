"""P009-A7: bitexact parallel SM120 Q/K/V projections."""

from __future__ import annotations

import ctypes
import os
import types
import weakref
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from instinctflash.backends.sm120_wan_gemm import CERTIFIED_CUBLASLT_VERSION
from instinctflash.backends.sm120_wan_stage2 import (
    CERTIFIED_CUDA_VERSION,
    CERTIFIED_TORCH_VERSION,
    REQUIRED_ALIGNMENT,
)

LIBRARY_ENV = "IFL_SM120_QKV_PARALLEL_LIBRARY"
LIBRARY_NAME = "libinstinctflash_sm120_wan_qkv_parallel.so"
ABI_VERSION = 1
CERTIFIED_DIM = 3072
CERTIFIED_MODULES = 30
CERTIFIED_CONFIGS = {
    (64, 3072, 3072): (21, 11, 14, 0, 0),
    (480, 3072, 3072): (21, 15, 25, 0, 0),
}


def library_candidates() -> tuple[Path, ...]:
    configured = os.environ.get(LIBRARY_ENV)
    packaged = Path(__file__).resolve().parents[1] / "native" / LIBRARY_NAME
    return tuple([Path(configured)] if configured else []) + (packaged,)


def _library_value(path: Path, symbol: str, restype) -> int | None:
    if not path.is_file():
        return None
    try:
        function = getattr(ctypes.CDLL(str(path)), symbol)
        function.argtypes = []
        function.restype = restype
        return int(function())
    except (OSError, AttributeError):
        return None


def _library_abi(path: Path) -> int | None:
    return _library_value(
        path, "instinctflash_sm120_wan_qkv_parallel_abi_version", ctypes.c_int
    )


def _library_cublaslt_version(path: Path) -> int | None:
    return _library_value(path, "wan_qkv_parallel_cublaslt_version", ctypes.c_uint64)


def available() -> bool:
    return any(
        _library_abi(path) == ABI_VERSION
        and _library_cublaslt_version(path) == CERTIFIED_CUBLASLT_VERSION
        for path in library_candidates()
    )


def resolve_library() -> Path:
    for path in library_candidates():
        if (
            _library_abi(path) == ABI_VERSION
            and _library_cublaslt_version(path) == CERTIFIED_CUBLASLT_VERSION
        ):
            return path
    searched = ", ".join(str(path) for path in library_candidates())
    raise RuntimeError(
        f"SM120 Wan parallel-QKV ABI v{ABI_VERSION} with cuBLASLt "
        f"{CERTIFIED_CUBLASLT_VERSION} unavailable; searched {searched}."
    )


@dataclass(frozen=True)
class ParallelQKVPlan:
    index: int
    m: int
    n: int
    k: int
    weight_ptrs: tuple[int, int, int]
    bias_ptrs: tuple[int, int, int]
    weight_versions: tuple[int, int, int]
    bias_versions: tuple[int, int, int]


class SM120WanParallelQKVKernels:
    def __init__(self, library: str | Path | None = None):
        if not torch.cuda.is_available():
            raise RuntimeError("P009-A7 requires CUDA")
        index = torch.cuda.current_device()
        if torch.cuda.get_device_capability(index) != (12, 0):
            raise RuntimeError("P009-A7 is certified only on SM120")
        if (
            torch.__version__ != CERTIFIED_TORCH_VERSION
            or torch.version.cuda != CERTIFIED_CUDA_VERSION
        ):
            raise RuntimeError(
                f"P009-A7 requires torch={CERTIFIED_TORCH_VERSION}, "
                f"CUDA={CERTIFIED_CUDA_VERSION}; got torch={torch.__version__}, "
                f"CUDA={torch.version.cuda}"
            )
        path = Path(library) if library is not None else resolve_library()
        if _library_abi(path) != ABI_VERSION:
            raise RuntimeError(f"expected A7 ABI v{ABI_VERSION} from {path}")
        version = _library_cublaslt_version(path)
        if version != CERTIFIED_CUBLASLT_VERSION:
            raise RuntimeError(
                f"P009-A7 requires cuBLASLt {CERTIFIED_CUBLASLT_VERSION}, got {version}"
            )
        self.path = path
        self.device = torch.device("cuda", index)
        self.library = ctypes.CDLL(str(path))
        u64 = ctypes.c_uint64
        self._create = self.library.wan_qkv_parallel_context_create
        self._create.argtypes = []
        self._create.restype = u64
        self._register = self.library.wan_qkv_parallel_plan_register
        self._register.argtypes = [
            u64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self._register.restype = ctypes.c_int
        self._run = self.library.wan_qkv_parallel_bf16
        self._run.argtypes = [u64, ctypes.c_int] + [u64] * 11
        self._run.restype = ctypes.c_int
        self._last = self.library.wan_qkv_parallel_context_last_status
        self._last.argtypes = [u64]
        self._last.restype = ctypes.c_int
        self._destroy = self.library.wan_qkv_parallel_context_destroy
        self._destroy.argtypes = [u64]
        self._destroy.restype = None
        self.context = int(self._create())
        if not self.context:
            raise RuntimeError(
                "P009-A7 could not create its streams/events/cuBLASLt handles"
            )
        self._finalizer = weakref.finalize(self, self._destroy, self.context)
        self._caller_stream = None
        self.calls = Counter()

    def _check_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be Tensor")
        if not tensor.is_cuda or tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{name} must be torch.bfloat16, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % REQUIRED_ALIGNMENT:
            raise ValueError(f"{name} must be {REQUIRED_ALIGNMENT}-byte aligned")

    def register_attention(self, attention, m: int) -> ParallelQKVPlan:
        linears = (attention.to_q, attention.to_k, attention.to_v)
        if not all(isinstance(linear, torch.nn.Linear) for linear in linears):
            raise TypeError("P009-A7 requires three nn.Linear projections")
        n, k = map(int, linears[0].weight.shape)
        config = CERTIFIED_CONFIGS.get((int(m), n, k))
        if config is None or any(
            tuple(linear.weight.shape) != (n, k) for linear in linears
        ):
            raise ValueError(f"uncertified A7 QKV shape {(m, n, k)}")
        if any(linear.bias is None for linear in linears):
            raise ValueError("P009-A7 requires three bias epilogues")
        for index, linear in enumerate(linears):
            self._check_tensor(f"weight[{index}]", linear.weight)
            self._check_tensor(f"bias[{index}]", linear.bias)
        weights = tuple(linear.weight.data_ptr() for linear in linears)
        biases = tuple(linear.bias.data_ptr() for linear in linears)
        pointers = tuple(
            pointer
            for linear in linears
            for pointer in (linear.weight.data_ptr(), linear.bias.data_ptr())
        )
        index = self._register(self.context, m, n, k, *pointers, *config)
        if index < 0:
            raise RuntimeError(
                f"A7 tactic registration failed for {(m, n, k)}: "
                f"status {self._last(self.context)}"
            )
        return ParallelQKVPlan(
            index,
            int(m),
            n,
            k,
            weights,
            biases,
            tuple(linear.weight._version for linear in linears),
            tuple(linear.bias._version for linear in linears),
        )

    def can_project(self, plan: ParallelQKVPlan, x, linears) -> bool:
        return (
            isinstance(x, torch.Tensor)
            and x.is_cuda
            and x.device == self.device
            and x.dtype is torch.bfloat16
            and x.is_contiguous()
            and x.shape[-1] == plan.k
            and x.numel() == plan.m * plan.k
            and tuple(linear.weight.data_ptr() for linear in linears)
            == plan.weight_ptrs
            and tuple(linear.bias.data_ptr() for linear in linears) == plan.bias_ptrs
            and tuple(linear.weight._version for linear in linears)
            == plan.weight_versions
            and tuple(linear.bias._version for linear in linears) == plan.bias_versions
        )

    def project_into(self, plan, x, linears, outputs):
        self._check_tensor("x", x)
        if not self.can_project(plan, x, linears):
            raise ValueError("A7 input or pinned QKV parameters changed")
        if len(outputs) != 3:
            raise ValueError("A7 requires three outputs")
        expected = (*x.shape[:-1], plan.n)
        for index, output in enumerate(outputs):
            self._check_tensor(f"output[{index}]", output)
            if tuple(output.shape) != expected:
                raise ValueError(f"A7 output[{index}] must have shape {expected}")
        output_ptrs = tuple(output.data_ptr() for output in outputs)
        protected = {x.data_ptr(), *plan.weight_ptrs, *plan.bias_ptrs}
        if len(set(output_ptrs)) != 3 or protected.intersection(output_ptrs):
            raise ValueError(
                "A7 outputs must be distinct and may not alias inputs or parameters"
            )
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if self._caller_stream is None:
            self._caller_stream = stream
        elif self._caller_stream != stream:
            raise RuntimeError("P009-A7 requires one caller stream per Runtime")
        weights = tuple(linear.weight.data_ptr() for linear in linears)
        biases = tuple(linear.bias.data_ptr() for linear in linears)
        status = self._run(
            self.context,
            plan.index,
            x.data_ptr(),
            weights[0],
            biases[0],
            weights[1],
            biases[1],
            weights[2],
            biases[2],
            outputs[0].data_ptr(),
            outputs[1].data_ptr(),
            outputs[2].data_ptr(),
            stream,
        )
        if status:
            raise RuntimeError(f"A7 parallel QKV launch failed with status {status}")
        self.calls[plan.m] += 1
        return outputs


def install_wan_qkv_parallel(
    transformer, kernels: SM120WanParallelQKVKernels | None = None
) -> SM120WanParallelQKVKernels:
    if getattr(transformer, "_ifl_wan_ring_concat_kernels", None) is None:
        raise RuntimeError("P009-A7 requires the installed A1-A6 chain")
    blocks = list(transformer.blocks)
    if len(blocks) != CERTIFIED_MODULES or not all(
        getattr(block.attn1, "_ifl_wan_ring_concat_installed", False)
        for block in blocks
    ):
        raise RuntimeError("P009-A7 requires all 30 A6 self-attention sites")
    kernels = kernels or SM120WanParallelQKVKernels()
    for block_index, block in enumerate(blocks):
        attention = block.attn1
        if getattr(attention, "_ifl_wan_qkv_parallel_installed", False):
            raise RuntimeError(f"P009-A7 duplicate install at block {block_index}")
        linears = (attention.to_q, attention.to_k, attention.to_v)
        originals = tuple(linear.forward for linear in linears)
        plans = {m: kernels.register_attention(attention, m) for m in (64, 480)}
        buffers = {}
        state = {"pending": None}

        def q_forward(
            self,
            x,
            _linears=linears,
            _original=originals[0],
            _plans=plans,
            _buffers=buffers,
            _state=state,
        ):
            _state["pending"] = None
            m = (
                x.numel() // x.shape[-1]
                if isinstance(x, torch.Tensor) and x.ndim
                else -1
            )
            plan = _plans.get(m)
            if plan is None or not kernels.can_project(plan, x, _linears):
                return _original(x)
            outputs = _buffers.get(m)
            if outputs is None:
                shape = (*x.shape[:-1], plan.n)
                outputs = _buffers[m] = tuple(
                    torch.empty(shape, device=x.device, dtype=x.dtype) for _ in range(3)
                )
            kernels.project_into(plan, x, _linears, outputs)
            _state["pending"] = (x, outputs[1], outputs[2])
            return outputs[0]

        def k_forward(self, x, _original=originals[1], _state=state):
            pending = _state["pending"]
            if pending is not None and pending[0] is x:
                return pending[1]
            _state["pending"] = None
            return _original(x)

        def v_forward(self, x, _original=originals[2], _state=state):
            pending = _state["pending"]
            if pending is not None and pending[0] is x:
                _state["pending"] = None
                return pending[2]
            _state["pending"] = None
            return _original(x)

        candidates = (
            types.MethodType(q_forward, linears[0]),
            types.MethodType(k_forward, linears[1]),
            types.MethodType(v_forward, linears[2]),
        )
        for linear, candidate in zip(linears, candidates):
            linear.forward = candidate
        attention._ifl_wan_qkv_parallel_originals = originals
        attention._ifl_wan_qkv_parallel_candidates = candidates
        attention._ifl_wan_qkv_parallel_buffers = buffers
        attention._ifl_wan_qkv_parallel_installed = True
    transformer._ifl_wan_qkv_parallel_kernels = kernels
    transformer._ifl_wan_qkv_parallel_sites = tuple(
        f"blocks.{index}.attn1" for index in range(len(blocks))
    )
    return kernels


__all__ = [
    "ABI_VERSION",
    "CERTIFIED_CONFIGS",
    "CERTIFIED_DIM",
    "CERTIFIED_MODULES",
    "LIBRARY_ENV",
    "LIBRARY_NAME",
    "ParallelQKVPlan",
    "SM120WanParallelQKVKernels",
    "available",
    "install_wan_qkv_parallel",
    "resolve_library",
]

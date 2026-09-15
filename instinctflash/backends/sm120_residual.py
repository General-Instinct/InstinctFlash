"""SM120 Wan gated-residual kernel and block installer.

This is the first narrow part of the LingBot-VA SM120 engine.  It replaces two eager chains per
Transformer block with one CUDA launch while preserving the eager FP32 multiply/add boundaries and
the final BF16 rounding bit-for-bit.  The raw-pointer ABI mirrors FlashRT and deliberately avoids a
PyTorch C++ ABI dependency.
"""

from __future__ import annotations

import ctypes
import os
import types
from pathlib import Path


LIBRARY_ENV = "IFL_SM120_KERNEL_LIBRARY"
LIBRARY_NAME = "libinstinctflash_sm120.so"
ABI_VERSION = 1
CERTIFIED_BLOCK_FORWARD_SHA256 = "8ea4a257a9bcfbb1c42f1a8a354e3d2a206151e3da1e1576ae14fe053d85b77e"
CERTIFIED_ROWS = frozenset({64, 480})
CERTIFIED_DIM = 3072


def library_candidates() -> tuple[Path, ...]:
    """Explicit override first, then the package-native build output."""
    configured = os.environ.get(LIBRARY_ENV)
    packaged = Path(__file__).resolve().parents[1] / "native" / LIBRARY_NAME
    return tuple([Path(configured)] if configured else []) + (packaged,)


def _library_abi(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        library = ctypes.CDLL(str(path))
        version = library.instinctflash_sm120_abi_version
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
        f"SM120 gated-residual ABI v{ABI_VERSION} is unavailable; searched {searched}. Build "
        "instinctflash/native/CMakeLists.txt or point IFL_SM120_KERNEL_LIBRARY at the compiled "
        f"{LIBRARY_NAME}. The pass refuses instead of claiming an optimization it did not run."
    )


class SM120GatedResidualKernel:
    """Validated raw-pointer launcher for the certified LingBot production shapes."""

    def __init__(self, library: str | Path | None = None):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("wan_gate_residual_bf16 requires CUDA")
        device_index = torch.cuda.current_device()
        if torch.cuda.get_device_capability(device_index) != (12, 0):
            raise RuntimeError("wan_gate_residual_bf16 is certified only on SM120")
        path = Path(library) if library is not None else resolve_library()
        abi = _library_abi(path)
        if abi != ABI_VERSION:
            raise RuntimeError(f"expected SM120 ABI v{ABI_VERSION}, got {abi} from {path}")
        self.path = path
        self.device = torch.device("cuda", device_index)
        self.lib = ctypes.CDLL(str(path))
        fn = self.lib.wan_gate_residual_bf16
        fn.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        fn.restype = ctypes.c_int
        self._fn = fn
        self.calls = 0
        self._stream = None
        self._validated_outputs = set()

    def _validate(self, hidden, update, gate, output) -> tuple[int, int]:
        import torch

        tensors = {"hidden": hidden, "update": update, "gate": gate, "output": output}
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be CUDA-resident")
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on {self.device}, got {tensor.device}")
        if hidden.dtype is not torch.bfloat16 or update.dtype is not torch.bfloat16:
            raise TypeError("hidden and update must be BF16")
        if output.dtype is not torch.bfloat16 or gate.dtype is not torch.float32:
            raise TypeError("output must be BF16 and gate must be FP32")
        if hidden.shape != update.shape or hidden.shape != output.shape:
            raise ValueError("hidden, update, and output shapes must match")
        if hidden.ndim not in (2, 3) or gate.ndim != hidden.ndim:
            raise ValueError("hidden/update/output/gate must all be rank 2 or rank 3")
        if not hidden.is_contiguous() or not update.is_contiguous() or not output.is_contiguous():
            raise ValueError("hidden, update, and output must be contiguous")
        if gate.shape != hidden.shape or gate.stride(-1) != 1:
            raise ValueError("gate must have the hidden shape with a contiguous feature dimension")
        if gate.stride(-2) < gate.shape[-1]:
            raise ValueError("gate rows overlap or run backwards")
        if gate.ndim == 3 and gate.stride(-3) != gate.shape[-2] * gate.stride(-2):
            raise ValueError("gate leading dimensions do not flatten to a constant row stride")
        spans = {}
        for name, tensor in tensors.items():
            if any(stride < 0 for stride in tensor.stride()):
                raise ValueError(f"{name} has a negative stride")
            last = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
            begin = tensor.data_ptr()
            spans[name] = (begin, begin + (last + 1) * tensor.element_size())
        names = tuple(spans)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                if max(spans[left][0], spans[right][0]) < min(spans[left][1], spans[right][1]):
                    raise ValueError(
                        f"restrict operands {left}/{right} overlap: {spans[left]} vs {spans[right]}"
                    )
        dim = hidden.shape[-1]
        rows = hidden.numel() // dim
        if dim != CERTIFIED_DIM or rows not in CERTIFIED_ROWS:
            raise ValueError(
                f"uncertified Wan residual shape: rows={rows}, dim={dim}; certified rows are "
                f"{sorted(CERTIFIED_ROWS)} at dim={CERTIFIED_DIM}"
            )
        return rows, dim

    def _launch(self, hidden, update, gate, output, rows, dim) -> None:
        import torch

        stream = torch.cuda.current_stream(hidden.device).cuda_stream
        if self._stream is None:
            self._stream = stream
        elif stream != self._stream:
            raise RuntimeError("P009-A1 is certified for one CUDA stream per Runtime")
        error = self._fn(
            hidden.data_ptr(),
            update.data_ptr(),
            gate.data_ptr(),
            output.data_ptr(),
            rows,
            dim,
            gate.stride(-2),
            stream,
        )
        if error:
            raise RuntimeError(f"wan_gate_residual_bf16 launch failed with CUDA error {error}")
        self.calls += 1

    def launch_into(self, hidden, update, gate, output) -> None:
        """Public fully-validated launch; safe for arbitrary callers."""
        rows, dim = self._validate(hidden, update, gate, output)
        self._launch(hidden, update, gate, output, rows, dim)

    def launch_certified(self, hidden, update, gate, output) -> None:
        """Amortize validation for installer's persistent per-block output buffers."""
        key = (output.data_ptr(), tuple(hidden.shape), tuple(gate.stride()))
        if key not in self._validated_outputs:
            rows, dim = self._validate(hidden, update, gate, output)
            self._validated_outputs.add(key)
        else:
            dim = hidden.shape[-1]
            rows = hidden.numel() // dim
        self._launch(hidden, update, gate, output, rows, dim)


def install_wan_blocks(transformer, kernel: SM120GatedResidualKernel | None = None):
    """Install the two exact residual fusions on every Wan block instance."""
    import hashlib
    import inspect
    import torch
    from einops import rearrange

    kernel = kernel or SM120GatedResidualKernel()
    blocks = list(transformer.blocks)
    if len(blocks) != 30:
        raise RuntimeError(
            f"sm120_gated_residual was certified for 30 Wan blocks, found {len(blocks)}"
        )

    if len({type(block) for block in blocks}) != 1:
        raise RuntimeError("sm120_gated_residual requires one unchanged Wan block class")
    try:
        source = inspect.getsource(type(blocks[0]).forward)
    except (OSError, TypeError) as error:
        raise RuntimeError("cannot certify upstream WanTransformerBlock.forward source") from error
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != CERTIFIED_BLOCK_FORWARD_SHA256:
        raise RuntimeError(
            "upstream WanTransformerBlock.forward changed; refusing a full-body rewrite: "
            f"expected {CERTIFIED_BLOCK_FORWARD_SHA256}, got {digest}"
        )

    for block in blocks:
        if getattr(block, "_ifl_sm120_gated_residual_installed", False):
            continue
        original = block.forward
        block._ifl_sm120_original_forward = original
        block._ifl_sm120_residual_buffers = {}

        def fused_forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb,
            update_cache=0,
            cache_name="pos",
        ):
            if (
                hidden_states.dtype is not torch.bfloat16
                or not hidden_states.is_contiguous()
                or hidden_states.shape[-1] != CERTIFIED_DIM
            ):
                # This is not an optimisation fallback: the model changed outside the certified
                # surface. Fail loud so the plan never reports work that was skipped.
                raise RuntimeError(
                    "sm120_gated_residual received an uncertified hidden layout: "
                    f"shape={tuple(hidden_states.shape)}, dtype={hidden_states.dtype}, "
                    f"contiguous={hidden_states.is_contiguous()}"
                )

            table = self.scale_shift_table[None] + temb.float()
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = rearrange(
                table, "b l n c -> b n l c"
            ).chunk(6, dim=1)
            shift_msa = shift_msa.squeeze(1)
            scale_msa = scale_msa.squeeze(1)
            gate_msa = gate_msa.squeeze(1)
            c_shift_msa = c_shift_msa.squeeze(1)
            c_scale_msa = c_scale_msa.squeeze(1)
            c_gate_msa = c_gate_msa.squeeze(1)

            norm_hidden_states = (
                self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa
            ).type_as(hidden_states)
            attn_output = self.attn1(
                norm_hidden_states,
                norm_hidden_states,
                norm_hidden_states,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )

            key = tuple(hidden_states.shape)
            residual_out = self._ifl_sm120_residual_buffers.get(key)
            if residual_out is None:
                residual_out = self._ifl_sm120_residual_buffers[key] = torch.empty_like(hidden_states)
            kernel.launch_certified(hidden_states, attn_output, gate_msa, residual_out)
            hidden_states = residual_out

            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states,
                encoder_hidden_states,
                None,
                update_cache=0,
                cache_name=cache_name,
            )
            hidden_states = hidden_states + attn_output

            norm_hidden_states = (
                self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa
            ).type_as(hidden_states)
            ff_output = self.ffn(norm_hidden_states)
            kernel.launch_certified(hidden_states, ff_output, c_gate_msa, residual_out)
            return residual_out

        block.forward = types.MethodType(fused_forward, block)
        block._ifl_sm120_gated_residual_installed = True

    transformer._ifl_sm120_gated_residual_kernel = kernel
    return kernel


__all__ = [
    "CERTIFIED_DIM",
    "CERTIFIED_ROWS",
    "LIBRARY_ENV",
    "SM120GatedResidualKernel",
    "available",
    "install_wan_blocks",
    "library_candidates",
    "resolve_library",
]

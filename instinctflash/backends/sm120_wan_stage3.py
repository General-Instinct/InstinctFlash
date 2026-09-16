"""P009-A3 exact norm1 + Ada modulation fusion for SM120 Wan blocks."""

from __future__ import annotations

import ctypes
import os
import types
from pathlib import Path

from instinctflash.backends.sm120_wan_stage2 import (
    CERTIFIED_CUDA_VERSION,
    CERTIFIED_DIM,
    CERTIFIED_EPS,
    CERTIFIED_ROWS,
    CERTIFIED_TORCH_VERSION,
    REQUIRED_ALIGNMENT,
    _assert_no_alias,
)

LIBRARY_ENV = "IFL_SM120_STAGE3_LIBRARY"
LIBRARY_NAME = "libinstinctflash_sm120_wan_stage3.so"
ABI_VERSION = 1


def library_candidates() -> tuple[Path, ...]:
    configured = os.environ.get(LIBRARY_ENV)
    packaged = Path(__file__).resolve().parents[1] / "native" / LIBRARY_NAME
    return tuple([Path(configured)] if configured else []) + (packaged,)


def _library_abi(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        library = ctypes.CDLL(str(path))
        version = library.instinctflash_sm120_wan_stage3_abi_version
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
        f"SM120 Wan stage3 ABI v{ABI_VERSION} is unavailable; searched {searched}. "
        "Build instinctflash/native/CMakeLists.txt or point IFL_SM120_STAGE3_LIBRARY "
        f"at {LIBRARY_NAME}. P009-A3 refuses instead of silently using eager."
    )


class SM120WanStage3Kernels:
    """Validated launcher for the certified norm1 + Ada region."""

    def __init__(self, library: str | Path | None = None):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("P009-A3 requires CUDA")
        device_index = torch.cuda.current_device()
        if torch.cuda.get_device_capability(device_index) != (12, 0):
            raise RuntimeError("P009-A3 is certified only on SM120")
        if torch.__version__ != CERTIFIED_TORCH_VERSION or torch.version.cuda != CERTIFIED_CUDA_VERSION:
            raise RuntimeError(
                "P009-A3's exact Welford claim is certified only for "
                f"torch={CERTIFIED_TORCH_VERSION}, CUDA={CERTIFIED_CUDA_VERSION}; got "
                f"torch={torch.__version__}, CUDA={torch.version.cuda}"
            )
        path = Path(library) if library is not None else resolve_library()
        abi = _library_abi(path)
        if abi != ABI_VERSION:
            raise RuntimeError(f"expected SM120 Wan stage3 ABI v{ABI_VERSION}, got {abi} from {path}")

        self.path = path
        self.device = torch.device("cuda", device_index)
        self.lib = ctypes.CDLL(str(path))
        u64, i32, f32 = ctypes.c_uint64, ctypes.c_int, ctypes.c_float
        self._norm1_ada = self.lib.wan_norm1_ada_layer_norm_bf16
        self._norm1_ada.argtypes = [
            u64, u64, u64, u64, u64, u64,
            i32, i32, i32, i32, f32, u64,
        ]
        self._norm1_ada.restype = i32
        self.calls = 0
        self._stream = None
        self._validated_outputs = set()

    def _check_tensor(self, name, tensor, dtype) -> None:
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if not tensor.is_cuda or tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {tensor.device}")
        if tensor.dtype is not dtype:
            raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
        if any(stride < 0 for stride in tensor.stride()):
            raise ValueError(f"{name} has a negative stride")
        if tensor.data_ptr() % REQUIRED_ALIGNMENT:
            raise ValueError(
                f"{name} pointer {tensor.data_ptr():#x} is not {REQUIRED_ALIGNMENT}-byte aligned"
            )

    def _validate(self, hidden, scale, shift, normed, means, rstds) -> tuple[int, int]:
        import torch

        for name, tensor in (("hidden", hidden), ("normed", normed)):
            self._check_tensor(name, tensor, torch.bfloat16)
        if hidden.ndim not in (2, 3) or normed.shape != hidden.shape:
            raise ValueError("hidden/normed must have the same rank-2 or rank-3 shape")
        if not hidden.is_contiguous() or not normed.is_contiguous():
            raise ValueError("hidden/normed must be contiguous")
        dim = hidden.shape[-1]
        rows = hidden.numel() // dim
        if dim != CERTIFIED_DIM or rows not in CERTIFIED_ROWS:
            raise ValueError(
                f"uncertified Wan stage3 shape: rows={rows}, dim={dim}; certified rows are "
                f"{sorted(CERTIFIED_ROWS)} at dim={CERTIFIED_DIM}"
            )
        for name, tensor in (("scale", scale), ("shift", shift)):
            self._check_tensor(name, tensor, torch.float32)
            if tensor.ndim != hidden.ndim or tensor.shape != hidden.shape:
                raise ValueError(f"{name} must have hidden's rank and shape")
            if tensor.stride(-1) != 1 or tensor.stride(-2) < dim:
                raise ValueError(f"{name} rows overlap, run backwards, or lack contiguous features")
            if tensor.ndim == 3 and tensor.stride(-3) != tensor.shape[-2] * tensor.stride(-2):
                raise ValueError(f"{name} leading dimensions do not flatten to one row stride")
        for name, tensor in (("means", means), ("rstds", rstds)):
            self._check_tensor(name, tensor, torch.float32)
            if tensor.shape != (rows,) or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous FP32 with shape ({rows},)")
        _assert_no_alias({
            "hidden": hidden, "scale": scale, "shift": shift,
            "normed": normed, "means": means, "rstds": rstds,
        })
        return rows, dim

    def _stream_for(self, hidden) -> int:
        import torch

        stream = torch.cuda.current_stream(hidden.device).cuda_stream
        if self._stream is None:
            self._stream = stream
        elif stream != self._stream:
            raise RuntimeError("P009-A3 is certified for one CUDA stream per Runtime")
        return stream

    def _launch(self, hidden, scale, shift, normed, means, rstds, rows, dim) -> None:
        error = self._norm1_ada(
            hidden.data_ptr(), scale.data_ptr(), shift.data_ptr(), normed.data_ptr(),
            means.data_ptr(), rstds.data_ptr(), rows, dim, scale.stride(-2),
            shift.stride(-2), CERTIFIED_EPS, self._stream_for(hidden),
        )
        if error:
            raise RuntimeError(f"P009-A3 norm1+Ada LayerNorm failed with CUDA error {error}")
        self.calls += 1

    def norm1_ada_into(self, hidden, scale, shift, normed, means, rstds) -> None:
        rows, dim = self._validate(hidden, scale, shift, normed, means, rstds)
        self._launch(hidden, scale, shift, normed, means, rstds, rows, dim)

    def norm1_ada_certified(self, hidden, scale, shift, normed, means, rstds) -> None:
        key = (
            normed.data_ptr(), means.data_ptr(), rstds.data_ptr(), tuple(hidden.shape),
            tuple(scale.stride()), tuple(shift.stride()),
        )
        if key not in self._validated_outputs:
            rows, dim = self._validate(hidden, scale, shift, normed, means, rstds)
            self._validated_outputs.add(key)
        else:
            dim = hidden.shape[-1]
            rows = hidden.numel() // dim
        self._launch(hidden, scale, shift, normed, means, rstds, rows, dim)


def install_wan_stage3(transformer, kernels: SM120WanStage3Kernels | None = None):
    """Replace A2's block body only after every A1/A2/A3 certificate passes."""
    import torch
    from einops import rearrange

    a1 = getattr(transformer, "_ifl_sm120_gated_residual_kernel", None)
    a2 = getattr(transformer, "_ifl_wan_stage2_kernels", None)
    if (a1 is None or not callable(getattr(a1, "launch_certified", None))
            or a2 is None
            or not callable(getattr(a2, "gate_affine_certified", None))
            or not callable(getattr(a2, "cross_ada_certified", None))):
        raise RuntimeError("P009-A3 requires installed P009-A1 and P009-A2 kernels")
    kernels = kernels or SM120WanStage3Kernels()
    blocks = list(transformer.blocks)
    if len(blocks) != 30:
        raise RuntimeError(f"P009-A3 was certified for 30 Wan blocks, found {len(blocks)}")
    installed = [getattr(block, "_ifl_wan_stage3_installed", False) for block in blocks]
    if any(installed):
        if all(installed) and hasattr(transformer, "_ifl_wan_stage3_kernels"):
            return transformer._ifl_wan_stage3_kernels
        raise RuntimeError("P009-A3 found a partially installed 30-block transformer")
    if not all(getattr(block, "_ifl_wan_stage2_installed", False) for block in blocks):
        raise RuntimeError("P009-A3 requires the full 30-block P009-A2 install")

    norm2_params = {}
    for block in blocks:
        if (getattr(block.forward, "__name__", "") != "stage2_forward"
                or not hasattr(block, "_ifl_stage2_buffers")):
            raise RuntimeError("P009-A3 requires the unchanged P009-A2 block rewrite")
        norm1, norm2 = block.norm1, block.norm2
        if type(norm1) is not type(norm2):
            raise RuntimeError("P009-A3 requires one FP32LayerNorm implementation")
        if tuple(norm1.normalized_shape) != (CERTIFIED_DIM,) or norm1.eps != CERTIFIED_EPS:
            raise RuntimeError("P009-A3 requires norm1 D=3072 and eps=1e-6")
        if getattr(norm1, "weight", None) is not None or getattr(norm1, "bias", None) is not None:
            raise RuntimeError("P009-A3 requires norm1 without affine parameters")
        if (tuple(norm2.normalized_shape) != (CERTIFIED_DIM,)
                or norm2.eps != CERTIFIED_EPS
                or norm2.weight is None or norm2.bias is None):
            raise RuntimeError("P009-A3 requires A2's affine norm2 D3072/eps parameters")
        allowed_param_dtypes = (torch.bfloat16, torch.float32)
        if (norm2.weight.dtype not in allowed_param_dtypes
                or norm2.bias.dtype not in allowed_param_dtypes
                or not norm2.weight.is_contiguous()
                or not norm2.bias.is_contiguous()):
            raise RuntimeError("P009-A3 requires contiguous BF16/FP32 norm2 parameters")
        weight, bias = norm2.weight.float(), norm2.bias.float()
        if (weight.device != kernels.device or bias.device != kernels.device
                or not weight.is_contiguous() or not bias.is_contiguous()):
            raise RuntimeError("P009-A3 requires contiguous promoted norm2 parameters on its device")
        norm2_params[id(block)] = (weight, bias)

    for block in blocks:
        block._ifl_stage3_a2_forward = block.forward
        block._ifl_stage3_buffers = {}

        def stage3_forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb,
            update_cache=0,
            cache_name="pos",
        ):
            rows = hidden_states.numel() // hidden_states.shape[-1]
            if (
                hidden_states.dtype is not torch.bfloat16
                or hidden_states.device != kernels.device
                or not hidden_states.is_contiguous()
                or hidden_states.shape[-1] != CERTIFIED_DIM
                or rows not in CERTIFIED_ROWS
            ):
                raise RuntimeError(
                    "P009-A3 received an uncertified Wan layout: "
                    f"shape={tuple(hidden_states.shape)}, dtype={hidden_states.dtype}, "
                    f"device={hidden_states.device}, contiguous={hidden_states.is_contiguous()}"
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

            key = (tuple(hidden_states.shape), hidden_states.device, hidden_states.dtype)
            stage3_buffers = self._ifl_stage3_buffers.get(key)
            if stage3_buffers is None:
                stage3_buffers = self._ifl_stage3_buffers[key] = (
                    torch.empty_like(hidden_states),
                    torch.empty(rows, device=hidden_states.device, dtype=torch.float32),
                    torch.empty(rows, device=hidden_states.device, dtype=torch.float32),
                )
            norm1_out, norm1_means, norm1_rstds = stage3_buffers
            kernels.norm1_ada_certified(
                hidden_states, scale_msa, shift_msa,
                norm1_out, norm1_means, norm1_rstds,
            )
            attn_output = self.attn1(
                norm1_out, norm1_out, norm1_out, rotary_emb,
                update_cache=update_cache, cache_name=cache_name,
            )

            stage2_buffers = self._ifl_stage2_buffers.get(key)
            if stage2_buffers is None:
                stage2_buffers = self._ifl_stage2_buffers[key] = (
                    torch.empty_like(hidden_states),
                    torch.empty_like(hidden_states),
                    torch.empty_like(hidden_states),
                    torch.empty(rows, device=hidden_states.device, dtype=torch.float32),
                    torch.empty(rows, device=hidden_states.device, dtype=torch.float32),
                )
            residual1, scratch2, scratch3, means, rstds = stage2_buffers
            weight, bias = norm2_params[id(self)]
            a2.gate_affine_certified(
                hidden_states, attn_output, gate_msa, weight, bias,
                residual1, scratch2, means, rstds,
            )
            attn_output = self.attn2(
                scratch2, encoder_hidden_states, encoder_hidden_states, None,
                update_cache=0, cache_name=cache_name,
            )
            a2.cross_ada_certified(
                residual1, attn_output, c_scale_msa, c_shift_msa,
                scratch2, scratch3, means, rstds,
            )
            ff_output = self.ffn(scratch3)
            a1.launch_certified(scratch2, ff_output, c_gate_msa, scratch3)
            return scratch3

        block.forward = types.MethodType(stage3_forward, block)
        block._ifl_wan_stage3_installed = True

    transformer._ifl_wan_stage3_kernels = kernels
    return kernels


__all__ = [
    "ABI_VERSION",
    "LIBRARY_ENV",
    "LIBRARY_NAME",
    "SM120WanStage3Kernels",
    "available",
    "install_wan_stage3",
    "library_candidates",
    "resolve_library",
]

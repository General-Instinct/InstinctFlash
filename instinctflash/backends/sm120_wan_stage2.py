"""P009-A2 raw-pointer kernels and Wan block installer for SM120.

This layer is additive to P009-A1. It fuses the self-attention residual with affine LayerNorm,
and the cross-attention residual with AdaLayerNorm, while retaining A1 for the final FFN residual.
The CUDA reduction exactly follows PyTorch 2.9's D=3072 vectorized Welford path, so both fused
regions preserve the eager BF16 materialization boundaries bit-for-bit.
"""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import os
import types
from pathlib import Path

from instinctflash.backends.sm120_residual import CERTIFIED_BLOCK_FORWARD_SHA256


LIBRARY_ENV = "IFL_SM120_STAGE2_LIBRARY"
LIBRARY_NAME = "libinstinctflash_sm120_wan_stage2.so"
ABI_VERSION = 1
CERTIFIED_ROWS = frozenset({64, 480})
CERTIFIED_DIM = 3072
CERTIFIED_EPS = 1e-6
CERTIFIED_TORCH_VERSION = "2.9.0+cu128"
CERTIFIED_CUDA_VERSION = "12.8"
CERTIFIED_FP32_LAYERNORM_FORWARD_SHA256 = "fb3a8812f75a9d8325df97c1e39495d4952622c76c846870a1aa2fba7855f869"
REQUIRED_ALIGNMENT = 16


def library_candidates() -> tuple[Path, ...]:
    configured = os.environ.get(LIBRARY_ENV)
    packaged = Path(__file__).resolve().parents[1] / "native" / LIBRARY_NAME
    return tuple([Path(configured)] if configured else []) + (packaged,)


def _library_abi(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        library = ctypes.CDLL(str(path))
        version = library.instinctflash_sm120_wan_stage2_abi_version
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
        f"SM120 Wan stage2 ABI v{ABI_VERSION} is unavailable; searched {searched}. Build "
        "instinctflash/native/CMakeLists.txt or point IFL_SM120_STAGE2_LIBRARY at the compiled "
        f"{LIBRARY_NAME}. P009-A2 refuses instead of claiming a fusion it did not run."
    )


def _logical_intervals(name, tensor) -> list[tuple[int, int, str]]:
    """Exact addressed byte intervals; strided style rows may share storage without aliasing."""
    element = tensor.element_size()
    begin = tensor.data_ptr()
    if tensor.ndim == 1 or tensor.is_contiguous():
        return [(begin, begin + tensor.numel() * element, name)]
    dim = tensor.shape[-1]
    rows = tensor.numel() // dim
    stride = tensor.stride(-2)
    return [
        (begin + row * stride * element, begin + (row * stride + dim) * element, name)
        for row in range(rows)
    ]


def _assert_no_alias(tensors: dict) -> None:
    intervals = sorted(
        interval for name, tensor in tensors.items()
        for interval in _logical_intervals(name, tensor)
    )
    active_end = -1
    active_name = None
    active_span = None
    for begin, end, name in intervals:
        if begin < active_end and name != active_name:
            raise ValueError(
                f"restrict operands {active_name}/{name} overlap: {active_span} vs "
                f"{(begin, end)}"
            )
        if end > active_end:
            active_end = end
            active_name = name
            active_span = (begin, end)


class SM120WanStage2Kernels:
    """Validated launchers for the two certified Wan stage2 regions."""

    def __init__(self, library: str | Path | None = None):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("P009-A2 requires CUDA")
        device_index = torch.cuda.current_device()
        if torch.cuda.get_device_capability(device_index) != (12, 0):
            raise RuntimeError("P009-A2 is certified only on SM120")
        if torch.__version__ != CERTIFIED_TORCH_VERSION or torch.version.cuda != CERTIFIED_CUDA_VERSION:
            raise RuntimeError(
                "P009-A2's exact Welford claim is certified only for "
                f"torch={CERTIFIED_TORCH_VERSION}, CUDA={CERTIFIED_CUDA_VERSION}; got "
                f"torch={torch.__version__}, CUDA={torch.version.cuda}"
            )
        path = Path(library) if library is not None else resolve_library()
        abi = _library_abi(path)
        if abi != ABI_VERSION:
            raise RuntimeError(f"expected SM120 Wan stage2 ABI v{ABI_VERSION}, got {abi} from {path}")

        self.path = path
        self.device = torch.device("cuda", device_index)
        self.lib = ctypes.CDLL(str(path))
        u64, i32, f32 = ctypes.c_uint64, ctypes.c_int, ctypes.c_float
        self._gate_affine = self.lib.wan_gate_residual_affine_layer_norm_bf16
        self._gate_affine.argtypes = [
            u64, u64, u64, u64, u64, u64, u64, u64, u64,
            i32, i32, i32, f32, u64,
        ]
        self._gate_affine.restype = i32
        self._cross_ada = self.lib.wan_cross_residual_ada_layer_norm_bf16
        self._cross_ada.argtypes = [
            u64, u64, u64, u64, u64, u64, u64, u64,
            i32, i32, i32, i32, f32, u64,
        ]
        self._cross_ada.restype = i32
        self.calls = 0
        self.gate_affine_calls = 0
        self.cross_ada_calls = 0
        self._stream = None
        self._validated_gate_outputs = set()
        self._validated_cross_outputs = set()

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

    def _hidden_group(self, tensors: dict) -> tuple[int, int]:
        import torch

        hidden = tensors["hidden"]
        for name, tensor in tensors.items():
            self._check_tensor(name, tensor, torch.bfloat16)
        if hidden.ndim not in (2, 3):
            raise ValueError("hidden/update/residual/normed must be rank 2 or rank 3")
        if any(tensor.shape != hidden.shape for tensor in tensors.values()):
            raise ValueError("hidden, update, residual, and normed shapes must match")
        if not all(tensor.is_contiguous() for tensor in tensors.values()):
            raise ValueError("hidden, update, residual, and normed must be contiguous")
        dim = hidden.shape[-1]
        rows = hidden.numel() // dim
        if dim != CERTIFIED_DIM or rows not in CERTIFIED_ROWS:
            raise ValueError(
                f"uncertified Wan stage2 shape: rows={rows}, dim={dim}; certified rows are "
                f"{sorted(CERTIFIED_ROWS)} at dim={CERTIFIED_DIM}"
            )
        return rows, dim

    def _style(self, name, tensor, hidden) -> None:
        import torch

        self._check_tensor(name, tensor, torch.float32)
        if tensor.ndim != hidden.ndim or tensor.shape != hidden.shape:
            raise ValueError(f"{name} must have hidden's rank and shape")
        if tensor.stride(-1) != 1 or tensor.stride(-2) < tensor.shape[-1]:
            raise ValueError(f"{name} rows overlap, run backwards, or lack contiguous features")
        if tensor.ndim == 3 and tensor.stride(-3) != tensor.shape[-2] * tensor.stride(-2):
            raise ValueError(f"{name} leading dimensions do not flatten to one row stride")

    def _stats(self, means, rstds, rows) -> None:
        import torch

        for name, tensor in (("means", means), ("rstds", rstds)):
            self._check_tensor(name, tensor, torch.float32)
            if tensor.shape != (rows,) or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous FP32 with shape ({rows},)")

    def _validate_gate(
        self, hidden, update, gate, weight, bias, residual, normed, means, rstds,
    ) -> tuple[int, int]:
        import torch

        group = {"hidden": hidden, "update": update, "residual": residual, "normed": normed}
        rows, dim = self._hidden_group(group)
        self._style("gate", gate, hidden)
        for name, tensor in (("weight", weight), ("bias", bias)):
            self._check_tensor(name, tensor, torch.float32)
            if tensor.shape != (dim,) or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous FP32 with shape ({dim},)")
        self._stats(means, rstds, rows)
        _assert_no_alias({
            **group, "gate": gate, "weight": weight, "bias": bias,
            "means": means, "rstds": rstds,
        })
        return rows, dim

    def _validate_cross(
        self, hidden, update, scale, shift, residual, normed, means, rstds,
    ) -> tuple[int, int]:
        group = {"hidden": hidden, "update": update, "residual": residual, "normed": normed}
        rows, dim = self._hidden_group(group)
        self._style("scale", scale, hidden)
        self._style("shift", shift, hidden)
        self._stats(means, rstds, rows)
        _assert_no_alias({
            **group, "scale": scale, "shift": shift, "means": means, "rstds": rstds,
        })
        return rows, dim

    def _stream_for(self, hidden) -> int:
        import torch

        stream = torch.cuda.current_stream(hidden.device).cuda_stream
        if self._stream is None:
            self._stream = stream
        elif stream != self._stream:
            raise RuntimeError("P009-A2 is certified for one CUDA stream per Runtime")
        return stream

    def _launch_gate(
        self, hidden, update, gate, weight, bias, residual, normed, means, rstds, rows, dim,
    ) -> None:
        error = self._gate_affine(
            hidden.data_ptr(), update.data_ptr(), gate.data_ptr(), weight.data_ptr(),
            bias.data_ptr(), residual.data_ptr(), normed.data_ptr(), means.data_ptr(),
            rstds.data_ptr(), rows, dim, gate.stride(-2), CERTIFIED_EPS,
            self._stream_for(hidden),
        )
        if error:
            raise RuntimeError(f"P009-A2 gate+affine LayerNorm failed with CUDA error {error}")
        self.calls += 1
        self.gate_affine_calls += 1

    def _launch_cross(
        self, hidden, update, scale, shift, residual, normed, means, rstds, rows, dim,
    ) -> None:
        error = self._cross_ada(
            hidden.data_ptr(), update.data_ptr(), scale.data_ptr(), shift.data_ptr(),
            residual.data_ptr(), normed.data_ptr(), means.data_ptr(), rstds.data_ptr(),
            rows, dim, scale.stride(-2), shift.stride(-2), CERTIFIED_EPS,
            self._stream_for(hidden),
        )
        if error:
            raise RuntimeError(f"P009-A2 cross+AdaLayerNorm failed with CUDA error {error}")
        self.calls += 1
        self.cross_ada_calls += 1

    def gate_affine_into(
        self, hidden, update, gate, weight, bias, residual, normed, means, rstds,
    ) -> None:
        rows, dim = self._validate_gate(
            hidden, update, gate, weight, bias, residual, normed, means, rstds,
        )
        self._launch_gate(
            hidden, update, gate, weight, bias, residual, normed, means, rstds, rows, dim,
        )

    def cross_ada_into(
        self, hidden, update, scale, shift, residual, normed, means, rstds,
    ) -> None:
        rows, dim = self._validate_cross(
            hidden, update, scale, shift, residual, normed, means, rstds,
        )
        self._launch_cross(
            hidden, update, scale, shift, residual, normed, means, rstds, rows, dim,
        )

    def gate_affine_certified(
        self, hidden, update, gate, weight, bias, residual, normed, means, rstds,
    ) -> None:
        key = (
            residual.data_ptr(), normed.data_ptr(), means.data_ptr(), rstds.data_ptr(),
            tuple(hidden.shape), tuple(gate.stride()), weight.data_ptr(), bias.data_ptr(),
        )
        if key not in self._validated_gate_outputs:
            rows, dim = self._validate_gate(
                hidden, update, gate, weight, bias, residual, normed, means, rstds,
            )
            self._validated_gate_outputs.add(key)
        else:
            dim = hidden.shape[-1]
            rows = hidden.numel() // dim
        self._launch_gate(
            hidden, update, gate, weight, bias, residual, normed, means, rstds, rows, dim,
        )

    def cross_ada_certified(
        self, hidden, update, scale, shift, residual, normed, means, rstds,
    ) -> None:
        key = (
            residual.data_ptr(), normed.data_ptr(), means.data_ptr(), rstds.data_ptr(),
            tuple(hidden.shape), tuple(scale.stride()), tuple(shift.stride()),
        )
        if key not in self._validated_cross_outputs:
            rows, dim = self._validate_cross(
                hidden, update, scale, shift, residual, normed, means, rstds,
            )
            self._validated_cross_outputs.add(key)
        else:
            dim = hidden.shape[-1]
            rows = hidden.numel() // dim
        self._launch_cross(
            hidden, update, scale, shift, residual, normed, means, rstds, rows, dim,
        )


def install_wan_stage2(transformer, kernels: SM120WanStage2Kernels | None = None):
    """Replace the middle A1 regions only after every block passes the stage2 certificate."""
    import torch
    from einops import rearrange

    a1 = getattr(transformer, "_ifl_sm120_gated_residual_kernel", None)
    if a1 is None or not callable(getattr(a1, "launch_certified", None)):
        raise RuntimeError("P009-A2 requires an installed P009-A1 transformer kernel")
    kernels = kernels or SM120WanStage2Kernels()
    blocks = list(transformer.blocks)
    if len(blocks) != 30:
        raise RuntimeError(f"P009-A2 was certified for 30 Wan blocks, found {len(blocks)}")
    installed = [getattr(block, "_ifl_wan_stage2_installed", False) for block in blocks]
    if any(installed):
        if all(installed) and hasattr(transformer, "_ifl_wan_stage2_kernels"):
            return transformer._ifl_wan_stage2_kernels
        raise RuntimeError("P009-A2 found a partially installed 30-block transformer")
    if not all(getattr(block, "_ifl_sm120_gated_residual_installed", False) for block in blocks):
        raise RuntimeError("P009-A2 requires all 30 blocks to carry the P009-A1 certificate")
    if len({type(block) for block in blocks}) != 1:
        raise RuntimeError("P009-A2 requires one unchanged Wan block class")
    try:
        source = inspect.getsource(type(blocks[0]).forward)
    except (OSError, TypeError) as error:
        raise RuntimeError("P009-A2 cannot certify upstream WanTransformerBlock.forward") from error
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != CERTIFIED_BLOCK_FORWARD_SHA256:
        raise RuntimeError(
            "upstream WanTransformerBlock.forward changed; P009-A2 refuses its full-body rewrite: "
            f"expected {CERTIFIED_BLOCK_FORWARD_SHA256}, got {digest}"
        )
    if len({type(block.norm2) for block in blocks}) != 1:
        raise RuntimeError("P009-A2 requires one unchanged FP32LayerNorm implementation")
    try:
        norm_source = inspect.getsource(type(blocks[0].norm2).forward)
    except (OSError, TypeError) as error:
        raise RuntimeError("P009-A2 cannot certify FP32LayerNorm.forward") from error
    norm_digest = hashlib.sha256(norm_source.encode()).hexdigest()
    if norm_digest != CERTIFIED_FP32_LAYERNORM_FORWARD_SHA256:
        raise RuntimeError(
            "FP32LayerNorm.forward changed; P009-A2 cannot assume FP32 parameter promotion: "
            f"expected {CERTIFIED_FP32_LAYERNORM_FORWARD_SHA256}, got {norm_digest}"
        )


    norm2_params = {}
    for block in blocks:
        if not hasattr(block, "_ifl_sm120_original_forward"):
            raise RuntimeError("P009-A2 cannot establish the P009-A1 source-certificate chain")
        norm2 = block.norm2
        norm3 = block.norm3
        if tuple(norm2.normalized_shape) != (CERTIFIED_DIM,) or norm2.eps != CERTIFIED_EPS:
            raise RuntimeError("P009-A2 requires affine norm2 with D=3072 and eps=1e-6")
        if norm2.weight is None or norm2.bias is None:
            raise RuntimeError("P009-A2 requires affine norm2 weight and bias")
        allowed_param_dtypes = (torch.bfloat16, torch.float32)
        if (norm2.weight.dtype not in allowed_param_dtypes
                or norm2.bias.dtype not in allowed_param_dtypes):
            raise RuntimeError("P009-A2 requires norm2 weight and bias to be BF16 or FP32")
        if not norm2.weight.is_contiguous() or not norm2.bias.is_contiguous():
            raise RuntimeError("P009-A2 requires contiguous norm2 weight and bias")
        if tuple(norm3.normalized_shape) != (CERTIFIED_DIM,) or norm3.eps != CERTIFIED_EPS:
            raise RuntimeError("P009-A2 requires no-affine norm3 with D=3072 and eps=1e-6")
        if getattr(norm3, "weight", None) is not None or getattr(norm3, "bias", None) is not None:
            raise RuntimeError("P009-A2 requires norm3 without affine parameters")
        if norm2.weight.device != kernels.device or norm2.bias.device != kernels.device:
            raise RuntimeError(f"P009-A2 norm2 parameters must be on {kernels.device}")
        # diffusers.FP32LayerNorm.forward promotes these exact checkpoint tensors on every call.
        # Keep one persistent promotion per block; the CUDA ABI still accepts FP32 only.
        weight = norm2.weight.float()
        bias = norm2.bias.float()
        if not weight.is_contiguous() or not bias.is_contiguous():
            raise RuntimeError("P009-A2 requires contiguous promoted norm2 parameters")
        norm2_params[id(block)] = (weight, bias)

    for block in blocks:
        block._ifl_stage2_a1_forward = block.forward
        block._ifl_stage2_buffers = {}

        def stage2_forward(
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
                    "P009-A2 received an uncertified Wan layout: "
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

            key = (tuple(hidden_states.shape), hidden_states.device, hidden_states.dtype)
            buffers = self._ifl_stage2_buffers.get(key)
            if buffers is None:
                buffers = self._ifl_stage2_buffers[key] = (
                    torch.empty_like(hidden_states),
                    torch.empty_like(hidden_states),
                    torch.empty_like(hidden_states),
                    torch.empty(rows, device=hidden_states.device, dtype=torch.float32),
                    torch.empty(rows, device=hidden_states.device, dtype=torch.float32),
                )
            residual1, scratch2, scratch3, means, rstds = buffers
            weight, bias = norm2_params[id(self)]

            kernels.gate_affine_certified(
                hidden_states, attn_output, gate_msa, weight, bias,
                residual1, scratch2, means, rstds,
            )
            attn_output = self.attn2(
                scratch2,
                encoder_hidden_states,
                encoder_hidden_states,
                None,
                update_cache=0,
                cache_name=cache_name,
            )
            kernels.cross_ada_certified(
                residual1, attn_output, c_scale_msa, c_shift_msa,
                scratch2, scratch3, means, rstds,
            )
            ff_output = self.ffn(scratch3)
            a1.launch_certified(scratch2, ff_output, c_gate_msa, scratch3)
            return scratch3

        block.forward = types.MethodType(stage2_forward, block)
        block._ifl_wan_stage2_installed = True

    transformer._ifl_wan_stage2_kernels = kernels
    return kernels


__all__ = [
    "ABI_VERSION",
    "CERTIFIED_DIM",
    "CERTIFIED_ROWS",
    "LIBRARY_ENV",
    "SM120WanStage2Kernels",
    "available",
    "install_wan_stage2",
    "library_candidates",
    "resolve_library",
]

"""``nn.Linear`` replacement that fuses a frozen FP8 base with BF16 LoRA.

Math (matching ``openpi.models.lora.Einsum``)::

    y = W_base @ x + (alpha / rank) * (B @ A @ x) + b

with ``W_base`` stored at FP8 E4M3 (uint8 bytes) and ``A``, ``B`` in
BF16 as the only trainable parameters. ``alpha == rank`` is the
``gemma_2b_lora`` / ``gemma_300m_lora`` default in upstream openpi
(``train_jax_lora_recap.py`` does not override it), so the default
scaling is ``1.0``. Initialisation matches upstream too:
``init_fn = nn.initializers.normal(stddev=0.01)`` for *both* ``A`` and
``B`` — note this differs from standard PEFT (Kaiming-A + zero-B),
which is a deliberate openpi choice and the LoRA contribution is
small but non-zero at step 0.

The module exposes a zero-element ``weight`` buffer of dtype
``torch.bfloat16`` so that upstream code that probes
``module.weight.dtype`` (notably ``PI0Pytorch.forward`` at the
prefix/suffix dtype gate) continues to work after FP8 injection,
without keeping a dequantised BF16 copy of the base weight in memory.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from .fp8_autograd import (
    Fp8MatmulFunction,
    dequant_fp8_to_bf16,
    quantize_weight_to_fp8,
)


class Fp8LinearWithLora(nn.Module):
    """Frozen FP8 base GEMM + BF16 LoRA adapter.

    Args:
        in_features: ``K`` — same as the source ``nn.Linear.in_features``.
        out_features: ``N`` — same as the source ``nn.Linear.out_features``.
        rank: LoRA rank ``r``. ``alpha == rank`` matches upstream openpi.
        alpha: LoRA scaling factor. Default to ``rank`` for openpi parity.
        bias: Whether the layer has a bias. Frozen if present.
        init_std: Std for both ``A`` and ``B`` Gaussian init (openpi: 0.01).
        rslora: If True, use ``alpha / sqrt(rank)`` scaling (rank-stabilised).

    Buffers:
        weight_fp8: ``(K, N) uint8`` — FP8 storage matching ``fp8_nn_dev``.
        weight_scale_dev: ``(1,) float32`` device buffer.
        act_scale_dev: ``(1,) float32`` device buffer, calibrated externally.
        weight: zero-element BF16 tensor — dtype proxy only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int = 16,
        alpha: float | None = None,
        bias: bool = False,
        init_std: float = 0.01,
        rslora: bool = False,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        if alpha is None:
            alpha = float(rank)

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.rslora = bool(rslora)
        self.scaling = self.alpha / (rank**0.5 if self.rslora else rank)

        self.register_buffer(
            "weight_fp8",
            torch.zeros(in_features, out_features, dtype=torch.uint8),
            persistent=True,
        )
        self.register_buffer(
            "weight_scale_dev",
            torch.zeros(1, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "act_scale_dev",
            torch.ones(1, dtype=torch.float32),
            persistent=True,
        )
        # Dtype proxy: HF code checks `module.weight.dtype` to decide
        # whether to cast embeddings to bf16. Zero-element tensor avoids
        # carrying a real BF16 copy of the frozen weight.
        self.register_buffer(
            "weight",
            torch.empty(0, dtype=torch.bfloat16),
            persistent=False,
        )

        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, dtype=torch.bfloat16)
        )
        self.lora_B = nn.Parameter(
            torch.empty(out_features, rank, dtype=torch.bfloat16)
        )
        nn.init.normal_(self.lora_A, mean=0.0, std=init_std)
        nn.init.normal_(self.lora_B, mean=0.0, std=init_std)

        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(out_features, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        # Tracks whether quantize_base_from_linear has run; calibration
        # may still leave act_scale_dev at the default ``1.0``.
        self._fp8_initialised = False
        self._weight_scale_f = 0.0
        # Optional dequantised BF16 copy of the frozen base weight,
        # populated by ``prepare_for_training``. When present, the
        # backward path uses it directly and skips the per-call FP8 →
        # BF16 dequant — closing the small-shape backward gap vs an
        # all-BF16 LoRA reference. Cost: 2x weight memory during training.
        self._w_bf16_cache: torch.Tensor | None = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int = 16,
        alpha: float | None = None,
        init_std: float = 0.01,
        rslora: bool = False,
    ) -> "Fp8LinearWithLora":
        """Build a wrapper from an existing ``nn.Linear``.

        The base weight is quantized to FP8 immediately; the source
        ``linear`` can be discarded by the caller after replacement.
        """
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            init_std=init_std,
            rslora=rslora,
        )
        module.quantize_base_from_linear(linear)
        return module

    def quantize_base_from_linear(self, linear: nn.Linear) -> None:
        """One-shot FP8 quantize of an ``nn.Linear``'s weight + bias."""
        if linear.in_features != self.in_features or linear.out_features != self.out_features:
            raise ValueError(
                f"shape mismatch: linear({linear.in_features},{linear.out_features}) "
                f"vs module({self.in_features},{self.out_features})"
            )

        # nn.Linear.weight is (N, K); fp8_nn_dev expects B as (K, N).
        w_kn = linear.weight.detach().t().contiguous()
        w_uint8, scale = quantize_weight_to_fp8(w_kn)
        device = self.weight_fp8.device
        self.weight_fp8.data = w_uint8.to(device)
        self.weight_scale_dev.data = torch.tensor(
            [scale], dtype=torch.float32, device=device
        )
        self._weight_scale_f = float(scale)

        if self.bias is not None and linear.bias is not None:
            self.bias.data = linear.bias.detach().to(torch.bfloat16).to(device)

        self._fp8_initialised = True

    def set_activation_scale(self, scale: float) -> None:
        """Install the calibrated per-tensor activation scale."""
        if scale <= 0:
            raise ValueError(f"activation scale must be > 0, got {scale}")
        device = self.act_scale_dev.device
        self.act_scale_dev.data = torch.tensor(
            [scale], dtype=torch.float32, device=device
        )

    def prepare_for_training(self, *, cache_bf16_weight: bool = True) -> None:
        """One-shot pre-training setup — populate the BF16 weight cache.

        Pre-dequantises the frozen FP8 base into a BF16 buffer that the
        backward path then reuses, skipping the per-call FP8 → BF16
        dequant. The transposed layout matches the
        ``Fp8MatmulFunction.backward`` GEMM (W_kn shape, BF16).

        Set ``cache_bf16_weight=False`` to opt out (e.g. memory-tight
        training where the per-call dequant cost is acceptable).
        """
        if not self._fp8_initialised:
            raise RuntimeError(
                "prepare_for_training requires quantize_base_from_linear "
                "to have run first"
            )
        if cache_bf16_weight:
            # Cache W transposed to (N, K) layout. The backward GEMM is
            # ``grad_x = grad_y @ W^T`` (in K-N-storage convention),
            # which equals ``grad_y @ self._w_bf16_cache`` directly —
            # so we pay the transpose memcpy ONCE here instead of every
            # backward call.
            w_kn_bf16 = dequant_fp8_to_bf16(
                self.weight_fp8, self._weight_scale_f
            )
            self._w_bf16_cache = w_kn_bf16.t().contiguous()
            del w_kn_bf16
        else:
            self._w_bf16_cache = None

    def release_training_cache(self) -> None:
        """Free the BF16 weight cache (e.g. before exporting / inference)."""
        self._w_bf16_cache = None

    @property
    def weight_scale_f(self) -> float:
        """Python float copy of the per-tensor weight scale (for backward)."""
        return self._weight_scale_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fp8_initialised:
            raise RuntimeError(
                "Fp8LinearWithLora.forward called before "
                "quantize_base_from_linear; build via from_linear or "
                "call quantize_base_from_linear explicitly."
            )

        orig_dtype = x.dtype
        x_bf16 = x.to(torch.bfloat16)

        y_base = Fp8MatmulFunction.apply(
            x_bf16,
            self.weight_fp8,
            self.act_scale_dev,
            self.weight_scale_dev,
            self._weight_scale_f,
            self._w_bf16_cache,
        )
        y_base = cast(torch.Tensor, y_base)

        # LoRA path. ``lora_A`` is (r, K) and ``lora_B`` is (N, r), so:
        # x_bf16 @ lora_A.t() → (..., r); ... @ lora_B.t() → (..., N).
        h = x_bf16 @ self.lora_A.t()
        y_lora = h @ self.lora_B.t()
        y = y_base + self.scaling * y_lora

        if self.bias is not None:
            y = y + self.bias

        if orig_dtype != torch.bfloat16:
            y = y.to(orig_dtype)
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"rslora={self.rslora}, bias={self.bias is not None}"
        )

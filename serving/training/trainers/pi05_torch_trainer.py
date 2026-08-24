"""Pi0.5 PyTorch trainer wrapper.

Thin facade over a vendored ``PI0Pytorch`` (or any ``nn.Module`` that
exposes Gemma-style ``q_proj``/.../``down_proj`` linears) that owns:

* the FP8 + LoRA injection (``inject_fp8_lora``);
* one-shot per-tensor activation calibration before training;
* a stable handle to the replaced layers and the trainable parameter
  list, so the W6/W7 RECAP loops can wire up an optimizer without
  needing to re-walk the module tree.

The trainer is **model-agnostic at the forward level** — the caller
still calls ``trainer.model(...)`` to run a step. This mirrors the
inference frontend convention (``Pi05TorchFrontendRtx`` exposes
``set_rl_mode`` / ``set_prompt`` / ``calibrate`` / ``infer``); the
training side keeps the same staged init: build → compile → step.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn

from training._vendor.openpi_pi0_pytorch import PI0Pytorch, Pi0Config
from training.lora.fp8_linear import Fp8LinearWithLora
from training.lora.inject import (
    InjectionConfig,
    calibrate_activation_scales,
    inject_fp8_lora,
)


@dataclass
class TrainerStats:
    """Snapshot of model size after compile()."""

    num_trainable_params: int
    num_lora_layers: int
    num_total_params: int


class Pi05Trainer:
    """Wraps a pi0/pi0.5 PyTorch model with FP8 + LoRA training plumbing.

    Args:
        model: A pre-built ``nn.Module``. Typical use cases:

            * ``PI0Pytorch(Pi0Config(...))`` — full pi0.5 model;
            * ``PaliGemmaWithExpertModel(...)`` — bare encoder+expert
              (useful for unit-level tests and ablations).

        device: CUDA device the model is moved to.
    """

    def __init__(self, model: nn.Module, *, device: str | torch.device = "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Pi05Trainer requires CUDA (FP8 kernels)")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self._compiled = False
        self._fp8_modules: dict[str, Fp8LinearWithLora] = {}
        self._lora_config: InjectionConfig | None = None

    @classmethod
    def from_pi0_config(
        cls,
        config: Pi0Config,
        *,
        device: str | torch.device = "cuda",
    ) -> "Pi05Trainer":
        """Build a trainer directly from a ``Pi0Config``."""
        model = PI0Pytorch(config)
        return cls(model, device=device)

    @property
    def is_compiled(self) -> bool:
        return self._compiled

    @property
    def fp8_modules(self) -> dict[str, Fp8LinearWithLora]:
        return dict(self._fp8_modules)

    @property
    def lora_config(self) -> InjectionConfig | None:
        return self._lora_config

    def compile(
        self,
        lora_config: InjectionConfig | None = None,
        calibration_passes: Iterable[Callable[[], None]] = (),
        *,
        safety_margin: float = 1.25,
        cache_bf16_weight: bool = True,
    ) -> TrainerStats:
        """Inject FP8 + LoRA into the model and (optionally) calibrate.

        Args:
            lora_config: Override the default openpi-aligned LoRA recipe
                (encoder rank=16, decoder rank=32, alpha=rank).
            calibration_passes: Iterable of zero-arg callables driving
                forward passes through the model with realistic input
                distributions. Each callable runs once. Empty iterable
                skips calibration — the activation scales stay at the
                ``1.0`` default and forward is incorrect, useful only
                for shape/grad-flow smoke tests.
            safety_margin: Multiplier on observed amax (passed through
                to ``calibrate_activation_scales``).

        Returns:
            ``TrainerStats`` describing the compiled model.

        Raises:
            RuntimeError: If ``compile`` has already been called.
        """
        if self._compiled:
            raise RuntimeError("Pi05Trainer.compile() may only be called once")

        cfg = lora_config or InjectionConfig()
        self._fp8_modules = inject_fp8_lora(self.model, cfg)
        if not self._fp8_modules:
            raise RuntimeError(
                "Pi05Trainer.compile: inject_fp8_lora found no target "
                "modules; check encoder/decoder prefix vs the model's "
                "actual module names."
            )

        passes = list(calibration_passes)
        if passes:
            calibrate_activation_scales(
                self.model,
                self._fp8_modules,
                passes,
                safety_margin=safety_margin,
            )

        # Pre-dequant the frozen base into a BF16 cache on each layer.
        # Closes the small-shape backward gap vs an all-BF16 reference
        # by skipping the per-call FP8 → BF16 dequant. Caller can opt
        # out to save ~3 GB on a full pi05.
        for module in self._fp8_modules.values():
            module.prepare_for_training(cache_bf16_weight=cache_bf16_weight)

        self._compiled = True
        self._lora_config = cfg
        return self._stats()

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return the LoRA adapter parameters for the optimizer."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def _stats(self) -> TrainerStats:
        return TrainerStats(
            num_trainable_params=self.num_trainable_parameters(),
            num_lora_layers=len(self._fp8_modules),
            num_total_params=sum(p.numel() for p in self.model.parameters()),
        )

    def reset_peak_memory(self) -> None:
        """Reset the CUDA peak-memory counter for benchmark windows."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory_bytes(self) -> int:
        if self.device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_allocated(self.device))

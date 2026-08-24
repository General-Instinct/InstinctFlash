"""Inject ``Fp8LinearWithLora`` over a base model + calibrate activations.

Replaces selected ``nn.Linear`` modules with FP8-base + BF16-LoRA
counterparts and freezes everything except the new ``lora_A`` /
``lora_B`` parameters. Module selection follows the
PEFT/openpi-aligned target-module naming convention
(``q_proj``, ``k_proj``, …) restricted by encoder/decoder path patterns
so the SigLIP vision tower, action heads, and embedding tables stay in
BF16 — matching the upstream ``train_jax_lora_recap.py`` freeze policy
("freeze vision encoder (img) to save memory").

The calibration helper runs caller-supplied forward passes through the
injected model with hooks that capture the per-layer activation
``amax``, then installs the static FP8 activation scales
(``amax * safety_margin / 448``). This is the same per-tensor static
calibration used by the inference pipeline; the scales never update
during training.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .fp8_linear import Fp8LinearWithLora

DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Substring patterns for the two LoRA-bearing trees inside (paligemma →
# language model) and the action expert. Substring instead of prefix so
# that BOTH wrappings work without per-caller config:
#
#   * bare ``PaliGemmaWithExpertModel`` (used by W5/W6 tests):
#       paligemma.model.language_model.layers.{N}.{q_proj,...}
#       gemma_expert.model.layers.{N}.{q_proj,...}
#   * full ``PI0Pytorch`` (W7+ production):
#       paligemma_with_expert.paligemma.model.language_model.layers...
#       paligemma_with_expert.gemma_expert.model.layers...
#
# The vision tower (``paligemma.model.vision_tower.*``) is naturally
# excluded — its path does not contain "language_model.layers." —
# so SigLIP stays fully frozen, matching ``train_jax_lora_recap.py``'s
# ``vision_filter``.
DEFAULT_ENCODER_PREFIX = "paligemma.model.language_model.layers."
DEFAULT_DECODER_PREFIX = "gemma_expert.model.layers."

_FP8_E4M3_MAX = 448.0


@dataclass(frozen=True)
class InjectionConfig:
    """Configuration object for ``inject_fp8_lora``.

    Defaults track openpi's ``gemma_2b_lora`` (rank=16, alpha=16) and
    ``gemma_300m_lora`` (rank=32, alpha=32) variants.
    """

    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES
    encoder_prefix: str = DEFAULT_ENCODER_PREFIX
    decoder_prefix: str = DEFAULT_DECODER_PREFIX
    encoder_rank: int = 16
    decoder_rank: int = 32
    encoder_alpha: float | None = None
    decoder_alpha: float | None = None
    init_std: float = 0.01
    rslora: bool = False


def _resolve_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    """Return ``(parent_module, child_attr_name)`` for a dotted path."""
    if "." not in qualified_name:
        return model, qualified_name
    parent_path, child_name = qualified_name.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    return parent, child_name


def _belongs_to(pattern: str, qualified_name: str) -> bool:
    """Substring containment — works for both bare and nested wrappings."""
    if not pattern:
        return False
    return pattern in qualified_name


def inject_fp8_lora(
    model: nn.Module,
    config: InjectionConfig | None = None,
) -> dict[str, Fp8LinearWithLora]:
    """Walk ``model``, swap matching ``nn.Linear`` modules for FP8+LoRA.

    Args:
        model: Root module (e.g. ``PaliGemmaWithExpertModel`` or
            full ``PI0Pytorch``).
        config: Optional ``InjectionConfig``; defaults to openpi's
            standard rank/alpha.

    Returns:
        Mapping from fully qualified module name to its replacement.
        Only modules that were actually replaced appear in the dict.

    Side effects:
        * In-place ``setattr`` on parents to install the replacements.
        * ``param.requires_grad_(False)`` on every non-LoRA parameter
          in the entire model after replacement.
    """
    cfg = config or InjectionConfig()

    target_set = set(cfg.target_modules)
    encoder_alpha = cfg.encoder_alpha if cfg.encoder_alpha is not None else float(cfg.encoder_rank)
    decoder_alpha = cfg.decoder_alpha if cfg.decoder_alpha is not None else float(cfg.decoder_rank)

    candidates: list[tuple[str, nn.Linear, str]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        last = name.rsplit(".", 1)[-1]
        if last not in target_set:
            continue
        if _belongs_to(cfg.encoder_prefix, name):
            tag = "encoder"
        elif _belongs_to(cfg.decoder_prefix, name):
            tag = "decoder"
        else:
            continue
        candidates.append((name, module, tag))

    replacements: dict[str, Fp8LinearWithLora] = {}
    for name, linear, tag in candidates:
        rank = cfg.encoder_rank if tag == "encoder" else cfg.decoder_rank
        alpha = encoder_alpha if tag == "encoder" else decoder_alpha
        replacement = Fp8LinearWithLora.from_linear(
            linear,
            rank=rank,
            alpha=alpha,
            init_std=cfg.init_std,
            rslora=cfg.rslora,
        )
        replacement = replacement.to(linear.weight.device)
        parent, child_name = _resolve_parent(model, name)
        setattr(parent, child_name, replacement)
        replacements[name] = replacement

    for pname, param in model.named_parameters():
        is_lora = pname.endswith(".lora_A") or pname.endswith(".lora_B")
        param.requires_grad_(is_lora)

    return replacements


def calibrate_activation_scales(
    model: nn.Module,
    fp8_modules: dict[str, Fp8LinearWithLora],
    calibration_passes: Iterable[Callable[[], None]],
    *,
    safety_margin: float = 1.25,
) -> dict[str, float]:
    """Run calibration passes; install static FP8 activation scales.

    Args:
        model: The injected model. Only used to satisfy a precondition
            check that ``fp8_modules`` are reachable.
        fp8_modules: Output of :func:`inject_fp8_lora`.
        calibration_passes: Iterable of zero-arg callables that each
            drive a forward pass through the model. Inputs and shapes
            must match the production training distribution.
        safety_margin: Multiplier on observed amax before deriving the
            scale. ``1.25`` follows the inference-side calibration
            cushion to absorb post-calibration distribution drift.

    Returns:
        Mapping ``{module_name: installed_scale}``.
    """
    if safety_margin <= 0:
        raise ValueError(f"safety_margin must be > 0, got {safety_margin}")

    amax_state: dict[str, float] = {n: 0.0 for n in fp8_modules}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(layer_name: str):
        def _hook(_mod: nn.Module, args: tuple, _kwargs: dict | None = None):
            if not args:
                return None
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return None
            cur = float(x.detach().abs().max().item())
            if cur > amax_state[layer_name]:
                amax_state[layer_name] = cur
            return None

        return _hook

    for name, mod in fp8_modules.items():
        if model.get_submodule(name) is not mod:
            raise RuntimeError(
                f"calibrate: module '{name}' has been replaced again "
                f"since inject_fp8_lora; refresh the fp8_modules dict."
            )
        handles.append(mod.register_forward_pre_hook(_make_hook(name), with_kwargs=True))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for run in calibration_passes:
                run()
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    scales: dict[str, float] = {}
    for name, mod in fp8_modules.items():
        amax = amax_state[name] * safety_margin
        scale = max(amax / _FP8_E4M3_MAX, 1e-12)
        mod.set_activation_scale(scale)
        scales[name] = scale
    return scales

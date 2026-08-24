"""Checkpoint I/O for the W9 long-form training driver.

Two surfaces:

* :func:`load_pi05_pretrained` — read the openpi PyTorch pi0.5 base
  weights (``model.safetensors``) into a freshly-built
  ``PI0Pytorch``. Handles the tied-embedding edge case
  (``paligemma.lm_head.weight`` shares storage with
  ``paligemma.model.language_model.embed_tokens.weight``; the ckpt
  only stores one of them).

* :func:`save_lora_state` / :func:`load_lora_state` — write / read
  *just* the LoRA adapters from a compiled :class:`Pi05Trainer`.
  A pi0.5 LoRA ckpt is ~5–20 MB depending on rank — a hundredth of
  the 14 GB base — so the user can keep many training run snapshots
  without burning disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from training._vendor.openpi_pi0_pytorch import PI0Pytorch, Pi0Config
from training.lora.fp8_linear import Fp8LinearWithLora
from training.trainers.pi05_torch_trainer import Pi05Trainer


@dataclass
class LoraCheckpointMetadata:
    """Per-layer LoRA shape + scale info, persisted alongside the weights."""

    layer_names: list[str]
    rank_per_layer: dict[str, int]
    alpha_per_layer: dict[str, float]
    scaling_per_layer: dict[str, float]
    weight_scale_per_layer: dict[str, float]


# ── Pi0.5 base ckpt ────────────────────────────────────────────────


def build_pi0_config_from_dir(ckpt_dir: str | Path) -> Pi0Config:
    """Read ``config.json`` next to ``model.safetensors`` and return a Pi0Config.

    The openpi PyTorch ckpts ship a tiny ``config.json`` carrying just
    the variant tags + dtype; the W5 vendor ``Pi0Config`` defaults
    cover the rest (action_horizon=50 by default; override after the
    fact for action_horizon=10 LIBERO).
    """
    ckpt_dir = Path(ckpt_dir)
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing {config_path}")
    raw = json.loads(config_path.read_text())
    return Pi0Config(
        pi05=(raw.get("type") == "pi05"),
        paligemma_variant=raw.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=raw.get("action_expert_variant", "gemma_300m"),
        dtype=raw.get("dtype", "bfloat16"),
    )


def load_pi05_pretrained(
    ckpt_dir: str | Path,
    *,
    action_horizon: int | None = None,
    device: str | torch.device = "cuda",
    strict_keys: bool = True,
) -> PI0Pytorch:
    """Build a ``PI0Pytorch`` and load its weights from ``ckpt_dir``.

    Args:
        ckpt_dir: Directory containing ``config.json`` and
            ``model.safetensors``.
        action_horizon: Optional override (LIBERO uses 10, default
            Pi0Config is 50). The action_horizon is a runtime hyper,
            not a stored tensor, so override-after-config is safe.
        device: Target device for the loaded model.
        strict_keys: If True, raise on unexpected keys in the ckpt.
            The tied-embedding miss
            (``paligemma.model.language_model.embed_tokens.weight``)
            is always tolerated even when ``strict_keys=True`` because
            its storage is shared with ``paligemma.lm_head.weight``.

    Returns:
        A loaded ``PI0Pytorch`` on ``device``, in ``train()`` mode.
    """
    ckpt_dir = Path(ckpt_dir)
    safetensors_path = ckpt_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"missing {safetensors_path}")

    cfg = build_pi0_config_from_dir(ckpt_dir)
    if action_horizon is not None:
        cfg = Pi0Config(
            pi05=cfg.pi05,
            paligemma_variant=cfg.paligemma_variant,
            action_expert_variant=cfg.action_expert_variant,
            dtype=cfg.dtype,
            action_horizon=action_horizon,
        )

    model = PI0Pytorch(cfg)
    sd = load_file(str(safetensors_path))
    result = model.load_state_dict(sd, strict=False)

    # Tied-embedding tolerance: lm_head and embed_tokens share storage,
    # but ``state_dict`` emits both keys; the ckpt only contains
    # lm_head, so embed_tokens shows up as "missing" while in fact
    # being filled via the shared storage.
    tolerable_missing = {
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
    }
    surprising_missing = [k for k in result.missing_keys if k not in tolerable_missing]
    if strict_keys and (surprising_missing or result.unexpected_keys):
        raise RuntimeError(
            f"unexpected ckpt key drift: missing={surprising_missing} "
            f"unexpected={list(result.unexpected_keys)}"
        )

    return model.to(torch.device(device))


# ── LoRA-only ckpt ─────────────────────────────────────────────────


def _collect_lora_state(trainer: Pi05Trainer) -> tuple[dict[str, torch.Tensor], LoraCheckpointMetadata]:
    if not trainer.is_compiled:
        raise RuntimeError(
            "Pi05Trainer must be compiled (FP8 + LoRA injected) before "
            "save_lora_state; call trainer.compile(...) first."
        )

    weights: dict[str, torch.Tensor] = {}
    layer_names: list[str] = []
    rank_per_layer: dict[str, int] = {}
    alpha_per_layer: dict[str, float] = {}
    scaling_per_layer: dict[str, float] = {}
    weight_scale_per_layer: dict[str, float] = {}

    for name, mod in trainer.fp8_modules.items():
        if not isinstance(mod, Fp8LinearWithLora):
            raise TypeError(f"module {name} is not Fp8LinearWithLora")
        layer_names.append(name)
        weights[f"{name}.lora_A"] = mod.lora_A.detach().contiguous().cpu()
        weights[f"{name}.lora_B"] = mod.lora_B.detach().contiguous().cpu()
        weights[f"{name}.act_scale"] = mod.act_scale_dev.detach().contiguous().cpu()
        rank_per_layer[name] = int(mod.rank)
        alpha_per_layer[name] = float(mod.alpha)
        scaling_per_layer[name] = float(mod.scaling)
        weight_scale_per_layer[name] = float(mod.weight_scale_f)

    metadata = LoraCheckpointMetadata(
        layer_names=layer_names,
        rank_per_layer=rank_per_layer,
        alpha_per_layer=alpha_per_layer,
        scaling_per_layer=scaling_per_layer,
        weight_scale_per_layer=weight_scale_per_layer,
    )
    return weights, metadata


def save_lora_state(
    trainer: Pi05Trainer,
    output_dir: str | Path,
) -> Path:
    """Persist LoRA adapters + activation scales from a compiled trainer.

    Layout::

        <output_dir>/
            lora.safetensors      # weights (lora_A, lora_B, act_scale per layer)
            lora_metadata.json    # per-layer rank / alpha / scaling / weight_scale

    Returns the directory path (created if missing).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights, metadata = _collect_lora_state(trainer)
    save_file(weights, str(output_dir / "lora.safetensors"))
    (output_dir / "lora_metadata.json").write_text(
        json.dumps(
            {
                "layer_names": metadata.layer_names,
                "rank_per_layer": metadata.rank_per_layer,
                "alpha_per_layer": metadata.alpha_per_layer,
                "scaling_per_layer": metadata.scaling_per_layer,
                "weight_scale_per_layer": metadata.weight_scale_per_layer,
            },
            indent=2,
        )
    )
    return output_dir


def load_lora_state(
    trainer: Pi05Trainer,
    ckpt_dir: str | Path,
    *,
    strict: bool = True,
) -> LoraCheckpointMetadata:
    """Load LoRA adapters into a compiled trainer.

    The trainer must already have run ``compile`` so the target
    ``Fp8LinearWithLora`` modules exist with matching shapes. Mismatch
    in any layer's rank / out / in dim raises immediately.

    Args:
        trainer: Compiled :class:`Pi05Trainer`.
        ckpt_dir: Directory produced by :func:`save_lora_state`.
        strict: If True, every layer in the metadata file must be
            present in ``trainer.fp8_modules`` and vice versa.

    Returns:
        The :class:`LoraCheckpointMetadata` parsed from the ckpt.
    """
    if not trainer.is_compiled:
        raise RuntimeError(
            "Pi05Trainer must be compiled before load_lora_state."
        )
    ckpt_dir = Path(ckpt_dir)
    weights = load_file(str(ckpt_dir / "lora.safetensors"))
    raw = json.loads((ckpt_dir / "lora_metadata.json").read_text())
    metadata = LoraCheckpointMetadata(
        layer_names=list(raw["layer_names"]),
        rank_per_layer={k: int(v) for k, v in raw["rank_per_layer"].items()},
        alpha_per_layer={k: float(v) for k, v in raw["alpha_per_layer"].items()},
        scaling_per_layer={k: float(v) for k, v in raw["scaling_per_layer"].items()},
        weight_scale_per_layer={k: float(v) for k, v in raw["weight_scale_per_layer"].items()},
    )

    fp8_modules = trainer.fp8_modules
    if strict:
        ckpt_set = set(metadata.layer_names)
        model_set = set(fp8_modules)
        if ckpt_set != model_set:
            raise RuntimeError(
                f"LoRA layer set mismatch: only_in_ckpt={ckpt_set - model_set}, "
                f"only_in_model={model_set - ckpt_set}"
            )

    for name in metadata.layer_names:
        if name not in fp8_modules:
            if strict:
                raise RuntimeError(f"layer '{name}' not in trainer")
            continue
        mod = fp8_modules[name]
        a = weights[f"{name}.lora_A"]
        b = weights[f"{name}.lora_B"]
        if a.shape != mod.lora_A.shape or b.shape != mod.lora_B.shape:
            raise RuntimeError(
                f"shape mismatch on '{name}': lora_A {tuple(a.shape)} vs "
                f"{tuple(mod.lora_A.shape)}; lora_B {tuple(b.shape)} vs "
                f"{tuple(mod.lora_B.shape)}"
            )
        with torch.no_grad():
            mod.lora_A.copy_(a.to(mod.lora_A.device, mod.lora_A.dtype))
            mod.lora_B.copy_(b.to(mod.lora_B.device, mod.lora_B.dtype))
            scale_t = weights[f"{name}.act_scale"].to(
                mod.act_scale_dev.device, mod.act_scale_dev.dtype
            )
            mod.act_scale_dev.copy_(scale_t)

    return metadata

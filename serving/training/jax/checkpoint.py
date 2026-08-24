"""LoRA-only checkpoint save/load — JAX-native.

Mirrors the surface of the PyTorch ``training.rl.checkpoint`` pair
(``save_lora_state`` / ``load_lora_state``) but in JAX-native
formats:

* ``lora_weights.npz`` — flat numpy archive keyed by openpi nnx
  param paths (e.g. ``"PaliGemma/llm/layers/attn/qkv_einsum/lora_a"``).
  Same format the upstream
  ``train_jax_lora_recap.py:save_checkpoint`` already produces, so a
  trained checkpoint from the upstream driver is directly consumable
  here without conversion.

* ``lora_metadata.json`` — sidecar carrying:
    - ``layer_paths`` : list[str]  every base layer that has a
                                    LoRA pair attached.
    - ``scaling_per_layer`` : dict[str, float]  the LoRA scaling
                                    factor (alpha / rank, or
                                    alpha / sqrt(rank) for rsLoRA).
                                    pi05's openpi default is
                                    ``alpha == rank`` → 1.0.
    - ``lora_rank`` : int           rank for sanity-checking.
    - ``base_path_per_layer`` : dict[str, str]  the base-weight
                                    param path for each layer (e.g.
                                    ``"…/qkv_einsum/w"`` for the
                                    Einsum case,
                                    ``"…/mlp/gating_einsum"`` for
                                    the FeedForward case).

The pair on disk is identical in spirit to the PyTorch
``lora.safetensors`` + ``lora_metadata.json`` pair, with safetensors
swapped for npz to stay in JAX-native land — no PyTorch dependency
on the JAX path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_LORA_A_SUFFIX = "lora_a"
_LORA_B_SUFFIX = "lora_b"


@dataclass
class JaxLoraMetadata:
    """Sidecar metadata for a saved JAX LoRA artifact."""

    layer_paths: list[str]
    base_path_per_layer: dict[str, str]
    scaling_per_layer: dict[str, float]
    lora_rank: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "layer_paths": list(self.layer_paths),
                "base_path_per_layer": dict(self.base_path_per_layer),
                "scaling_per_layer": {k: float(v) for k, v in self.scaling_per_layer.items()},
                "lora_rank": int(self.lora_rank),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "JaxLoraMetadata":
        raw = json.loads(text)
        return cls(
            layer_paths=list(raw["layer_paths"]),
            base_path_per_layer=dict(raw["base_path_per_layer"]),
            scaling_per_layer={k: float(v) for k, v in raw["scaling_per_layer"].items()},
            lora_rank=int(raw["lora_rank"]),
        )


# ── path conventions ────────────────────────────────────────────────


def _resolve_base_path(lora_a_path: str) -> tuple[str, str]:
    """Return ``(layer_path, base_param_path)`` for a ``lora_a`` key.

    Two openpi naming patterns are supported, both seen in
    ``train_jax_lora_recap.py``'s output ``lora_weights.npz``:

    * Einsum     — ``parent/lora_a``       → base ``parent/w``,
                                              layer key ``parent``.
    * FeedForward — ``parent/X_lora_a``    → base ``parent/X`` (where
                                              X ∈ {gating_einsum, linear}),
                                              layer key ``parent/X``.

    Anything else raises — caller is consuming an unfamiliar param
    naming convention and should update this resolver explicitly.
    """
    if "/" not in lora_a_path:
        raise ValueError(f"unexpected LoRA path with no separator: {lora_a_path}")
    parent, name = lora_a_path.rsplit("/", 1)
    if name == _LORA_A_SUFFIX:
        # Einsum case: parent/lora_a → parent/w
        return parent, f"{parent}/w"
    suffix = "_" + _LORA_A_SUFFIX
    if not name.endswith(suffix):
        raise ValueError(
            f"LoRA-A path does not end in 'lora_a' or '_lora_a': {lora_a_path}"
        )
    prefix = name[: -len(suffix)]                    # 'gating_einsum' or 'linear'
    base = f"{parent}/{prefix}"
    return base, base


def _b_path_from_a(lora_a_path: str) -> str:
    if lora_a_path.endswith("/" + _LORA_A_SUFFIX):
        return lora_a_path[: -len(_LORA_A_SUFFIX)] + _LORA_B_SUFFIX
    if lora_a_path.endswith("_" + _LORA_A_SUFFIX):
        return lora_a_path[: -len("_" + _LORA_A_SUFFIX)] + "_" + _LORA_B_SUFFIX
    raise ValueError(f"not a LoRA-A path: {lora_a_path}")


def discover_lora_pairs(
    flat_params: dict[str, np.ndarray],
) -> list[tuple[str, str, str, str]]:
    """Find ``(layer_path, base_path, lora_a_path, lora_b_path)`` tuples.

    Walks ``flat_params`` (path-string → ndarray dict, the same shape
    ``np.load(...).items()`` returns from a saved ``lora_weights.npz``)
    and pairs every ``lora_a`` with its corresponding ``lora_b``.
    """
    pairs: list[tuple[str, str, str, str]] = []
    for path in sorted(flat_params):
        if path.endswith(_LORA_A_SUFFIX) is False and not path.endswith("_" + _LORA_A_SUFFIX):
            continue
        # Be strict: the suffix must be the trailing component (or the
        # tail of the trailing component).
        b_path = _b_path_from_a(path)
        if b_path not in flat_params:
            raise KeyError(
                f"LoRA pair incomplete: have '{path}' but not '{b_path}'."
            )
        layer_path, base_path = _resolve_base_path(path)
        pairs.append((layer_path, base_path, path, b_path))
    return pairs


# ── save / load ─────────────────────────────────────────────────────


def save_lora_state(
    flat_params: dict[str, np.ndarray],
    output_dir: str | Path,
    *,
    scaling: float | dict[str, float] = 1.0,
    overwrite: bool = False,
) -> Path:
    """Save a flat LoRA-param dict + metadata to ``output_dir``.

    Args:
        flat_params: Path-string → ndarray map containing exactly the
            LoRA-bearing tensors (everything ending in ``lora_a`` or
            ``lora_b``). Anything else in the dict is silently
            ignored — only LoRA pairs are persisted.
        output_dir: Destination directory. Created if missing.
        scaling: LoRA scaling factor. ``float`` applies uniformly to
            every layer; ``dict[layer_path, float]`` applies per
            layer (matching the PyTorch
            ``checkpoint.JaxLoraMetadata.scaling_per_layer`` field).
        overwrite: If True, overwrite existing files in
            ``output_dir``.

    Returns:
        ``output_dir`` as a :class:`Path`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "lora_weights.npz"
    meta_path = output_dir / "lora_metadata.json"
    if npz_path.exists() and not overwrite:
        raise FileExistsError(f"{npz_path} exists; pass overwrite=True")
    if meta_path.exists() and not overwrite:
        raise FileExistsError(f"{meta_path} exists; pass overwrite=True")

    # Filter to LoRA pairs only — caller may pass a dict that contains
    # base weights too (e.g. the full upstream training state). That's
    # fine; we extract just the lora_a/lora_b entries.
    lora_only = {
        k: np.asarray(v)
        for k, v in flat_params.items()
        if k.endswith(_LORA_A_SUFFIX) or k.endswith(_LORA_B_SUFFIX)
    }
    pairs = discover_lora_pairs(lora_only)
    layer_paths = [layer for layer, _, _, _ in pairs]
    base_path_per_layer = {layer: base for layer, base, _, _ in pairs}

    if isinstance(scaling, dict):
        scaling_per_layer = {layer: float(scaling.get(layer, 1.0)) for layer in layer_paths}
    else:
        scaling_per_layer = {layer: float(scaling) for layer in layer_paths}

    # rank = trailing axis of any lora_a tensor (all should match for a single pi05 config).
    if not pairs:
        raise ValueError("no LoRA pairs found in flat_params — nothing to save")
    sample_a = lora_only[pairs[0][2]]
    lora_rank = int(sample_a.shape[-1])

    np.savez(str(npz_path), **lora_only)
    metadata = JaxLoraMetadata(
        layer_paths=layer_paths,
        base_path_per_layer=base_path_per_layer,
        scaling_per_layer=scaling_per_layer,
        lora_rank=lora_rank,
    )
    meta_path.write_text(metadata.to_json())
    return output_dir


def load_lora_state(
    lora_dir: str | Path,
) -> tuple[dict[str, np.ndarray], JaxLoraMetadata]:
    """Read back a directory written by :func:`save_lora_state`."""
    lora_dir = Path(lora_dir)
    npz_path = lora_dir / "lora_weights.npz"
    meta_path = lora_dir / "lora_metadata.json"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    with np.load(str(npz_path)) as z:
        flat = {k: np.asarray(z[k]) for k in z.files}
    metadata = JaxLoraMetadata.from_json(meta_path.read_text())
    return flat, metadata

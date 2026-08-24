"""PaliGemma tokenizer wrapper for the policy training driver.

Loads the PaliGemma SentencePiece tokenizer from a *local* path —
no network calls, no HuggingFace Hub downloads. Resolution order:

1. The ``path`` argument when explicitly provided.
2. The ``FLASHVLA_TOKENIZER_PATH`` environment variable.

A tokenizer directory is expected to contain:

* ``tokenizer.model``           — SentencePiece model bytes
* ``tokenizer_config.json``     — HF tokenizer metadata
* ``special_tokens_map.json``   — BOS / EOS / PAD ids

The wrapper exposes a single ``__call__(list[str]) → (tokens, mask)``
surface so the W7 ``train_policy`` driver can plug it in via the
``tokenize_fn`` argument unchanged. ``max_token_len`` defaults to
``200`` (Pi0Config default for pi0.5).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def _resolve_tokenizer_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("FLASHVLA_TOKENIZER_PATH")
    if env:
        return Path(env)
    raise FileNotFoundError(
        "Pass an explicit `path` or set FLASHVLA_TOKENIZER_PATH to a "
        "directory containing tokenizer.model + tokenizer_config.json."
    )


class PaligemmaTokenizer:
    """Local PaliGemma SentencePiece tokenizer wrapper.

    Args:
        path: Directory containing ``tokenizer.model`` +
            ``tokenizer_config.json``.
        max_token_len: Pad / truncate to this length. Pi0.5 default 200.
        device: Device for the returned tensors.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_token_len: int = 200,
        device: str | torch.device = "cuda",
    ):
        path = _resolve_tokenizer_path(path)
        if not (path / "tokenizer.model").exists():
            raise FileNotFoundError(
                f"PaliGemma tokenizer not found at {path}. Provide an "
                "explicit `path=` or set FLASHVLA_TOKENIZER_PATH; no "
                "download fallback."
            )
        # Lazy import — keeps the module light when only metadata is read.
        from transformers import AutoTokenizer

        self._tok: Any = AutoTokenizer.from_pretrained(str(path))
        self.max_token_len = int(max_token_len)
        self.device = torch.device(device)
        self.vocab_size = int(self._tok.vocab_size)

    def __call__(self, tasks: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a list of task strings to ``(int64[B, L], bool[B, L])``."""
        if not isinstance(tasks, list) or not all(isinstance(t, str) for t in tasks):
            raise TypeError("tasks must be list[str]")
        out = self._tok(
            tasks,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_len,
            return_tensors="pt",
        )
        tokens = out["input_ids"].to(dtype=torch.long, device=self.device)
        mask = out["attention_mask"].to(dtype=torch.bool, device=self.device)
        return tokens, mask

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        """Decode an ``int[B, L]`` tensor back to a list of strings."""
        if token_ids.ndim != 2:
            raise ValueError(
                f"decode expects 2D tensor, got shape={tuple(token_ids.shape)}"
            )
        return self._tok.batch_decode(token_ids, skip_special_tokens=True)

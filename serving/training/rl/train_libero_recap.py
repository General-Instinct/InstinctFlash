r"""LIBERO entry point for the RECAP / ACP policy training driver.

Thin LIBERO-specific wrapper around the **dataset-agnostic** core
in :mod:`training.rl.train_recap`. The training loop semantics
(AdamW + warmup-cosine + ACP injection at 30% dropout + flow-
matching loss + LoRA adapters) are paper-aligned (π\\*0.6
arXiv:2511.14759) and not LIBERO-specific — that's why the actual
math lives in ``train_recap.py``. This module supplies:

* :class:`LeRobotLiberoDataset` (concrete impl of
  :class:`training.rl.dataset_protocol.RecapPolicyDataset`),
* :func:`training.rl.observation.decoded_to_observation` (LIBERO 2-cam
  → pi0.5 3-cam :class:`Observation` adapter),
* sensible defaults for env-var resolution.

Aliases preserved for back-compat:

* ``LiberoTrainConfig``  → :class:`training.rl.train_recap.RecapTrainConfig`
* ``LiberoTrainResult``  → :class:`training.rl.train_recap.RecapTrainResult`
* ``train_libero_recap`` → calls :func:`train_recap_policy` with the
  LIBERO observation adapter wired in.

For a non-LIBERO dataset, implement :class:`RecapPolicyDataset` and
call :func:`training.rl.train_recap.train_recap_policy` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from training.rl.lerobot_libero import LeRobotLiberoDataset
from training.rl.observation import decoded_to_observation
from training.rl.tokenizer import PaligemmaTokenizer
from training.rl.train_recap import (
    RecapTrainConfig,
    RecapTrainResult,
    make_warmup_cosine_lr_lambda,
    train_recap_policy,
)
from training.trainers.pi05_torch_trainer import Pi05Trainer

# Back-compat aliases — any user code or tests that import these names
# keeps working unchanged. New code should import the generic names
# directly from ``training.rl.train_recap``.
LiberoTrainConfig = RecapTrainConfig
LiberoTrainResult = RecapTrainResult


__all__ = [
    "LiberoTrainConfig",       # back-compat alias for RecapTrainConfig
    "LiberoTrainResult",       # back-compat alias for RecapTrainResult
    "make_warmup_cosine_lr_lambda",
    "train_libero_recap",
]


def train_libero_recap(
    trainer: Pi05Trainer,
    dataset: LeRobotLiberoDataset,
    tokenizer: PaligemmaTokenizer,
    *,
    config: RecapTrainConfig | None = None,
    output_dir: str | Path | None = None,
    derive_acp_if_missing: bool = True,
    progress_cb: Callable[[int, float, float], None] | None = None,
) -> RecapTrainResult:
    """LIBERO-specific shim around :func:`train_recap_policy`.

    Wires the LIBERO observation adapter
    (:func:`training.rl.observation.decoded_to_observation`) into
    the generic driver and forwards everything else.

    Args mirror :func:`training.rl.train_recap.train_recap_policy` —
    see that function's docstring for parameter semantics.
    """
    return train_recap_policy(
        trainer=trainer,
        dataset=dataset,
        tokenizer=tokenizer,
        observation_builder=decoded_to_observation,
        config=config,
        output_dir=output_dir,
        derive_acp_if_missing=derive_acp_if_missing,
        progress_cb=progress_cb,
    )

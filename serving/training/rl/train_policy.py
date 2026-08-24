"""Advantage-conditioned policy training driver for pi0.5.

Wraps a compiled ``Pi05Trainer`` (FP8 + LoRA) with the ACP prompt
injection step so each training mini-batch:

1. Reads the ``acp_indicator`` column populated by ``value_infer``;
2. Modifies ``batch["task"]`` via :class:`ACPPromptHook` (30 % dropout
   per paper recommendation, rest get ``"Advantage: positive/negative"``
   appended);
3. Re-tokenises the modified prompts and rebuilds the
   :class:`Observation` slot for ``tokenized_prompt`` /
   ``tokenized_prompt_mask``;
4. Runs ``trainer.model.forward(observation, actions)`` — the same
   flow-matching loss as the non-ACP path;
5. Backprops and steps the optimiser over **only** the LoRA adapters.

The driver is intentionally model-agnostic at the data layer: caller
supplies an iterable of ``(observation, actions, indicators, tasks)``
tuples and a ``tokenize_fn``. The W7 unit test feeds synthetic
observations of the right shape; W8 swaps in a LeRobotDataset adapter.

Source: paper-aligned port of openpi-compiler/RL/recap/train_policy.py
(2026-04-25). The upstream reference uses a standalone MLP for
pipeline testing — we drive a real ``PI0Pytorch`` directly because
the FP8 + LoRA infrastructure was W5's deliverable and does not
need a stand-in.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from training.rl.acp_hook import ACPPromptHook
from training.trainers.pi05_torch_trainer import Pi05Trainer

#: Signature of a tokenizer callable.
#: ``(list[str]) → (tokens int64[B, L], mask bool[B, L])``.
TokenizeFn = Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class PolicyTrainResult:
    """Aggregate output of :func:`run_policy_training`."""

    loss_history: list[float] = field(default_factory=list)
    peak_memory_bytes: int = 0
    num_steps: int = 0


def _replace_tokenized_prompt(observation: Any, tokens: torch.Tensor, mask: torch.Tensor) -> Any:
    """Return a copy of ``observation`` with the prompt slot swapped in.

    Uses ``dataclasses.replace`` if available, otherwise falls back to
    setting attributes on a copy. The vendored openpi ``Observation``
    is a frozen dataclass, so the dataclass path is the common one.
    """
    try:
        import dataclasses

        if dataclasses.is_dataclass(observation):
            return dataclasses.replace(
                observation,
                tokenized_prompt=tokens,
                tokenized_prompt_mask=mask,
            )
    except (TypeError, ImportError):
        pass

    # Fallback: SimpleNamespace-style object — copy + setattr.
    import copy

    new_obs = copy.copy(observation)
    new_obs.tokenized_prompt = tokens
    new_obs.tokenized_prompt_mask = mask
    return new_obs


def policy_train_step(
    trainer: Pi05Trainer,
    optimizer: torch.optim.Optimizer,
    observation: Any,
    actions: torch.Tensor,
    *,
    indicators: torch.Tensor | None = None,
    tasks: list[str] | None = None,
    hook: ACPPromptHook | None = None,
    tokenize_fn: TokenizeFn | None = None,
    grad_clip_norm: float | None = 1.0,
) -> float:
    """Single ACP-aware policy training step.

    Args:
        trainer: A *compiled* :class:`Pi05Trainer`.
        optimizer: Optimizer over ``trainer.trainable_parameters()``.
        observation: Vendored openpi Observation dataclass instance.
        actions: ``(B, action_horizon, action_dim)`` ground-truth
            action chunk for the flow-matching loss.
        indicators: Optional ``(B,) int64`` indicator tensor from
            ``value_infer``. Required when ``hook`` is set.
        tasks: Optional ``list[str]`` of base task prompts; required
            when ``hook`` is set so the hook can mutate them.
        hook: Optional :class:`ACPPromptHook`. ``None`` disables ACP
            (useful for the no-ACP baseline curve).
        tokenize_fn: Optional tokenizer; required when ``hook`` is set
            so the modified prompts can flow back into the
            ``tokenized_prompt`` slot.
        grad_clip_norm: Optional gradient-norm clip applied before
            ``optimizer.step()``. Default 1.0 — Evo-RL convention.

    Returns:
        Scalar loss value as a Python float.
    """
    if hook is not None:
        if tokenize_fn is None:
            raise ValueError("tokenize_fn is required when hook is set")
        if indicators is None or tasks is None:
            raise ValueError("indicators and tasks are required when hook is set")
        if len(tasks) != actions.shape[0]:
            raise ValueError(
                f"tasks count {len(tasks)} != actions batch {actions.shape[0]}"
            )

        batch = {"task": list(tasks), "acp_indicator": indicators}
        modified = hook(batch)
        new_tokens, new_mask = tokenize_fn(modified["task"])
        observation = _replace_tokenized_prompt(observation, new_tokens, new_mask)

    trainer.model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_per_step = trainer.model(observation, actions)
    loss = loss_per_step.mean()
    loss.backward()
    if grad_clip_norm is not None and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            trainer.trainable_parameters(), grad_clip_norm
        )
    optimizer.step()
    return float(loss.item())


def run_policy_training(
    trainer: Pi05Trainer,
    batch_iter: Iterable[tuple[Any, torch.Tensor, torch.Tensor | None, list[str] | None]],
    *,
    num_steps: int,
    hook: ACPPromptHook | None = None,
    tokenize_fn: TokenizeFn | None = None,
    lr: float = 5e-5,
    grad_clip_norm: float | None = 1.0,
    optimizer: torch.optim.Optimizer | None = None,
) -> PolicyTrainResult:
    """Run a fixed-step policy training loop with optional ACP injection.

    Args:
        trainer: A compiled :class:`Pi05Trainer`.
        batch_iter: Iterable yielding tuples
            ``(observation, actions, indicators, tasks)``. ``indicators``
            and ``tasks`` may be ``None`` when ``hook`` is ``None`` (no-ACP).
        num_steps: Number of optimiser steps. The iterable is consumed
            up to this count; if it terminates early the loop ends.
        hook, tokenize_fn: As in :func:`policy_train_step`.
        lr: Used only when ``optimizer is None`` to build a default
            AdamW over ``trainer.trainable_parameters()``.
        grad_clip_norm: As in :func:`policy_train_step`.
        optimizer: Optional pre-built optimiser.

    Returns:
        :class:`PolicyTrainResult` with the loss curve, peak GPU
        memory bytes observed during the loop, and the realised step
        count.
    """
    if not trainer.is_compiled:
        raise RuntimeError(
            "Pi05Trainer must be compiled (FP8 + LoRA injected) before "
            "run_policy_training; call trainer.compile(...) first."
        )

    optim_ = optimizer
    if optim_ is None:
        optim_ = torch.optim.AdamW(trainer.trainable_parameters(), lr=lr)

    trainer.reset_peak_memory()
    losses: list[float] = []
    step = 0
    for batch in batch_iter:
        if step >= num_steps:
            break
        observation, actions, indicators, tasks = batch
        loss = policy_train_step(
            trainer,
            optim_,
            observation,
            actions,
            indicators=indicators,
            tasks=tasks,
            hook=hook,
            tokenize_fn=tokenize_fn,
            grad_clip_norm=grad_clip_norm,
        )
        losses.append(loss)
        step += 1

    return PolicyTrainResult(
        loss_history=losses,
        peak_memory_bytes=trainer.peak_memory_bytes(),
        num_steps=step,
    )

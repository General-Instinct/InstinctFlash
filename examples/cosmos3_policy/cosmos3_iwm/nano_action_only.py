"""Construction-time action-only residency for Cosmos3-Nano on 32 GiB SM120.

Nano's Qwen language wrapper owns an untied 151936x4096 BF16 ``lm_head``. The action-policy
forward never reads it when ``predict_text_tokens`` is false, but materializing that dead head is
enough to make the 32 GiB card fail before or during its first request. This module consumes a
one-shot constructor token while the model is still on ``meta`` and installs a loud zero-parameter
sentinel before the parent ``to_empty(cuda)`` traversal.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import torch

CERTIFIED_HEAD_SHAPE = (151936, 4096)
CERTIFIED_DTYPE = torch.bfloat16


class ElidedActionOnlyLMHead(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Cosmos3-Nano action-only residency removed lm_head. Text-logit or prompt-"
            "upsampling requests require a Runtime built with "
            "IFL_COSMOS3_NANO_ACTION_ONLY=0."
        )


def arm_nano_action_only_lm_head() -> None:
    from cosmos_framework.model.generator.mot.unified_mot import (
        Qwen3VLTextForCausalLM,
    )

    state = getattr(Qwen3VLTextForCausalLM, "_ifl_action_only_state", None)
    if state is None:
        state = threading.local()
        original_init = Qwen3VLTextForCausalLM.__init__

        def _init(self, *args, **kwargs):
            pending = getattr(state, "pending", False)
            if pending:
                del state.pending
            original_init(self, *args, **kwargs)
            if not pending:
                return
            head = self.lm_head
            if not isinstance(head, torch.nn.Linear) or head.bias is not None:
                raise RuntimeError(
                    "Nano action-only residency requires an unbiased nn.Linear lm_head"
                )
            if tuple(head.weight.shape) != CERTIFIED_HEAD_SHAPE:
                raise RuntimeError(
                    f"Nano action-only residency requires lm_head {CERTIFIED_HEAD_SHAPE}, "
                    f"got {tuple(head.weight.shape)}"
                )
            # The released safetensors value is BF16, but upstream intentionally creates
            # its meta placeholder in FP32 and applies checkpoint dtype while loading. Shape,
            # device and tying are the only facts observable at this construction boundary.
            if head.weight.device.type != "meta":
                raise RuntimeError(
                    "Nano lm_head must be elided before CUDA materialization"
                )
            embedding = self.get_input_embeddings()
            if head.weight is embedding.weight:
                raise RuntimeError(
                    "Nano action-only residency requires an untied lm_head"
                )
            self._ifl_action_only_lm_head_shape = tuple(head.weight.shape)
            self._ifl_action_only_lm_head_numel = head.weight.numel()
            self.lm_head = ElidedActionOnlyLMHead()

        Qwen3VLTextForCausalLM.__init__ = _init
        Qwen3VLTextForCausalLM._ifl_action_only_state = state
        Qwen3VLTextForCausalLM._ifl_action_only_original_init = original_init
    if getattr(state, "pending", False):
        raise RuntimeError(
            "Nano action-only constructor is already armed on this thread"
        )
    state.pending = True


def verify_nano_action_only_model(model) -> int:
    network = getattr(model, "net", None)
    language_model = getattr(network, "language_model", None)
    if network is None or language_model is None:
        raise RuntimeError(
            "Nano action-only residency cannot find model.net.language_model"
        )
    if bool(getattr(network, "predict_text_tokens", True)):
        raise RuntimeError("Nano model requests text logits, so lm_head is not dead")
    if not isinstance(getattr(language_model, "lm_head", None), ElidedActionOnlyLMHead):
        raise TypeError("Nano action-only lm_head constructor token was not consumed")
    numel = getattr(language_model, "_ifl_action_only_lm_head_numel", None)
    shape = getattr(language_model, "_ifl_action_only_lm_head_shape", None)
    if (
        shape != CERTIFIED_HEAD_SHAPE
        or numel != CERTIFIED_HEAD_SHAPE[0] * CERTIFIED_HEAD_SHAPE[1]
    ):
        raise RuntimeError(
            "Nano action-only lm_head certificate metadata is inconsistent"
        )
    language_model._ifl_action_only_lm_head_bytes = int(numel) * 2
    return language_model._ifl_action_only_lm_head_bytes


__all__ = [
    "CERTIFIED_DTYPE",
    "CERTIFIED_HEAD_SHAPE",
    "ElidedActionOnlyLMHead",
    "arm_nano_action_only_lm_head",
    "verify_nano_action_only_model",
]


@contextmanager
def nano_action_only_construction(enabled=True):
    """Never leave an unconsumed constructor token after a failed model load."""
    if not enabled:
        yield
        return
    arm_nano_action_only_lm_head()
    from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLTextForCausalLM
    state = Qwen3VLTextForCausalLM._ifl_action_only_state
    try:
        yield
    finally:
        if getattr(state, "pending", False):
            del state.pending

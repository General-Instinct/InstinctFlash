"""Construction-time and fail-closed gates for Nano action-only LM-head elision."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "cosmos3_policy"
sys.path.insert(0, str(PLUGIN))

from cosmos3_iwm.nano_action_only import (
    CERTIFIED_HEAD_SHAPE,
    ElidedActionOnlyLMHead,
    arm_nano_action_only_lm_head,
    verify_nano_action_only_model,
)


def fake_class(*, shape=CERTIFIED_HEAD_SHAPE, tied=False, fail=False):
    class FakeQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if fail:
                raise RuntimeError("synthetic constructor failure")
            out_features, in_features = shape
            self.embed = torch.nn.Embedding(
                out_features, in_features, device="meta", dtype=torch.bfloat16
            )
            self.lm_head = torch.nn.Linear(
                in_features,
                out_features,
                bias=False,
                device="meta",
                dtype=torch.bfloat16,
            )
            if tied:
                self.lm_head.weight = self.embed.weight

        def get_input_embeddings(self):
            return self.embed

    return FakeQwen


def install_fake(monkey_class):
    module = ModuleType("cosmos_framework.model.generator.mot.unified_mot")
    module.Qwen3VLTextForCausalLM = monkey_class
    sys.modules[module.__name__] = module


def test_one_shot_constructor_elides_only_the_armed_instance():
    cls = fake_class()
    install_fake(cls)
    arm_nano_action_only_lm_head()
    first = cls()
    assert isinstance(first.lm_head, ElidedActionOnlyLMHead)
    assert first._ifl_action_only_lm_head_shape == CERTIFIED_HEAD_SHAPE
    second = cls()
    assert isinstance(second.lm_head, torch.nn.Linear)
    arm_nano_action_only_lm_head()
    third = cls()
    assert isinstance(third.lm_head, ElidedActionOnlyLMHead)


def test_verifier_requires_action_only_network_and_returns_bf16_bytes():
    cls = fake_class()
    install_fake(cls)
    arm_nano_action_only_lm_head()
    language = cls()
    model = SimpleNamespace(
        net=SimpleNamespace(language_model=language, predict_text_tokens=False)
    )
    size = verify_nano_action_only_model(model)
    assert size == CERTIFIED_HEAD_SHAPE[0] * CERTIFIED_HEAD_SHAPE[1] * 2
    model.net.predict_text_tokens = True
    try:
        verify_nano_action_only_model(model)
    except RuntimeError as error:
        assert "text logits" in str(error)
    else:
        raise AssertionError("text-producing model accepted action-only elision")


def test_wrong_shape_and_tied_weight_fail_closed():
    for cls, message in (
        (fake_class(shape=(8, 4)), "requires lm_head"),
        (fake_class(tied=True), "untied"),
    ):
        install_fake(cls)
        arm_nano_action_only_lm_head()
        try:
            cls()
        except RuntimeError as error:
            assert message in str(error)
        else:
            raise AssertionError("uncertified Nano head was elided")


def test_failed_constructor_consumes_token():
    cls = fake_class(fail=True)
    install_fake(cls)
    arm_nano_action_only_lm_head()
    try:
        cls()
    except RuntimeError as error:
        assert "synthetic" in str(error)
    else:
        raise AssertionError("synthetic constructor did not fail")
    # No stale pending token: arming again is legal.
    arm_nano_action_only_lm_head()


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))


def test_failure_before_constructor_does_not_poison_next_model():
    import pytest
    from cosmos3_iwm.nano_action_only import nano_action_only_construction
    cls = fake_class()
    install_fake(cls)
    with pytest.raises(RuntimeError, match="checkpoint resolution failed"):
        with nano_action_only_construction():
            raise RuntimeError("checkpoint resolution failed")
    assert isinstance(cls().lm_head, torch.nn.Linear)
    with nano_action_only_construction():
        assert isinstance(cls().lm_head, ElidedActionOnlyLMHead)
    assert isinstance(cls().lm_head, torch.nn.Linear)

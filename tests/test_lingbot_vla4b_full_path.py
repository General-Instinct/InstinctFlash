from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "examples" / "lingbot_vla"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from lingbot_vla_iwm.adapter import LingBotVLA4BAdapter, _env_flag
from lingbot_vla_iwm.full_capture import StaticFullPath, _StaticLinear


def test_full_graph_installer_is_selected_explicitly():
    from lingbot_vla_iwm import full_capture

    installed = {}
    driver = SimpleNamespace(captured=False, replays=0)

    def fake_install(model, on_self_check=None):
        installed["model"] = model
        installed["on_self_check"] = on_self_check
        return driver

    capture = SimpleNamespace(
        name="graph_capture",
        applies=True,
        params={},
    )
    plan = SimpleNamespace(results=[capture])
    policy_model = object()
    server = SimpleNamespace(vla=SimpleNamespace(model=policy_model))

    previous_backend = os.environ.pop("IFL_VLA4B_BACKEND", None)
    previous_kill = os.environ.pop("IFL_VLA4B_NO_CAPTURE", None)
    try:
        output = io.StringIO()
        with (
            mock.patch.object(full_capture, "install_full_capture", fake_install),
            contextlib.redirect_stdout(output),
        ):
            result = LingBotVLA4BAdapter().install(
                server,
                plan,
                device="cuda:0",
                full_graph=True,
            )
    finally:
        if previous_backend is not None:
            os.environ["IFL_VLA4B_BACKEND"] = previous_backend
        if previous_kill is not None:
            os.environ["IFL_VLA4B_NO_CAPTURE"] = previous_kill

    assert result is driver
    assert installed["model"] is policy_model
    assert callable(installed["on_self_check"])
    assert "full vision/prefix + ten-step flow" in output.getvalue()


def test_full_graph_schedule_matches_upstream_bf16_while_loop():
    noise = torch.zeros((1, 50, 75), dtype=torch.bfloat16)
    dt, schedule = StaticFullPath._schedule(noise, 10)

    reference = []
    current = torch.tensor(1.0, dtype=noise.dtype)
    while current >= -dt / 2:
        reference.append(current.expand(1).clone())
        current += dt

    assert len(schedule) == 10
    assert all(torch.equal(left, right) for left, right in zip(schedule, reference))


def test_changed_full_graph_inputs_keep_signatures_and_change_values():
    images = torch.arange(24).reshape(1, 2, 3, 4)
    image_masks = torch.tensor([[True, False]])
    tokens = torch.tensor([[1, 2, 3, 4]])
    token_masks = torch.tensor([[True, True, False, False]])
    state = torch.arange(6).reshape(1, 6)

    changed = StaticFullPath._changed_inputs(
        images,
        image_masks,
        tokens,
        token_masks,
        state,
    )
    for original, staged in zip(
        (images, image_masks, tokens, token_masks, state),
        changed,
    ):
        assert original.shape == staged.shape
        assert original.dtype == staged.dtype
    assert not torch.equal(images, changed[0])
    assert torch.equal(image_masks, changed[1])
    assert not torch.equal(tokens, changed[2])
    assert not torch.equal(state, changed[4])


def test_static_linear_returns_precomputed_tensor_without_arithmetic():
    real = torch.nn.Linear(4, 3)
    output = torch.randn(2, 3)
    table = _StaticLinear(real, output)

    got = table(torch.randn(2, 4))

    assert got is output


def test_full_graph_environment_flag_is_strict():
    name = "IFL_VLA4B_FULL_GRAPH"
    previous = os.environ.get(name)
    try:
        os.environ.pop(name, None)
        assert _env_flag(name, default=True) is True
        os.environ[name] = "off"
        assert _env_flag(name, default=True) is False
        os.environ[name] = "1"
        assert _env_flag(name, default=False) is True
        os.environ[name] = "sometimes"
        try:
            _env_flag(name, default=False)
        except ValueError as error:
            assert name in str(error)
        else:
            raise AssertionError("invalid full-graph flag must fail closed")
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))


def test_initial_full_path_admission_error_rejects_before_serving():
    # Exercise the failure before graph allocation using the real __call__ entry point.
    driver = StaticFullPath.__new__(StaticFullPath)
    driver.fm = SimpleNamespace(config=SimpleNamespace(num_steps=10))
    driver.rejected = False
    driver.patch_proved = False
    driver.original_sample_actions = lambda *args, noise=None, **kwargs: noise
    def fail(*args, **kwargs):
        raise RuntimeError("upstream API drift")
    driver._call_original = fail
    def reject(reason):
        driver.rejected = True
        assert "upstream API drift" in reason
    driver._reject = reject
    x = torch.zeros(1, 2)
    noise = torch.ones(1, 2)
    assert driver(x, x, x, x, x, noise=noise, num_steps=10) is noise
    assert driver.rejected


def test_eager_vision_hoists_only_sequence_boundaries():
    from lingbot_vla_iwm.full_capture import _eager_vision_with_host_boundaries
    calls = []
    def original(hidden, boundaries, **kwargs):
        calls.append((hidden, boundaries, kwargs))
        return hidden + 1
    patched = _eager_vision_with_host_boundaries(original)
    hidden = torch.ones(2, 3)
    indices = torch.tensor([0, 1, 2])
    positions = (torch.ones(2, 3), torch.zeros(2, 3))
    with mock.patch('torch.cuda.is_current_stream_capturing', return_value=False):
        expected = patched(None, hidden, indices, position_embeddings=positions)
    with mock.patch('torch.cuda.is_current_stream_capturing', return_value=True):
        actual = patched(None, hidden, indices, position_embeddings=positions)
        try:
            patched(None, hidden, indices.clone(), position_embeddings=positions)
        except RuntimeError as error:
            assert 'not warmed' in str(error)
        else:
            raise AssertionError('unwarmed boundaries admitted during capture')
    assert torch.equal(actual, expected)
    assert calls[0][0] is hidden and calls[0][2]['position_embeddings'] is positions
    assert calls[0][1] is calls[1][1] and torch.equal(calls[0][1], indices)


def test_eager_prefix_setup_does_not_import_or_select_flash_attention():
    import types
    from lingbot_vla_iwm.full_capture import CaptureSafePrefixPatches
    class Qwen2_5_VLVisionAttention:
        def forward(self, hidden_states, cu_seqlens, **kwargs):
            return hidden_states
    attention = Qwen2_5_VLVisionAttention()
    expert = SimpleNamespace(
        qwenvl=SimpleNamespace(visual=SimpleNamespace(blocks=[SimpleNamespace(attn=attention)])),
        embed_image=lambda image: image, rotary_pos_emb=torch.ones(1),
        window_index=torch.arange(4), cu_window_seqlens=torch.tensor([0, 4]),
        cu_seqlens=torch.tensor([0, 4]))
    modules = {name: types.ModuleType(name) for name in (
        'lingbotvla', 'lingbotvla.models', 'lingbotvla.models.vla',
        'lingbotvla.models.vla.pi0', 'lingbotvla.models.vla.pi0.qwenvl_in_vla')}
    patches = CaptureSafePrefixPatches(SimpleNamespace(qwenvl_with_expert=expert))
    original = attention.forward
    with mock.patch.dict(sys.modules, modules):
        patches.install(torch.ones(1, 4, 3))
    assert len(patches.patched_attention) == 1
    patches.uninstall()
    assert attention.forward == original

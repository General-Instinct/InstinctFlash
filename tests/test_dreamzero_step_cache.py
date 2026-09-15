"""Native-loop integration invariants, without constructing a model or GPU."""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import sys
import textwrap
from threading import Barrier
from types import MethodType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/dreamzero"))
from dreamzero_iwm import dynamic_cache as integration
from dreamzero_iwm import schedule
from instinctflash.runtime.step_cache_policy import ResolvedStepCache


class ToyHead:
    num_inference_steps = 16
    dynamic_cache_schedule = True
    ip_size = 1
    cfg_scale = 5.0

    def __init__(self):
        self.fail_at = None
        self.mask = []
        self.scheduler_calls = 0

    def _run_diffusion_steps(self, *, index=0, kv_cache_metadata):
        if index == self.fail_at:
            raise RuntimeError("injected native forward failure")
        video = torch.ones(1, 2, 3)
        action = torch.full((1, 4, 2), float(index + 1))
        return [(video, action), (video.clone(), action.clone())]

    def should_run_model(self, index, timestep, predictions):
        if len(predictions) < 2:
            return True
        if self.skip_countdown > 1:
            self.skip_countdown -= 1
            return False
        if self.skip_countdown == 1:
            self.skip_countdown = 0
            return True
        similarity = torch.nn.functional.cosine_similarity(
            predictions[-1][1].flatten(1).float(), predictions[-2][1].flatten(1).float(), dim=1).mean()
        for threshold, count in ((0.95, 4), (0.93, 2)):
            if similarity > threshold:
                self.skip_countdown = count
                return False
        return True

    def lazy_joint_video_action(self):
        self.mask = []
        self.scheduler_calls = 0
        self._run_diffusion_steps(kv_cache_metadata={"update_kv_cache": True})
        history = []
        self.skip_countdown = 0
        video_latent = torch.zeros(1, 2, 3)
        action_latent = torch.zeros(1, 4, 2)
        for index in range(16):
            compute = self.should_run_model(index, 1000 - index * 60, history)
            self.mask.append(compute)
            if compute:
                positive, negative = self._run_diffusion_steps(
                    index=index, kv_cache_metadata={"update_kv_cache": False})
                video = negative[0] + self.cfg_scale * (positive[0] - negative[0])
                action = positive[1]
                history.append((index, video, action))
                history[:] = history[-2:]
            else:
                _, video, action = history[-1]
            video_latent = video_latent + video / 16
            action_latent = action_latent + action / 16
            self.scheduler_calls += 2
        return video_latent, action_latent


@pytest.fixture
def fake_source_gate(monkeypatch):
    monkeypatch.setattr(integration, "_verify_native_methods", lambda _: {"fixture": "toy-only"})


def test_hook_preserves_native_dynamic_outputs_and_all_solver_kv_calls(fake_source_gate):
    head = ToyHead()
    with torch.inference_mode():
        reference = head.lazy_joint_video_action()
        reference_mask = head.mask.copy()
        hook = integration.install(head)
        for _ in range(3):
            actual = head.lazy_joint_video_action()
            assert all(torch.equal(a, b) for a, b in zip(actual, reference))
            assert head.mask == reference_mask
            assert head.scheduler_calls == 32
            stats = hook.report()["last_generation"]
            assert stats["computed_steps"] == stats["denoiser_calls"] == 4
            assert stats["denoiser_branch_forwards"] == 8
            assert stats["kv_update_calls"] == 1
            assert stats["kv_update_branch_forwards"] == 2
        hook.close()
        hook.close()
        assert "should_run_model" not in vars(head)
        assert "lazy_joint_video_action" not in vars(head)
        restored = head.lazy_joint_video_action()
        assert all(torch.equal(a, b) for a, b in zip(restored, reference))


def test_unstable_final_compute_is_recorded_before_generation_ends(fake_source_gate):
    class UnstableHead(ToyHead):
        def _run_diffusion_steps(self, *, index=0, kv_cache_metadata):
            video = torch.full((1, 2, 3), (-1.0) ** index)
            action = torch.full((1, 4, 2), float(index + 1))
            return [(video, action), (video.clone(), action.clone())]

    assert not torch.cuda.is_initialized()
    head = UnstableHead()
    with torch.no_grad():
        reference = head.lazy_joint_video_action()
        hook = integration.install(head)
        try:
            for _ in range(2):
                actual = head.lazy_joint_video_action()
                assert all(torch.equal(a, b) for a, b in zip(actual, reference))
                stats = hook.report()["last_generation"]
                assert stats["status"] == "completed"
                assert stats["compute_mask"] == [True] * 16
                assert stats["computed_steps"] == stats["denoiser_calls"] == 16
                assert stats["reused_steps"] == 0
                assert stats["denoiser_branch_forwards"] == 32
                assert stats["kv_update_calls"] == 1
                assert stats["kv_update_branch_forwards"] == 2
                assert stats["trace"][-1]["index"] == 15
                assert stats["trace"][-1]["compute"]
                assert stats["trace"][-1]["consumed"]
                assert head.scheduler_calls == 32
                controller = hook.controller.report()
                assert controller["history_entries"] == controller["cache_bytes"] == 0
                assert controller["pending_decision"] is None
        finally:
            hook.close()
    assert not torch.cuda.is_initialized()


def test_forward_failure_drops_history_and_next_generation_starts_fresh(fake_source_gate):
    head = ToyHead()
    hook = integration.install(head)
    head.fail_at = 6
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="injected"):
            head.lazy_joint_video_action()
        assert hook.report()["last_generation"]["status"] == "aborted"
        assert not hook.controller.report()["active"]
        head.fail_at = None
        head.lazy_joint_video_action()
        assert [i for i, compute in enumerate(head.mask) if compute] == [0, 1, 6, 11]
    hook.close()


def test_configuration_mutation_and_concurrent_reset_refused(fake_source_gate):
    head = ToyHead()
    hook = integration.install(head)
    head.dynamic_cache_schedule = False
    with torch.inference_mode(), pytest.raises(RuntimeError, match="configuration changed"):
        head.lazy_joint_video_action()
    hook._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="reset"):
            hook.reset()
        with pytest.raises(RuntimeError, match="concurrent"):
            head.lazy_joint_video_action()
    finally:
        hook._lock.release()
    hook.close()


def test_foreign_wrapper_conflict_releases_owned_resources(fake_source_gate):
    from dreamzero_iwm.adapter import _DreamZeroLoop
    from unittest.mock import Mock

    head = ToyHead()
    hook = integration.install(head)
    foreign = MethodType(lambda self, *args: True, head)
    head.should_run_model = foreign
    view = SimpleNamespace(cleanup=Mock())
    loop = _DreamZeroLoop(SimpleNamespace(head=head), dynamic_cache=True,
                         checkpoint_view=view, step_cache_hook=hook)
    with pytest.raises(RuntimeError, match="owned cache released"):
        loop.close()
    assert head.should_run_model is foreign
    assert "lazy_joint_video_action" not in vars(head)
    assert "_run_diffusion_steps" not in vars(head)
    assert hook.closed and hook.controller.report()["closed"]
    assert loop._wrapper is loop._step_cache_hook is loop._checkpoint_view is None
    view.cleanup.assert_called_once()
    loop.close()
    view.cleanup.assert_called_once()


def test_active_generation_close_refusal_retains_model_and_view(fake_source_gate):
    from dreamzero_iwm.adapter import _DreamZeroLoop
    from unittest.mock import Mock

    head = ToyHead()
    hook = integration.install(head)
    wrapper, view = SimpleNamespace(head=head), SimpleNamespace(cleanup=Mock())
    loop = _DreamZeroLoop(wrapper, dynamic_cache=True, checkpoint_view=view, step_cache_hook=hook)
    hook._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="during a generation"):
            loop.close()
        assert loop._wrapper is wrapper and loop._step_cache_hook is hook
        assert loop._checkpoint_view is view and not hook.closed
        view.cleanup.assert_not_called()
    finally:
        hook._lock.release()
    loop.close()
    assert hook.closed
    view.cleanup.assert_called_once()


def test_public_runtime_close_keeps_refused_backend_for_retry(fake_source_gate):
    from dreamzero_iwm.adapter import _DreamZeroLoop
    from instinctflash.runtime.execution import InProcessBackend
    from instinctflash.runtime.facade import Runtime
    from unittest.mock import Mock

    head = ToyHead()
    hook = integration.install(head)
    view = SimpleNamespace(cleanup=Mock())
    loop = _DreamZeroLoop(SimpleNamespace(head=head), dynamic_cache=True,
                         checkpoint_view=view, step_cache_hook=hook)
    backend = InProcessBackend(None, None, None)
    backend._impl = loop
    runtime = Runtime(None, None, None, backend)
    hook._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="during a generation"):
            runtime.close()
        assert backend._impl is loop and not hook.closed
        view.cleanup.assert_not_called()
    finally:
        hook._lock.release()
    with torch.inference_mode():
        head.lazy_joint_video_action()
    runtime.close()
    assert backend._impl is None and hook.closed
    view.cleanup.assert_called_once()


def test_real_source_gate_accepts_audited_methods_and_rejects_modified_loop():
    source = Path("/home/ubuntu/dreamzero-repo/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py")
    if not source.is_file():
        pytest.skip("Pinned DreamZero source checkout unavailable")
    tree = ast.parse(source.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
             and node.name in integration._METHOD_HASHES]
    namespace = {}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
                        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    head = SimpleNamespace(num_inference_steps=16, ip_size=1, dynamic_cache_schedule=True)
    for name in integration._METHOD_HASHES:
        setattr(head, name, MethodType(namespace[name], head))
    assert integration._verify_native_methods(head) == integration._METHOD_HASHES
    head.should_run_model = MethodType(ToyHead.should_run_model, head)
    with pytest.raises(ValueError, match="re-audit"):
        integration.install(head)


def test_source_gate_rejects_unscoped_fake_head():
    with pytest.raises(ValueError, match="re-audit"):
        integration.install(ToyHead())


class ScheduleBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.super_initialized = True
        self.base_env_value = os.getenv("IFL_SCHEDULE_TEST_OTHER")


class ScheduleToy(ScheduleBase):
    def __init__(self, config, *, marker="original-default"):
        super().__init__()
        self.config = config
        self.marker = marker
        self.dynamic = os.getenv("DYNAMIC_CACHE_SCHEDULE", "False").lower() == "true"
        if config.get("barrier") is not None:
            config["barrier"].wait(timeout=5)
        self.fixed_steps = int(os.getenv("NUM_DIT_STEPS", "8"))
        self.unrelated = os.getenv("IFL_SCHEDULE_TEST_OTHER")
        self.fallback = os.getenv("IFL_SCHEDULE_TEST_MISSING", "fallback")
        self.path_value = os.path.join("a", "b")
        self.environment_seen = dict(os.environ)


def test_owned_constructor_ignores_invalid_schedule_env_and_preserves_original(monkeypatch):
    monkeypatch.setenv("DYNAMIC_CACHE_SCHEDULE", "invalid-ambient-flag")
    monkeypatch.setenv("NUM_DIT_STEPS", "invalid-ambient-count")
    monkeypatch.setenv("IFL_SCHEDULE_TEST_OTHER", "preserved")
    monkeypatch.delenv("IFL_SCHEDULE_TEST_MISSING", raising=False)
    original = ScheduleToy.__init__
    code, namespace, closure = original.__code__, original.__globals__, original.__closure__
    globals_before = dict(namespace)
    environment_before = dict(os.environ)
    config = {"sentinel": object()}
    selected = ResolvedStepCache(True, 8, "dreamzero_velocity_v1", "test")

    result = schedule._construct(ScheduleToy, config, dynamic=selected.dynamic,
                                 fixed_steps=selected.fixed_steps)

    assert type(result) is ScheduleToy
    assert result.config is config
    assert result.super_initialized and result._parameters == {}
    assert result.marker == "original-default"
    assert result.dynamic is True and result.fixed_steps == 8
    assert result.unrelated == result.base_env_value == "preserved"
    assert result.fallback == "fallback"
    assert result.path_value == os.path.join("a", "b")
    assert result.environment_seen == environment_before == dict(os.environ)
    assert ScheduleToy.__init__ is original
    assert original.__code__ is code
    assert original.__globals__ is namespace
    assert original.__closure__ is closure
    assert namespace.keys() == globals_before.keys()
    assert all(namespace[key] is value for key, value in globals_before.items())
    assert namespace["os"] is os
    # The original constructor still sees the original invalid ambient value.
    with pytest.raises(ValueError, match="invalid-ambient-count"):
        ScheduleToy(config)


def test_owned_constructors_have_independent_concurrent_selections(monkeypatch):
    monkeypatch.setenv("DYNAMIC_CACHE_SCHEDULE", "invalid-ambient-flag")
    monkeypatch.setenv("NUM_DIT_STEPS", "invalid-ambient-count")
    monkeypatch.setenv("IFL_SCHEDULE_TEST_OTHER", "shared-unrelated")
    barrier = Barrier(2)
    selections = [ResolvedStepCache(True, 5, "dreamzero_velocity_v1", "first"),
                  ResolvedStepCache(False, 8, None, "second")]
    environment_before = dict(os.environ)
    original = ScheduleToy.__init__

    def reject_environment_mutation(*args, **kwargs):
        raise AssertionError("Construction must not mutate the process environment")

    def construct(selection):
        return schedule._construct(ScheduleToy, {"barrier": barrier},
                                   dynamic=selection.dynamic, fixed_steps=selection.fixed_steps)

    # A barrier pauses both constructors between their two getenv reads. Any
    # shared temporary environment selection would either trip these guards or
    # mix the two instances' values. Thread-local copies need no global lock.
    with monkeypatch.context() as patch:
        patch.setattr(os, "putenv", reject_environment_mutation)
        patch.setattr(os, "unsetenv", reject_environment_mutation)
        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = list(executor.map(construct, selections))
    assert (first.dynamic, first.fixed_steps) == (True, 5)
    assert (second.dynamic, second.fixed_steps) == (False, 8)
    assert type(first) is type(second) is ScheduleToy
    assert first.super_initialized and second.super_initialized
    assert first.unrelated == second.unrelated == "shared-unrelated"
    assert first.environment_seen == second.environment_seen == environment_before
    assert dict(os.environ) == environment_before
    assert ScheduleToy.__init__ is original
    assert original.__globals__["os"] is os


def test_owned_constructor_rejects_changed_allocation_protocol():
    class CustomAllocation(ScheduleToy):
        def __new__(cls):
            return object.__new__(cls)

    with pytest.raises(ValueError, match="construction changed"):
        schedule._construct(CustomAllocation, {}, dynamic=False, fixed_steps=8)


def test_owned_constructor_gate_matches_pinned_native_source():
    source = Path("/home/ubuntu/dreamzero-repo/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py")
    if not source.is_file():
        pytest.skip("Pinned DreamZero source checkout unavailable")
    data = source.read_text()
    native = next(node for node in ast.parse(data).body
                  if isinstance(node, ast.ClassDef) and node.name == "WANPolicyHead")
    initializer = next(node for node in native.body
                       if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    lines = data.splitlines(keepends=True)[initializer.lineno - 1:initializer.end_lineno]
    normalized = textwrap.dedent("".join(lines)).strip()
    assert hashlib.sha256(normalized.encode()).hexdigest() == schedule._INIT_HASH

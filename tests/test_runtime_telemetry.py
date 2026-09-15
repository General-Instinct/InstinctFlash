"""Public statistics are detached observations, never a trigger to load or use a GPU."""
from __future__ import annotations

import builtins
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instinctflash.runtime.engine_backend import EngineBackend
from instinctflash.runtime.execution import InProcessBackend, WorkerBackend
from instinctflash.runtime.facade import Runtime

raises = unittest.TestCase().assertRaisesRegex


def runtime_for(backend_type, implementation=None):
    backend = object.__new__(backend_type)
    if isinstance(backend, InProcessBackend):
        backend._impl = implementation
    elif isinstance(backend, EngineBackend):
        backend._loop = implementation
    backend._ensure = Mock(side_effect=AssertionError("statistics attempted to load"))
    runtime = object.__new__(Runtime)
    runtime._backend = backend
    return runtime


def test_native_property_returns_available_snapshot():
    class Loop:
        @property
        def backend_stats(self):
            return {"precision": "native", "computed_steps": 6}
    runtime = runtime_for(InProcessBackend, Loop())
    assert runtime.backend_stats == {
        "status": "available", "backend": "InProcessBackend",
        "stats": {"precision": "native", "computed_steps": 6}}
    runtime._backend._ensure.assert_not_called()


def test_engine_snapshot_preserves_h100_native_nesting():
    loop = SimpleNamespace(backend_stats={
        "native_backend": {"step_cache": {"last_generation": {"computed_steps": 6}}},
        "fp8_recipe": {"precision": "fp8"}})
    runtime = runtime_for(EngineBackend, loop)
    report = runtime.backend_stats
    assert report["backend"] == "EngineBackend"
    assert report["stats"]["native_backend"]["step_cache"]["last_generation"]["computed_steps"] == 6
    assert report["stats"]["fp8_recipe"] == {"precision": "fp8"}
    runtime._backend._ensure.assert_not_called()


def test_unloaded_and_closed_backends_do_not_import_torch_or_load():
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("statistics imported Torch")
        return original_import(name, *args, **kwargs)
    with patch("builtins.__import__", side_effect=guarded_import):
        for backend_type in (InProcessBackend, EngineBackend):
            runtime = runtime_for(backend_type)
            report = runtime.backend_stats
            assert report["status"] == "not_loaded" and report["stats"] is None
            runtime._backend._ensure.assert_not_called()


def test_worker_does_not_request_live_or_handshake_statistics():
    runtime = runtime_for(WorkerBackend)
    runtime._backend._client = Mock()
    runtime._backend._transport_client = SimpleNamespace(metadata={"computed_steps": 999})
    report = runtime.backend_stats
    assert report["status"] == "unsupported" and report["stats"] is None
    assert report["backend"] == "WorkerBackend" and "RPC" in report["reason"]
    runtime._backend._ensure.assert_not_called()
    assert runtime._backend._client.mock_calls == []


def test_legacy_callable_and_mapping_provider_are_supported():
    provider = Mock(return_value=MappingProxyType({"computed_steps": 3}))
    runtime = runtime_for(InProcessBackend, SimpleNamespace(backend_stats=provider))
    assert runtime.backend_stats["stats"] == {"computed_steps": 3}
    provider.assert_called_once_with()


def test_snapshot_cannot_mutate_loaded_statistics_and_refreshes_on_next_read():
    source = {"step_cache": {"trace": [{"compute": True}], "config": {"thresholds": [0.95, 0.93]}}}
    runtime = runtime_for(InProcessBackend, SimpleNamespace(backend_stats=source))
    snapshot = runtime.backend_stats
    snapshot["stats"]["step_cache"]["trace"][0]["compute"] = False
    snapshot["stats"]["step_cache"]["config"]["thresholds"][0] = 0.1
    assert source["step_cache"]["trace"][0]["compute"] is True
    assert source["step_cache"]["config"]["thresholds"][0] == 0.95
    source["step_cache"]["trace"].append({"compute": False})
    assert len(runtime.backend_stats["stats"]["step_cache"]["trace"]) == 2
    assert len(snapshot["stats"]["step_cache"]["trace"]) == 1


def test_absent_stats_and_unknown_backends_are_explicitly_unsupported():
    runtime = runtime_for(InProcessBackend, object())
    assert runtime.backend_stats["status"] == "unsupported"
    foreign = SimpleNamespace(_impl=SimpleNamespace(backend_stats={"should_not_read": True}))
    runtime._backend = foreign
    report = runtime.backend_stats
    assert report["status"] == "unsupported" and report["stats"] is None


def test_loaded_provider_exceptions_including_attribute_error_propagate():
    for error in (RuntimeError("broken statistics"), AttributeError("broken getter")):
        class BrokenProperty:
            @property
            def backend_stats(self):
                raise error
        runtime = runtime_for(InProcessBackend, BrokenProperty())
        with raises(type(error), str(error)):
            _ = runtime.backend_stats
        runtime = runtime_for(InProcessBackend, SimpleNamespace(backend_stats=Mock(side_effect=error)))
        with raises(type(error), str(error)):
            _ = runtime.backend_stats


def test_invalid_loaded_results_and_copy_errors_propagate():
    for invalid in (None, [], "not statistics"):
        runtime = runtime_for(InProcessBackend, SimpleNamespace(backend_stats=invalid))
        with raises(TypeError, "must return a mapping"):
            _ = runtime.backend_stats
    class BrokenCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("copy failed")
    runtime = runtime_for(InProcessBackend, SimpleNamespace(backend_stats={"value": BrokenCopy()}))
    with raises(RuntimeError, "copy failed"):
        _ = runtime.backend_stats


def test_public_stats_property_has_no_setter():
    runtime = runtime_for(InProcessBackend)
    with raises(AttributeError, "backend_stats"):
        runtime.backend_stats = {}


if __name__ == "__main__":
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))

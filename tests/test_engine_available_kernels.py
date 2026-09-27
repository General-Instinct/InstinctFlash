"""Engine availability names the missing piece; GI engines need only Triton and FP8 matmul."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from instinctflash.runtime.engine_backend import GI_ENGINE_BACKBONES, engine_available


def _torch(capability=(11, 0), scaled_mm=True):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                                                 get_device_capability=lambda: capability))
    if scaled_mm:
        torch._scaled_mm = lambda *a, **k: None
    return torch


@pytest.mark.parametrize('backbone', GI_ENGINE_BACKBONES)
def test_gi_engine_available_with_triton_and_scaled_mm(monkeypatch, backbone):
    monkeypatch.setitem(sys.modules, 'torch', _torch())
    monkeypatch.setitem(sys.modules, 'triton', ModuleType('triton'))
    ok, reason = engine_available(backbone)
    assert ok and 'Triton' in reason


def test_missing_triton_is_named(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', _torch())
    monkeypatch.setitem(sys.modules, 'triton', None)
    ok, reason = engine_available('pi05')
    assert not ok and 'requires Triton' in reason


def test_missing_scaled_mm_is_named(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', _torch(scaled_mm=False))
    monkeypatch.setitem(sys.modules, 'triton', ModuleType('triton'))
    ok, reason = engine_available('groot_n17')
    assert not ok and '_scaled_mm' in reason


def test_unregistered_backbone_is_not_available(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', _torch())
    ok, reason = engine_available('wam')
    assert not ok and 'no SM110 engine' in reason


def test_other_architectures_decline(monkeypatch):
    # SM89/SM120 have their own RTX executors (sm89_fp8/sm120_fp8); others decline.
    monkeypatch.setitem(sys.modules, 'torch', _torch(capability=(8, 6)))
    ok, reason = engine_available('pi05')
    assert not ok and 'SM86' in reason

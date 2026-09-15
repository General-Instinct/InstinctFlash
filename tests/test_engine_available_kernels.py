"""A Python package alone is not an installed Thor engine."""
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from instinctflash.runtime.engine_backend import engine_available


@pytest.fixture
def thor_python_package(monkeypatch):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,
                            get_device_capability=lambda:(11,0)))
    package = ModuleType('flash_rt')
    package.__path__ = []
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'flash_rt',package)


def test_python_package_without_binary_is_not_available(thor_python_package,monkeypatch):
    monkeypatch.setitem(sys.modules,'flash_rt.flash_rt_kernels',None)
    ok, reason = engine_available()
    assert not ok and 'compiled kernels cannot load' in reason


@pytest.mark.parametrize('error',[ImportError('undefined symbol'),OSError('missing CUDA library')])
def test_incompatible_binary_has_actionable_reason(thor_python_package,error):
    with patch('importlib.import_module',side_effect=error):
        ok, reason = engine_available()
    assert not ok and str(error) in reason and 'CUDA environment' in reason


def test_incomplete_binary_is_not_available(thor_python_package,monkeypatch):
    monkeypatch.setitem(sys.modules,'flash_rt.flash_rt_kernels',SimpleNamespace(GemmRunner=lambda:None))
    ok, reason = engine_available()
    assert not ok and 'FvkContext' in reason


def test_loadable_binary_is_available(thor_python_package,monkeypatch):
    monkeypatch.setitem(sys.modules,'flash_rt.flash_rt_kernels',
                        SimpleNamespace(GemmRunner=lambda:None,FvkContext=lambda:None))
    assert engine_available()[0]

"""Regression for the installed config lookup using the package's parent directory."""
import importlib.util
from pathlib import Path
import sys


def test_builtin_configuration_loads_outside_checkout(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "serving/flash_rt/core/config.py"
    spec = importlib.util.spec_from_file_location("ifl_config_resource_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    config = module.load_config("pi05")
    assert config.name == "pi05"
    assert config.decoder_steps > 0
    assert config.vision.hidden_dim > 0

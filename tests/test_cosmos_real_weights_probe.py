"""A partially loaded checkpoint must never produce a benchmark result."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "cosmos_real_weights_probe",
    Path(__file__).resolve().parents[1] / "eval/cosmos3_edge/probe_real_weights.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

@pytest.mark.parametrize("key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_incomplete_checkpoint_is_rejected(key):
    with pytest.raises(RuntimeError, match="refusing to benchmark"):
        probe.validate_loading_info({key: ["failed tensor"]})

def test_clean_checkpoint_is_accepted():
    probe.validate_loading_info({"missing_keys": [], "unexpected_keys": []})

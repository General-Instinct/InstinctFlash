"""Hub-id resolution honors HF_HUB_OFFLINE even when huggingface_hub would otherwise query the Hub."""
import sys
from types import ModuleType

import pytest

from instinctflash.descriptors import package


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("", False), ("0", False)])
def test_offline_environment_selects_local_files_only(monkeypatch, tmp_path, value, expected):
    seen = {}

    def snapshot_download(repo, revision=None, local_files_only=False):
        seen.update(repo=repo, revision=revision, local_files_only=local_files_only)
        raise RuntimeError("stop after resolution")

    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    with pytest.raises(RuntimeError, match="stop after resolution"):
        package.from_pretrained("org/model-not-local", revision="a" * 40)
    assert seen == {"repo": "org/model-not-local", "revision": "a" * 40, "local_files_only": expected}

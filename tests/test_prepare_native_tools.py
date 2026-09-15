from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("native_tools", ROOT / "scripts/prepare_native_tools.py")
tools = module_from_spec(spec)
spec.loader.exec_module(tools)


def test_catalog_closes_native_tool_without_torch():
    catalog = tools.load_catalog(tools.CATALOG)
    assert len(catalog["packages"]) == 23
    assert catalog["packages"]["hf"] == catalog["packages"]["huggingface-hub"] == "1.16.4"
    assert "torch" not in catalog["packages"]
    assert catalog["native_argv_prefix"] == ["uvx", "--with", "click", "hf@1.16.4"]


def test_environment_separates_tool_offline_from_explicit_model_download(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_INDEX", "https://invalid.example/simple")
    monkeypatch.setenv("UV_CONSTRAINT", "/old/constraints.txt")
    monkeypatch.setenv("PYTHONPATH", "/old/vendor")
    values = tools.runtime_environment(tmp_path, tmp_path / "venv/bin/python", tmp_path / "venv/bin/uvx")
    env = tools.clean_environment(values)
    assert env["UV_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in values
    assert env["UV_CONSTRAINT"] == str(tmp_path / "constraints.txt")
    assert env["UV_PYTHON"] == str(tmp_path / "venv/bin/python")
    assert env["PATH"].split(":")[0] == str(tmp_path / "venv/bin")
    assert "UV_INDEX" not in env and "PYTHONPATH" not in env
    assert "UV_OFFLINE" not in tools.clean_environment(values, online=True)
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_existing_root_rejected_before_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(tools.subprocess, "run", lambda *a, **k: pytest.fail("unexpected subprocess"))
    with pytest.raises(ValueError, match="root must be new"):
        tools.prepare(tmp_path, Path(__file__), Path(__file__), tools.CATALOG)


def test_changed_constraint_rejected(tmp_path):
    catalog = json.loads(tools.CATALOG.read_text())
    (tmp_path / "hf_tool.json").write_text(json.dumps(catalog))
    (tmp_path / "hf_tool_constraints.txt").write_text("hf==9.9.9\n")
    with pytest.raises(ValueError, match="constraint source"):
        tools.load_catalog(tmp_path / "hf_tool.json")


def test_prepare_rejects_wrong_uv_before_any_tool_install(tmp_path, monkeypatch):
    calls = []

    def execute(root, name, command, env, **kwargs):
        calls.append(name)
        return "uvx 0.0.1 (wrong)"

    monkeypatch.setattr(tools, "execute", execute)
    root = tmp_path / "new"
    with pytest.raises(ValueError, match="uvx version"):
        tools.prepare(root, Path(__file__), Path(__file__), tools.CATALOG)
    assert calls == ["uvx_version"]
    assert json.loads((root / "failure.json").read_text())["automatic_retry"] is False
    assert not (root / "completion.json").exists()


def setup_tool(root):
    root.mkdir()
    (root / "constraints.txt").write_text("hf==1.16.4\n")
    uvx = root / "uvx"
    uvx.write_text("source binding")
    completion = {"status": "tool_packages_checked_offline_reuse_passed", "uvx": str(uvx),
                  "python": "/owned/venv/bin/python", "constraint_sha256": tools.sha(root / "constraints.txt"),
                  "uvx_sha256": tools.sha(uvx)}
    (root / "completion.json").write_text(json.dumps(completion))
    return completion


def test_probe_missing_asset_fails_before_native_call(tmp_path, monkeypatch):
    root = tmp_path / "tool"
    setup_tool(root)
    monkeypatch.setattr(tools, "execute", lambda *a, **k: pytest.fail("must not attempt a download"))
    with pytest.raises(ValueError, match="prepared Wan asset"):
        tools.probe(root, tmp_path / "cache", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("different_result", [False, True])
def test_probe_native_argv_offline_and_returned_path(tmp_path, monkeypatch, different_result):
    root, cache = tmp_path / "tool", tmp_path / "cache"
    setup_tool(root)
    expected = cache / "models--Wan-AI--Wan2.2-TI2V-5B/snapshots" / tools.WAN_REVISION / tools.WAN_FILE
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"tiny exact test payload")
    monkeypatch.setattr(tools, "WAN_BYTES", expected.stat().st_size)
    other = tmp_path / "wrong"
    other.write_bytes(expected.read_bytes())
    calls = []

    def execute(root, name, command, env, **kwargs):
        calls.append(command)
        assert command == ["uvx", "--with", "click", "hf@1.16.4", "download", "--format=json",
                           tools.WAN_REPO, "--repo-type", "model", "--revision", tools.WAN_REVISION, tools.WAN_FILE]
        assert env["UV_OFFLINE"] == env["HF_HUB_OFFLINE"] == "1"
        assert env["HF_HUB_CACHE"] == str(cache)
        return json.dumps({"path": str(other if different_result else expected)})

    monkeypatch.setattr(tools, "execute", execute)
    if different_result:
        with pytest.raises(ValueError, match="different asset"):
            tools.probe(root, cache, tmp_path / "output")
        assert not (tmp_path / "output/completion.json").exists()
    else:
        result = tools.probe(root, cache, tmp_path / "output")
        assert result["status"] == "passed"
        assert result["weight_rehashed"] is result["weights_downloaded"] is result["model_constructed"] is False
    assert len(calls) == 1

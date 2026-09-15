"""CPU checks for installed-package discovery and metadata-only planning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from instinctflash.cli import main
from instinctflash.runtime import facade, loader


def _metadata(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    declaration = directory / "instinctflash.json"
    declaration.write_text(json.dumps({
        "instinctflash_schema": 1,
        "execution": {"model_id": "org/model", "backbone": "wan_va", "servable": True,
                      "guidance": {"video": "cfg", "action": "positive_only"},
                      "nfe": {"video": 2, "action": 4}},
    }))
    (directory / "config.json").write_text('{"tokens_per_frame": 880}')
    return declaration


class _GeometryAdapter:
    PLANNING_FILES = ("config.json",)

    def spec_for_checkpoint(self, checkpoint):
        from instinctflash.adapters.lingbot_va import LingBotVA
        config = json.loads((Path(checkpoint.path) / "config.json").read_text())
        assert config["tokens_per_frame"] == 880
        return LingBotVA().spec()


def test_local_preflight_passes_checkpoint_directory_and_stays_offline(tmp_path, monkeypatch):
    import huggingface_hub
    _metadata(tmp_path)
    monkeypatch.setattr(loader, "load", lambda _: _GeometryAdapter())

    def forbidden(*args, **kwargs):
        raise AssertionError("local preflight attempted network access or weight download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    checkpoint, _, plan, device = facade.plan_declaration(tmp_path, probe_device=False)
    assert checkpoint.path == str(tmp_path)
    assert plan.results and device is None


def test_remote_metadata_uses_declaration_commit_and_no_weight_snapshot(tmp_path, monkeypatch):
    import huggingface_hub
    revision = "a" * 40
    root = tmp_path / "models--org--model" / "snapshots" / revision
    _metadata(root)
    calls = []

    def one_file(repo, name, revision=None):
        calls.append((repo, name, revision))
        return str(root / name)

    monkeypatch.setattr(loader, "load", lambda _: _GeometryAdapter())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", one_file)
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **kw: pytest.fail("weight snapshot requested"))
    checkpoint, _, _, _ = facade.plan_declaration("org/model", revision="release", probe_device=False)
    assert calls == [("org/model", "instinctflash.json", "release"),
                     ("org/model", "config.json", revision)]
    assert checkpoint.path == str(root)


def test_known_declaration_pins_remaining_metadata_after_first_download(tmp_path, monkeypatch):
    import huggingface_hub
    revision = "b" * 40
    root = tmp_path / "snapshots" / revision
    declaration = _metadata(root)
    (root / "metadata").mkdir()
    (root / "metadata" / "extra.json").write_text("{}")
    decl, doc, _ = facade.load_declaration_ref(root)
    monkeypatch.setattr(facade, "load_declaration_ref", lambda *a, **kw: (decl, doc, "known:org/model"))
    adapter = _GeometryAdapter()
    adapter.PLANNING_FILES = ("config.json", "metadata/extra.json")
    monkeypatch.setattr(loader, "load", lambda _: adapter)
    calls = []

    def one_file(repo, name, revision=None):
        calls.append((name, revision))
        return str(root / name)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", one_file)
    checkpoint, _, _, _ = facade.plan_declaration("org/model", revision="tag", probe_device=False)
    assert calls == [("config.json", "tag"), ("metadata/extra.json", revision)]
    assert checkpoint.path == str(declaration.parent)


@pytest.mark.parametrize("name", ["../config.json", "/config.json", "a/../config.json",
                                 "a\\config.json", "./config.json", "weights.safetensors",
                                 "weights.bin", "a//config.json", "C:/config.json"])
def test_remote_preflight_rejects_unsafe_or_weight_metadata_requests(tmp_path, monkeypatch, name):
    import huggingface_hub
    from types import SimpleNamespace
    adapter = SimpleNamespace(PLANNING_FILES=(name,))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **kw: pytest.fail("invalid filename reached Hub client"))
    with pytest.raises(ValueError, match="unsafe or non-metadata"):
        facade._prepare_planning_metadata(adapter, SimpleNamespace(path=str(tmp_path)), "org/model", None)


def test_metadata_commit_mismatch_is_rejected(tmp_path, monkeypatch):
    import huggingface_hub
    from types import SimpleNamespace
    first = tmp_path / "snapshots" / ("a" * 40)
    second = tmp_path / "snapshots" / ("b" * 40) / "config.json"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(second))
    with pytest.raises(RuntimeError, match="requested commit"):
        facade._prepare_planning_metadata(_GeometryAdapter(), SimpleNamespace(path=str(first)), "org/model", None)


def test_models_reports_registration_separately_from_host_support(monkeypatch, capsys):
    monkeypatch.setattr(loader, "available_models", lambda: ["wan_va"])
    monkeypatch.setattr(loader, "discover_plugins", lambda: ["broken_adapter: missing dependency"])
    monkeypatch.setattr(loader, "load", lambda _: pytest.fail("catalog instantiated a model adapter"))
    assert main(["models", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["host_dependencies_checked"] is False
    assert result["registered_adapters"] == ["wan_va"]
    assert result["adapter_discovery_errors"] == ["broken_adapter: missing dependency"]
    rows = {row["model_id"]: row for row in result["known_checkpoints"]}
    assert rows["GEAR-Dreams/DreamZero-DROID"]["adapter_registered"] is False
    assert rows["GEAR-Dreams/DreamZero-DROID"]["adapter_source_directory"] == "examples/dreamzero"
    assert rows["robbyant/lingbot-va-posttrain-robotwin"]["adapter_registered"] is True


def test_help_lists_public_precision_schedule_and_permission_choices(capsys):
    assert main(["serve", "--help"]) == 0
    out = capsys.readouterr().out
    assert "--runtime.precision=native|fp8" in out
    assert "--runtime.step_cache=dynamic|checkpoint|null" in out
    assert "--runtime.tier_ceiling=bitexact|numeric|behavioral|null" in out
    assert "--serve.dry_run=true|false" in out

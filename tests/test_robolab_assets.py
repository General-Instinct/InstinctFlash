"""CPU gates for complete multi-root assets and native URL byte bindings."""

from __future__ import annotations

import copy
import hashlib

import pytest

from benchmarks.vla.robolab_assets import (
    freeze_roots, hydrate_urls, mdl_texture_references, verified_asset_file,
    verify_asset_inventory, verify_remote_assets,
)
from benchmarks.vla.util import ConfigurationError


URL = "https://assets.example.test/Materials/Wood.mdl"


def setup_manifest(tmp_path):
    roots = {name: tmp_path / name for name in ("scenes", "mdl", "remote")}
    for root in roots.values():
        root.mkdir()
    (roots["scenes"] / "scene.usd").write_bytes(b"native scene")
    (roots["mdl"] / "OmniPBR.mdl").write_bytes(b"native MDL library")
    (roots["remote"] / "material.mdl").write_bytes(b"unchanged remote material")
    bindings = {URL: {"root": "remote", "relative": "material.mdl",
                      "sha256": hashlib.sha256(b"unchanged remote material").hexdigest()}}
    return freeze_roots(roots, url_bindings=bindings), roots


def test_multiple_complete_roots_cover_native_library_and_remote_copy(tmp_path):
    manifest, roots = setup_manifest(tmp_path)
    inventory = verify_asset_inventory(manifest)
    assert len(inventory["files"]) == 3
    assert verified_asset_file(str(roots["mdl"] / "OmniPBR.mdl"), inventory)["bytes"] == 18
    assert verified_asset_file(URL, inventory)["path"] == str(roots["remote"] / "material.mdl")
    assert manifest["roots"]["mdl"]["files"]["OmniPBR.mdl"]


@pytest.mark.parametrize("name", ["scenes", "mdl", "remote"])
def test_each_declared_root_must_be_complete_and_unchanged(tmp_path, name):
    manifest, roots = setup_manifest(tmp_path)
    (roots[name] / "extra.png").write_bytes(b"unbound texture")
    with pytest.raises(ConfigurationError, match="complete actual asset root"):
        verify_asset_inventory(manifest)
    (roots[name] / "extra.png").unlink()
    next(roots[name].iterdir()).write_bytes(b"changed")
    with pytest.raises(ConfigurationError, match="hash differs"):
        verify_asset_inventory(manifest)


def test_overlapping_roots_refused(tmp_path):
    manifest, _ = setup_manifest(tmp_path)
    manifest["roots"]["duplicate"] = copy.deepcopy(manifest["roots"]["scenes"])
    with pytest.raises(ConfigurationError, match="disjoint"):
        verify_asset_inventory(manifest)


def test_hidden_symlink_directory_refused(tmp_path):
    manifest, roots = setup_manifest(tmp_path)
    (roots["scenes"] / "hidden").symlink_to(roots["remote"], target_is_directory=True)
    with pytest.raises(ConfigurationError, match="symlinked directory"):
        verify_asset_inventory(manifest)


@pytest.mark.parametrize("change", ["root", "path", "hash", "unbound"])
def test_url_bindings_cannot_escape_or_invent_hashes(tmp_path, change):
    manifest, _ = setup_manifest(tmp_path)
    binding = manifest["url_bindings"][URL]
    if change == "root":
        binding["root"] = "missing"
    elif change == "path":
        binding["relative"] = "../scenes/scene.usd"
    elif change == "hash":
        binding["sha256"] = "0" * 64
    else:
        binding["relative"] = "unknown.mdl"
    with pytest.raises((ConfigurationError, FileNotFoundError)):
        verify_asset_inventory(manifest)


def test_remote_verification_reads_original_url_and_never_rewrites_asset(tmp_path):
    manifest, roots = setup_manifest(tmp_path)
    inventory = verify_asset_inventory(manifest)
    calls = []

    def reader(url):
        calls.append(url)
        return b"unchanged remote material"

    receipt = verify_remote_assets(inventory, reader=reader)
    assert calls == [URL]
    assert receipt[URL]["sha256"] == manifest["url_bindings"][URL]["sha256"]
    assert (roots["scenes"] / "scene.usd").read_bytes() == b"native scene"
    with pytest.raises(ConfigurationError, match="remote asset differs"):
        verify_remote_assets(inventory, reader=lambda _: b"changed remote material")
    with pytest.raises(ConfigurationError, match="unbound remote"):
        verified_asset_file(URL + ".other", inventory)


def test_remote_dependency_copy_rechecked_during_native_read(tmp_path):
    manifest, roots = setup_manifest(tmp_path)
    inventory = verify_asset_inventory(manifest)

    def reader(_):
        (roots["remote"] / "material.mdl").write_bytes(b"tampered during request")
        return b"unchanged remote material"

    with pytest.raises(ConfigurationError, match="copy changed"):
        verify_remote_assets(inventory, reader=reader)


def test_mdl_texture_extraction_preserves_strings_and_ignores_comments():
    source = '''
    diffuse: texture_2d("./Oak/color.png", gamma),
    // texture_2d("fake.png")
    normal: texture_2d(), // "also_fake.png"
    /* texture_2d("fake2.png") */
    distant: texture_2d("https://assets.example.test/actual.png"),
    '''
    assert mdl_texture_references(source) == ["./Oak/color.png", "https://assets.example.test/actual.png"]


def test_hydration_binds_native_bytes_and_relative_mdl_textures_additively(tmp_path):
    root = tmp_path / "copies"
    texture = "https://assets.example.test/Materials/textures/wood.png"
    source = b'mdl 1.7; texture_2d("./textures/wood.png"); // texture_2d("absent.png")'
    data = {URL: source, texture: b"raw pixels"}
    calls = []

    def reader(url):
        calls.append(url)
        return data[url]

    bindings = hydrate_urls([URL], root, reader=reader)
    assert calls == [URL, texture]
    manifest = freeze_roots({"remote_assets": root}, url_bindings=bindings)
    inventory = verify_asset_inventory(manifest)
    assert len(verify_remote_assets(inventory, reader=reader)) == 2
    with pytest.raises(FileExistsError):
        hydrate_urls([URL], root, reader=reader)


@pytest.mark.parametrize("limit", ["files", "bytes", "host"])
def test_hydration_stops_at_prospective_bounds_and_preserves_partial_evidence(tmp_path, limit):
    root = tmp_path / "copies"
    if limit == "host":
        content = b'texture_2d("https://unknown.example.test/other.png")'
        kwargs = {}
    else:
        content = b'texture_2d("./child.png")'
        kwargs = {"max_files": 1} if limit == "files" else {"max_bytes": 1}
    with pytest.raises(ConfigurationError, match="scope|byte bound"):
        hydrate_urls([URL], root, reader=lambda _: content, **kwargs)
    assert root.is_dir()


def test_real_usd_layer_and_native_url_binding_without_renderer(tmp_path, monkeypatch):
    """Optional installed USD parser test; it never starts Isaac or a GPU."""
    import sys
    from types import ModuleType

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd

    from benchmarks.vla import robolab_assets
    from benchmarks.vla.robolab_driver import _scene_configuration

    manifest, roots = setup_manifest(tmp_path)
    usd = roots["scenes"] / "scene.usd"
    usd.unlink()
    stage = Usd.Stage.CreateNew(str(usd))
    stage.DefinePrim("/Prototype", "Xform")
    material = stage.DefinePrim("/Prototype/Material", "Scope")
    material.CreateAttribute("test:material", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(URL))
    instance = stage.DefinePrim("/Instance", "Xform")
    instance.GetReferences().AddInternalReference("/Prototype")
    instance.SetInstanceable(True)
    stage.GetRootLayer().Save()
    manifest = freeze_roots(roots, url_bindings=manifest["url_bindings"])
    inventory = verify_asset_inventory(manifest)
    fake_omni = ModuleType("omni")
    fake_usd = ModuleType("omni.usd")
    fake_usd.get_context = lambda: type("Context", (), {"get_stage": lambda _: stage})()
    fake_omni.usd = fake_usd
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.usd", fake_usd)
    monkeypatch.setattr(robolab_assets, "read_native_url", lambda _: b"unchanged remote material")
    config = _scene_configuration({"native": "unchanged"}, tmp_path / "episode", inventory)
    assert config["loaded_usd_layers"][0]["path"] == str(usd)
    assert URL in config["remote_asset_receipts"]
    assert len(config["resolved_asset_attributes"]) == 2
    monkeypatch.setattr(robolab_assets, "read_native_url", lambda _: b"different bytes")
    with pytest.raises(ConfigurationError, match="remote asset differs"):
        _scene_configuration({}, tmp_path / "episode", inventory)

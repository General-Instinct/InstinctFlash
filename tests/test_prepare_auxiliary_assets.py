import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_auxiliary_assets", ROOT / "scripts/prepare_auxiliary_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def fixture(tmp_path):
    data = assets.load_catalog(assets.CATALOG)
    repo = data["models"]["vla4"]["repositories"][0]
    info = data["repositories"][repo]
    payload = b"original processor bytes"
    info["files"] = {"tokenizer.json": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps(data))
    cache = tmp_path / "original_cache"
    source = assets.snapshot(cache, repo, info["revision"]) / "tokenizer.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    return data, cat, cache, source


@pytest.mark.parametrize("family", assets.FAMILIES)
def test_every_family_exact_plan_cpu_only(family):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_auxiliary_assets.py"), "plan", family],
                       check=True, text=True, capture_output=True)
    assert json.loads(r.stdout)["GPU_verified"] is False


def test_dreamzero_retains_native_external_initialization_weights():
    catalog = assets.load_catalog(assets.CATALOG)
    plan = assets.plan("dreamzero", catalog)
    assert set(plan["repositories"]) == {
        "google/umt5-xxl", "Wan-AI/Wan2.1-I2V-14B-480P"}
    wan = plan["repositories"]["Wan-AI/Wan2.1-I2V-14B-480P"]
    assert wan["revision"] == "6b73f84e66371cdfe870c72acd6826e1d61cf279"
    assert wan["files"] == {
        "models_t5_umt5-xxl-enc-bf16.pth": {"bytes": 11361920418,
            "sha256": "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"},
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": {"bytes": 4772359047,
            "sha256": "628c9998b613391f193eb67ff68da9667d75f492911e4eb3decf23460a158c38"},
        "Wan2.1_VAE.pth": {"bytes": 507609880,
            "sha256": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"}}
    assert not plan["GPU_verified"]


def test_groot_cold_constructor_assets_match_framework_reference():
    catalog = assets.load_catalog(assets.CATALOG)
    plan = assets.plan("groot", catalog)
    repo = "nvidia/Cosmos-Reason2-2B"
    assert set(plan["repositories"]) == {repo}
    backbone = plan["repositories"][repo]
    assert backbone["revision"] == "9ce19a195e423419c349abfc86fd07178b230561"
    assert backbone["files"]["model.safetensors"] == {
        "bytes": 4877470304,
        "sha256": "fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede"}
    assert set(backbone["files"]) == {
        "chat_template.json", "config.json", "generation_config.json", "merges.txt",
        "model.safetensors", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json"}
    assert plan["file_count"] == 10
    assert plan["payload_bytes"] == 4888970298
    frameworks = json.loads((ROOT / "benchmarks/regression/fixtures/frameworks/catalog.json").read_text())
    reference = next(cell for cell in frameworks["cells"] if cell["family"] == "groot")
    assert reference["auxiliary_repositories"][repo] == backbone
    assert reference["assets"]["checkpoint"] == plan["requirements"]["primary_checkpoint"]


def groot_cold_fixture(tmp_path):
    data = assets.load_catalog(assets.CATALOG)
    repo = "nvidia/Cosmos-Reason2-2B"
    info = data["repositories"][repo]
    cache = tmp_path / "original_cache"
    source = assets.snapshot(cache, repo, info["revision"])
    source.mkdir(parents=True)
    for filename in info["files"]:
        payload = ("cold-cache fixture: " + filename).encode()
        (source / filename).write_bytes(payload)
        info["files"][filename] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps(data))
    return data, cat, cache, source


def test_groot_prepares_complete_original_backbone_in_empty_cache(tmp_path, monkeypatch):
    data, cat, cache, source = groot_cold_fixture(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    receipt = assets.prepare("groot", data, tmp_path / "new", cache_dir=cache,
                             local_files_only=True, method="hardlink", catalog_path=cat)
    hub = Path(receipt["environment"]["HF_HUB_CACHE"])
    staged = assets.snapshot(hub, "nvidia/Cosmos-Reason2-2B", source.name)
    assert {p.name for p in staged.iterdir()} == {p.name for p in source.iterdir()}
    assert len(receipt["files"]) == 10
    for row in receipt["files"]:
        before = source / row["filename"]
        after = staged / row["filename"]
        assert after.stat().st_ino == before.stat().st_ino
        assert hashlib.sha256(after.read_bytes()).hexdigest() == row["sha256"]
    assert not (source.parent.parent / "refs").exists()
    assert (staged.parent.parent / "refs/main").read_text() == source.name
    assert receipt["model_constructed"] is receipt["GPU_verified"] is False
    assert (tmp_path / "new/run.env").is_file()


def test_groot_metadata_only_cache_cannot_be_admitted_for_cold_loading(tmp_path, monkeypatch):
    data, cat, cache, source = groot_cold_fixture(tmp_path)
    (source / "model.safetensors").unlink()
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(FileNotFoundError, match="exact cached asset.*model.safetensors"):
        assets.prepare("groot", data, tmp_path / "new", cache_dir=cache,
                       local_files_only=True, method="hardlink", catalog_path=cat)
    failure = json.loads((tmp_path / "new/failure.json").read_text())
    assert failure["automatic_retry"] is False
    assert not (tmp_path / "new/run.env").exists()
    assert len(list(source.iterdir())) == 9


def native_dreamzero_config(tmp_path):
    prefix = "groot.vla.model.dreamzero."
    config = {"action_head_cfg": {
        "_target_": prefix + "action_head.wan_flow_matching_action_tf.WANPolicyHead",
        "config": {
            "text_encoder_cfg": {"_target_": prefix + "modules.wan_video_text_encoder.WanTextEncoder"},
            "image_encoder_cfg": {"_target_": prefix + "modules.wan_video_image_encoder.WanImageEncoder"},
            "vae_cfg": {"_target_": prefix + "modules.wan_video_vae.WanVideoVAE"},
            "skip_component_loading": True}}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, config


def test_actual_native_zdim16_config_selects_wan21_and_rejects_prior_incomplete_catalog(tmp_path):
    path, _ = native_dreamzero_config(tmp_path)
    catalog = assets.load_catalog(assets.CATALOG)
    checked = assets.dreamzero_native_asset_contract(path, catalog)
    assert checked["vae_z_dim"] == 16
    assert [row["filename"] for row in checked["required_native_files"]] == [
        "models_t5_umt5-xxl-enc-bf16.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", "Wan2.1_VAE.pth"]
    # Reproduce the real v3 omission: a complete Wan2.2 file did not satisfy
    # the active native Wan2.1 branch even with skip_component_loading=True.
    del catalog["repositories"]["Wan-AI/Wan2.1-I2V-14B-480P"]["files"]["Wan2.1_VAE.pth"]
    catalog["models"]["dreamzero"]["repositories"].append("Wan-AI/Wan2.2-TI2V-5B")
    with pytest.raises(ValueError, match="omits an actual native"):
        assets.dreamzero_native_asset_contract(path, catalog)


@pytest.mark.parametrize("mutation", ["unknown_vae", "latent48", "unknown_text"])
def test_uncovered_native_component_variants_fail_before_weights(tmp_path, mutation):
    path, config = native_dreamzero_config(tmp_path)
    head = config["action_head_cfg"]["config"]
    if mutation == "unknown_vae":
        head["vae_cfg"]["_target_"] = "some.other.VAE"
    elif mutation == "latent48":
        head["vae_cfg"]["z_dim"] = 48
    else:
        head["text_encoder_cfg"]["_target_"] = "some.other.TextEncoder"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unqualified|omits an actual native"):
        assets.dreamzero_native_asset_contract(path, assets.load_catalog(assets.CATALOG))


def test_config_bound_plan_is_a_cpu_only_command(tmp_path):
    path, _ = native_dreamzero_config(tmp_path)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_auxiliary_assets.py"),
                             "plan", "dreamzero", "--primary-config", str(path)],
                            check=True, text=True, capture_output=True)
    assert json.loads(result.stdout)["native_asset_contract"]["vae_z_dim"] == 16


@pytest.mark.parametrize("method", ["copy", "hardlink", "auto"])
def test_exact_isolated_cache_and_override_without_original_ref_changes(tmp_path, method):
    data, cat, cache, source = fixture(tmp_path)
    original = source.read_bytes()
    receipt = assets.prepare("vla4", data, tmp_path / "new", cache_dir=cache,
                             local_files_only=True, method=method, catalog_path=cat)
    assert source.read_bytes() == original
    assert not (source.parent.parent.parent / "refs").exists()
    proc = Path(receipt["environment"]["QWEN25_PATH"])
    assert (proc / "tokenizer.json").read_bytes() == original
    assert (proc.parent.parent / "refs/main").read_text() == proc.name
    assert receipt["native_processor_loaded"] is False
    assert receipt["files"][0]["materialization"] == ("hardlink" if method == "auto" else method)
    assert "unset TRANSFORMERS_CACHE" in (tmp_path / "new/run.env").read_text()
    assert "HF_HUB_OFFLINE" not in receipt["environment"]
    assert "TRANSFORMERS_OFFLINE" not in receipt["environment"]
    assert "HF_HOME" not in receipt["environment"]


def test_hash_mismatch_retains_failure_and_no_ready_env(tmp_path):
    data, cat, cache, source = fixture(tmp_path)
    source.write_bytes(b"wrong processor")
    with pytest.raises(ValueError, match="content mismatch"):
        assets.prepare("vla4", data, tmp_path / "new", cache_dir=cache,
                       local_files_only=True, method="copy", catalog_path=cat)
    assert json.loads((tmp_path / "new/failure.json").read_text())["automatic_retry"] is False
    assert not (tmp_path / "new/run.env").exists()


def test_missing_revision_never_falls_back_to_main_or_network(tmp_path, monkeypatch):
    data, cat, cache, source = fixture(tmp_path)
    source.unlink()
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(FileNotFoundError, match="exact cached asset"):
        assets.prepare("vla4", data, tmp_path / "new", cache_dir=cache,
                       local_files_only=True, method="copy", catalog_path=cat)


def test_existing_root_not_modified(tmp_path):
    data, cat, cache, source = fixture(tmp_path)
    with pytest.raises(ValueError, match="must be new"):
        assets.prepare("vla4", data, cache, cache_dir=cache,
                       local_files_only=True, method="copy", catalog_path=cat)
    assert source.read_bytes() == b"original processor bytes"
    assert not (cache / "plan.json").exists()


def test_explicit_primary_import_binds_commit_symlink_target_and_bytes(tmp_path):
    data, cat, cache, _ = fixture(tmp_path)
    primary = data["models"]["vla4"]["primary_checkpoint"]
    source = assets.snapshot(cache, primary["model_id"], primary["revision"])
    source.mkdir(parents=True)
    blob = tmp_path / "original_blob"
    blob.write_bytes(b"test checkpoint payload")
    (source / "model.safetensors").symlink_to(blob)
    (source / "empty_marker").write_bytes(b"")
    receipt = assets.prepare("vla4", data, tmp_path / "new", cache_dir=cache,
                             local_files_only=True, method="hardlink", catalog_path=cat, include_primary=True)
    bound = receipt["primary_checkpoint"]
    assert bound["revision"] == primary["revision"]
    row = next(x for x in bound["files"] if x["filename"] == "model.safetensors")
    assert row["physical_source"] == str(blob)
    assert row["sha256"] == hashlib.sha256(blob.read_bytes()).hexdigest()
    assert Path(row["destination"]).read_bytes() == blob.read_bytes()
    assert not (source.parent.parent / "refs").exists()


def test_primary_is_not_required_or_imported_without_explicit_flag(tmp_path):
    data, cat, cache, _ = fixture(tmp_path)
    receipt = assets.prepare("vla4", data, tmp_path / "new", cache_dir=cache,
                             local_files_only=True, method="copy", catalog_path=cat)
    assert "primary_checkpoint" not in receipt


@pytest.mark.parametrize("mutation", ["revision", "traversal", "env", "repository"])
def test_catalog_rejects_undeclared_paths_and_unpinned_inputs(tmp_path, mutation):
    data = copy.deepcopy(assets.load_catalog(assets.CATALOG))
    repo = data["models"]["vla4"]["repositories"][0]
    if mutation == "revision":
        data["repositories"][repo]["revision"] = "main"
    elif mutation == "traversal":
        data["repositories"][repo]["files"]["../../escape"] = next(iter(data["repositories"][repo]["files"].values()))
    elif mutation == "env":
        data["models"]["vla4"]["overrides"]["PYTHONPATH"] = repo
    else:
        data["repositories"]["https://user:secret@example.com"] = data["repositories"].pop(repo)
    cat = tmp_path / "bad.json"
    cat.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        assets.load_catalog(cat)

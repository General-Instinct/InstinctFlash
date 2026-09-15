"""Prospective native-renderer pairing contracts; CPU-only, no render claims."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.vla.robolab_driver import (
    CAMERAS,
    NATIVE_IMAGE_GROUPS,
    NATIVE_RENDERER_PAIRING_MODE,
    RENDER_PRODUCT_IDENTITY_ALIASES,
    bind_initial_state,
    canonicalize_render_product_scene,
    capture_initial_state,
    freeze_value,
    native_renderer_observation_evidence,
)
from benchmarks.vla.util import ConfigurationError, sha256_json


def observation():
    return {
        **{group: {name: np.zeros((1, 2, 3, 3), np.uint8) for name in names}
           for group, names in NATIVE_IMAGE_GROUPS.items()},
        "proprio_obs": {"arm_joint_pos": np.arange(7, dtype=np.float32)[None]},
        "other_native_buffer": np.array([-0.0], dtype=np.float32),
    }


def pairing_args(tmp_path):
    return {
        "anchor_path": tmp_path / "anchor.json", "protocol_sha256": "a" * 64,
        "episode": {"pair_id": "smoke/BananaInBowlTask/0000", "task_id": "BananaInBowlTask",
                    "scene_seed": 268439552},
        "state": freeze_value({"robot": np.arange(7, dtype=np.float32)}),
        "observation": observation(), "simulator_fingerprint": {"source": "fixed"},
        "scene_config": {"seed": 268439552}, "asset_inventory": {"files": ["banana.usd"]},
    }


def test_legacy_exact_observation_default_is_unchanged(tmp_path):
    args = pairing_args(tmp_path)
    baseline = bind_initial_state(family="edge", arm="baseline", output=tmp_path / "eb", **args)
    assert "pairing_mode" not in baseline
    assert "initial_observation_sha256" in baseline
    assert not (tmp_path / "eb" / "initial_observation.json").exists()
    args["observation"]["image_obs"]["wrist_cam"][0, 0, 0, 0] = 1
    with pytest.raises(ConfigurationError, match="paired initial"):
        bind_initial_state(family="edge", arm="candidate", output=tmp_path / "ec", **args)


def test_native_images_remain_independent_and_original_bytes_are_preserved(tmp_path):
    args = pairing_args(tmp_path)
    args["pairing_mode"] = NATIVE_RENDERER_PAIRING_MODE
    baseline = bind_initial_state(family="edge", arm="baseline", output=tmp_path / "eb", **args)
    anchor = json.loads(args["anchor_path"].read_text())
    assert "initial_observation_sha256" not in anchor
    assert anchor["pairing_mode"] == NATIVE_RENDERER_PAIRING_MODE
    for index, (family, arm) in enumerate((("edge", "candidate"), ("nano", "baseline"), ("nano", "candidate")), 1):
        args["observation"]["image_obs"]["wrist_cam"][0, 0, 0, 0] = index
        output = tmp_path / (family + arm)
        candidate = bind_initial_state(family=family, arm=arm, output=output, **args)
        full = json.loads((output / "initial_observation.json").read_text())
        assert full == freeze_value(args["observation"])
        assert candidate["initial_observation_sha256"] == sha256_json(full)
        assert candidate["initial_observation_sha256"] != baseline["initial_observation_sha256"]
        assert {k: v for k, v in candidate.items() if k != "initial_observation_sha256"} == anchor
        assert json.loads((output / "initial_state.json").read_text())["binding"] == anchor
    with pytest.raises(FileExistsError):
        bind_initial_state(family="edge", arm="baseline", output=tmp_path / "overwrite", **args)


@pytest.mark.parametrize("difference", ["nonvisual", "unrecognized_group", "shape", "state", "scene_config", "asset_inventory", "simulator_fingerprint"])
def test_mode3_rejects_every_nondeclared_pair_difference_and_preserves_evidence(tmp_path, difference):
    args = pairing_args(tmp_path)
    args["pairing_mode"] = NATIVE_RENDERER_PAIRING_MODE
    bind_initial_state(family="edge", arm="baseline", output=tmp_path / "eb", **args)
    if difference == "nonvisual":
        args["observation"]["other_native_buffer"][0] = 0.0
    elif difference == "unrecognized_group":
        args["observation"]["new_visual_group"] = {"image": np.ones((1, 2, 3, 3), np.uint8)}
    elif difference == "shape":
        args["observation"]["image_obs"]["wrist_cam"] = np.zeros((1, 3, 3, 3), np.uint8)
    else:
        args[difference] = {"changed": True}
    with pytest.raises(ConfigurationError, match="paired initial"):
        bind_initial_state(family="edge", arm="candidate", output=tmp_path / "ec", **args)
    assert (tmp_path / "ec" / "initial_observation.json").is_file()
    assert (tmp_path / "ec" / "initial_state.json").is_file()


@pytest.mark.parametrize("difference", ["dtype", "batch", "channels", "camera", "group", "base64", "length", "unknown_leaf"])
def test_invalid_or_incomplete_camera_evidence_fails_closed(difference):
    full = freeze_value(observation())
    camera = full["image_obs"]["wrist_cam"]
    if difference == "dtype":
        camera["dtype"] = "<f4"
    elif difference == "batch":
        camera["shape"][0] = 2
    elif difference == "channels":
        camera["shape"][-1] = 4
    elif difference == "camera":
        full["image_obs"]["other_camera"] = full["image_obs"].pop("head_camera")
    elif difference == "group":
        del full["viewport_cam"]
    elif difference == "base64":
        camera["data_base64"] = "??"
    elif difference == "length":
        camera["data_base64"] = "AA=="
    else:
        camera["metadata"] = "not declared"
    with pytest.raises(ConfigurationError):
        native_renderer_observation_evidence(full, frozen=True)


def test_all_observed_camera_poses_and_intrinsics_are_paired():
    names = [name for group in NATIVE_IMAGE_GROUPS.values() for name in group]
    scene = SimpleNamespace(
        get_state=lambda is_relative: {"articulation": {"robot": {"joints": np.zeros((1, 7), np.float32)}}},
        env_origins=np.zeros((1, 3), np.float32),
        sensors={name: SimpleNamespace(data=SimpleNamespace(
            pos_w=np.zeros((1, 3), np.float32), quat_w_world=np.array([[1, 0, 0, 0]], np.float32),
            intrinsic_matrices=np.eye(3, dtype=np.float32)[None])) for name in names},
    )
    env = SimpleNamespace(scene=scene, episode_length_buf=np.zeros(1, np.int64),
                          _frozen_envs=np.zeros(1, bool), _has_stepped=False)
    original = capture_initial_state(env, camera_names=names)
    assert set(original["cameras"]) == set(names)
    assert set(capture_initial_state(env)["cameras"]) == set(CAMERAS)
    for name in names:
        scene.sensors[name].data.intrinsic_matrices[0, 0, 0] += 0.001
        assert capture_initial_state(env, camera_names=names) != original
        scene.sensors[name].data.intrinsic_matrices[0, 0, 0] = 1.0


@pytest.fixture
def usd_scene():
    pytest.importorskip("pxr.Sdf")
    from pxr import Sdf, Usd

    root = Sdf.Layer.CreateAnonymous("root.usda")
    session = Sdf.Layer.CreateAnonymous("session.usda")
    stage = Usd.Stage.Open(root, session)
    for path in RENDER_PRODUCT_IDENTITY_ALIASES:
        stage.DefinePrim(Sdf.Path(path).GetPrimPath(), "RenderProduct")
    with Usd.EditContext(stage, session):
        for index, path in enumerate(RENDER_PRODUCT_IDENTITY_ALIASES):
            prim = stage.GetPrimAtPath(Sdf.Path(path).GetPrimPath())
            prim.CreateAttribute("viewPickingId", Sdf.ValueTypeNames.UInt64, custom=True).Set(700000000000000 + index)
            prim.CreateAttribute("exposure", Sdf.ValueTypeNames.Double, custom=True).Set(0.125)

    def config():
        return {"environment_config": {"native_camera": "unchanged"},
                "loaded_usd_layers": [{"path": "banana.usd", "sha256": "b" * 64}],
                "anonymous_usd_layers": [{"role": "root", "content": root.ExportToString()},
                                         {"role": "session", "content": session.ExportToString()}],
                "resolved_asset_attributes": [{"attribute": "material.texture", "file": "original.png"}],
                "remote_asset_receipts": ["unchanged"]}
    return Sdf, Usd, stage, config


def test_only_six_session_identity_digits_are_canonicalized_without_live_writes(usd_scene):
    _, _, stage, config = usd_scene
    raw = config()
    untouched = copy.deepcopy(raw)
    canonical, identity = canonicalize_render_product_scene(raw, pairing_mode=NATIVE_RENDERER_PAIRING_MODE, stage=stage)
    assert raw == untouched == config()
    assert identity["aliases"] == RENDER_PRODUCT_IDENTITY_ALIASES
    assert identity["raw_scene_config_sha256"] == sha256_json(raw)
    assert identity["scene_config_sha256"] == sha256_json(canonical)
    session = raw["anonymous_usd_layers"][1]["content"]
    assert identity["session_raw_text_sha256"] == hashlib.sha256(session.encode("utf-8")).hexdigest()
    assert len(identity["rows"]) == len(identity["substitutions"]) == 6
    for row in reversed(identity["substitutions"]):
        assert session[row["start"]:row["end"]] == row["raw_decimal"]
        session = session[:row["start"]] + row["alias_decimal"] + session[row["end"]:]
    assert identity["session_canonical_text_sha256"] == hashlib.sha256(session.encode("utf-8")).hexdigest()
    replay = copy.deepcopy(raw)
    replay["anonymous_usd_layers"][1]["content"] = session
    assert replay == canonical


def test_unrelated_scene_values_are_never_removed_from_pairing(usd_scene):
    Sdf, Usd, stage, config = usd_scene
    original, _ = canonicalize_render_product_scene(config(), pairing_mode=NATIVE_RENDERER_PAIRING_MODE, stage=stage)
    path = next(iter(RENDER_PRODUCT_IDENTITY_ALIASES))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        stage.GetPrimAtPath(Sdf.Path(path).GetPrimPath()).GetAttribute("exposure").Set(0.126)
    changed, _ = canonicalize_render_product_scene(config(), pairing_mode=NATIVE_RENDERER_PAIRING_MODE, stage=stage)
    assert changed != original
    assert "0.126" in changed["anonymous_usd_layers"][1]["content"]


@pytest.mark.parametrize("defect", ["wrong_type", "inactive", "not_custom", "wrong_dtype", "timesample", "connection", "missing", "extra", "duplicate", "not_session", "raw_mismatch"])
def test_undeclared_or_unverified_identity_changes_fail_closed(usd_scene, defect):
    Sdf, Usd, stage, config = usd_scene
    path = next(iter(RENDER_PRODUCT_IDENTITY_ALIASES))
    prim = stage.GetPrimAtPath(Sdf.Path(path).GetPrimPath())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        attr = prim.GetAttribute("viewPickingId")
        if defect == "wrong_type":
            prim.SetTypeName("Camera")
        elif defect == "inactive":
            prim.SetActive(False)
        elif defect == "not_custom":
            attr.SetCustom(False)
        elif defect == "wrong_dtype":
            prim.RemoveProperty("viewPickingId")
            prim.CreateAttribute("viewPickingId", Sdf.ValueTypeNames.Int64, custom=True).Set(7)
        elif defect == "timesample":
            attr.Set(7, 1)
        elif defect == "connection":
            attr.SetConnections(["/Other.source"])
        elif defect == "missing":
            prim.RemoveProperty("viewPickingId")
        elif defect == "extra":
            extra = stage.DefinePrim("/Extra", "RenderProduct")
            extra.CreateAttribute("viewPickingId", Sdf.ValueTypeNames.UInt64, custom=True).Set(1234)
        elif defect == "duplicate":
            attr.Set(700000000000001)
        elif defect == "not_session":
            value = attr.Get()
            prim.RemoveProperty("viewPickingId")
            with Usd.EditContext(stage, stage.GetRootLayer()):
                prim.CreateAttribute("viewPickingId", Sdf.ValueTypeNames.UInt64, custom=True).Set(value)
    raw = config()
    if defect == "raw_mismatch":
        raw["anonymous_usd_layers"][1]["content"] = raw["anonymous_usd_layers"][1]["content"].replace("700000000000000", "700000000000050")
    with pytest.raises(ConfigurationError):
        canonicalize_render_product_scene(raw, pairing_mode=NATIVE_RENDERER_PAIRING_MODE, stage=stage)


@pytest.mark.parametrize("mode", [None, "legacy", "native_renderer_physical_v2"])
def test_canonicalization_requires_exact_explicit_mode(mode):
    with pytest.raises(ConfigurationError, match="explicit"):
        canonicalize_render_product_scene({}, pairing_mode=mode, stage=None)

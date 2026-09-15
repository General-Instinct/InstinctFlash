"""CPU contracts for paired RoboLab execution; no renderer qualification implied."""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.vla.robolab_driver import (
    CAMERAS, ObservedEnv, bind_initial_state, capture_initial_state, check_native_result,
    freeze_value, installed_simulator_sources, make_paired_client, recheck_simulator_sources,
    recheck_renderer_compatibility, verify_asset_inventory, verify_renderer_compat_manifest,
    verify_renderer_runtime_bindings,
)
from benchmarks.vla.robolab_protocol import model_seed
from benchmarks.vla.util import ConfigurationError, sha256_json


EPISODE = {
    "pair_id": "smoke/BananaInBowlTask/0000", "task_id": "BananaInBowlTask",
    "episode_index": 0, "scene_seed": 100, "model_seed_base": 1000,
    "max_episode_steps": 64, "max_policy_chunks": 2, "instruction": "put banana in bowl",
}
IDENTITY = {"family": "edge", "arm": "candidate", "execution": "numeric"}


class Env:
    def __init__(self):
        camera_data = SimpleNamespace(pos_w=np.zeros((1, 3), np.float32),
                                      quat_w_world=np.array([[1, 0, 0, 0]], np.float32),
                                      intrinsic_matrices=np.eye(3, dtype=np.float32)[None])
        self.state = {"articulation": {"robot": {
            "joint_position": np.arange(7, dtype=np.float32)[None],
            "joint_velocity": np.zeros((1, 7), np.float32),
        }}, "rigid_object": {"banana": {"root_pose": np.zeros((1, 7), np.float32)}}}
        self.scene = SimpleNamespace(
            get_state=lambda is_relative: self.state,
            sensors={name: SimpleNamespace(data=camera_data) for name in CAMERAS},
            env_origins=np.zeros((1, 3), np.float32),
        )
        self.episode_length_buf = np.zeros(1, dtype=np.int64)
        self._frozen_envs = np.zeros(1, dtype=bool)
        self._has_stepped = False
        self.native_resets = 0
        self.early_termination = False

    def _reset_idx(self, ids):
        self.native_resets += 1

    def reset(self):
        self._reset_idx(np.array([0]))
        return {"image": np.zeros((2, 2, 3), dtype=np.uint8)}, {}

    def step(self, action):
        self._has_stepped = True
        self.episode_length_buf += 1
        if self.early_termination:
            self._reset_idx(np.array([0]))
        return {}, 0, False, False, {}


class NativeClient:
    IMAGE_W = 640
    IMAGE_H = 360
    OPEN_LOOP_HORIZON = 32

    def begin_episode(self, episode_idx):
        self._eval_episode_idx = episode_idx

    def _query_server(self, request):
        return self._infer_with_retry(request)

    def _postprocess_chunk(self, action):
        self.postprocess_calls = getattr(self, "postprocess_calls", 0) + 1
        chunk = action.copy()
        chunk[..., -1] = chunk[..., -1] > 0.5
        return chunk


class Remote:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.wrong = None
        self.action = np.zeros((32, 8), np.float32)
        self.action[:, -1] = np.linspace(0, 1, 32)

    def infer(self, request):
        self.calls.append(request)
        if self.fail:
            raise OSError("transport disconnected")
        if request.get("reset"):
            return {key: value for key, value in request.items() if key != "prompt"}
        result = {"action": self.action, "episode_id": request["episode_id"],
                  "request_id": request["request_id"],
                  "request_seed": model_seed(EPISODE, request["request_id"]),
                  "benchmark_identity_sha256": sha256_json(IDENTITY)}
        if self.wrong:
            result[self.wrong] = "changed"
        return result

    def close(self):
        pass


def test_array_snapshot_preserves_dtype_and_signed_zero():
    plus = freeze_value(np.array([0.0], dtype=np.float32))
    minus = freeze_value(np.array([-0.0], dtype=np.float32))
    assert plus != minus
    assert plus != freeze_value(np.array([0.0], dtype=np.float64))
    assert plus != freeze_value(np.array(0.0, dtype=np.float32))
    assert json.loads(json.dumps(plus)) == plus
    with pytest.raises(ConfigurationError, match="object arrays"):
        freeze_value(np.array([{}], dtype=object))


def test_scene_snapshot_includes_object_robot_camera_and_eval_state():
    env = Env()
    original = capture_initial_state(env)
    env.state["rigid_object"]["banana"]["root_pose"][0, 0] = 0.01
    assert capture_initial_state(env) != original
    assert set(original["cameras"]) == set(CAMERAS)
    assert "joint_velocity" in original["scene"]["articulation"]["robot"]
    del env.state["articulation"]["robot"]
    with pytest.raises(ConfigurationError, match="robot"):
        capture_initial_state(env)


@pytest.mark.parametrize("field", ["robot", "object", "camera"])
def test_nonfinite_initial_physical_state_refused(field):
    env = Env()
    if field == "robot":
        env.state["articulation"]["robot"]["joint_position"][0, 0] = np.nan
    elif field == "object":
        env.state["rigid_object"]["banana"]["root_pose"][0, 0] = np.inf
    else:
        env.scene.sensors[CAMERAS[0]].data.pos_w[0, 0] = -np.inf
    with pytest.raises(ConfigurationError, match="nonfinite native initial"):
        capture_initial_state(env)


def anchor_args(tmp_path):
    return dict(anchor_path=tmp_path / "anchor.json", protocol_sha256="a" * 64,
                episode=EPISODE, state=freeze_value({"x": np.array([1], np.float32)}),
                observation={"image": np.zeros((2, 2, 3), np.uint8)},
                simulator_fingerprint={"software": "fixed"}, scene_config={"seed": 100},
                asset_inventory={"files": ["banana.usd"]})


def test_four_arms_share_edge_baseline_anchor_and_no_overwrite(tmp_path):
    args = anchor_args(tmp_path)
    baseline = bind_initial_state(family="edge", arm="baseline", output=tmp_path / "eb", **args)
    for family, arm in (("edge", "candidate"), ("nano", "baseline"), ("nano", "candidate")):
        assert bind_initial_state(family=family, arm=arm, output=tmp_path / (family + arm), **args) == baseline
    with pytest.raises(FileExistsError):
        bind_initial_state(family="edge", arm="baseline", output=tmp_path / "retry", **args)


@pytest.mark.parametrize("field", ["state", "observation", "scene_config", "asset_inventory", "simulator_fingerprint"])
def test_any_initial_pair_difference_refused_and_snapshot_preserved(tmp_path, field):
    args = anchor_args(tmp_path)
    bind_initial_state(family="edge", arm="baseline", output=tmp_path / "eb", **args)
    args[field] = {"changed": True}
    with pytest.raises(ConfigurationError, match="paired initial"):
        bind_initial_state(family="edge", arm="candidate", output=tmp_path / "ec", **args)
    assert (tmp_path / "ec" / "initial_state.json").is_file()


def test_missing_anchor_refused_before_candidate(tmp_path):
    with pytest.raises(ConfigurationError, match="missing JSON"):
        bind_initial_state(family="nano", arm="baseline", output=tmp_path / "nb", **anchor_args(tmp_path))


def test_native_double_reset_and_only_actual_steps_recorded(tmp_path):
    env = Env()
    snapshots = []
    observed = ObservedEnv(env, on_initial_state=lambda e, o: snapshots.append(o) or {"ok": True},
                           max_steps=3, output=tmp_path)
    with pytest.raises(ConfigurationError, match="pairing gate"):
        observed.step(np.zeros((1, 8)))
    observed.reset()
    assert snapshots == []
    observed.reset()
    assert len(snapshots) == 1 and env.native_resets == 2
    for _ in range(3):
        observed.step(np.zeros((1, 8)))
    assert observed.executed_steps == observed.attempted_steps == 3
    assert len(list((tmp_path / "steps").glob("*.json"))) == 3
    with pytest.raises(ConfigurationError, match="step bound"):
        observed.step(np.zeros((1, 8)))
    with pytest.raises(ConfigurationError, match="two native"):
        observed.reset()
    observed.restore_hooks()


def test_upstream_hidden_early_physics_retry_fails_without_reset(tmp_path):
    env = Env()
    observed = ObservedEnv(env, on_initial_state=lambda e, o: {"ok": True}, max_steps=3, output=tmp_path)
    observed.reset()
    observed.reset()
    env.early_termination = True
    with pytest.raises(ConfigurationError, match="early-termination retry"):
        observed.step(np.zeros((1, 8)))
    assert env.native_resets == 2
    assert observed.attempted_steps == 1 and observed.executed_steps == 0
    assert (tmp_path / "steps" / "000000.json").is_file()


def make_client(tmp_path):
    remote = Remote()
    client = make_paired_client(NativeClient, remote, identity=IDENTITY, episode=EPISODE, output=tmp_path)
    return client, remote


def test_reset_uses_model_seed_and_bounded_chunk_requests_preserve_raw_action(tmp_path):
    client, remote = make_client(tmp_path)
    with pytest.raises(ConfigurationError, match="seeded reset"):
        client._query_server({"prompt": EPISODE["instruction"]})
    client.begin_episode(0)
    reset = remote.calls[0]
    assert reset["benchmark_seed"] == 1000 and reset["max_policy_chunks"] == 2
    for index in range(2):
        result = client._query_server({"prompt": EPISODE["instruction"]})
        assert remote.calls[-1]["request_id"] == index
        assert result["action"] is remote.action
        processed = client._postprocess_chunk(result["action"])
        assert set(processed[:, -1]) == {0.0, 1.0}
    assert client.seeds == [1000, 1001]
    assert np.any((remote.action[:, -1] > 0) & (remote.action[:, -1] < 1))
    assert client.postprocess_calls == 2
    with pytest.raises(ConfigurationError, match="chunk bound"):
        client._query_server({"prompt": EPISODE["instruction"]})
    assert len(remote.calls) == 3
    with pytest.raises(ConfigurationError, match="fresh simulator"):
        client.begin_episode(1)


def test_transport_disconnect_is_not_retried(tmp_path):
    client, remote = make_client(tmp_path)
    client.begin_episode(0)
    remote.fail = True
    with pytest.raises(OSError, match="disconnected"):
        client._query_server({"prompt": EPISODE["instruction"]})
    assert len(remote.calls) == 2 and client.requests == 1 and client.seeds == []


@pytest.mark.parametrize("field", ["episode_id", "request_id", "request_seed", "benchmark_identity_sha256"])
def test_mismatched_remote_receipt_refused(tmp_path, field):
    client, remote = make_client(tmp_path)
    client.begin_episode(0)
    remote.wrong = field
    with pytest.raises(ConfigurationError, match="response identity"):
        client._query_server({"prompt": EPISODE["instruction"]})


@pytest.mark.parametrize("action", [np.zeros((31, 8), np.float32), np.zeros((32, 8), np.float64),
                                   np.full((32, 8), np.nan, np.float32)])
def test_invalid_raw_actions_never_enter_simulator(tmp_path, action):
    client, remote = make_client(tmp_path)
    client.begin_episode(0)
    remote.action = action
    with pytest.raises(ConfigurationError, match="finite raw float32"):
        client._query_server({"prompt": EPISODE["instruction"]})


@pytest.mark.parametrize("success", [None, 0, "false", np.bool_(True)])
def test_incomplete_or_nonboolean_native_outcome_is_not_a_failure_label(tmp_path, success):
    observed = SimpleNamespace(executed_steps=3, reset_count=2, action_digest=hashlib.sha256())
    with pytest.raises(ConfigurationError, match="terminal success/failure"):
        check_native_result([{"env_id": 0, "success": success, "step": 3}], observed,
                            SimpleNamespace(seeds=[1000]))


def test_native_failed_task_is_preserved_as_completed_false():
    observed = SimpleNamespace(executed_steps=3, reset_count=2, action_digest=hashlib.sha256())
    record = check_native_result([{"env_id": 0, "success": False, "step": 3}], observed,
                                 SimpleNamespace(seeds=[1000]))
    assert record["status"] == "completed" and record["success"] is False


def test_asset_content_tamper_and_escape_refused(tmp_path):
    asset = tmp_path / "banana.usd"
    asset.write_bytes(b"USD scene")
    manifest = {"schema_version": 1, "root": str(tmp_path),
                "files": {asset.name: hashlib.sha256(asset.read_bytes()).hexdigest()}}
    verified = verify_asset_inventory(manifest)
    assert verified["files"][0]["path"] == str(asset)
    asset.write_bytes(b"changed scene")
    with pytest.raises(ConfigurationError, match="hash differs"):
        verify_asset_inventory(manifest)
    outside = tmp_path.parent / (tmp_path.name + "-outside.usd")
    outside.write_bytes(b"outside")
    try:
        manifest["files"] = {"../" + outside.name: hashlib.sha256(outside.read_bytes()).hexdigest()}
        with pytest.raises(ConfigurationError, match="escapes"):
            verify_asset_inventory(manifest)
    finally:
        outside.unlink()


def test_asset_inventory_cannot_omit_existing_textures(tmp_path):
    (tmp_path / "texture.png").write_bytes(b"pixels")
    (tmp_path / "scene.usd").write_bytes(b"scene")
    manifest = {"schema_version": 1, "root": str(tmp_path),
                "files": {"scene.usd": hashlib.sha256(b"scene").hexdigest()}}
    with pytest.raises(ConfigurationError, match="complete actual asset root"):
        verify_asset_inventory(manifest)


def test_actual_simulator_sources_and_binaries_bound_and_rechecked(tmp_path, monkeypatch):
    import sys

    for name in ("isaaclab", "isaacsim"):
        root = tmp_path / name
        root.mkdir()
        (root / "__init__.py").write_text("# source\n")
        (root / "physics.so").write_bytes(b"native physics")
        cache = root / "__pycache__"
        cache.mkdir()
        (cache / "init.pyc").write_bytes(b"derived bytecode")
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=str(root / "__init__.py")))
    inventory = installed_simulator_sources()
    assert len(inventory["files"]) == 4
    recheck_simulator_sources(inventory)
    (tmp_path / "isaacsim" / "physics.so").write_bytes(b"different math")
    with pytest.raises(ConfigurationError, match="changed during"):
        recheck_simulator_sources(inventory)


def warp_fixture(tmp_path, monkeypatch):
    import sys

    for name in ("isaaclab", "isaacsim", "warp"):
        root = tmp_path / name
        root.mkdir()
        (root / "__init__.py").write_text("# imported package\n")
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=str(root / "__init__.py")))
    overlay = tmp_path / "external_overlay"
    overlay.mkdir()
    context = overlay / "context.py"
    context.write_text("# explicitly qualified patched context\n")
    library = overlay / "actual_warp.so"
    library.write_bytes(b"actual loaded runtime, outside bundled Isaac tree")
    native = SimpleNamespace(_name=str(library))
    monkeypatch.setitem(sys.modules, "warp.context", SimpleNamespace(
        __file__=str(context), runtime=SimpleNamespace(core=native, llvm=None),
    ))
    return context, library, native


@pytest.mark.parametrize("artifact", ["context", "native"])
def test_external_executed_warp_code_and_native_drift_fail(tmp_path, monkeypatch, artifact):
    context, native, _ = warp_fixture(tmp_path, monkeypatch)
    inventory = installed_simulator_sources()
    assert inventory["roots"]["warp"] == str(tmp_path / "warp")
    assert any(row["path"] == str(context) for row in inventory["warp_imported_modules"])
    assert inventory["warp_runtime_libraries"][0]["path"] == str(native)
    recheck_simulator_sources(inventory)
    (context if artifact == "context" else native).write_bytes(b"changed actual executable")
    with pytest.raises(ConfigurationError, match="changed during"):
        recheck_simulator_sources(inventory)


def test_warp_runtime_cannot_silently_switch_native_library_targets(tmp_path, monkeypatch):
    _, library, handle = warp_fixture(tmp_path, monkeypatch)
    inventory = installed_simulator_sources()
    replacement = library.with_name("other.so")
    replacement.write_bytes(library.read_bytes())
    handle._name = str(replacement)
    with pytest.raises(ConfigurationError, match="changed during"):
        recheck_simulator_sources(inventory)
    handle._name = "unresolved-soname.so"
    with pytest.raises(ConfigurationError, match="absolute file target"):
        installed_simulator_sources()


def test_lazy_warp_import_must_be_inside_already_bound_source_tree(tmp_path, monkeypatch):
    import sys

    warp_fixture(tmp_path, monkeypatch)
    covered = tmp_path / "warp" / "already_bound.py"
    covered.write_text("# complete package inventory already covers these bytes\n")
    inventory = installed_simulator_sources()
    monkeypatch.setitem(sys.modules, "warp.already_bound", SimpleNamespace(__file__=str(covered)))
    recheck_simulator_sources(inventory)
    external = tmp_path / "unbound_late_overlay.py"
    external.write_text("# not prospectively bound by the original package tree\n")
    monkeypatch.setitem(sys.modules, "warp.unbound_late", SimpleNamespace(__file__=str(external)))
    with pytest.raises(ConfigurationError, match="changed during"):
        recheck_simulator_sources(inventory)


def compatibility_fixture(tmp_path):
    artifact = tmp_path / "explicit_profile_or_library.bin"
    artifact.write_bytes(b"prospectively declared compatibility bytes")
    environment = {"LD_LIBRARY_PATH": "/explicit/compat:/native", "VK_LAYER_PATH": "/explicit/layers",
                   "VK_INSTANCE_LAYERS": "explicit_layer", "VK_LAYER_SETTINGS_PATH": None,
                   "CUDA_VISIBLE_DEVICES": None, "EXTRA_DECLARED_SETTING": "preserved"}
    manifest = {"schema_version": 1, "files": {
        str(artifact): hashlib.sha256(artifact.read_bytes()).hexdigest()}, "environment": environment}
    path = tmp_path / "compatibility.json"
    path.write_text(json.dumps(manifest))
    actual_environment = {key: value for key, value in environment.items() if value is not None}
    return path, manifest, artifact, actual_environment


@pytest.mark.parametrize("changed", ["file", "manifest", "environment"])
def test_compatibility_file_manifest_and_environment_drift_fail(tmp_path, changed):
    path, manifest, artifact, environment = compatibility_fixture(tmp_path)
    before = verify_renderer_compat_manifest(path, environment=environment)
    assert before["files"][0]["path"] == str(artifact)
    assert before["environment"]["CUDA_VISIBLE_DEVICES"] is None
    assert recheck_renderer_compatibility(before, environment=environment) == before
    if changed == "file":
        artifact.write_bytes(b"changed Vulkan or bootstrap executable")
    elif changed == "manifest":
        # Even equivalent parsed JSON must preserve the frozen manifest bytes.
        path.write_text(json.dumps(manifest, indent=2))
    else:
        environment["VK_LAYER_PATH"] = "/different/layer/search/path"
    with pytest.raises(ConfigurationError, match="compatibility"):
        recheck_renderer_compatibility(before, environment=environment)


@pytest.mark.parametrize("value", ["0", ""])
def test_compatibility_forbids_even_explicit_or_empty_cuda_visibility_override(tmp_path, value):
    path, manifest, _, environment = compatibility_fixture(tmp_path)
    environment["CUDA_VISIBLE_DEVICES"] = value
    with pytest.raises(ConfigurationError, match="environment differs"):
        verify_renderer_compat_manifest(path, environment=environment)
    manifest["environment"]["CUDA_VISIBLE_DEVICES"] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError, match="must not override"):
        verify_renderer_compat_manifest(path, environment=environment)


def test_compatibility_requires_all_selected_environment_and_absolute_files(tmp_path):
    path, manifest, artifact, environment = compatibility_fixture(tmp_path)
    del manifest["environment"]["VK_INSTANCE_LAYERS"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError, match="omits required"):
        verify_renderer_compat_manifest(path, environment=environment)
    manifest["environment"]["VK_INSTANCE_LAYERS"] = environment["VK_INSTANCE_LAYERS"]
    manifest["files"] = {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError, match="absolute paths"):
        verify_renderer_compat_manifest(path, environment=environment)


def test_fingerprint_binds_optional_compatibility_without_changing_old_callers(tmp_path, monkeypatch):
    import sys
    from benchmarks.vla import robolab_driver as driver

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda _: SimpleNamespace(name="NVIDIA GeForce RTX 4090", total_memory=24 << 30),
    ), version=SimpleNamespace(cuda="12.8"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(driver.importlib.metadata, "distributions", lambda: [])
    source, simulator = {"source": "bound"}, {"warp": "actual executable"}
    original = driver._simulator_fingerprint(source, simulator)
    assert "renderer_compatibility_sha256" not in original
    path, _, _, environment = compatibility_fixture(tmp_path)
    compatibility = verify_renderer_compat_manifest(path, environment=environment)
    explicit = driver._simulator_fingerprint(source, simulator, compatibility)
    assert explicit.pop("renderer_compatibility_sha256") == sha256_json(compatibility)
    assert explicit == original


def runtime_binding_fixture(tmp_path, monkeypatch):
    context, core, handle = warp_fixture(tmp_path, monkeypatch)
    path, manifest, vulkan, environment = compatibility_fixture(tmp_path)
    module = tmp_path / "warp" / "__init__.py"
    manifest["files"].update({str(file): hashlib.sha256(file.read_bytes()).hexdigest()
                              for file in (module, core)})
    manifest["runtime_bindings"] = {
        "warp_module": str(module), "warp_core": str(core),
        "vulkan_layer_libraries": [str(vulkan)],
    }
    path.write_text(json.dumps(manifest))
    return path, manifest, environment, context, handle


def test_runtime_declaration_is_bound_before_startup_without_imports(tmp_path, monkeypatch):
    import sys

    path, manifest, environment, _, _ = runtime_binding_fixture(tmp_path, monkeypatch)
    monkeypatch.delitem(sys.modules, "warp")
    receipt = verify_renderer_compat_manifest(path, environment=environment)
    assert receipt["runtime_bindings"] == manifest["runtime_bindings"]
    assert "warp" not in sys.modules
    with pytest.raises(ConfigurationError, match="not actually imported"):
        verify_renderer_runtime_bindings(receipt, environment=environment)
    assert "warp" not in sys.modules


@pytest.mark.parametrize("target", ["warp_module", "warp_core", "warp_llvm", "vulkan_layer_libraries"])
def test_runtime_bindings_cannot_name_unhashed_files(tmp_path, monkeypatch, target):
    path, manifest, environment, _, _ = runtime_binding_fixture(tmp_path, monkeypatch)
    unbound = tmp_path / "unbound_target.so"
    unbound.write_bytes(b"file exists but is not bound")
    manifest["runtime_bindings"][target] = [str(unbound)] if target == "vulkan_layer_libraries" else str(unbound)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError, match="explicit absolute file hash"):
        verify_renderer_compat_manifest(path, environment=environment)


@pytest.mark.parametrize("drift", ["module", "core", "inactive", "missing_llvm"])
def test_runtime_gate_rejects_unused_warp_targets_or_inactive_core(tmp_path, monkeypatch, drift):
    import sys

    path, manifest, environment, _, handle = runtime_binding_fixture(tmp_path, monkeypatch)
    if drift == "missing_llvm":
        # A valid declaration cannot claim LLVM was selected when no handle exists.
        manifest["runtime_bindings"]["warp_llvm"] = manifest["runtime_bindings"]["warp_core"]
        path.write_text(json.dumps(manifest))
    receipt = verify_renderer_compat_manifest(path, environment=environment)
    if drift == "module":
        wrong = tmp_path / "unused_alternate_init.py"
        wrong.write_text("# different actual imported module\n")
        sys.modules["warp"].__file__ = str(wrong)
    elif drift == "core":
        wrong = tmp_path / "unused_core.so"
        wrong.write_bytes(b"different actual native handle")
        handle._name = str(wrong)
    elif drift == "inactive":
        sys.modules["warp.context"].runtime = None
    with pytest.raises(ConfigurationError, match="Warp"):
        verify_renderer_runtime_bindings(receipt, environment=environment)


def test_vulkan_binding_requires_actual_executable_library_mapping(tmp_path, monkeypatch):
    import ctypes
    import shutil
    import subprocess

    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("CPU shared-library compiler unavailable")
    path, manifest, environment, _, _ = runtime_binding_fixture(tmp_path, monkeypatch)
    library = tmp_path / "libbounded_compat_cpu_test.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-x", "c", "-", "-o", str(library)],
                   input="int bounded_compat_probe(void) { return 7; }\n", text=True, check=True,
                   capture_output=True)
    manifest["files"][str(library)] = hashlib.sha256(library.read_bytes()).hexdigest()
    manifest["runtime_bindings"]["vulkan_layer_libraries"] = [str(library)]
    path.write_text(json.dumps(manifest))
    receipt = verify_renderer_compat_manifest(path, environment=environment)
    with pytest.raises(ConfigurationError, match="not actually mapped"):
        verify_renderer_runtime_bindings(receipt, environment=environment)
    loaded = ctypes.CDLL(str(library))
    assert loaded.bounded_compat_probe() == 7
    actual = verify_renderer_runtime_bindings(receipt, environment=environment)
    assert actual["vulkan_layer_libraries"] == [{
        "expected_path": str(library), "resolved_path": str(library), "mapped_paths": [str(library)],
    }]
    # Optional field remains disabled for callers that only use the original gate.
    del manifest["runtime_bindings"]
    path.write_text(json.dumps(manifest))
    old_receipt = verify_renderer_compat_manifest(path, environment=environment)
    assert verify_renderer_runtime_bindings(old_receipt, environment=environment) is None


@pytest.mark.parametrize("exit_code,prior_failure,env_failure,expected", [
    (0, False, False, "completed"), (7, False, False, "completed"),
    (0, True, False, "failed"), (0, False, True, "failed"),
])
def test_native_fast_exit_preserves_outcome_but_never_erases_prior_failure(
    tmp_path, exit_code, prior_failure, env_failure, expected,
):
    import subprocess
    import sys

    code = """
import json, os, sys
from pathlib import Path
from benchmarks.vla.robolab_driver import finalize_episode_resources
out, code, failed, env_failed = sys.argv[1:]
class Env:
    def close(self):
        if env_failed == '1':
            raise RuntimeError('environment cleanup failed')
class App:
    def close(self):
        os._exit(int(code))
record = {'status': 'failed' if failed == '1' else 'completed',
          'success': None if failed == '1' else True}
finalize_episode_resources(record, Path(out), env=Env(), app=App())
raise AssertionError('native exit unexpectedly returned')
"""
    process = subprocess.run([sys.executable, "-c", code, str(tmp_path), str(exit_code),
                              str(int(prior_failure)), str(int(env_failure))],
                             capture_output=True, text=True, timeout=20)
    assert process.returncode == exit_code, process.stderr
    record = json.loads((tmp_path / "result.json").read_text())
    assert record["status"] == expected
    assert record["native_app_close"] == "pending_external_exit_check"
    # A shutdown crash leaves its pre-close evidence but cannot be admitted.
    admitted = record["status"] == "completed" and process.returncode == 0
    assert admitted is (exit_code == 0 and not prior_failure and not env_failure)


@pytest.mark.skipif(not os.environ.get("ROBOLAB_ROOT"), reason="optional pinned RoboLab client environment")
def test_official_cosmos_client_cpu_mosaic_and_chunk_parity(tmp_path, monkeypatch):
    """Uses actual upstream preprocessing and public infer; never launches Isaac."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("openpi_client")
    monkeypatch.syspath_prepend(os.environ["ROBOLAB_ROOT"])
    from policies.cosmos3.client import Cosmos3Client

    raw = {
        "image_obs": {
            name: torch.from_numpy(np.random.default_rng(index).integers(
                0, 256, (1, 231, 333, 3), dtype=np.uint8))
            for index, name in enumerate(CAMERAS)
        },
        "proprio_obs": {"arm_joint_pos": torch.zeros((1, 7)), "gripper_pos": torch.zeros((1, 1))},
    }
    remote = Remote()
    client = make_paired_client(Cosmos3Client, remote, identity=IDENTITY, episode=EPISODE, output=tmp_path)
    native = Cosmos3Client.__new__(Cosmos3Client)
    native._image_h, native._image_w = 360, 640
    actual = client._pack_request(client._extract_observation(raw), EPISODE["instruction"])
    expected = native._pack_request(native._extract_observation(raw), EPISODE["instruction"])
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], np.ndarray):
            assert (actual[key].dtype, actual[key].shape, actual[key].tobytes()) == (
                expected[key].dtype, expected[key].shape, expected[key].tobytes())
        else:
            assert actual[key] == expected[key]
    assert actual["observation/image"].shape == (540, 640, 3)
    client.begin_episode(0)
    actions = np.stack([client.infer(raw, EPISODE["instruction"])["action"] for _ in range(33)])
    expected_chunk = native._postprocess_chunk(remote.action)
    np.testing.assert_array_equal(actions, np.concatenate((expected_chunk, expected_chunk[:1])))
    assert client.requests == 2 and client.seeds == [1000, 1001]

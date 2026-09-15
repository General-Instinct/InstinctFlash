"""One paired Cosmos episode per fresh, pinned RoboLab/Isaac Sim process.

The native episode loop, camera preprocessing, action chunking, gripper threshold,
and task predicates belong to RoboLab. This bridge adds seeded remote requests,
immutable traces, and an initial-state pairing gate. CPU tests exercise the bridge;
an actual renderer qualification is still required before any task-quality claim.
"""

from __future__ import annotations

import argparse
import base64
import copy
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from . import robolab_assets
from .robolab_assets import (
    remote_url, verified_asset_file, verify_asset_inventory, verify_remote_assets,
)
from .util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic

ROBOLAB_REVISION = "9db0aaf09d9fe5d4f37b168320788258c7012463"
CAMERAS = ("wrist_cam", "over_shoulder_left_camera", "over_shoulder_right_camera")
HORIZON = 32
NATIVE_RENDERER_PAIRING_MODE = "native_renderer_physical_v1"
NATIVE_IMAGE_GROUPS = {
    "image_obs": ("head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam"),
    "viewport_cam": ("egocentric_mirrored_camera",),
}
RENDER_PRODUCT_IDENTITY_ALIASES = {
    f"/Render/OmniverseKit/HydraTextures/{name}.viewPickingId": index
    for index, name in enumerate((
        "Replicator", "Replicator_01", "Replicator_02", "Replicator_03", "Replicator_04",
        "omni_kit_widget_viewport_ViewportTexture_0",
    ), start=1)
}


def freeze_value(value: Any, *, output_root: str | None = None) -> Any:
    """Serialize actual array bytes, without lossy float/list conversion or repr IDs."""
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ConfigurationError("object arrays cannot attest simulator state")
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            # Some configuration limits legitimately use infinities. Preserve them
            # explicitly rather than emitting nonstandard JSON NaN/Infinity.
            return {"float_hex": value.hex()}
        return value
    if isinstance(value, (str, Path)):
        text = str(value)
        return text.replace(output_root, "<episode-output>") if output_root else text
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ConfigurationError("state/config mapping keys must be strings")
        return {key: freeze_value(item, output_root=output_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [freeze_value(item, output_root=output_root) for item in value]
    if isinstance(value, functools.partial):
        return {
            "partial": freeze_value(value.func),
            "args": freeze_value(value.args, output_root=output_root),
            "keywords": freeze_value(value.keywords, output_root=output_root),
        }
    if isinstance(value, slice):
        return {"slice": freeze_value([value.start, value.stop, value.step])}
    if callable(value) and hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return {"callable": value.__module__ + "." + value.__qualname__}
    if hasattr(value, "to_dict"):
        return freeze_value(value.to_dict(), output_root=output_root)
    raise ConfigurationError(f"unsupported state/config value type: {type(value).__name__}")


def capture_initial_state(env: Any, *, camera_names=None) -> dict[str, Any]:
    """Capture every entity in native scene.get_state plus cameras and eval buffers.

    This is an auditable physical-state readout, not a portable PhysX checkpoint:
    private solver/contact caches are not exposed by InteractiveScene.get_state.
    No state is restored or synthesized by this driver.
    """
    import numpy as np

    state = env.scene.get_state(is_relative=False)
    if not isinstance(state, dict) or "robot" not in state.get("articulation", {}):
        # Isaac Lab versions use singular asset category names in get_state.
        if not isinstance(state, dict) or "robot" not in state.get("articulations", {}):
            raise ConfigurationError("native scene state lacks the complete robot articulation")
    cameras = {}
    for name in CAMERAS if camera_names is None else sorted(camera_names):
        sensor = env.scene.sensors[name]
        data = sensor.data
        cameras[name] = {
            field: getattr(data, field)
            for field in ("pos_w", "quat_w_world", "intrinsic_matrices")
        }
        if any(value is None for value in cameras[name].values()):
            raise ConfigurationError("camera state is not initialized")

    def require_finite(value):
        if isinstance(value, dict):
            for item in value.values():
                require_finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                require_finite(item)
        else:
            array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise ConfigurationError("nonfinite native initial scene state")

    require_finite({"scene": state, "cameras": cameras})
    if getattr(env.scene, "rigid_object_collections", {}):
        raise ConfigurationError("rigid object collections need a complete native state exporter")
    return freeze_value({
        "scope": "native_interactive_scene_state_world_frame_and_camera_eval_buffers_v1",
        "scene": state,
        "env_origins": env.scene.env_origins,
        "cameras": cameras,
        "episode_length_buf": env.episode_length_buf,
        "frozen_envs": env._frozen_envs,
        "has_stepped": env._has_stepped,
    })


def native_renderer_observation_evidence(observation: Any, *, frozen: bool = False) -> dict:
    """Retain each native image; separate only the declared visual groups for pairing.

    This does not change or supply observations to the policy. The full frozen
    observation remains per-arm evidence. All other observation values compare
    exactly, and the fixed camera names, data types and dimensions also compare.
    """
    observed = copy.deepcopy(observation) if frozen else freeze_value(observation)
    if not isinstance(observed, dict) or any(not isinstance(key, str) for key in observed):
        raise ConfigurationError("native renderer observation must be a string-keyed mapping")
    schema = {}
    for group, names in NATIVE_IMAGE_GROUPS.items():
        images = observed.get(group)
        if not isinstance(images, dict) or set(images) != set(names):
            raise ConfigurationError(f"native renderer image group differs from the declared cameras: {group}")
        schema[group] = {}
        for name in sorted(images):
            image = images[name]
            if not isinstance(image, dict) or set(image) != {"dtype", "shape", "data_base64"}:
                raise ConfigurationError("native renderer images require complete frozen array bytes")
            shape = image["shape"]
            if (image["dtype"] != "|u1" or not isinstance(shape, list) or len(shape) != 4
                    or any(type(value) is not int or value <= 0 for value in shape)
                    or shape[0] != 1 or shape[-1] != 3):
                raise ConfigurationError("native renderer images must have uint8 [1,H,W,3] schema")
            try:
                data = base64.b64decode(image["data_base64"], validate=True)
            except (ValueError, TypeError) as error:
                raise ConfigurationError("native renderer image bytes are invalid base64") from error
            if len(data) != math.prod(shape):
                raise ConfigurationError("native renderer image bytes differ from the declared dimensions")
            schema[group][name] = {"dtype": image["dtype"], "shape": list(shape)}
    return {
        "observation": observed,
        "nonvisual_observation": {key: value for key, value in observed.items() if key not in NATIVE_IMAGE_GROUPS},
        "image_schema": schema,
    }


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def finalize_episode_resources(record: dict, output: Path, *, remote=None, env=None, app=None) -> None:
    """Persist the checked outcome before native fast shutdown can exit Python.

    A completed record still requires a separately observed process exit code 0.
    In particular, an application shutdown crash cannot be admitted from this
    file alone. Prior body/transport/environment failures are never cleared.
    """
    for name, resource in (("remote", remote), ("environment", env)):
        if resource is not None:
            try:
                resource.close()
            except BaseException as error:
                record.setdefault("cleanup_errors", []).append(f"{name}: {type(error).__name__}: {error}")
                record.update(status="failed", success=None)
    record["native_app_close"] = "pending_external_exit_check" if app is not None else "not_created"
    write_json_atomic(Path(output) / "result.json", record)
    if app is not None:
        try:
            # Isaac's native fast_shutdown can terminate here without returning.
            app.close()
            record["native_app_close"] = "returned"
        except BaseException as error:
            record.setdefault("cleanup_errors", []).append(f"application: {type(error).__name__}: {error}")
            record.update(status="failed", success=None, native_app_close="failed")
    write_json_atomic(Path(output) / "result.json", record)


def bind_initial_state(
    *, family: str, arm: str, anchor_path: Path, protocol_sha256: str, episode: dict,
    state: dict, observation: Any, simulator_fingerprint: dict, scene_config: dict,
    asset_inventory: dict, output: Path, pairing_mode: str | None = None,
) -> dict:
    """Baseline creates a write-once anchor; candidate must match before inference."""
    if pairing_mode not in (None, NATIVE_RENDERER_PAIRING_MODE):
        raise ConfigurationError("unsupported native initial pairing mode")
    frozen_observation = freeze_value(observation)
    observation_sha = sha256_json(frozen_observation)
    snapshot = {
        "pair_id": episode["pair_id"],
        "task_id": episode["task_id"],
        "scene_seed": episode["scene_seed"],
        "protocol_sha256": protocol_sha256,
        "initial_state_sha256": sha256_json(state),
        "initial_observation_sha256": observation_sha,
        "simulator_fingerprint_sha256": sha256_json(simulator_fingerprint),
        "scene_config_sha256": sha256_json(scene_config),
        "asset_inventory_sha256": sha256_json(asset_inventory),
    }
    if pairing_mode is not None:
        evidence = native_renderer_observation_evidence(frozen_observation, frozen=True)
        del snapshot["initial_observation_sha256"]
        snapshot.update(
            pairing_mode=pairing_mode,
            initial_nonvisual_observation_sha256=sha256_json(evidence["nonvisual_observation"]),
            initial_image_schema_sha256=sha256_json(evidence["image_schema"]),
        )
        _write_exclusive(output / "initial_observation.json", evidence["observation"])
    _write_exclusive(output / "initial_state.json", {"binding": snapshot, "state": state})
    if family == "edge" and arm == "baseline":
        _write_exclusive(anchor_path, snapshot)
    elif family in ("edge", "nano") and arm in ("baseline", "candidate"):
        if load_json(anchor_path) != snapshot:
            raise ConfigurationError("paired initial scene, observation, or simulator differs")
    else:
        raise ConfigurationError("unknown paired arm")
    if pairing_mode is not None:
        return dict(snapshot, initial_observation_sha256=observation_sha)
    return snapshot


class ObservedEnv:
    """Delegate native semantics while recording resets and actually executed actions."""

    def __init__(self, env: Any, *, on_initial_state, max_steps: int, output: Path):
        self.env = env
        self.on_initial_state = on_initial_state
        self.max_steps = max_steps
        self.output = output
        self.reset_count = 0
        self.executed_steps = 0
        self.attempted_steps = 0
        self.action_digest = hashlib.sha256()
        self.binding = None
        self._native_reset_idx = env._reset_idx

        def checked_reset_idx(env_ids):
            if env._has_stepped:
                ids = env_ids.tolist()
                if any(int(env.episode_length_buf[index].item()) <= 2 for index in ids):
                    raise ConfigurationError("native early-termination retry refused; episode failed")
            return self._native_reset_idx(env_ids)

        # Native normal termination still freezes the environment and produces
        # its own success result. Only the hidden early-physics retry is refused.
        env._reset_idx = checked_reset_idx

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        if self.executed_steps or self.reset_count >= 2:
            raise ConfigurationError("only the two native initial resets are permitted")
        result = self.env.reset(*args, **kwargs)
        self.reset_count += 1
        if self.reset_count == 2:
            self.binding = self.on_initial_state(self.env, result[0])
        return result

    def step(self, actions):
        if self.reset_count != 2 or self.binding is None:
            raise ConfigurationError("initial pairing gate must pass before stepping")
        if self.attempted_steps >= self.max_steps:
            raise ConfigurationError("prospective episode step bound exceeded")
        frozen = freeze_value(actions)
        _write_exclusive(self.output / "steps" / f"{self.attempted_steps:06d}.json", frozen)
        self.attempted_steps += 1
        result = self.env.step(actions)
        self.action_digest.update(bytes.fromhex(sha256_json(frozen)))
        self.executed_steps += 1
        return result

    def restore_hooks(self):
        self.env._reset_idx = self._native_reset_idx


def make_paired_client(native_client_class, remote, *, identity: dict, episode: dict, output: Path):
    """Reuse the pinned official image pipeline and chunk postprocessing unchanged."""
    import numpy as np

    from .robolab_protocol import model_seed

    identity_sha = sha256_json(identity)

    class PairedCosmosClient(native_client_class):
        def __init__(self):
            # Do not call Cosmos3Client.__init__: its OpenPI transport reconnects.
            self._chunks = {}
            self._counters = {}
            self._eval_episode_idx = 0
            self._image_w = self.IMAGE_W
            self._image_h = self.IMAGE_H
            self.open_loop_horizon = self.OPEN_LOOP_HORIZON
            if self.open_loop_horizon != HORIZON:
                raise ConfigurationError("upstream Cosmos action horizon changed")
            self.client = remote
            self.requests = 0
            self.seeds = []
            self.started = False

        def begin_episode(self, episode_idx):
            if self.started:
                raise ConfigurationError("fresh simulator/client process required for each episode")
            super().begin_episode(episode_idx)
            self.started = True
            request = {
                "reset": True,
                "episode_id": episode["pair_id"],
                "prompt": episode["instruction"],
                "benchmark_seed": episode["model_seed_base"],
                "max_policy_chunks": episode["max_policy_chunks"],
                "benchmark_identity_sha256": identity_sha,
            }
            result = self.client.infer(request)
            if any(result.get(key) != request[key] for key in (
                "reset", "episode_id", "benchmark_seed", "benchmark_identity_sha256", "max_policy_chunks"
            )):
                raise ConfigurationError("server did not acknowledge the exact episode reset")
            _write_exclusive(output / "reset_receipt.json", freeze_value(result))

        def _infer_with_retry(self, request, max_retries=1):
            del max_retries
            if not self.started:
                raise ConfigurationError("seeded reset required before first inference")
            if request.get("prompt") != episode["instruction"]:
                raise ConfigurationError("native instruction differs from the frozen manifest")
            if self.requests >= episode["max_policy_chunks"]:
                raise ConfigurationError("prospective policy chunk bound exceeded")
            index = self.requests
            request = dict(request, episode_id=episode["pair_id"], request_id=index,
                           benchmark_identity_sha256=identity_sha)
            self.requests += 1  # A transport error consumes this request; never retry.
            result = self.client.infer(request)
            seed = model_seed(episode, index)
            expected = {"episode_id": episode["pair_id"], "request_id": index,
                        "request_seed": seed, "benchmark_identity_sha256": identity_sha}
            if any(result.get(key) != value for key, value in expected.items()):
                raise ConfigurationError("server response identity, request order, or seed differs")
            action = np.asarray(result.get("action"))
            if action.shape != (32, 8) or action.dtype != np.float32 or not np.isfinite(action).all():
                raise ConfigurationError("server must return finite raw float32 [32,8] actions")
            _write_exclusive(output / "chunks" / f"{index:06d}.json", {
                # RemotePolicy's immutable wire_trace carries full request images.
                "request_sha256": sha256_json(freeze_value(request)),
                "response": freeze_value(result),
            })
            self.seeds.append(seed)
            return result

        def _postprocess_chunk(self, chunk):
            processed = super()._postprocess_chunk(chunk)
            _write_exclusive(output / "processed_chunks" / f"{len(self.seeds) - 1:06d}.json",
                             freeze_value(processed))
            return processed

        def close(self):
            self.client.close()

    return PairedCosmosClient()


def check_native_result(results: Any, observed: ObservedEnv, client: Any) -> dict:
    if not isinstance(results, list) or len(results) != 1 or results[0].get("env_id") != 0:
        raise ConfigurationError("expected exactly one native episode outcome")
    native = results[0]
    if type(native.get("success")) is not bool:
        raise ConfigurationError("native episode has no terminal success/failure outcome")
    if type(native.get("step")) is not int or native["step"] != observed.executed_steps:
        raise ConfigurationError("native termination step differs from recorded executed steps")
    if observed.reset_count != 2 or observed.executed_steps < 1 or not client.seeds:
        raise ConfigurationError("incomplete episode or missing native reset/inference evidence")
    return {
        "status": "completed", "success": native["success"],
        "executed_steps": observed.executed_steps, "reset_count": observed.reset_count,
        "generated_chunks": len(client.seeds), "model_seeds": client.seeds,
        "action_trace_sha256": observed.action_digest.hexdigest(),
        "native_outcome": native,
    }


def _source_identity(root: Path) -> dict:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != ROBOLAB_REVISION:
        raise ConfigurationError("RoboLab source revision differs from the pinned protocol")
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    )
    if changed.strip():
        raise ConfigurationError("tracked RoboLab sources are modified")
    return {"revision": revision, "driver_sha256": sha256_file(Path(__file__)),
            "asset_driver_sha256": sha256_file(Path(robolab_assets.__file__))}


def installed_simulator_sources() -> dict:
    """Hash imported Isaac/Warp trees and the actual Warp runtime library targets.

    Warp can come from an Isaac extension, an installed wheel, or an explicit
    compatibility overlay. Its imported modules and loaded library handles are
    bound separately so an external context.py or native library is not mistaken
    for the unused bundled copy. This function never imports or initializes Warp.
    """
    rows = []
    roots = {}
    packages = ["isaaclab", "isaacsim"]
    if "warp" in sys.modules:
        packages.append("warp")
    for name in packages:
        module = sys.modules.get(name)
        if module is None or not getattr(module, "__file__", None):
            raise ConfigurationError(f"actual imported {name} package location is unavailable")
        root = Path(module.__file__).resolve(strict=True).parent
        roots[name] = str(root)
        files = sorted(path for path in root.rglob("*")
                       if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
        if not files:
            raise ConfigurationError(f"empty actual {name} package source inventory")
        for path in files:
            rows.append({"package": name, "relative": str(path.relative_to(root)),
                         "sha256": sha256_file(path), "bytes": path.stat().st_size})
    result = {"roots": roots, "files": rows, "excluded": ["__pycache__", "*.pyc"]}
    if "warp" in roots:
        imported, libraries = [], []
        for name, module in sorted(list(sys.modules.items())):
            if name != "warp" and not name.startswith("warp."):
                continue
            filename = getattr(module, "__file__", None)
            if filename is not None:
                imported.append({"module": name, **_local_artifact(filename)})
            if name not in {"warp.context", "warp._src.context"}:
                continue
            runtime = getattr(module, "runtime", None)
            if runtime is None:
                continue
            for role in ("core", "llvm"):
                library = getattr(runtime, role, None)
                if library is None:
                    if role == "core":
                        raise ConfigurationError("initialized Warp runtime has no core library handle")
                    continue
                filename = getattr(library, "_name", None)
                if isinstance(filename, bytes):
                    filename = os.fsdecode(filename)
                if not isinstance(filename, (str, Path)) or not Path(filename).is_absolute():
                    raise ConfigurationError("actual loaded Warp library has no absolute file target")
                libraries.append({"module": name, "role": role, **_local_artifact(filename)})
        result["warp_imported_modules"] = imported
        result["warp_runtime_libraries"] = libraries
    return result


def recheck_simulator_sources(inventory: dict) -> None:
    actual = installed_simulator_sources()
    old_modules = {row["module"]: row for row in inventory.get("warp_imported_modules", [])}
    new_modules = {row["module"]: row for row in actual.get("warp_imported_modules", [])}
    # Lazy imports from an already bound complete tree introduce no new source
    # bytes. Existing import targets and any external overlays must stay fixed.
    unchanged = (
        {key: value for key, value in actual.items() if key != "warp_imported_modules"}
        == {key: value for key, value in inventory.items() if key != "warp_imported_modules"}
        and all(new_modules.get(name) == row for name, row in old_modules.items())
    )
    covered = {
        (str((Path(inventory["roots"][row["package"]]) / row["relative"]).resolve()),
         row["sha256"], row["bytes"])
        for row in inventory["files"] if row["package"] == "warp"
    }
    unchanged &= all(
        (row["resolved_path"], row["sha256"], row["bytes"]) in covered
        for name, row in new_modules.items() if name not in old_modules
    )
    if not unchanged:
        raise ConfigurationError("actual Isaac/Warp package or runtime files changed during this episode")


RENDERER_COMPAT_ENV = {
    "LD_LIBRARY_PATH", "VK_LAYER_PATH", "VK_INSTANCE_LAYERS",
    "VK_LAYER_SETTINGS_PATH", "CUDA_VISIBLE_DEVICES",
}


def _local_artifact(filename: str | Path) -> dict:
    path = Path(filename).absolute()
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ConfigurationError(f"bound renderer artifact is not a local file: {path}")
    return {"path": str(path), "resolved_path": str(resolved),
            "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _renderer_runtime_expectations(bindings: Any, files: dict) -> dict:
    required = {"warp_module", "warp_core", "vulkan_layer_libraries"}
    if (not isinstance(bindings, dict) or not required.issubset(bindings)
            or set(bindings) - required - {"warp_llvm"}):
        raise ConfigurationError("renderer runtime bindings require Warp module/core and Vulkan libraries")
    libraries = bindings["vulkan_layer_libraries"]
    if (not isinstance(libraries, list) or not libraries
            or any(not isinstance(path, str) for path in libraries)
            or len(set(libraries)) != len(libraries)):
        raise ConfigurationError("renderer runtime Vulkan libraries require a nonempty unique path list")
    expected = [bindings["warp_module"], bindings["warp_core"], *libraries]
    if "warp_llvm" in bindings:
        expected.append(bindings["warp_llvm"])
    if any(not isinstance(path, str) or not Path(path).is_absolute() or path not in files
           for path in expected):
        raise ConfigurationError("every renderer runtime target must have an explicit absolute file hash")
    return dict(bindings, vulkan_layer_libraries=list(libraries))


def verify_renderer_compat_manifest(path: Path, *, environment=None) -> dict:
    """Bind an explicit compatibility configuration without applying any changes.

    Schema 1 contains ``files: {absolute_path: sha256}`` and ``environment``.
    The five required environment entries may be exact strings or null (unset);
    CUDA_VISIBLE_DEVICES must be null. Additional declared environment entries
    are also checked exactly. Supplied artifact names and values are not inferred.
    This attests declared files/environment, not complete Vulkan execution coverage.
    Optional runtime_bindings additionally declare the Warp package/core/LLVM and
    mapped Vulkan library targets. This early function binds those expectations;
    verify_renderer_runtime_bindings checks their actual use after native startup.
    """
    path = Path(path).absolute()
    before = _local_artifact(path)
    manifest = load_json(path)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ConfigurationError("unsupported renderer compatibility manifest schema")
    files, declared = manifest.get("files"), manifest.get("environment")
    if not isinstance(files, dict) or not files:
        raise ConfigurationError("renderer compatibility manifest requires explicit local files")
    if not isinstance(declared, dict) or not RENDERER_COMPAT_ENV.issubset(declared):
        raise ConfigurationError("renderer compatibility manifest omits required environment entries")
    if any(not isinstance(key, str) or not key or "=" in key or "\x00" in key
           or (value is not None and (not isinstance(value, str) or "\x00" in value))
           for key, value in declared.items()):
        raise ConfigurationError("renderer compatibility environment must declare exact strings or null")
    if declared["CUDA_VISIBLE_DEVICES"] is not None:
        raise ConfigurationError("renderer compatibility must not override CUDA_VISIBLE_DEVICES")
    actual_environment = os.environ if environment is None else environment
    observed = {key: actual_environment.get(key) for key in sorted(declared)}
    if observed != declared:
        raise ConfigurationError("renderer compatibility environment differs from its explicit manifest")
    receipts = []
    for filename, digest in sorted(files.items()):
        if (not isinstance(filename, str) or not Path(filename).is_absolute()
                or not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ConfigurationError("renderer compatibility files require absolute paths and SHA256")
        artifact = _local_artifact(filename)
        if artifact["sha256"] != digest:
            raise ConfigurationError(f"renderer compatibility artifact changed: {filename}")
        receipts.append(artifact)
    if _local_artifact(path) != before:
        raise ConfigurationError("renderer compatibility manifest changed while verifying it")
    result = {"schema_version": 1, "manifest": before, "files": receipts,
              "environment": observed, "cuda_visible_devices_override": False,
              "scope": "explicit local compatibility files and declared environment; no implicit substitution"}
    if "runtime_bindings" in manifest:
        result["runtime_bindings"] = _renderer_runtime_expectations(manifest["runtime_bindings"], files)
    return result


def recheck_renderer_compatibility(receipt: dict, *, environment=None) -> dict:
    actual = verify_renderer_compat_manifest(Path(receipt["manifest"]["path"]), environment=environment)
    if actual != receipt:
        raise ConfigurationError("renderer compatibility manifest, targets or environment changed during this episode")
    return actual


def _mapped_executable_libraries() -> list[dict]:
    """Read this process's existing executable file mappings; load nothing."""
    rows = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or "x" not in fields[1]:
            continue
        filename = fields[5]
        if not filename.startswith("/") or filename.endswith(" (deleted)"):
            continue
        path = Path(filename)
        if not path.is_file():
            continue
        major, minor = fields[3].split(":")
        rows.append({"path": filename, "resolved_path": str(path.resolve(strict=True)),
                     "device_major": int(major, 16), "device_minor": int(minor, 16),
                     "inode": int(fields[4])})
    return rows


def verify_renderer_runtime_bindings(receipt: dict, *, environment=None) -> dict | None:
    """Prove declared compatibility targets are selected, without extra imports.

    Warp must already have an active core handle. Vulkan libraries must have
    executable mappings for the exact real files/inodes bound in the declaration.
    This proves selection/loading, not calls through every exported entry point.
    """
    recheck_renderer_compatibility(receipt, environment=environment)
    expected = receipt.get("runtime_bindings")
    if expected is None:
        return None
    warp = sys.modules.get("warp")
    filename = vars(warp).get("__file__") if warp is not None else None
    if filename is None:
        raise ConfigurationError("declared Warp package is not actually imported")
    actual_module = _local_artifact(filename)
    if actual_module["resolved_path"] != str(Path(expected["warp_module"]).resolve(strict=True)):
        raise ConfigurationError("actual imported Warp package differs from the declared runtime target")
    runtime_rows = []
    for name in ("warp.context", "warp._src.context"):
        module = sys.modules.get(name)
        runtime = vars(module).get("runtime") if module is not None else None
        if runtime is None:
            continue
        for role in ("core", "llvm"):
            if role == "llvm" and "warp_llvm" not in expected:
                continue
            library = vars(runtime).get(role)
            filename = vars(library).get("_name") if library is not None else None
            if isinstance(filename, bytes):
                filename = os.fsdecode(filename)
            if not isinstance(filename, (str, Path)) or not Path(filename).is_absolute():
                raise ConfigurationError(f"actual Warp {role} library is not active with an absolute target")
            artifact = _local_artifact(filename)
            target = Path(expected[f"warp_{role}"]).resolve(strict=True)
            if artifact["resolved_path"] != str(target):
                raise ConfigurationError(f"actual Warp {role} library differs from the declared runtime target")
            runtime_rows.append({"module": name, "role": role, **artifact})
    if not any(row["role"] == "core" for row in runtime_rows):
        raise ConfigurationError("declared Warp runtime has no active native core")
    mappings = _mapped_executable_libraries()
    native_rows = []
    for filename in expected["vulkan_layer_libraries"]:
        target = Path(filename).resolve(strict=True)
        stat = target.stat()
        matches = [row for row in mappings if row["resolved_path"] == str(target)
                   and row["inode"] == stat.st_ino
                   and row["device_major"] == os.major(stat.st_dev)
                   and row["device_minor"] == os.minor(stat.st_dev)]
        if not matches:
            raise ConfigurationError(f"declared Vulkan layer library is not actually mapped: {filename}")
        native_rows.append({"expected_path": filename, "resolved_path": str(target),
                            "mapped_paths": sorted({row["path"] for row in matches})})
    return {"schema_version": 1, "runtime_bindings": expected, "warp_module": actual_module,
            "warp_runtime_libraries": runtime_rows, "vulkan_layer_libraries": native_rows,
            "scope": "existing imports, active native handles and executable process mappings; no runtime mutation"}


def _simulator_fingerprint(
    source: dict, simulator_sources: dict, renderer_compatibility: dict | None = None,
) -> dict:
    import platform

    import torch

    if platform.machine() != "x86_64":
        raise ConfigurationError("this pinned RoboLab/Isaac renderer requires qualified x86_64 hardware")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if "RTX" not in properties.name or properties.total_memory < 16_000_000_000:
        raise ConfigurationError("renderer requires a qualified RTX GPU with at least 16 GB memory")
    result = {
        "source": source,
        "installed_simulator_source_sha256": sha256_json(simulator_sources),
        "packages": sorted(f"{dist.metadata['Name']}=={dist.version}"
                           for dist in importlib.metadata.distributions()),
        "renderer_gpu": properties.name,
        "torch_cuda": torch.version.cuda,
        "renderer": "realtime",
        "rendering_mode": "native_default",
    }
    if renderer_compatibility is not None:
        result["renderer_compatibility_sha256"] = sha256_json(renderer_compatibility)
    return result


def _scene_configuration(env_cfg: Any, output: Path, inventory: dict) -> dict:
    import omni.usd
    from pxr import Ar, Sdf, Usd

    stage = omni.usd.get_context().get_stage()
    # These native reads attest the renderer's original resource URLs. The
    # frozen local copies are evidence only; no USD or material is redirected.
    remote_receipts = verify_remote_assets(inventory)
    layers = []
    used_layers = stage.GetUsedLayers()
    anonymous = {}
    for layer in used_layers:
        if not layer.anonymous:
            continue
        role = ("root" if layer == stage.GetRootLayer() else
                "session" if layer == stage.GetSessionLayer() else layer.GetDisplayName())
        if role in anonymous.values():
            raise ConfigurationError("ambiguous anonymous USD layer identity")
        anonymous[layer.identifier] = role
    anonymous_layers = []
    for layer in used_layers:
        if layer.realPath:
            layers.append(verified_asset_file(layer.realPath, inventory))
        elif not layer.anonymous:
            raise ConfigurationError("loaded nonanonymous USD layer has no hashable local file")
        else:
            text = layer.ExportToString().replace(str(output.resolve()), "<episode-output>")
            for identifier, role in sorted(anonymous.items(), key=lambda item: -len(item[0])):
                text = text.replace(identifier, "anon:<" + role + ">")
            anonymous_layers.append({"role": anonymous[layer.identifier], "content": text})
    if not layers:
        raise ConfigurationError("no resolved USD asset inventory captured")
    asset_attributes = []

    def resolve_asset(value, attribute):
        if isinstance(value, Sdf.AssetPath):
            if not value.path:
                return
            # The plain USD resolver can normalize an unsupported HTTP URL
            # into a malformed filesystem path. Keep its exact authored URL;
            # native client reads above bind the remote bytes independently.
            resolved = value.path if remote_url(value.path) else value.resolvedPath
            if not resolved:
                stack = attribute.GetPropertyStack()
                if not stack:
                    raise ConfigurationError("USD asset attribute has no authored layer")
                anchored = Sdf.ComputeAssetPathRelativeToLayer(stack[0].layer, value.path)
                resolved = str(Ar.GetResolver().Resolve(anchored))
                # HTTP assets may remain URI-valued under the native resolver.
                # Explicit URL binding plus native byte reads are required.
                if not resolved and remote_url(anchored):
                    resolved = anchored
            if not resolved:
                raise ConfigurationError(f"unresolved USD asset dependency: {value.path}")
            file = verified_asset_file(resolved, inventory)
            asset_attributes.append({"attribute": str(attribute.GetPath()), "file": file,
                                     "authored_asset": value.path, "resolved_asset": resolved})
        elif value is not None and type(value).__name__ == "AssetPathArray":
            for item in value:
                resolve_asset(item, attribute)

    # Materials inside instance prototypes also carry texture/module assets.
    # Traversing only the non-instance scene could omit those dependencies.
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        for attribute in prim.GetAttributes():
            if str(attribute.GetTypeName()) not in ("asset", "asset[]"):
                continue
            resolve_asset(attribute.Get(), attribute)
            for sample_time in attribute.GetTimeSamples():
                resolve_asset(attribute.Get(sample_time), attribute)
    return {
        "environment_config": freeze_value(env_cfg, output_root=str(output.resolve())),
        "loaded_usd_layers": sorted(layers, key=lambda row: row["path"]),
        "anonymous_usd_layers": sorted(anonymous_layers, key=lambda row: row["role"]),
        "resolved_asset_attributes": sorted(asset_attributes, key=lambda row: (row["attribute"], row["file"]["path"])),
        "remote_asset_receipts": remote_receipts,
    }


def canonicalize_render_product_scene(scene_config: dict, *, pairing_mode: str, stage) -> tuple[dict, dict]:
    """Canonicalize only six verified session picking handles in a detached copy.

    No live USD attribute or renderer setting is written. Every original scene
    value remains in the raw evidence. A separate receipt records each exact
    property path, native value and character span, allowing independent replay
    without interpreting any other field as an identity or numerical tolerance.
    """
    if pairing_mode != NATIVE_RENDERER_PAIRING_MODE:
        raise ConfigurationError("render-product identity canonicalization requires explicit native pairing mode")
    from pxr import Sdf

    anonymous = scene_config.get("anonymous_usd_layers")
    if not isinstance(anonymous, list):
        raise ConfigurationError("native scene has no anonymous layer evidence")
    sessions = [(index, row) for index, row in enumerate(anonymous) if row.get("role") == "session"]
    if len(sessions) != 1 or not isinstance(sessions[0][1].get("content"), str):
        raise ConfigurationError("native scene requires one complete session layer")
    session_index, session_row = sessions[0]
    raw_text = session_row["content"]
    session = stage.GetSessionLayer()
    if session is None or not session.anonymous:
        raise ConfigurationError("native picking handles must belong to the live anonymous session layer")
    detached = Sdf.Layer.CreateAnonymous("ifl_identity_review.usda")
    if not detached.ImportFromString(raw_text) or detached.ExportToString() != raw_text:
        raise ConfigurationError("session layer cannot roundtrip without changing other text")
    rows = []
    for attribute_path, alias in RENDER_PRODUCT_IDENTITY_ALIASES.items():
        path = Sdf.Path(attribute_path)
        prim_path = path.GetPrimPath()
        prim = stage.GetPrimAtPath(prim_path)
        if (not prim or prim.GetTypeName() != "RenderProduct" or not prim.IsActive()
                or not prim.IsDefined() or not prim.IsLoaded() or prim.IsInstanceProxy()):
            raise ConfigurationError(f"picking handle is not on a live RenderProduct: {attribute_path}")
        attribute = prim.GetAttribute("viewPickingId")
        native_spec = session.GetAttributeAtPath(path)
        copied_spec = detached.GetAttributeAtPath(path)
        if (not attribute or not attribute.IsCustom() or str(attribute.GetTypeName()) != "uint64"
                or native_spec is None or copied_spec is None
                or not native_spec.custom or not copied_spec.custom
                or str(native_spec.typeName) != "uint64" or str(copied_spec.typeName) != "uint64"
                or attribute.GetTimeSamples() or attribute.GetConnections()
                or session.ListTimeSamplesForPath(path) or detached.ListTimeSamplesForPath(path)):
            raise ConfigurationError(f"picking identity is not a plain session custom uint64: {attribute_path}")
        stack = attribute.GetPropertyStack()
        if not stack or stack[0].layer != session or stack[0].path != path:
            raise ConfigurationError(f"picking identity is not supplied by the live session: {attribute_path}")
        value = attribute.Get()
        if (type(value) is not int or not 0 <= value < 2**64
                or native_spec.default != value or copied_spec.default != value):
            raise ConfigurationError(f"saved picking identity differs from its live native value: {attribute_path}")
        rows.append({"attribute_path": attribute_path, "prim_path": str(prim_path),
                     "prim_type": "RenderProduct", "type_name": "uint64", "custom": True,
                     "layer_role": "session", "raw_value": value, "alias": alias})
    by_value = {row["raw_value"]: row for row in rows}
    if len(by_value) != len(rows):
        raise ConfigurationError("native picking identities must be distinct for the declared products")
    matches = list(re.finditer(r"(?m)^[ \t]+custom uint64 viewPickingId = ([0-9]+)[ \t]*$", raw_text))
    if len(matches) != len(rows) or {int(match.group(1)) for match in matches} != set(by_value):
        raise ConfigurationError("session has an additional or unrecognized picking identity declaration")
    substitutions = []
    for match in matches:
        row = by_value[int(match.group(1))]
        substitutions.append({"attribute_path": row["attribute_path"], "start": match.start(1),
                              "end": match.end(1), "raw_decimal": match.group(1),
                              "alias_decimal": str(row["alias"])})
    canonical_text = raw_text
    for substitution in reversed(substitutions):
        canonical_text = (canonical_text[:substitution["start"]] + substitution["alias_decimal"]
                          + canonical_text[substitution["end"]:])
    # Independently confirm that these exact digit edits represent only the
    # declared USD properties, not another occurrence of the same integer.
    for row in rows:
        detached.GetAttributeAtPath(row["attribute_path"]).default = row["alias"]
    if detached.ExportToString() != canonical_text:
        raise ConfigurationError("picking canonicalization changed a non-identity USD field or text")
    canonical = copy.deepcopy(scene_config)
    canonical["anonymous_usd_layers"][session_index]["content"] = canonical_text
    identity = {
        "schema_version": 1, "pairing_mode": pairing_mode,
        "raw_scene_config_sha256": sha256_json(scene_config),
        "scene_config_sha256": sha256_json(canonical),
        "aliases": dict(RENDER_PRODUCT_IDENTITY_ALIASES), "rows": rows,
        "session_raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "session_canonical_text_sha256": hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
        "substitutions": substitutions,
    }
    return canonical, identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--family", choices=("edge", "nano"), required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--renderer-compat-manifest", type=Path,
                        help="Optional exact local compatibility files/environment gate; applies no patches")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--rpc-timeout", type=float, default=120)
    # Parse without importing Isaac. Startup failures also get an immutable record.
    args, launcher_args = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"status": "failed", "success": None, "family": args.family, "arm": args.arm,
              "pair_id": args.pair_id, "task_quality_validated": False}
    app = env = observed = client = remote = None
    compatibility = None
    try:
        from .remote_policy import RemotePolicy
        from .robolab_protocol import CHECKPOINTS, validate_protocol

        protocol = load_json(args.protocol)
        protocol_sha = validate_protocol(protocol)
        pairing_mode = protocol.get("contract", {}).get("pairing_mode")
        if pairing_mode is not None:
            if protocol["schema_version"] != 3 or pairing_mode != NATIVE_RENDERER_PAIRING_MODE:
                raise ConfigurationError("native renderer pairing requires explicit prospective schema 3")
            record["pairing_mode"] = pairing_mode
            if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization != 0:
                raise ConfigurationError("native renderer interpreter must start with PYTHONHASHSEED=0")
            record["python_hash_interpreter_startup"] = {
                "environment": "0", "hash_randomization": sys.flags.hash_randomization,
            }
        episodes = [row for row in protocol["episodes"] if row["pair_id"] == args.pair_id]
        if len(episodes) != 1:
            raise ConfigurationError("pair_id must identify exactly one prospective episode")
        episode = dict(episodes[0])
        record.update({key: episode[key] for key in ("task_id", "episode_index", "scene_seed")})
        record["protocol_sha256"] = protocol_sha
        identity = load_json(args.identity)
        identity = identity.get("benchmark_identity", identity)
        identity_sha = sha256_json(identity)
        expected_cell = args.family + ("-eager_native" if args.arm == "baseline" else "-runtime_selected")
        expected_checkpoint = CHECKPOINTS[args.family]
        if (identity.get("cell_id") != expected_cell
                or identity.get("model_id") != expected_checkpoint["model_id"]
                or identity.get("revision") != expected_checkpoint["revision"]):
            raise ConfigurationError("server identity belongs to a different family, arm, or checkpoint")
        if protocol["execution_bindings"][args.family][args.arm] != identity_sha:
            raise ConfigurationError("exact server identity must be prospectively bound in protocol")
        if protocol["stage"] == "formal" and protocol["acceptance"]["mode"] == "pending":
            raise ConfigurationError("formal capture requires prospective certification or report-only scope")
        record["execution_binding_sha256"] = identity_sha
        record["robolab_revision"] = ROBOLAB_REVISION
        record["checkpoint_revision"] = CHECKPOINTS[args.family]["revision"]
        if (args.family, args.arm) != ("edge", "baseline") and not args.anchor.is_file():
            raise ConfigurationError("this arm requires Edge baseline's immutable initial anchor")
        source = _source_identity(args.robolab_root)
        assets = verify_asset_inventory(load_json(args.asset_manifest))
        _write_exclusive(args.output / "verified_asset_inventory.json", assets)
        if args.renderer_compat_manifest is not None:
            compatibility = verify_renderer_compat_manifest(args.renderer_compat_manifest)
            _write_exclusive(args.output / "renderer_compatibility_before.json", compatibility)
        sys.path.insert(0, str(args.robolab_root.resolve()))
        import cv2  # noqa: F401 -- required before importing IsaacLab.
        from isaaclab.app import AppLauncher

        launcher_parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(launcher_parser)
        launch = launcher_parser.parse_args(launcher_args)
        launch.enable_cameras = True
        launch.headless = True
        app = AppLauncher(launch, multi_gpu=False).app
        if compatibility is not None:
            runtime_binding = verify_renderer_runtime_bindings(compatibility)
            if runtime_binding is not None:
                _write_exclusive(args.output / "renderer_runtime_after_launcher.json", runtime_binding)

        import numpy as np
        import torch
        from policies.cosmos3.client import Cosmos3Client
        from robolab.constants import set_output_dir
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.eval.episode import run_episode
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

        random.seed(episode["scene_seed"])
        np.random.seed(episode["scene_seed"])
        torch.manual_seed(episode["scene_seed"])
        torch.cuda.manual_seed_all(episode["scene_seed"])
        set_output_dir(str(args.output.resolve()))
        auto_register_droid_envs(task=[episode["task_id"]], cameras=WRIST_LEFT_RIGHT_HEAD)
        environments = get_envs(task=[episode["task_id"]])
        if len(environments) != 1:
            raise ConfigurationError("task registration must resolve to exactly one frozen environment")
        env, cfg = create_env(
            environments[0], device=launch.device, seed=episode["scene_seed"], num_envs=1,
            instruction_type="default", policy="cosmos3", renderer="realtime", rendering_mode=None,
        )
        if pairing_mode is not None:
            if (os.environ.get("PYTHONHASHSEED") != str(episode["scene_seed"])
                    or sys.flags.hash_randomization != 0):
                raise ConfigurationError("native seed lifecycle differs from the prospective scene seed")
            record["native_seed_environment"] = {
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                "interpreter_hash_randomization": sys.flags.hash_randomization,
            }
        if env.num_envs != 1 or env.max_episode_length != episode["max_episode_steps"]:
            raise ConfigurationError("native environment count or episode time limit changed")
        inventory_rows = [row for row in protocol["inventory"]["tasks"]
                          if row["task_id"] == episode["task_id"]]
        if len(inventory_rows) != 1 or cfg.instruction != inventory_rows[0]["instruction_default"]:
            raise ConfigurationError("native instruction differs from the frozen task inventory")
        episode["instruction"] = cfg.instruction
        simulator_sources = installed_simulator_sources()
        _write_exclusive(args.output / "simulator_source_inventory.json", simulator_sources)
        if compatibility is not None:
            recheck_renderer_compatibility(compatibility)
        fingerprint = _simulator_fingerprint(source, simulator_sources, compatibility)
        scene_config = _scene_configuration(cfg, args.output, assets)
        _write_exclusive(args.output / "simulator_fingerprint.json", fingerprint)
        _write_exclusive(args.output / "scene_config.json", scene_config)

        def on_initial_state(native_env, observation):
            # Reset can resolve additional assets. Bind what was actually loaded
            # after the native second reset, immediately before first inference.
            actual_scene_config = _scene_configuration(cfg, args.output, assets)
            if pairing_mode is not None:
                import omni.usd

                _write_exclusive(args.output / "scene_config_after_reset_raw.json", actual_scene_config)
                actual_scene_config, render_identity = canonicalize_render_product_scene(
                    actual_scene_config, pairing_mode=pairing_mode, stage=omni.usd.get_context().get_stage(),
                )
                _write_exclusive(args.output / "render_product_identity.json", render_identity)
                record["raw_scene_config_sha256"] = render_identity["raw_scene_config_sha256"]
                record["render_product_identity_sha256"] = sha256_json(render_identity)
            if compatibility is not None:
                runtime_binding = verify_renderer_runtime_bindings(compatibility)
                if runtime_binding is not None:
                    _write_exclusive(args.output / "renderer_runtime_before_inference.json", runtime_binding)
            _write_exclusive(args.output / "scene_config_after_reset.json", actual_scene_config)
            return bind_initial_state(
                family=args.family, arm=args.arm, anchor_path=args.anchor,
                protocol_sha256=record["protocol_sha256"],
                episode=episode, state=capture_initial_state(
                    native_env,
                    camera_names=[name for names in NATIVE_IMAGE_GROUPS.values() for name in names]
                    if pairing_mode is not None else None,
                ), observation=observation,
                simulator_fingerprint=fingerprint, scene_config=actual_scene_config,
                asset_inventory=assets, output=args.output, pairing_mode=pairing_mode,
            )

        observed = ObservedEnv(env, on_initial_state=on_initial_state,
                               max_steps=episode["max_episode_steps"], output=args.output)
        remote = RemotePolicy(args.endpoint, identity, timeout=args.rpc_timeout,
                              record_dir=args.output / "wire_trace")
        client = make_paired_client(Cosmos3Client, remote, identity=identity,
                                    episode=episode, output=args.output)
        outcomes, subtask_status, timing = run_episode(
            observed, cfg, episode["episode_index"], client, headless=True,
            save_videos=False, video_mode="none", enable_gt_state=False,
        )
        record.update(check_native_result(outcomes, observed, client))
        record.update(observed.binding)
        if pairing_mode is not None and (
            os.environ.get("PYTHONHASHSEED") != str(episode["scene_seed"])
            or sys.flags.hash_randomization != 0
        ):
            raise ConfigurationError("native seed lifecycle changed during the episode")
        recheck_simulator_sources(simulator_sources)
        if _source_identity(args.robolab_root) != source:
            raise ConfigurationError("RoboLab source files changed during this episode")
        if verify_asset_inventory(load_json(args.asset_manifest)) != assets:
            raise ConfigurationError("asset inventory changed during this episode")
        if verify_remote_assets(assets) != scene_config["remote_asset_receipts"]:
            raise ConfigurationError("native remote assets changed during this episode")
        _write_exclusive(args.output / "native_subtask_status.json", freeze_value(subtask_status))
        record["native_timing"] = timing
        record["evaluation_mode"] = "paused_simulation"
    except BaseException as error:
        record.update(status="failed", success=None, error_type=type(error).__name__, error=str(error))
        record["traceback"] = traceback.format_exc()
        if observed is not None:
            record.update(reset_count=observed.reset_count, executed_steps=observed.executed_steps,
                          attempted_steps=observed.attempted_steps)
        raise
    finally:
        if compatibility is not None:
            record["renderer_compatibility_environment_after"] = {
                key: os.environ.get(key) for key in compatibility["environment"]
            }
            try:
                after = recheck_renderer_compatibility(compatibility)
                _write_exclusive(args.output / "renderer_compatibility_after.json", after)
                runtime_binding = verify_renderer_runtime_bindings(compatibility)
                if runtime_binding is not None:
                    _write_exclusive(args.output / "renderer_runtime_after.json", runtime_binding)
                record["renderer_compatibility_sha256"] = sha256_json(after)
            except BaseException as error:
                record["renderer_compatibility_error"] = f"{type(error).__name__}: {error}"
                record.update(status="failed", success=None)
        finalize_episode_resources(record, args.output, remote=remote, env=env, app=app)
        if record["status"] != "completed" and sys.exc_info()[0] is None:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Finite eight-episode Cosmos smoke controller; no formal follow-on or retries.

The same file supplies a stdlib-only remote job wrapper. Copy its exact bytes to
the explicitly configured Thor/RTX runner paths before invoking the controller.
No host, qualified renderer, asset set, identity or scientific protocol is inferred.
"""

from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from urllib.parse import urlparse


CELLS = ("edge-eager_native", "edge-runtime_selected", "nano-eager_native", "nano-runtime_selected")
ANCHOR_FIELDS = ("pair_id", "task_id", "scene_seed", "protocol_sha256", "initial_state_sha256",
                 "initial_observation_sha256", "simulator_fingerprint_sha256",
                 "scene_config_sha256", "asset_inventory_sha256")
PHYSICAL_PAIRING_MODE = "native_renderer_physical_v1"
PHYSICAL_ANCHOR_FIELDS = tuple(name for name in ANCHOR_FIELDS if name != "initial_observation_sha256") + (
    "pairing_mode", "initial_nonvisual_observation_sha256", "initial_image_schema_sha256",
)
RENDER_PRODUCT_PATHS = tuple("/Render/OmniverseKit/HydraTextures/" + name for name in (
    "Replicator", "Replicator_01", "Replicator_02", "Replicator_03", "Replicator_04",
    "omni_kit_widget_viewport_ViewportTexture_0",
))
NATIVE_IMAGE_GROUPS = {
    "image_obs": {"head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam"},
    "viewport_cam": {"egocentric_mirrored_camera"},
}


def anchor_fields(protocol):
    """Only the prospectively explicit v3 mode separates native image bytes."""
    schema, mode = protocol.get("schema_version"), protocol.get("contract", {}).get("pairing_mode")
    if schema == 2 and mode is None:
        return ANCHOR_FIELDS
    require(schema == 3 and mode == PHYSICAL_PAIRING_MODE, "unknown native observation pairing mode")
    return PHYSICAL_ANCHOR_FIELDS


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def driver_digest(value):
    """Driver artifacts use ASCII JSON escaping; protocol manifests use UTF-8."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if exclusive:
        # Publish complete JSON atomically while preserving write-once semantics.
        # Remote completion receipts are polled by another process.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(text)
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink()
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def process_identity(pid):
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text()
    except FileNotFoundError:
        return None
    fields = text.rsplit(")", 1)[1].split()
    return {"pid": int(pid), "start_ticks": fields[19], "state": fields[0],
            "parent_pid": int(fields[1]), "group_id": int(fields[2]), "session_id": int(fields[3])}


def same_process(identity):
    current = process_identity(identity["pid"])
    return current is not None and current["start_ticks"] == identity["start_ticks"] and current["state"] != "Z"


def verify_files(files):
    require(isinstance(files, dict) and files, "explicit bound files are required")
    for filename, digest in files.items():
        require(Path(filename).is_absolute() and sha(filename) == digest, f"bound file differs: {filename}")


def become_subreaper():
    """Keep orphaned native descendants owned by this finite Linux wrapper."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER, this process only.
        raise OSError(ctypes.get_errno(), "cannot establish native child supervision")


def remote_spawn(spec):
    require(spec["runner_sha256"] == sha(__file__), "remote wrapper source differs")
    verify_files(spec["bound_files"])
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=False)
    write(output / "spec.json", spec, exclusive=True)
    with (output / "launcher.log").open("x") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--remote-job",
                                  str(output / "spec.json"), "--spec-sha256", sha(output / "spec.json")], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(child.pid)
    require(identity is not None, "remote wrapper exited before its PID could be bound")
    result = {"process": identity, "job_id": spec["job_id"], "runner_sha256": sha(__file__),
              "spec_sha256": sha(output / "spec.json"), "output": str(output)}
    write(output / "launch.json", result, exclusive=True)
    return result


def remote_job(spec_path, expected_spec_sha256):
    require(sha(spec_path) == expected_spec_sha256, "remote job specification differs before execution")
    spec = read(spec_path)
    output = Path(spec["output"])
    state = {"status": "starting", "job_id": spec["job_id"], "wrapper": process_identity(os.getpid()),
             "started": time.time(), "automatic_retry": False, "task_quality_validated": False}
    stop = {"reason": None}
    child = lock = None
    owned = {}

    def requested(number, _frame):
        stop["reason"] = signal.Signals(number).name

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, requested)
    write(output / "live_status.json", state)

    def owned_live():
        # Subreaping also catches descendants that create another session and
        # outlive the Vulkan wrapper or the native Thor supervisor.
        current = {}
        for directory in Path("/proc").iterdir():
            if directory.name.isdigit():
                try:
                    identity = process_identity(int(directory.name))
                except (OSError, ProcessLookupError):
                    continue
                if identity is not None:
                    current[identity["pid"]] = identity
        pending = True
        while pending:
            pending = False
            parents = {os.getpid()} | {
                pid for pid, identity in owned.items()
                if pid in current and identity["start_ticks"] == current[pid]["start_ticks"]
            }
            for pid, identity in current.items():
                if pid not in owned and identity["parent_pid"] in parents:
                    owned[pid] = identity
                    pending = True
        return [identity for pid, identity in owned.items() if pid in current
                and current[pid]["start_ticks"] == identity["start_ticks"] and current[pid]["state"] != "Z"]

    termination = {"deadline": None, "signaled": set()}

    def terminate():
        live = owned_live()
        if not live:
            if child is not None:
                child.wait()
            return False
        if termination["deadline"] is None:
            termination["deadline"] = time.monotonic() + spec["grace_seconds"]

        def request_once(identity, scope):
            key = identity["pid"], identity["start_ticks"]
            if key not in termination["signaled"] and same_process(identity):
                try:
                    os.kill(identity["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    return
                termination["signaled"].add(key)
                state.setdefault("termination_events", []).append({
                    "signal": "SIGTERM", "scope": scope, "pid": identity["pid"],
                    "start_ticks": identity["start_ticks"], "time": time.time(),
                })

        while live:
            # The direct child may itself supervise a worker in another process
            # group. Let it own graceful worker shutdown and teardown first.
            # Signaling both levels races a second SIGTERM against interpreter
            # teardown after the worker has restored its default signal handler.
            leader = state.get("child")
            leader_live = child is not None and child.poll() is None and leader is not None and same_process(leader)
            if leader_live:
                request_once(leader, "direct_supervisor")
            else:
                for identity in live:
                    request_once(identity, "survivor_after_supervisor_exit")
            if time.monotonic() >= termination["deadline"]:
                for identity in live:
                    if same_process(identity):
                        try:
                            os.kill(identity["pid"], signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                if child is not None:
                    child.wait(timeout=10)
                for _ in range(100):
                    if not owned_live():
                        return True
                    time.sleep(.05)
                raise RuntimeError("owned native descendants did not exit after forced cleanup")
            time.sleep(.05)
            live = owned_live()
        if child is not None:
            child.wait()
        return False

    try:
        require(spec["runner_sha256"] == sha(__file__), "remote wrapper changed before launch")
        verify_files(spec["bound_files"])
        become_subreaper()
        if spec.get("gpu_lock") is not None:
            require(spec["gpu_index"] == 0, "this bounded renderer study owns GPU0 only")
            lock = open(spec["gpu_lock"], "a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            occupied = subprocess.check_output([
                spec["nvidia_smi"], "--id=0", "--query-compute-apps=pid",
                "--format=csv,noheader,nounits"], text=True).strip()
            require(not occupied, f"renderer GPU0 has competing compute processes: {occupied}")
        require(stop["reason"] is None, "remote wrapper stopped before child launch")
        environment = dict(os.environ)
        for key, value in spec["environment"].items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        with (output / "worker.log").open("x") as log:
            child = subprocess.Popen(spec["command"], env=environment, cwd=spec["cwd"],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            state.update(status="running", child=process_identity(child.pid), command=spec["command"])
            owned[child.pid] = state["child"]
            write(output / "live_status.json", state)
            deadline = time.monotonic() + spec["max_seconds"]
            while child.poll() is None:
                owned_live()
                if stop["reason"] or time.monotonic() >= deadline:
                    state["stop_reason"] = stop["reason"] or "deadline"
                    state["forced_kill"] = terminate()
                    break
                try:
                    child.wait(timeout=min(1, max(0.01, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
        state["exit_code"] = child.returncode
        require(not owned_live(), "native wrapper leader exited while owned descendants remained active")
        verify_files(spec["bound_files"])
        require(sha(spec_path) == expected_spec_sha256, "remote job specification changed during execution")
        require(sha(__file__) == spec["runner_sha256"], "remote wrapper source changed during execution")
        require(child.returncode == 0 and not state.get("forced_kill")
                and state.get("stop_reason") != "deadline", "remote child failed or exceeded its lifetime")
        state["status"] = "stopped" if state.get("stop_reason") else "passed"
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if owned_live():
            state["cleanup_forced_kill"] = terminate()
        state["owned_processes"] = list(owned.values())
        state["owned_processes_still_live"] = owned_live()
        if child is not None:
            child.poll()
        # The direct child's exit code is already collected. Reap adopted native
        # descendants without confusing subprocess.Popen's exit bookkeeping.
        while child is None or child.returncode is not None:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
        if lock is not None:
            lock.close()
        state["ended"] = time.time()
        write(output / "completion.json", state, exclusive=True)
        write(output / "live_status.json", state)


def remote_status(output):
    output = Path(output)
    completion = read(output / "completion.json") if (output / "completion.json").exists() else None
    launch = read(output / "launch.json")
    return {"completion": completion, "live": same_process(launch["process"]), "launch": launch}


def remote_stop(output):
    launch = read(Path(output) / "launch.json")
    stopped = same_process(launch["process"])
    if stopped:
        os.kill(launch["process"]["pid"], signal.SIGTERM)
    return {"signal_sent": stopped, "owned_process": launch["process"]}


class SSHTransport:
    """One attempt per SSH operation or evidence transfer; no experiment retries."""

    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.index = 0

    def call(self, host, args, *, payload=None):
        entry = self.config[host]
        command = ["ssh", *entry["ssh"]["options"], entry["ssh"]["target"],
                   shlex.join([entry["control_python"], entry["runner"], *args])]
        self.index += 1
        stem = self.output / "control" / f"{self.index:05d}-{host}"
        try:
            result = subprocess.run(command, input=json.dumps(payload) if payload is not None else None,
                                    text=True, capture_output=True,
                                    timeout=self.config["limits"]["control_seconds"])
        except subprocess.TimeoutExpired as error:
            write(stem.with_suffix(".json"), {"command": command, "error": str(error)}, exclusive=True)
            raise
        write(stem.with_suffix(".json"), {"command": command, "returncode": result.returncode,
                                          "stderr": result.stderr, "stdout": result.stdout if result.returncode else None},
              exclusive=True)
        require(result.returncode == 0, f"{host} control failed; evidence retained at {stem}")
        return json.loads(result.stdout)

    def prepare(self, host):
        return self.call(host, ["--remote-prepare", self.config[host]["output_root"]])

    def spawn(self, host, spec):
        return self.call(host, ["--remote-spawn"], payload=spec)

    def status(self, host, output):
        return self.call(host, ["--remote-status", output])

    def read_json(self, host, path):
        return self.call(host, ["--remote-read-json", path])

    def stop(self, host, output):
        return self.call(host, ["--remote-stop", output])

    def copy(self, host, source, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        entry = self.config[host]
        remote = f"{entry['ssh']['target']}:{source.rstrip('/')}/"
        command = ["rsync", "-a", "--checksum", "--protect-args", "-e",
                   shlex.join(["ssh", *entry["ssh"]["options"]]), remote, str(destination) + "/"]
        # Copy/transport failures stop the study. Retain partial copied evidence.
        with destination.with_name(destination.name + "-rsync.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=self.config["limits"]["copy_seconds"])


def load_study(config):
    from benchmarks.vla import robolab_protocol as protocol_api

    require(config["schema_version"] == 1, "unknown controller config")
    require(config.get("template_only") is not True, "fill the explicit study template after renderer qualification")
    protocol_file = config["protocol"]
    require(sha(protocol_file["local"]) == protocol_file["sha256"], "local frozen protocol changed")
    protocol = read(protocol_file["local"])
    protocol_api.validate_protocol(protocol)
    anchor_fields(protocol)
    require(protocol["stage"] == "smoke" and protocol["expected_records"] == 8
            and protocol["episodes_per_task"] == 1 and len(protocol["episodes"]) == 2
            and set(protocol["task_ids"]) == set(protocol_api.SMOKE_TASKS),
            "controller is restricted to the two fixed smoke tasks and eight episodes")
    require(set(config["identities"]) == set(CELLS), "all four pre-collected identities are required")
    identities = {}
    for cell in CELLS:
        paths = config["identities"][cell]
        require(sha(paths["local"]) == paths["sha256"], f"local identity changed: {cell}")
        metadata = read(paths["local"])
        identity = metadata["benchmark_identity"]
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        family, kind = cell.split("-", 1)
        arm = "baseline" if kind == "eager_native" else "candidate"
        require(identity["cell_id"] == cell and digest == metadata["benchmark_identity_sha256"]
                and digest == protocol["execution_bindings"][family][arm],
                f"exact identity was not prospectively bound: {cell}")
        identities[cell] = metadata
    qualification = config["renderer_qualification"]
    require(sha(qualification["local"]) == qualification["sha256"], "renderer qualification receipt changed")
    qualified = read(qualification["local"])
    require(qualified.get("renderer_qualified") is True and qualified.get("task_quality_validated") is False,
            "a separate passed renderer qualification receipt is required before smoke")
    for name in ("asset_manifest", "compat_manifest"):
        require(qualified[f"{name}_sha256"] == config["renderer"]["bound_files"][config["renderer"][name]],
                f"qualified renderer {name} differs from execution")
    for key, path in (("bootstrap_sha256", config["renderer"]["bootstrap"]),
                      ("driver_sha256", config["renderer"]["flash_root"] + "/benchmarks/vla/robolab_driver.py")):
        require(qualified[key] == config["renderer"]["bound_files"][path],
                f"qualified renderer source differs: {key}")
    limits = config["limits"]
    for key in ("server_start_seconds", "episode_seconds", "server_seconds", "grace_seconds",
                "copy_seconds", "control_seconds", "poll_seconds", "rpc_seconds"):
        require(type(limits[key]) in (int, float) and 0 < limits[key] <= 172800, f"invalid {key}")
    require(limits["server_seconds"] > limits["server_start_seconds"] + 2 * limits["episode_seconds"]
            + 4 * limits["copy_seconds"] + limits["grace_seconds"], "server lifetime cannot cover its assigned smoke work")
    require(limits["grace_seconds"] > config["thor"]["worker_grace_seconds"] + 30,
            "outer cleanup allowance must exceed the native Thor supervisor's cleanup bound")
    endpoint = urlparse(config["renderer"]["endpoint"])
    require(endpoint.scheme == "ws" and endpoint.hostname in {"127.0.0.1", "localhost"}
            and endpoint.port == config["thor"]["port"], "explicit private relay endpoint must match the server port")
    protected = {"--protocol", "--pair-id", "--family", "--arm", "--robolab-root", "--endpoint",
                 "--identity", "--asset-manifest", "--renderer-compat-manifest", "--output", "--anchor", "--device"}
    protected |= {"--selector", "--build-result", "--flash-root", "--bootstrap-receipt",
                  "--entry-script", "--episode-driver", "--rpc-timeout"}
    require(not any(item.split("=", 1)[0] in protected for item in config["renderer"]["launcher_args"]),
            "launcher arguments cannot override frozen episode fields or GPU0")
    require("CUDA_VISIBLE_DEVICES" in config["renderer"]["environment"]
            and config["renderer"]["environment"]["CUDA_VISIBLE_DEVICES"] is None,
            "renderer CUDA_VISIBLE_DEVICES must remain unset")
    require(isinstance(config["renderer"]["launch_prefix"], list)
            and all(isinstance(item, str) for item in config["renderer"]["launch_prefix"]),
            "renderer launch_prefix must be an explicit argv list")
    for path in config["renderer"]["launch_prefix_sources"]:
        require(path in config["renderer"]["bound_files"], "renderer launch prefix source must be hash-bound")
    for host in ("thor", "renderer"):
        entry = config[host]
        require(Path(entry["output_root"]).is_absolute() and Path(entry["runner"]).is_absolute(),
                "remote runner and output roots must be explicit absolute paths")
        require(entry["bound_files"].get(entry["runner"]) == sha(__file__), "exact controller source must be bound on both hosts")
        require(isinstance(entry["ssh"]["options"], list) and isinstance(entry["ssh"]["target"], str),
                "explicit SSH destination/options required")
        for cell in CELLS:
            identity_path = config["identities"][cell][host]
            require(entry["bound_files"].get(identity_path) == config["identities"][cell]["sha256"],
                    f"identity file must be remotely hash-bound: {host}/{cell}")
        required_paths = ("supervisor", "installation") if host == "thor" else (
            "bootstrap", "selector", "build_result", "asset_manifest", "compat_manifest")
        for key in required_paths:
            require(entry[key] in entry["bound_files"], f"remote execution source must be hash-bound: {host}/{key}")
    require(config["thor"]["bound_files"][config["thor"]["installation"]] == config["thor"]["installation_sha256"],
            "native supervisor and outer wrapper must bind the same installation manifest")
    require(config["renderer"]["bound_files"].get(config["protocol"]["renderer"]) == protocol_file["sha256"],
            "remote renderer protocol must be hash-bound")
    return protocol, identities, protocol_api


def spec_for(config, host, job_id, command, *, gpu_lock=None, environment=None):
    entry, limits = config[host], config["limits"]
    return {"schema_version": 1, "job_id": job_id, "output": f"{entry['output_root']}/{job_id}",
            "runner_sha256": sha(__file__), "bound_files": entry["bound_files"],
            "command": command, "cwd": entry["cwd"], "environment": environment or {},
            "max_seconds": limits["episode_seconds"] if host == "renderer" else limits["server_seconds"] + limits["grace_seconds"],
            "grace_seconds": limits["grace_seconds"], "gpu_lock": gpu_lock,
            "gpu_index": 0 if gpu_lock else None, "nvidia_smi": entry.get("nvidia_smi")}


def server_spec(config, cell):
    entry, limits = config["thor"], config["limits"]
    output = f"{entry['output_root']}/servers/{cell}"
    command = [entry["control_python"], entry["supervisor"], "--bundle", entry["bundle"],
               "--installation", entry["installation"], "--installation-sha256", entry["installation_sha256"],
               "--mode", "serve", "--cells", cell, "--port", str(entry["port"]),
               "--max-seconds", str(limits["server_seconds"]), "--grace-seconds", str(entry["worker_grace_seconds"]),
               "--expected-identity", config["identities"][cell]["thor"], "--output", output + "/supervisor"]
    return spec_for(config, "thor", f"servers/{cell}", command)


def episode_spec(config, cell, episode):
    entry, limits = config["renderer"], config["limits"]
    family, kind = cell.split("-", 1)
    arm = "baseline" if kind == "eager_native" else "candidate"
    job_id = f"episodes/{cell}/{episode['task_id']}"
    output = f"{entry['output_root']}/{job_id}"
    anchor = f"{entry['output_root']}/anchors/{episode['task_id']}.json"
    command = [entry["python"], entry["bootstrap"], "--selector", entry["selector"],
               "--build-result", entry["build_result"], "--flash-root", entry["flash_root"],
               "--bootstrap-receipt", output + "/bootstrap.json", "--episode-driver",
               "--protocol", config["protocol"]["renderer"], "--pair-id", episode["pair_id"],
               "--family", family, "--arm", arm, "--robolab-root", entry["robolab_root"],
               "--endpoint", entry["endpoint"], "--identity", config["identities"][cell]["renderer"],
               "--asset-manifest", entry["asset_manifest"], "--renderer-compat-manifest", entry["compat_manifest"],
               "--output", output + "/episode", "--anchor", anchor,
               "--rpc-timeout", str(limits["rpc_seconds"]), "--device", "cuda:0", *entry["launcher_args"]]
    command = [argument.replace("{job_output}", output) for argument in entry["launch_prefix"]] + command
    return spec_for(config, "renderer", job_id, command, gpu_lock=entry["gpu0_lock"], environment=entry["environment"]), anchor


def inventory_tree(root):
    root = Path(root)
    files, links = {}, {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            links[relative] = os.readlink(path)
        elif path.is_file():
            files[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return {"files": files, "symlinks": links}


def physical_observation_parts(observation):
    """Recompute shared fields without transforming retained native pixels."""
    groups = NATIVE_IMAGE_GROUPS
    require(isinstance(observation, dict) and all(group in observation for group in groups),
            "native observation must retain both image groups")
    schemas = {}
    for group in groups:
        images = observation[group]
        require(isinstance(images, dict) and set(images) == groups[group],
                "native observation image group differs from the declared cameras")
        schemas[group] = {}
        for camera, value in images.items():
            require(isinstance(camera, str) and camera and isinstance(value, dict)
                    and set(value) == {"dtype", "shape", "data_base64"}, "invalid frozen native image")
            shape = value["shape"]
            require(value["dtype"] == "|u1" and isinstance(shape, list) and len(shape) == 4
                    and all(type(size) is int and size > 0 for size in shape)
                    and shape[0] == 1 and shape[3] == 3, "native RGB image dtype or shape differs")
            require(isinstance(value["data_base64"], str), "native image bytes are missing")
            data = base64.b64decode(value["data_base64"], validate=True)
            require(len(data) == shape[1] * shape[2] * 3, "native image byte length differs from shape")
            require(base64.b64encode(data).decode("ascii") == value["data_base64"],
                    "native image bytes have a noncanonical representation")
            schemas[group][camera] = {"dtype": value["dtype"], "shape": list(shape)}
    return {key: value for key, value in observation.items() if key not in groups}, schemas


def validate_render_product_identity(raw_scene, scene, identity, record):
    """Replay only the driver's attested session ID digit spans, preserving all else."""
    require(isinstance(identity, dict) and type(identity.get("schema_version")) is int and identity["schema_version"] == 1
            and identity.get("pairing_mode") == PHYSICAL_PAIRING_MODE,
            "invalid native RenderProduct identity receipt")
    require(identity.get("raw_scene_config_sha256") == record["raw_scene_config_sha256"]
            and identity.get("scene_config_sha256") == record["scene_config_sha256"],
            "native RenderProduct identity belongs to another scene")
    aliases = {path + ".viewPickingId": index for index, path in enumerate(RENDER_PRODUCT_PATHS, 1)}
    require(identity.get("aliases") == aliases, "native RenderProduct aliases differ from the fixed paths")
    rows = identity.get("rows")
    require(isinstance(rows, list) and len(rows) == len(aliases), "native RenderProduct identity rows missing")
    by_path = {}
    for row in rows:
        require(isinstance(row, dict), "invalid native RenderProduct identity row")
        path = row.get("attribute_path")
        require(path in aliases and path not in by_path and row.get("prim_path") == path.removesuffix(".viewPickingId")
                and row.get("prim_type") == "RenderProduct" and row.get("type_name") == "uint64"
                and row.get("custom") is True and row.get("layer_role") == "session"
                and type(row.get("alias")) is int and row["alias"] == aliases[path] and type(row.get("raw_value")) is int
                and 0 <= row["raw_value"] < 2**64, "native RenderProduct identity row is not a permitted session ID")
        by_path[path] = row
    require(len({row["raw_value"] for row in rows}) == len(aliases), "native RenderProduct values are not distinct")
    layers = raw_scene.get("anonymous_usd_layers") if isinstance(raw_scene, dict) else None
    require(isinstance(layers, list), "raw native scene lacks anonymous layer evidence")
    session_indices = [i for i, layer in enumerate(layers) if isinstance(layer, dict) and layer.get("role") == "session"]
    require(len(session_indices) == 1, "raw native scene has missing or ambiguous session layer")
    index = session_indices[0]
    text = layers[index].get("content")
    require(isinstance(text, str) and hashlib.sha256(text.encode()).hexdigest() == identity.get("session_raw_text_sha256"),
            "raw session text differs from native RenderProduct receipt")
    substitutions = identity.get("substitutions")
    require(isinstance(substitutions, list) and len(substitutions) == len(aliases), "native ID substitutions missing")
    require(all(isinstance(change, dict) for change in substitutions), "invalid native ID substitution")
    permitted_spans = {(match.start(1), match.end(1)) for match in re.finditer(
        r"(?m)^[ \t]+custom uint64 viewPickingId = ([0-9]+)[ \t]*$", text)}
    require(len(permitted_spans) == len(aliases), "raw session has missing or additional native picking identities")
    previous, seen, parts = 0, set(), []
    for change in sorted(substitutions, key=lambda item: item.get("start", -1)):
        path, start, end = change.get("attribute_path"), change.get("start"), change.get("end")
        require(path in by_path and path not in seen and type(start) is int and type(end) is int
                and previous <= start < end <= len(text) and (start, end) in permitted_spans,
                "native ID substitution span is invalid or duplicated")
        original, alias = str(by_path[path]["raw_value"]), str(aliases[path])
        require(change.get("raw_decimal") == original and change.get("alias_decimal") == alias
                and text[start:end] == original, "native ID substitution does not match retained raw bytes")
        prefix = text[text.rfind("\n", 0, start) + 1:start]
        suffix = text[end:text.find("\n", end) if "\n" in text[end:] else len(text)]
        require(re.fullmatch(r"\s*custom uint64 viewPickingId\s*=\s*", prefix) is not None
                and not suffix.strip(), "native canonicalization attempted to alter a non-ID value")
        parts.extend((text[previous:start], alias))
        seen.add(path)
        previous = end
    parts.append(text[previous:])
    derived_text = "".join(parts)
    require(hashlib.sha256(derived_text.encode()).hexdigest() == identity.get("session_canonical_text_sha256"),
            "native canonical session hash does not match the exact ID-only replay")
    derived = copy.deepcopy(raw_scene)
    derived["anonymous_usd_layers"][index]["content"] = derived_text
    require(derived == scene, "semantic scene changed fields beyond the six native picking IDs")


def validate_physical_observation_evidence(directory, record):
    """Audit each arm's originals without equating or substituting its pixels."""
    directory = Path(directory)
    artifacts = {}
    for filename, field in (("initial_observation.json", "initial_observation_sha256"),
                            ("scene_config_after_reset_raw.json", "raw_scene_config_sha256"),
                            ("render_product_identity.json", "render_product_identity_sha256")):
        artifacts[filename] = read(directory / filename)
        require(driver_digest(artifacts[filename]) == record[field],
                f"persisted {filename} differs from its per-arm outcome binding")
    nonvisual, image_schema = physical_observation_parts(artifacts["initial_observation.json"])
    require(driver_digest(nonvisual) == record["initial_nonvisual_observation_sha256"],
            "actual nonvisual observation differs from its shared anchor")
    require(driver_digest(image_schema) == record["initial_image_schema_sha256"],
            "actual image schema differs from its shared anchor")
    validate_render_product_identity(artifacts["scene_config_after_reset_raw.json"],
                                     read(directory / "scene_config_after_reset.json"),
                                     artifacts["render_product_identity.json"], record)


def collected_renderer_record(directory, remote_completion, protocol):
    """Bind successful external collection without altering the native result."""
    directory = Path(directory)
    completion_path = directory / "renderer" / "completion.json"
    completed = read(completion_path)
    require(completed == remote_completion, "copied renderer completion differs from the observed remote receipt")
    require(completed.get("status") == "passed" and type(completed.get("exit_code")) is int
            and completed["exit_code"] == 0 and not completed.get("forced_kill")
            and not completed.get("cleanup_forced_kill") and not completed.get("stop_reason")
            and not completed.get("owned_processes_still_live"),
            "native renderer did not complete with externally observed clean exit0")
    native = read(directory / "renderer" / "episode" / "result.json")
    if protocol["schema_version"] == 3:
        fields = {"renderer_process_exit_code", "renderer_process_completion_sha256", "collector_source_sha256"}
        require(not fields.intersection(native), "native result cannot predeclare collector execution proof")
        return dict(native, renderer_process_exit_code=0,
                    renderer_process_completion_sha256=sha(completion_path), collector_source_sha256=sha(__file__))
    return native


class SmokeController:
    def __init__(self, config, protocol, identities, protocol_api, transport, output, *, clock=time.monotonic, sleep=time.sleep):
        self.config, self.protocol, self.identities, self.api = config, protocol, identities, protocol_api
        self.transport, self.output, self.clock, self.sleep = transport, Path(output), clock, sleep
        self.server = self.renderer = None
        self.records = []
        self.state = {"status": "prepared", "episodes_completed": 0, "expected_episodes": 8,
                      "task_quality_validated": False, "formal_follow_on": False, "automatic_retry": False}

    def save(self):
        write(self.output / "live_status.json", self.state)

    def wait(self, host, output, seconds):
        deadline = self.clock() + seconds
        while True:
            status = self.transport.status(host, output)
            if status["completion"] is not None:
                return status["completion"]
            require(status["live"], f"{host} job exited without a completion receipt")
            if self.clock() >= deadline:
                raise TimeoutError(f"{host} job exceeded the controller's finite wait")
            self.sleep(min(self.config["limits"]["poll_seconds"], max(0, deadline - self.clock())))

    def ready(self, cell, spec):
        deadline = self.clock() + self.config["limits"]["server_start_seconds"]
        path = f"{spec['output']}/supervisor/{cell}/ready.json"
        while True:
            status = self.transport.status("thor", spec["output"])
            require(status["completion"] is None and status["live"], "server ended before completing assigned smoke work")
            ready = self.transport.read_json("thor", path)
            if ready is not None:
                require(ready.get("status") == "listening" and ready.get("host") == "127.0.0.1"
                        and ready.get("port") == self.config["thor"]["port"]
                        and ready.get("metadata") == self.identities[cell], "server readiness identity differs")
                write(self.output / "cells" / cell / "ready.json", ready, exclusive=True)
                return
            if self.clock() >= deadline:
                raise TimeoutError("server startup deadline reached; no automatic retry")
            self.sleep(min(self.config["limits"]["poll_seconds"], max(0, deadline - self.clock())))

    def stop_job(self, host, spec):
        self.transport.stop(host, spec["output"])
        return self.wait(host, spec["output"], self.config["limits"]["grace_seconds"] + self.config["limits"]["control_seconds"])

    def validate_episode(self, cell, episode, directory, anchor, completion):
        directory = Path(directory)
        record = collected_renderer_record(directory, completion, self.protocol)
        expected = {row["pair_id"]: row for row in self.protocol["episodes"]}
        key = self.api._validate_record(record, self.protocol, expected, self.api.digest(self.protocol))
        family, kind = cell.split("-", 1)
        arm = "baseline" if kind == "eager_native" else "candidate"
        require(key == (family, arm, episode["pair_id"]), "returned episode belongs to another cell or pair")
        fields = anchor_fields(self.protocol)
        require({name: record[name] for name in fields} == anchor, "record does not match its complete native anchor")
        saved = read(directory / "renderer" / "episode" / "initial_state.json")
        require(saved["binding"] == anchor and driver_digest(saved["state"]) == anchor["initial_state_sha256"],
                "persisted native initial state does not bind the anchor")
        for filename, field in (("simulator_fingerprint.json", "simulator_fingerprint_sha256"),
                                ("scene_config_after_reset.json", "scene_config_sha256"),
                                ("verified_asset_inventory.json", "asset_inventory_sha256")):
            require(driver_digest(read(directory / "renderer" / "episode" / filename)) == record[field],
                    f"persisted {filename} differs from outcome binding")
        if fields == PHYSICAL_ANCHOR_FIELDS:
            validate_physical_observation_evidence(directory / "renderer" / "episode", record)
        golden = self.output / "anchors" / f"{episode['task_id']}.json"
        if cell == "edge-eager_native":
            write(golden, anchor, exclusive=True)
        else:
            require(read(golden) == anchor, "paired native initial anchors differ across arms")
        return record

    def validate_closed_cell(self, cell, final_server):
        family, kind = cell.split("-", 1)
        arm = "baseline" if kind == "eager_native" else "candidate"
        records = [row for row in self.records if row["family"] == family and row["arm"] == arm]
        requests = Path(final_server) / "supervisor" / cell / "requests"
        closed = read(requests / "closed.json")
        identity = self.identities[cell]["benchmark_identity_sha256"]
        require(len(records) == 2 and closed.get("failed") is False and closed.get("episodes") == 2
                and closed.get("requests") == sum(row["generated_chunks"] for row in records)
                and closed.get("benchmark_identity_sha256") == identity,
                "owned Thor server did not execute exactly its two recorded smoke episodes")
        for ordinal, episode in enumerate(self.protocol["episodes"]):
            reset = read(requests / f"episode_{ordinal:06d}.json")
            require(reset.get("reset") is True and reset.get("episode_id") == episode["pair_id"]
                    and reset.get("benchmark_seed") == episode["model_seed_base"]
                    and reset.get("max_policy_chunks") == episode["max_policy_chunks"]
                    and reset.get("benchmark_identity_sha256") == identity,
                    "owned server reset receipt differs from the paired smoke episode")

    def run(self):
        self.save()
        try:
            self.transport.prepare("thor")
            self.transport.prepare("renderer")
            for cell in CELLS:
                server = server_spec(self.config, cell)
                self.state.update(status="starting_server", cell=cell)
                self.save()
                self.server = server
                write(self.output / "cells" / cell / "launch.json", self.transport.spawn("thor", server), exclusive=True)
                cell_finished = False
                try:
                    self.ready(cell, server)
                    for episode in self.protocol["episodes"]:
                        job, anchor_path = episode_spec(self.config, cell, episode)
                        directory = self.output / "episodes" / cell / episode["task_id"]
                        directory.mkdir(parents=True, exist_ok=False)
                        self.state.update(status="running_episode", pair_id=episode["pair_id"])
                        self.save()
                        self.renderer = job
                        write(directory / "launch.json", self.transport.spawn("renderer", job), exclusive=True)
                        completion = self.wait("renderer", job["output"], self.config["limits"]["episode_seconds"]
                                               + self.config["limits"]["grace_seconds"] + self.config["limits"]["control_seconds"])
                        self.renderer = None
                        # Persist both processes' evidence before inspecting success or advancing.
                        self.transport.copy("renderer", job["output"], directory / "renderer")
                        self.transport.copy("thor", server["output"], directory / "thor_snapshot")
                        anchor = self.transport.read_json("renderer", anchor_path)
                        write(directory / "native_anchor.json", anchor, exclusive=True)
                        write(directory / "persisted_inventory.json", inventory_tree(directory), exclusive=True)
                        require(completion["status"] == "passed" and completion.get("exit_code") == 0,
                                "renderer process failed; full attempt evidence retained")
                        require(self.transport.status("thor", server["output"])["completion"] is None,
                                "server ended before its assigned episode evidence was persisted")
                        record = self.validate_episode(cell, episode, directory, anchor, completion)
                        self.records.append(record)
                        write(self.output / "records.json", self.records)
                        self.state["episodes_completed"] = len(self.records)
                        self.save()
                    cell_finished = True
                finally:
                    completion = self.stop_job("thor", server)
                    self.transport.copy("thor", server["output"], self.output / "cells" / cell / "final_server")
                    self.server = None
                    require(completion["status"] in {"passed", "stopped"}, "owned server did not shut down cleanly")
                    native = read(self.output / "cells" / cell / "final_server" / "supervisor" / "completion.json")
                    require(native["status"] == "passed", "Thor native supervisor gates did not pass after cleanup")
                    if cell_finished:
                        self.validate_closed_cell(cell, self.output / "cells" / cell / "final_server")
            report = self.api.build_report(self.protocol, self.records)
            require(len(self.records) == 8 and report["status"] == "NOT_CERTIFIABLE"
                    and report["task_quality_validated"] is False, "smoke must never become a formal certificate")
            write(self.output / "smoke_report.json", report, exclusive=True)
            self.state.update(status="completed_smoke_only", ended=time.time())
            self.save()
        except BaseException:
            self.state.update(status="failed", error=traceback.format_exc(), ended=time.time())
            self.save()
            raise
        finally:
            # Exceptional exits may occur before the per-cell cleanup was entered.
            for host, spec in (("renderer", self.renderer), ("thor", self.server)):
                if spec is not None:
                    try:
                        self.stop_job(host, spec)
                        self.transport.copy(host, spec["output"], self.output / "failure_evidence" / host)
                    except BaseException:
                        self.state.setdefault("cleanup_errors", []).append(traceback.format_exc())
                        self.state["status"] = "failed"
            self.save()
            write(self.output / "completion.json", self.state, exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--remote-spawn", action="store_true")
    mode.add_argument("--remote-job", type=Path)
    mode.add_argument("--remote-status", type=Path)
    mode.add_argument("--remote-stop", type=Path)
    mode.add_argument("--remote-read-json", type=Path)
    mode.add_argument("--remote-prepare", type=Path)
    parser.add_argument("--config-sha256")
    parser.add_argument("--spec-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.remote_spawn:
        result = remote_spawn(json.load(sys.stdin))
    elif args.remote_job:
        remote_job(args.remote_job, args.spec_sha256)
        return
    elif args.remote_status:
        result = remote_status(args.remote_status)
    elif args.remote_stop:
        result = remote_stop(args.remote_stop)
    elif args.remote_read_json:
        result = read(args.remote_read_json) if args.remote_read_json.is_file() else None
    elif args.remote_prepare:
        args.remote_prepare.mkdir(parents=True, exist_ok=False)
        result = {"created": str(args.remote_prepare)}
    else:
        require(args.output is not None and args.config_sha256 == sha(args.config), "explicit fresh output and config SHA required")
        config = read(args.config)
        sys.path.insert(0, config["local_flash_root"])
        protocol, identities, protocol_api = load_study(config)
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / "config.json", config, exclusive=True)
        write(args.output / "protocol.json", protocol, exclusive=True)
        write(args.output / "identities.json", identities, exclusive=True)
        write(args.output / "source.json", {"controller_sha256": sha(__file__),
                                             "protocol_validator_sha256": sha(protocol_api.__file__),
                                             "config_sha256": args.config_sha256}, exclusive=True)

        def interrupted(number, _frame):
            raise InterruptedError(f"controller received {signal.Signals(number).name}")

        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, interrupted)
        SmokeController(config, protocol, identities, protocol_api,
                        SSHTransport(config, args.output), args.output).run()
        return
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

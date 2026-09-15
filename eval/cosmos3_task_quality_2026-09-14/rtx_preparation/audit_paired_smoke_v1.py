"""CPU-only audit of persisted native Cosmos smoke evidence; never executes jobs.

This independently checks transport, native action processing, step/outcome and
source/exit chains. The frozen controller's scene-text replay is reused explicitly.
Each output is write-once; partial runs make no final-study or quality claim.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import sys

import msgpack
import numpy as np


CONFIG_SHA256 = "9461bcc4636d4eb01768a7ca4f75920e26ee24e743b5f4644d362b98db54a6ec"
CONTROLLER_SHA256 = "22ea0e076781f26a2be33fe40f608cfa33abdb98ee7f441f10276068f21da5a7"
PROTOCOL_SHA256 = "e3322c0b0d8ba30458604c46e31ab53fdb9a44ccd22b799aa49f9e83355e0849"
CELLS = ("edge-eager_native", "edge-runtime_selected", "nano-eager_native", "nano-runtime_selected")
PUBLIC_KEYS = {"observation/image", "observation/joint_position", "observation/gripper_position", "prompt"}
REQUEST_KEYS = PUBLIC_KEYS | {"episode_id", "request_id", "benchmark_identity_sha256"}
PROOF_FIELDS = {"renderer_process_exit_code", "renderer_process_completion_sha256", "collector_source_sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value, *, ascii=True):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=ascii, allow_nan=False).encode()).hexdigest()


def freeze(value):
    if isinstance(value, np.ndarray):
        require(value.dtype.kind not in "OVc", "unsupported wire array dtype")
        return {"dtype": value.dtype.str, "shape": list(value.shape),
                "data_base64": base64.b64encode(value.tobytes(order="C")).decode("ascii")}
    if isinstance(value, np.generic):
        return freeze(value.item())
    if isinstance(value, dict):
        return {key: freeze(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [freeze(child) for child in value]
    return value


def array(value):
    require(isinstance(value, dict) and set(value) == {"dtype", "shape", "data_base64"}, "invalid frozen array")
    dtype, shape = np.dtype(value["dtype"]), value["shape"]
    require(dtype.kind not in "OVc" and isinstance(shape, list)
            and all(type(n) is int and n >= 0 for n in shape), "unsafe frozen array schema")
    raw = base64.b64decode(value["data_base64"], validate=True)
    require(len(raw) == math.prod(shape) * dtype.itemsize, "frozen array byte count differs")
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def same_array(left, right, message):
    require(isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
            and left.shape == right.shape and left.dtype.str == right.dtype.str
            and left.tobytes() == right.tobytes(), message)


def public_hash(value):
    """Independent implementation of the installed public-input hash grammar."""
    result = hashlib.sha256()

    def add(item):
        if isinstance(item, dict):
            result.update(b"dict:")
            for key in sorted(item):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(f"list:{len(item)}:".encode())
            for child in item:
                add(child)
        elif isinstance(item, np.ndarray):
            result.update(json.dumps(["array", item.dtype.str, item.shape]).encode())
            result.update(np.ascontiguousarray(item).tobytes())
        else:
            result.update(json.dumps([type(item).__name__, item], sort_keys=True).encode())
    add(value)
    return result.hexdigest()


def decode_wire(packet):
    def hook(value):
        if b"__ndarray__" in value:
            require(set(value) == {b"__ndarray__", b"dtype", b"shape", b"data"}
                    and value[b"__ndarray__"] is True, "invalid msgpack array marker")
            return array({"dtype": value[b"dtype"], "shape": list(value[b"shape"]),
                          "data_base64": base64.b64encode(value[b"data"]).decode("ascii")})
        require(b"__npgeneric__" not in value, "unexpected scalar extension on native wire")
        return value
    return msgpack.unpackb(packet, object_hook=hook, strict_map_key=False)


def safe_file(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and relative.parts and ".." not in relative.parts,
            "unsafe evidence relative path")
    path = Path(root) / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()),
            "missing or escaping evidence file")
    return path


def trace_calls(root, expected_identity):
    manifest = read(root / "trace.json")
    require(manifest.get("schema_version") == 1 and manifest.get("complete") is True
            and manifest.get("identity") == expected_identity, "trace identity or completion differs")
    calls = []
    for index, row in enumerate(manifest["calls"]):
        values = []
        for kind in ("request", "response"):
            ref = row[kind]
            require(ref["path"] == f"{index:05d}.{kind}.msgpack", "noncontiguous trace")
            path = safe_file(root, ref["path"])
            require(sha(path) == ref["sha256"], "wire frame hash differs")
            values.append(decode_wire(path.read_bytes()))
        calls.append(tuple(values))
    require(len(list(root.glob("*.msgpack"))) == 2 * len(calls), "extra wire frames")
    return calls


def contiguous(root, count):
    expected = [root / f"{index:06d}.json" for index in range(count)]
    require(sorted(root.glob("*.json")) == expected, f"noncontiguous or excess files: {root.name}")
    return expected


def validate_outcome(record, episode):
    require(record.get("status") == "completed" and type(record.get("success")) is bool,
            "only completed real native outcomes are scored")
    require(not record.get("cleanup_errors") and not record.get("renderer_compatibility_error")
            and record.get("native_app_close") in {"pending_external_exit_check", "returned"},
            "native cleanup failed or application was never created")
    steps, chunks = record["executed_steps"], record["generated_chunks"]
    require(type(steps) is int and 3 <= steps <= episode["max_episode_steps"], "native step bound differs")
    require(record["success"] or steps == episode["max_episode_steps"], "failure before native timeout")
    require(type(chunks) is int and chunks == (steps + 31) // 32, "native chunk/step count differs")
    require(record["reset_count"] == 2 and type(record["reset_count"]) is int, "native reset count differs")
    native = record["native_outcome"]
    require(type(native.get("env_id")) is int and native["env_id"] == 0
            and type(native.get("step")) is int and native["step"] == steps
            and type(native.get("success")) is bool and native["success"] == record["success"],
            "record differs from native terminal outcome")
    expected_seeds = list(range(episode["model_seed_base"], episode["model_seed_base"] + chunks))
    require(record["model_seeds"] == expected_seeds, "policy request seeds differ")


def postprocess_action(raw):
    require(raw.shape == (32, 8) and raw.dtype == np.dtype("float32") and np.isfinite(raw).all(),
            "native action geometry or finiteness differs")
    result = raw.copy()
    result[:, 7] = (raw[:, 7] > 0.5).astype(raw.dtype)
    return result


def validate_execution(receipt, seed, cell):
    expected = {"branches": 8, "callbacks": 4,
                "generation": [{"guidance": 3.0, "num_steps": 4, "seed": [seed], "shift": 5.0}],
                "sampler": [{"num_steps": 4, "seed": [seed], "shift": 5.0}],
                "normal_request_rng_unchanged": True, "normal_seed_config_restored": True,
                "observers_restored": True, "request_seed": seed}
    require(receipt["execution"] == expected, "observed UniPC4/CFG3/seed/branch execution differs")
    owners = receipt["actual_optimizations"]
    if cell.endswith("-eager_native"):
        require(owners == {"route": "eager_native"}, "native baseline changed route")
        return
    require(owners["numeric_attention"]["backend"] == "cudnn", "NUMERIC cuDNN owner not selected")
    conditioning, regions = owners["conditioning_cache"], owners["generation_regions"]
    require(conditioning.get("disabled") is False and conditioning.get("closed") is False
            and not conditioning.get("rejected"), "selected conditioning owner failed")
    require(regions.get("closed") is False and not regions.get("rejected")
            and type(regions.get("compiled_calls")) is int and regions["compiled_calls"] > 0,
            "compiled generation region never executed or failed")
    require(regions.get("regions") and all(row.get("state") == "ready" and row.get("error") is None
                                          for row in regions["regions"]), "compiled generation region not ready")
    require(owners["timestep_cache"].get("closed") is False, "selected timestep owner closed")


def reconstruct_initial_request(observation, instruction):
    """CPU-only original geometry, with exact integer 2x bilinear box reduction.

    At scale 1/2, align_corners=False bilinear has four exact 1/4 weights;
    uint8 samples fit exactly in float32. Integer floor reproduces native cast.
    PIL is used only for the original resize-with-padding stage when required.
    """
    from PIL import Image

    def resize(value):
        pixels = array(value)[0]
        require(pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3,
                "native camera geometry differs")
        if pixels.shape[:2] == (360, 640):
            return pixels
        h, w = pixels.shape[:2]
        ratio = max(w / 640, h / 360)
        new_h, new_w = int(h / ratio), int(w / ratio)
        target = Image.new("RGB", (640, 360), 0)
        target.paste(Image.fromarray(pixels).resize((new_w, new_h), Image.Resampling.BILINEAR),
                     (int((640 - new_w) / 2), int((360 - new_h) / 2)))
        return np.asarray(target)

    def half(pixels):
        return (pixels.astype(np.uint16).reshape(180, 2, 320, 2, 3).sum(axis=(1, 3)) // 4).astype(np.uint8)

    images = observation["image_obs"]
    wrist = resize(images["wrist_cam"])
    bottom = np.concatenate([half(resize(images[name])) for name in
                             ("over_shoulder_left_camera", "over_shoulder_right_camera")], axis=1)
    return {"observation/image": np.concatenate([wrist, bottom], axis=0), "prompt": instruction,
            "observation/joint_position": array(observation["proprio_obs"]["arm_joint_pos"])[0],
            "observation/gripper_position": array(observation["proprio_obs"]["gripper_pos"])[0]}


def validate_inventory(directory):
    inventory = read(directory / "persisted_inventory.json")
    for relative, row in inventory["files"].items():
        path = safe_file(directory, relative)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                f"persisted file changed: {relative}")
    require(not inventory["symlinks"], "unexpected copied smoke symlink")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    require(actual == set(inventory["files"]) | {"persisted_inventory.json"}, "uninventoried episode files")
    return len(inventory["files"])


def validate_external_completion(directory, record, source, control_receipts):
    path = directory / "renderer" / "completion.json"
    completion = read(path)
    require(completion.get("status") == "passed" and type(completion.get("exit_code")) is int
            and completion["exit_code"] == 0 and not completion.get("forced_kill")
            and not completion.get("cleanup_forced_kill") and not completion.get("stop_reason")
            and not completion.get("owned_processes_still_live"), "renderer external completion failed")
    spec = read(directory / "renderer" / "spec.json")
    require(any(row.get("returncode") == 0 and len(row.get("command", []))
                and shlex.split(row["command"][-1])[-2:] == ["--remote-status", spec["output"]]
                for row in control_receipts), "no successful owned renderer status RPC")
    native = read(directory / "renderer" / "episode" / "result.json")
    require(not PROOF_FIELDS.intersection(native), "native result predeclares collector proof")
    expected = dict(native, renderer_process_exit_code=0, renderer_process_completion_sha256=sha(path),
                    collector_source_sha256=source["controller_sha256"])
    require(record == expected, "collected record differs from native result plus external proof")
    require(spec["runner_sha256"] == source["controller_sha256"], "executed renderer wrapper source differs")
    return native


def validate_job_identity(directory, spec, launch):
    """Replay immutable launch/command/process bindings across local copies."""
    require(read(directory / "spec.json") == spec and read(directory / "launch.json") == launch,
            "copied job spec or launch differs from prospective owned job")
    require(launch["spec_sha256"] == sha(directory / "spec.json")
            and all(launch[key] == spec[key] for key in ("job_id", "output", "runner_sha256")),
            "owned job launch does not bind its exact specification")
    if (directory / "completion.json").exists():
        complete = read(directory / "completion.json")
        require(complete["command"] == spec["command"] and complete["job_id"] == spec["job_id"]
                and all(complete["wrapper"][key] == launch["process"][key]
                        for key in ("pid", "start_ticks", "session_id", "group_id")),
                "external completion belongs to another command or process lifetime")


def validate_ledger_files(ledger, requests, episodes, *, closed=False):
    names = {"identity.json"}
    names.update(f"request_{i:06d}{suffix}" for i in range(requests)
                 for suffix in (".intent.json", ".json", ".npz", ".sources.json"))
    names.update(f"episode_{i:06d}{suffix}" for i in range(episodes) for suffix in (".intent.json", ".json"))
    if closed:
        names.add("closed.json")
    require({p.name for p in ledger.iterdir()} == names
            and all(p.is_file() and not p.is_symlink() for p in ledger.iterdir()),
            "server ledger has extra, missing or noncontiguous request/reset evidence")


def validate_cell_close(study, cell, episodes, identity, config, helper):
    final = study / "cells" / cell / "final_server"
    require(final.is_dir(), "completed cell lacks final owned server copy")
    launch = read(study / "cells" / cell / "launch.json")
    validate_job_identity(final, helper.server_spec(config, cell), launch)
    completed = read(final / "completion.json")
    require(completed["status"] in {"passed", "stopped"} and type(completed["exit_code"]) is int
            and completed["exit_code"] == 0 and not completed.get("forced_kill")
            and not completed.get("cleanup_forced_kill") and not completed.get("owned_processes_still_live")
            and completed.get("stop_reason") in {None, "SIGTERM"}, "server did not stop gracefully")
    require(read(final / "supervisor" / "completion.json")["status"] == "passed",
            "native Thor supervisor failed cleanup gates")
    ledger = final / "supervisor" / cell / "requests"
    closed = read(ledger / "closed.json")
    total = sum(row["generated_chunks"] for row in episodes)
    require(closed == {"failed": False, "requests": total, "episodes": 2,
                       "benchmark_identity_sha256": identity["benchmark_identity_sha256"]},
            "server closed ledger does not match exactly two audited episodes")
    validate_ledger_files(ledger, total, 2, closed=True)
    require(read(ledger / "identity.json") == identity
            and read(final / "supervisor" / cell / "ready.json")["metadata"] == identity,
            "final server identity differs")
    for row in episodes:
        snapshot = study / "episodes" / cell / row["task_id"] / "thor_snapshot"
        validate_job_identity(snapshot, helper.server_spec(config, cell), launch)
        for before in (snapshot / "supervisor" / cell / "requests").iterdir():
            require(sha(before) == sha(ledger / before.name), "immutable request ledger changed between episode and final close")
    require(not list((final / "supervisor" / cell).glob("connection-*-failure.json")),
            "server recorded a failed connection")
    return {"cell": cell, "closed_requests": total, "closed_episodes": 2,
            "final_completion_sha256": sha(final / "completion.json"),
            "one_owned_process_lifetime": True, "immutable_request_ledger_unchanged": True}


def audit_transport(directory, record, episode, identity, request_offset, episode_ordinal, instruction):
    output = directory / "renderer" / "episode"
    ledger = directory / "thor_snapshot" / "supervisor" / identity["benchmark_identity"]["cell_id"] / "requests"
    require(read(ledger / "identity.json") == identity, "actual policy identity differs")
    identity_hash = identity["benchmark_identity_sha256"]
    require(digest(identity["benchmark_identity"]) == identity_hash, "identity digest differs")
    calls = trace_calls(output / "wire_trace", identity["benchmark_identity"])
    chunks, steps = record["generated_chunks"], record["executed_steps"]
    validate_ledger_files(ledger, request_offset + chunks, episode_ordinal + 1)
    require(len(calls) == chunks + 1, "wire calls must contain one reset then exactly the generated chunks")
    reset_request, reset_response = calls[0]
    reset = {"reset": True, "episode_id": episode["pair_id"], "prompt": instruction,
             "benchmark_seed": episode["model_seed_base"], "max_policy_chunks": episode["max_policy_chunks"],
             "benchmark_identity_sha256": identity_hash}
    reset_path = ledger / f"episode_{episode_ordinal:06d}.json"
    require(reset_request == reset and read(ledger / f"episode_{episode_ordinal:06d}.intent.json") == reset,
            "reset request or native server reset intent differs")
    reset_ack = {key: value for key, value in reset.items() if key != "prompt"}
    require(read(reset_path) == reset and reset_response == dict(reset_ack, receipt_sha256=sha(reset_path))
            and read(output / "reset_receipt.json") == reset_response, "reset acknowledgement differs")
    chunk_files = contiguous(output / "chunks", chunks)
    processed_files = contiguous(output / "processed_chunks", chunks)
    step_files = contiguous(output / "steps", steps)
    full_initial = read(output / "initial_observation.json")
    expected_initial = reconstruct_initial_request(full_initial, instruction)
    action_digest = hashlib.sha256()
    latency = []
    source_counts = []
    for index, ((request, response), chunk_file, processed_file) in enumerate(zip(calls[1:], chunk_files, processed_files)):
        require(set(request) == REQUEST_KEYS and request["episode_id"] == episode["pair_id"]
                and type(request["request_id"]) is int and request["request_id"] == index
                and request["benchmark_identity_sha256"] == identity_hash and request["prompt"] == instruction,
                "request identity, prompt, ordinal or keys differ")
        observation = {key: request[key] for key in PUBLIC_KEYS}
        image = observation["observation/image"]
        require(image.shape == (540, 640, 3) and image.dtype == np.uint8, "wire image geometry differs")
        for key, shape in (("observation/joint_position", (7,)), ("observation/gripper_position", (1,))):
            value = observation[key]
            require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                    "wire measured proprioception differs")
        if index == 0:
            for key in PUBLIC_KEYS - {"prompt"}:
                same_array(observation[key], expected_initial[key], "first actual request differs from native own observation")
        prefix = ledger / f"request_{request_offset + index:06d}"
        intent = {"episode_id": episode["pair_id"], "request_id": index,
                  "request_seed": episode["model_seed_base"] + index, "input_sha256": public_hash(observation),
                  "benchmark_identity_sha256": identity_hash}
        require(read(str(prefix) + ".intent.json") == intent, "Thor request hash/seed intent differs")
        receipt_path = Path(str(prefix) + ".json")
        receipt = read(receipt_path)
        require(all(receipt.get(k) == v for k, v in intent.items()) and receipt.get("status") == "passed"
                and receipt.get("task_quality_certified") is False, "Thor successful request receipt differs")
        validate_execution(receipt, intent["request_seed"], identity["benchmark_identity"]["cell_id"])
        action_path = safe_file(ledger, receipt["action_file"])
        require(action_path == Path(str(prefix) + ".npz") and sha(action_path) == receipt["action_file_sha256"],
                "raw Thor action archive binding differs")
        with np.load(action_path, allow_pickle=False) as archive:
            require(archive.files == ["action"], "action archive has unexpected fields")
            raw = archive["action"]
        require(receipt["action_sha256"] == public_hash(raw) and receipt["action_shape"] == [32, 8]
                and receipt["action_dtype"] == "float32", "raw Thor action digest or geometry differs")
        require(set(response) == {"action", "episode_id", "request_id", "request_seed",
                                  "benchmark_identity_sha256", "receipt_sha256"}
                and all(response[k] == intent[k] for k in ("episode_id", "request_id", "request_seed", "benchmark_identity_sha256"))
                and response["receipt_sha256"] == sha(receipt_path), "response does not bind Thor receipt")
        same_array(response["action"], raw, "wire action differs from Thor archive")
        require(read(chunk_file) == {"request_sha256": digest(freeze(request)), "response": freeze(response)},
                "native client chunk differs from wire")
        processed = array(read(processed_file))
        same_array(processed, postprocess_action(raw), "native gripper processing or joint columns differ")
        for step in range(index * 32, min((index + 1) * 32, steps)):
            frozen = read(step_files[step])
            same_array(array(frozen), processed[step % 32:step % 32 + 1], "executed row differs from native processed chunk")
            action_digest.update(bytes.fromhex(digest(frozen)))
        source_file = safe_file(ledger, receipt["source_inventory_file"])
        require(source_file == Path(str(prefix) + ".sources.json") and sha(source_file) == receipt["source_inventory_sha256"],
                "Thor actual request source inventory binding differs")
        sources = read(source_file)
        require(all(sources.get(path) == value for path, value in identity["benchmark_identity"]["source_inventory"].items()),
                "prospectively bound source changed during actual prediction")
        source_counts.append(len(sources))
        seconds = receipt["predict_host_seconds"]
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0, "invalid host prediction timing")
        latency.append(seconds * 1000)
    require(action_digest.hexdigest() == record["action_trace_sha256"], "executed action trace digest differs")
    return {"wire_calls": len(calls), "policy_resets": 1, "native_resets": 2, "chunks": chunks,
            "executed_steps": steps, "first_request_camera_composition_exact": True,
            "all_wire_observations_bound_to_thor_receipts": True,
            "all_raw_processed_executed_actions_exact": True,
            "observed_unipc_steps": 4, "observed_cfg": 3.0, "observed_branches": 8 * chunks,
            "request_source_count_range": [min(source_counts), max(source_counts)],
            "predict_host_ms": {"min": min(latency), "median": float(np.median(latency)), "max": max(latency)},
            "latency_scope": "actual per-request Thor host prediction; includes cold request; not steady latency or realtime"}


def load_controller():
    path = Path(__file__).with_name("run_paired_smoke_v1.py")
    require(sha(path) == CONTROLLER_SHA256, "frozen scene replay helper source changed")
    spec = importlib.util.spec_from_file_location("frozen_smoke_scene_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_report(config, source, protocol, records):
    root = Path(config["local_flash_root"])
    path = root / "benchmarks" / "vla" / "robolab_protocol.py"
    require(sha(path) == source["protocol_validator_sha256"], "frozen report validator source changed")
    certificate = root / "instinctflash" / "verify" / "certify.py"
    remote = config["renderer"]["flash_root"].rstrip("/") + "/instinctflash/verify/certify.py"
    require(sha(certificate) == config["renderer"]["bound_files"][remote], "frozen certificate math source changed")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("frozen_smoke_protocol_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_report(protocol, records)


def audit(study, *, require_complete=False):
    study = Path(study).resolve()
    config, protocol, source, identities = (read(study / name) for name in
                                           ("config.json", "protocol.json", "source.json", "identities.json"))
    require(sha(study / "config.json") == CONFIG_SHA256 and source["config_sha256"] == CONFIG_SHA256,
            "frozen launched config changed")
    require(source["controller_sha256"] == CONTROLLER_SHA256 and digest(protocol, ascii=False) == PROTOCOL_SHA256,
            "frozen launched source or protocol changed")
    require(protocol["stage"] == "smoke" and protocol["expected_records"] == 8
            and protocol["acceptance"]["mode"] == "report_only", "audit cannot promote or substitute a formal protocol")
    records = read(study / "records.json")
    require(records and (not require_complete or len(records) == 8), "no completed records or full study incomplete")
    controls = [read(path) for path in sorted((study / "control").glob("*.json"))]
    helper = load_controller()
    expected = [(cell, episode) for cell in CELLS for episode in protocol["episodes"]]
    require(len(records) <= len(expected), "extra smoke records")
    offsets = dict.fromkeys(CELLS, 0)
    rows = []
    for record, (cell, episode) in zip(records, expected):
        family, route = cell.split("-", 1)
        arm = "baseline" if route == "eager_native" else "candidate"
        require((record["family"], record["arm"], record["pair_id"]) == (family, arm, episode["pair_id"]),
                "record schedule differs or contains duplicate/reordered pair")
        require(all(record[key] == episode[key] for key in ("scene_seed", "task_id", "episode_index"))
                and record["protocol_sha256"] == PROTOCOL_SHA256
                and record["checkpoint_revision"] == protocol["families"][family]["revision"]
                and record["robolab_revision"] == protocol["inventory"]["robolab_revision"]
                and record["evaluation_mode"] == "paused_simulation"
                and record["task_quality_validated"] is False, "native record contract differs")
        identity = identities[cell]
        declaration = config["identities"][cell]
        require(sha(declaration["local"]) == declaration["sha256"] and read(declaration["local"]) == identity,
                "collected identity differs from prelaunch four-cell identity")
        require(record["execution_binding_sha256"] == identity["benchmark_identity_sha256"]
                == protocol["execution_bindings"][family][arm], "actual deployment identity differs from protocol")
        require(read(study / "cells" / cell / "ready.json")["metadata"] == identity, "server ready identity differs")
        directory = study / "episodes" / cell / episode["task_id"]
        count = validate_inventory(directory)
        renderer_spec, _ = helper.episode_spec(config, cell, episode)
        validate_job_identity(directory / "renderer", renderer_spec, read(directory / "launch.json"))
        validate_job_identity(directory / "thor_snapshot", helper.server_spec(config, cell),
                              read(study / "cells" / cell / "launch.json"))
        renderer = config["renderer"]
        expected_rpc = ["ssh", *renderer["ssh"]["options"], renderer["ssh"]["target"],
                        shlex.join([renderer["control_python"], renderer["runner"], "--remote-status", renderer_spec["output"]])]
        require(any(row.get("returncode") == 0 and row.get("command") == expected_rpc for row in controls),
                "no successful status RPC to the exact configured owned renderer job")
        validate_external_completion(directory, record, source, controls)
        validate_outcome(record, episode)
        out = directory / "renderer" / "episode"
        anchor = read(study / "anchors" / f"{episode['task_id']}.json")
        require({field: record[field] for field in helper.anchor_fields(protocol)} == anchor
                and read(directory / "native_anchor.json") == anchor, "shared physical/camera anchor differs")
        initial = read(out / "initial_state.json")
        require(initial["binding"] == anchor and digest(initial["state"]) == record["initial_state_sha256"],
                "native physical/camera state bytes differ from anchor")
        for filename, field in (("simulator_fingerprint.json", "simulator_fingerprint_sha256"),
                                ("scene_config_after_reset.json", "scene_config_sha256"),
                                ("verified_asset_inventory.json", "asset_inventory_sha256")):
            require(digest(read(out / filename)) == record[field], f"native {filename} binding differs")
        fingerprint = read(out / "simulator_fingerprint.json")
        bound = config["renderer"]["bound_files"]
        flash = config["renderer"]["flash_root"].rstrip("/")
        require(fingerprint["source"] == {
            "revision": protocol["inventory"]["robolab_revision"],
            "driver_sha256": bound[flash + "/benchmarks/vla/robolab_driver.py"],
            "asset_driver_sha256": bound[flash + "/benchmarks/vla/robolab_assets.py"]},
            "actual native driver/source differs from frozen transferred source")
        bootstrap = read(directory / "renderer" / "bootstrap.json")
        require(bootstrap["bootstrap_sha256"] == bound[config["renderer"]["bootstrap"]]
                and bootstrap["entry_script"] is None and bootstrap["entry_script_sha256"] is None
                and bootstrap["episode_driver"] is True, "actual selected module bootstrap source differs")
        require(digest(read(out / "simulator_source_inventory.json")) == fingerprint["installed_simulator_source_sha256"],
                "actual renderer source inventory differs from simulator fingerprint")
        helper.validate_physical_observation_evidence(out, record)
        state = initial["state"]
        require(array(state["episode_length_buf"]).tolist() == [0]
                and array(state["frozen_envs"]).tolist() == [False] and state["has_stepped"] is False,
                "native initial reset state is not fresh")
        status = read(out / "native_subtask_status.json")
        require(isinstance(status, list) and len(status) == record["executed_steps"]
                and all(isinstance(row, list) and len(row) == 1 and isinstance(row[0], dict) for row in status),
                "native per-step single-env diagnostic coverage differs")
        instruction = next(row["instruction_default"] for row in protocol["inventory"]["tasks"] if row["task_id"] == episode["task_id"])
        transport = audit_transport(directory, record, episode, identity, offsets[cell],
                                    protocol["episodes"].index(episode), instruction)
        offsets[cell] += record["generated_chunks"]
        rows.append({"cell": cell, "pair_id": record["pair_id"], "success": record["success"],
                     "persisted_files_verified": count, "execution_binding_sha256": record["execution_binding_sha256"],
                     "result_sha256": sha(out / "result.json"), "external_exit_code": 0, **transport})
    closed_cells = []
    for cell in CELLS:
        family, route = cell.split("-", 1)
        arm = "baseline" if route == "eager_native" else "candidate"
        cell_records = [row for row in records if (row["family"], row["arm"]) == (family, arm)]
        # During live collection, two records may be copied just before graceful
        # server close finishes. Partial audits state which closures are verified.
        if len(cell_records) == 2 and (study / "cells" / cell / "final_server").exists():
            closed_cells.append(validate_cell_close(study, cell, cell_records, identities[cell], config, helper))
    if require_complete:
        require(read(study / "completion.json")["status"] == "completed_smoke_only", "controller did not finish successfully")
        report = replay_report(config, source, protocol, records)
        require(read(study / "smoke_report.json") == report and report["status"] == "NOT_CERTIFIABLE"
                and report["task_quality_validated"] is False, "persisted smoke aggregate differs from verified outcome replay")
        require(len(closed_cells) == 4, "all four owned servers have not closed")
    return {"schema_version": 1, "status": "passed", "scope": "complete_eight_episode_smoke" if require_complete else "completed_records_only",
            "task_quality_certified": False, "formal_follow_on": False, "auditor_source_sha256": sha(__file__),
            "reused_scene_replay_source_sha256": CONTROLLER_SHA256, "study": str(study),
            "config_sha256": CONFIG_SHA256, "protocol_sha256": PROTOCOL_SHA256,
            "records_sha256": sha(study / "records.json"), "records_verified": len(rows),
            "report_replay_scope": "frozen protocol validator and certificate math" if require_complete else "not final",
            "closed_cells": closed_cells,
            "total_wire_calls": sum(row["wire_calls"] for row in rows),
            "total_chunks": sum(row["chunks"] for row in rows), "total_executed_steps": sum(row["executed_steps"] for row in rows),
            "limitations": ["Two-task smoke does not establish the approved 5pp task-quality margin.",
                            "Native images are retained separately per arm; no pixel equivalence claim.",
                            "First request is independently reconstructed from own initial cameras; later raw cameras are not persisted.",
                            "Outcome is source-bound native terminal authority; subtask scores are diagnostics.",
                            "Successful control RPC payloads were not separately persisted; copied external completion is hash-bound by the frozen collector that compared the live response.",
                            "Source inventories bind files, not every executed instruction."], "episodes": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists(), "audit output exists; preserve previous evidence")
    try:
        report = audit(args.study, require_complete=args.require_complete)
    except Exception as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                  "task_quality_certified": False, "auditor_source_sha256": sha(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "records_verified", "error") if key in report}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

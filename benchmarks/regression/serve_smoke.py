"""Run a prepared model through the installed CLI and an actual WebSocket client.

This verifies transport, action shapes, metadata, history, reset and shutdown.
The six requests include startup effects and are not a latency benchmark or a
robot task evaluation. Use reproduce.run for paired prediction measurements.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .hardware import (
    TARGETS,
    bound_target,
    positive_timeout,
    probe_device,
    target_record,
    validate_device_receipt,
)
from .reproduce import (
    child_environment,
    encoded,
    new_directory,
    require,
    require_installed,
    sha,
    validate_bundle,
    write_new,
)


def selected_cell(plan, cell_id=None):
    cells = plan["matrix"]["cells"]
    if cell_id is None:
        cell = cells[-1]  # selected Runtime, or the explicitly requested operating point
    else:
        matches = [item for item in cells if item["id"] == cell_id]
        require(len(matches) == 1, "unknown prepared cell")
        cell = matches[0]
    require(cell["arm"] != "eager_native", "CLI smoke requires a Runtime cell")
    return cell


def serve_config(cell, checkpoint_path, port, seed):
    serve = {"model": str(checkpoint_path), "host": "127.0.0.1", "port": port}
    if cell["expected_runtime_kwargs"].get("precision", "native") == "fp8":
        require(seed is None, "the FP8 serving engine does not support --seed")
    elif seed is not None:
        serve["seed"] = seed
    return {"serve": serve,
            "runtime": cell["expected_runtime_kwargs"], "output": {"format": "json"}}


def receive(connection, timeout):
    from instinctflash.serving import msgpack_numpy
    frame = connection.recv(timeout=timeout)
    require(isinstance(frame, bytes), "server returned an error text frame; inspect server.log")
    result = msgpack_numpy.unpackb(frame)
    require(isinstance(result, dict), "server response must be a mapping")
    return result


def validate_metadata(metadata, cell):
    require(metadata.get("model_id") == cell["model_id"], "server model differs from prepared checkpoint")
    protocol = metadata.get("protocol", {})
    require(protocol.get("wire") == "openpi-websocket-msgpack-numpy"
            and protocol.get("reset_extension") is True, "unexpected serving protocol")
    policy = metadata.get("execution_policy", {})
    options = cell["expected_runtime_kwargs"]
    require(metadata.get("precision") == options.get("precision", "native"),
            "server precision differs from requested cell")
    require(policy.get("nfe") == cell["effective_schedule"]["nfe"],
            "server schedule differs from prepared cell")
    require(policy.get("tier_ceiling") == options.get("tier_ceiling", "bitexact"),
            "server transformation ceiling differs from prepared cell")


def client_requests(connection, cell, fixture, timeout):
    import numpy as np

    from instinctflash.serving import msgpack_numpy

    from .user_e2e import RecordedInputs, prompt, request_hash

    inputs = RecordedInputs(fixture)
    metadata = receive(connection, timeout)
    validate_metadata(metadata, cell)
    calls, actions = [], []
    family = cell["family"]
    for episode in range(2):
        instruction = prompt(episode)
        connection.send(msgpack_numpy.packb({"reset": True, "prompt": instruction}))
        reset_reply = receive(connection, timeout)
        require(set(reset_reply) <= {"server_timing"}, "reset unexpectedly returned actions")
        for cycle in range(3):
            index = episode * 3 + cycle
            observation = inputs.observation(family, index, cycle)
            observation["prompt"] = instruction
            if family == "va":
                observation["executed_action"] = inputs.feedback[cycle].copy()
            request_digest = request_hash(observation)
            start = time.perf_counter()
            connection.send(msgpack_numpy.packb(observation))
            reply = receive(connection, timeout)
            elapsed = (time.perf_counter() - start) * 1000
            require("action" in reply, "server response has no public action")
            action = np.asarray(reply["action"])
            require(list(action.shape) == cell["action_shape"]
                    and np.issubdtype(action.dtype, np.number) and np.isfinite(action).all(),
                    "unexpected action shape/dtype or nonfinite action")
            actions.append(action.copy())
            calls.append({"episode": episode, "cycle": cycle,
                          "request_sha256": request_digest, "roundtrip_ms": elapsed,
                          "action_shape": list(action.shape), "action_dtype": str(action.dtype),
                          "server_timing": reply.get("server_timing")})
    return metadata, calls, np.stack(actions)


def connect_started_process(process, port, timeout):
    """Retry connection refusal only while this exact new process is alive."""
    from websockets.sync.client import connect
    deadline = time.monotonic() + timeout
    while True:
        require(process.poll() is None, "CLI server exited before connection; inspect server.log")
        remaining = deadline - time.monotonic()
        require(remaining > 0, "CLI server startup deadline exceeded")
        try:
            return connect(f"ws://127.0.0.1:{port}", open_timeout=min(5, remaining),
                           compression=None, max_size=None, ping_timeout=None)
        except (ConnectionRefusedError, TimeoutError):
            time.sleep(min(.25, max(0, deadline - time.monotonic())))


def stop_owned_process(process):
    """Gracefully stop only the server process group we just created."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    return process.returncode


def run(prepared, output, *, cell_id=None, startup_timeout=1200, request_timeout=600, seed=None):
    startup_timeout = positive_timeout(startup_timeout, name="startup timeout")
    request_timeout = positive_timeout(request_timeout, name="request timeout")
    require_installed()
    prepared = Path(prepared).resolve(strict=True)
    plan, preparation = validate_bundle(prepared)
    target = bound_target(plan.get("target"))
    cell = selected_cell(plan, cell_id)
    require(not plan["unresolved_library_options"], "required native library options are unresolved")
    # This is a transport/episode check, not a paired RNG experiment. Native
    # serving accepts a constructor seed; the FP8 engine explicitly rejects it.
    fp8 = cell["expected_runtime_kwargs"].get("precision", "native") == "fp8"
    require(not fp8 or seed is None, "the FP8 serving engine does not support --seed")
    effective_seed = seed if seed is not None else None if fp8 else 9173
    root = new_directory(output)
    environment = child_environment(plan, cell, preparation)
    receipt = {"schema": "instinctflash.serving_smoke.v1", "status": "failed",
               "target": target,
               "cell_id": cell["id"], "checkpoint": plan["checkpoint"],
               "plan_sha256": sha(prepared / "plan.json"),
               "fixture_sha256": plan["fixture_sha256"],
               "task_quality_validated": False, "latency_benchmark": False,
               "scope": "Installed CLI and loopback WebSocket; two episodes, three calls each",
               "startup_timeout_s": startup_timeout, "request_timeout_s": request_timeout,
               "interpreter": sys.executable, "seed": effective_seed,
               "rng_scope": "unseeded FP8 transport check" if fp8 else "native serving constructor seed"}
    process = None
    try:
        # Resolve a declared local view using the same pinned primary snapshot.
        # CLI has no revision flag; passing the Hub name directly could select a
        # newer revision. Preparation is offline and separate from the GPU server.
        script = ("from instinctflash.descriptors.package import from_pretrained; "
                  f"print(from_pretrained({cell['model_id']!r}, revision={cell['revision']!r}).path)")
        resolved = subprocess.run([sys.executable, "-I", "-c", script], cwd=root,
                                  env=environment, capture_output=True, text=True,
                                  timeout=startup_timeout)
        write_new(root / "checkpoint_prepare.log", (resolved.stdout + resolved.stderr).encode())
        require(resolved.returncode == 0, "checkpoint preparation failed; inspect checkpoint_prepare.log")
        checkpoint = Path(resolved.stdout.strip().splitlines()[-1])
        require(checkpoint.is_absolute() and (checkpoint / "instinctflash.json").is_file(),
                "preparation did not return a declared checkpoint directory")
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        config = root / "serve.json"
        write_new(config, encoded(serve_config(cell, checkpoint, port, effective_seed)))
        device_receipt = root / "server_device.json"
        command = [sys.executable, "-I", "-m", "benchmarks.regression.serve_smoke", "_serve",
                   "--target", target["name"], "--config", str(config),
                   "--device-output", str(device_receipt)]
        receipt.update(command=command, config_sha256=sha(config), checkpoint_path=str(checkpoint))
        with (root / "server.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            receipt["server_pid"] = process.pid
            with connect_started_process(process, port, startup_timeout) as connection:
                hardware = json.loads(device_receipt.read_text())
                validate_device_receipt(hardware, target)
                receipt.update(hardware=hardware, server_device_sha256=sha(device_receipt))
                metadata, calls, actions = client_requests(
                    connection, cell, prepared / "inputs/recorded_inputs_v1.npz", request_timeout)
            import numpy as np
            np.savez_compressed(root / "actions.npz", actions=actions)
            receipt.update(status="passed", metadata=metadata, calls=calls,
                           action_archive_sha256=sha(root / "actions.npz"))
    except BaseException as error:
        receipt.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        if process is not None:
            receipt["server_exit_code"] = stop_owned_process(process)
            if receipt["server_exit_code"] != 0:
                receipt["status"] = "failed"
                receipt["shutdown_error"] = "server did not shut down cleanly"
        device_receipt = root / "server_device.json"
        if device_receipt.is_file():
            receipt["server_device_sha256"] = sha(device_receipt)
            # A failed target probe remains visible even if no WebSocket opened.
            try:
                receipt["hardware"] = json.loads(device_receipt.read_text())
            except (OSError, ValueError) as error:
                receipt.update(status="failed", server_device_error=repr(error))
        write_new(root / "receipt.json", encoded(receipt))
    return receipt


def serve_on_target(argv):
    """Probe in the actual server process, then invoke the public CLI unchanged."""
    parser = argparse.ArgumentParser(description="Internal target-bound public CLI launcher")
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device-output", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    target = target_record(args.target)
    hardware = probe_device(target, torch)
    write_new(args.device_output, encoded(hardware))
    validate_device_receipt(hardware, target)
    from instinctflash.cli import main as cli_main
    return cli_main(["serve", f"--config_path={args.config}"])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["_serve"]:
        return serve_on_target(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cell")
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, help="Native serving seed; defaults to 9173. Unsupported by FP8 serving.")
    args = parser.parse_args(argv)
    result = run(args.prepared, args.output, cell_id=args.cell,
                 startup_timeout=args.startup_timeout, request_timeout=args.request_timeout,
                 seed=args.seed)
    print(json.dumps({"status": result["status"], "receipt": str(args.output / "receipt.json")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

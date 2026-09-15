"""Replay recorded Cosmos SCREEN observations for latency and raw actions only.

The original simulator outcomes are never replayed or certified by this command.
Fixtures are explicit separately licensed inputs, not installed package data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import traceback

from benchmarks.vla.cosmos_quality_policy import CHECKPOINTS


SCHEMA = "instinctflash.cosmos_screen_replay.v1"
SOURCE_LATENCY_SHA = "dfc360c8bec3e87fc16617b52a4979378b28b6db353d0296dc172271f4504c73"
CELLS = {
    "edge-eager_native": 109,
    "edge-runtime_selected": 89,
    "nano-eager_native": 56,
    "nano-runtime_selected": 62,
}
STUDY = Path("eval/cosmos3_task_quality_2026-09-14/timed_screen_v1")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def contained(root, name):
    relative = Path(name)
    require(
        not relative.is_absolute() and ".." not in relative.parts, "unsafe fixture path"
    )
    root = Path(root).resolve(strict=True)
    path = root / relative
    require(
        path.is_file() and path.resolve().is_relative_to(root),
        "fixture leaves its root",
    )
    return path


def matrix_cell(cell_id, library=None):
    require(cell_id in CELLS, "unknown fixed SCREEN cell")
    family, arm = cell_id.split("-", 1)
    selected = arm == "runtime_selected"
    environment = (
        {}
        if not selected
        else {
            "IFL_COSMOS3_GEN_REGIONS": "1",
            "IFL_COSMOS3_TIMESTEP_CACHE": "1",
            "IFL_COSMOS3_SPLIT_PREFILL": "1",
            "IFL_COSMOS3_CONTIGUOUS_KV": "0",
            "IFL_BF16_LINEAR_RELU2": "1" if family == "edge" else "0",
        }
    )
    if selected and family == "edge":
        require(
            library is not None and Path(library).is_file(),
            "Edge NUMERIC requires --bf16-library",
        )
        environment["IFL_BF16_KERNEL_LIBRARY"] = str(Path(library).resolve())
    else:
        require(library is None, "this cell does not use a supplied BF16 library")
    model, revision = CHECKPOINTS[family]
    return {
        "id": cell_id,
        "family": family,
        "arm": arm,
        "model_id": model,
        "revision": revision,
        "action_shape": [32, 8],
        "expected_runtime_kwargs": {"device": "cuda:0", "tier_ceiling": "numeric"}
        if selected
        else {},
        "expected_optimizer_environment": environment,
        "effective_schedule": {
            "sampler": "unipc",
            "steps": 4,
            "shift": 5.0,
            "guidance": 3.0,
            "nfe": {"prefix": 1, "action": 4},
        },
    }


def fixture_catalog(root, expected):
    path = contained(root, "manifest.json")
    require(sha(path) == expected, "fixture manifest hash changed")
    value = json.loads(path.read_text())
    require(
        value["schema"] == SCHEMA and set(value["cells"]) == set(CELLS),
        "unexpected fixture catalog",
    )
    require(
        value["source_latency_sha256"] == SOURCE_LATENCY_SHA,
        "fixture names another source study",
    )
    require(
        value["task_quality_certified"] is False, "fixture cannot certify task quality"
    )
    for name, count in CELLS.items():
        require(
            len(value["cells"][name]["requests"]) == count,
            "fixed request count changed",
        )
    return value


def load_arrays(root, cell):
    import numpy as np

    path = contained(root, cell["file"])
    require(
        path.stat().st_size == cell["bytes"] and sha(path) == cell["sha256"],
        "fixture archive changed",
    )
    n = len(cell["requests"])
    with np.load(path, allow_pickle=False) as archive:
        require(
            set(archive.files) == {"images", "joints", "grippers", "actions"},
            "unexpected fixture arrays",
        )
        arrays = {name: archive[name] for name in archive.files}
    for name, shape, dtype in (
        ("images", (n, 540, 640, 3), np.uint8),
        ("joints", (n, 7), np.float32),
        ("grippers", (n, 1), np.float32),
        ("actions", (n, 32, 8), np.float32),
    ):
        require(
            arrays[name].shape == shape and arrays[name].dtype == dtype,
            f"invalid {name}",
        )
        require(np.isfinite(arrays[name]).all(), f"nonfinite {name}")
    return arrays


def observation(arrays, row, index):
    return {
        "observation/image": arrays["images"][index].copy(),
        "observation/joint_position": arrays["joints"][index].copy(),
        "observation/gripper_position": arrays["grippers"][index].copy(),
        "prompt": row["prompt"],
    }


def action_comparison(action, reference):
    import numpy as np

    require(
        action.dtype == reference.dtype and action.shape == reference.shape,
        "action comparison dtype/shape differs",
    )
    return {
        "bitwise_equal": action.tobytes() == reference.tobytes(),
        "max_abs_error": float(
            np.max(np.abs(action.astype(np.float64) - reference.astype(np.float64)))
        ),
    }


def verify_inputs(arrays, cell):
    from benchmarks.regression.user_e2e import request_hash

    resets = []
    previous = None
    for i, row in enumerate(cell["requests"]):
        require(
            request_hash(observation(arrays, row, i)) == row["input_sha256"],
            "decoded observation hash differs",
        )
        require(
            request_hash(arrays["actions"][i]) == row["action_sha256"],
            "reference action hash differs",
        )
        if row["episode_id"] != previous:
            require(
                row["request_id"] == 0 and row["episode_id"] not in resets,
                "invalid reset sequence",
            )
            resets.append(row["episode_id"])
        else:
            require(
                row["request_id"] == cell["requests"][i - 1]["request_id"] + 1,
                "request order changed",
            )
        require(
            row["request_seed"] == row["benchmark_seed"] + row["request_id"],
            "request seed changed",
        )
        previous = row["episode_id"]
    require(len(resets) == 6, "exactly six original resets required")


def export_fixture(study, latency, output):
    """CPU-only lossless conversion; originals remain untouched and unpublished."""
    import numpy as np
    from instinctflash.serving.msgpack_numpy import unpackb
    from benchmarks.regression.user_e2e import request_hash

    study, output = Path(study).resolve(strict=True), Path(output).absolute()
    require(
        not output.exists() and not output.resolve().is_relative_to(study),
        "use fresh output outside study",
    )
    original = json.loads(Path(latency).read_text())
    require(sha(latency) == SOURCE_LATENCY_SHA, "original latency aggregate changed")
    require(
        {c["cell"]: c["request_count"] for c in original["cells"]} == CELLS,
        "wrong source request roster",
    )
    output.mkdir(parents=True, exist_ok=False)
    catalog = {
        "schema": SCHEMA,
        "source_latency_sha256": sha(latency),
        "cells": {},
        "scope": "Exact recorded observation/action inputs, not simulator assets or a replay of task success.",
        "rights_status": "Separate rendered-data license review required; not covered by the code license.",
        "publication_ready": False,
        "task_quality_certified": False,
    }
    try:
        for cell in original["cells"]:
            arrays = {key: [] for key in ("images", "joints", "grippers", "actions")}
            rows = []
            for binding in cell["requests"]:
                relative = Path(binding["path"]).relative_to(STUDY)
                receipt_path = contained(study, relative)
                require(
                    sha(receipt_path) == binding["sha256"],
                    "source request receipt changed",
                )
                receipt = json.loads(receipt_path.read_text())
                trace_root = (
                    study
                    / "batches/0000/episodes"
                    / cell["cell"]
                    / receipt["episode_id"]
                    / "renderer/episode/wire_trace"
                )
                trace = json.loads(contained(trace_root, "trace.json").read_text())
                call = trace["calls"][receipt["request_id"] + 1]
                packet = contained(trace_root, call["request"]["path"])
                reply = contained(trace_root, call["response"]["path"])
                require(
                    sha(packet) == call["request"]["sha256"]
                    and sha(reply) == call["response"]["sha256"],
                    "wire payload changed",
                )
                request, response = (
                    unpackb(packet.read_bytes()),
                    unpackb(reply.read_bytes()),
                )
                reset_ref = trace["calls"][0]["request"]
                reset_path = contained(trace_root, reset_ref["path"])
                require(sha(reset_path) == reset_ref["sha256"], "reset payload changed")
                reset = unpackb(reset_path.read_bytes())
                obs = {
                    k: request[k]
                    for k in (
                        "observation/image",
                        "observation/joint_position",
                        "observation/gripper_position",
                        "prompt",
                    )
                }
                require(
                    request_hash(obs) == receipt["input_sha256"],
                    "original input hash differs",
                )
                require(
                    response["receipt_sha256"] == binding["sha256"],
                    "wire response lost receipt binding",
                )
                action_path = contained(receipt_path.parent, receipt["action_file"])
                require(
                    sha(action_path) == receipt["action_file_sha256"],
                    "original action archive changed",
                )
                with np.load(action_path, allow_pickle=False) as saved:
                    require(saved.files == ["action"], "unexpected action archive")
                    action = saved["action"]
                require(
                    action_comparison(action, response["action"])["bitwise_equal"],
                    "wire action differs",
                )
                for key, value in (
                    ("images", obs["observation/image"]),
                    ("joints", obs["observation/joint_position"]),
                    ("grippers", obs["observation/gripper_position"]),
                    ("actions", action),
                ):
                    arrays[key].append(value)
                rows.append(
                    {
                        "episode_id": receipt["episode_id"],
                        "request_id": receipt["request_id"],
                        "request_seed": receipt["request_seed"],
                        "benchmark_seed": reset["benchmark_seed"],
                        "max_policy_chunks": reset["max_policy_chunks"],
                        "prompt": request["prompt"],
                        "input_sha256": receipt["input_sha256"],
                        "action_sha256": receipt["action_sha256"],
                        "source_receipt_sha256": binding["sha256"],
                        "source_wire_sha256": sha(packet),
                        "source_reply_sha256": sha(reply),
                        "source_reset_sha256": sha(reset_path),
                        "historical_predict_host_seconds": receipt[
                            "predict_host_seconds"
                        ],
                    }
                )
            name = cell["cell"] + ".npz"
            arrays = {key: np.stack(value) for key, value in arrays.items()}
            entry = {"file": name, "requests": rows}
            verify_inputs(arrays, entry)
            with (output / name).open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            entry.update(
                bytes=(output / name).stat().st_size, sha256=sha(output / name)
            )
            verify_inputs(load_arrays(output, entry), entry)
            catalog["cells"][cell["cell"]] = entry
        write(output / "manifest.json", catalog)
        return catalog
    except BaseException:
        write(
            output / "failure.json",
            {"traceback": traceback.format_exc(), "task_quality_certified": False},
        )
        raise


def plan(fixture, expected, cell_id, library=None):
    catalog = fixture_catalog(fixture, expected)
    cell = matrix_cell(cell_id, library)
    return {
        "schema": SCHEMA,
        "fixture": str(Path(fixture).resolve()),
        "fixture_sha256": expected,
        "cell": cell,
        "request_count": len(catalog["cells"][cell_id]["requests"]),
        "library_sha256": sha(library) if library else None,
        "task_quality_certified": False,
        "historical_outcomes_replayed": False,
        "scope": "Fixed recorded trajectory; cold-inclusive seeded_predict host timer, no network/rendering.",
        "requirements": "Install Cosmos vendor/core/adapter and prepare the pinned auxiliary cache before run.",
    }


def prepare(fixture, expected, cell_id, output, library=None, cache_dir=None):
    value = plan(fixture, expected, cell_id, library)
    entry = fixture_catalog(fixture, expected)["cells"][cell_id]
    verify_inputs(load_arrays(fixture, entry), entry)
    output = Path(output).absolute()
    require(not output.exists(), "preparation output exists")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", value)
    write(output / "matrix.json", {"cells": [value["cell"]]})
    try:
        from huggingface_hub import snapshot_download

        reference = Path(
            snapshot_download(
                value["cell"]["model_id"],
                revision=value["cell"]["revision"],
                cache_dir=cache_dir,
            )
        ).absolute()
        require(
            reference.name == value["cell"]["revision"] and reference.is_dir(),
            "wrong checkpoint reference",
        )
        write(
            output / "prepared.json",
            {
                "plan_sha256": sha(output / "plan.json"),
                "matrix_sha256": sha(output / "matrix.json"),
                "checkpoint_reference": str(reference),
                "checkpoint_path": str(reference.resolve()),
                "cache_dir": str(Path(cache_dir).resolve()) if cache_dir else None,
            },
        )
    except BaseException:
        write(output / "failure.json", {"traceback": traceback.format_exc()})
        raise


def loaded_plan(prepared):
    prepared = Path(prepared).resolve(strict=True)
    bound = json.loads(contained(prepared, "prepared.json").read_text())
    require(
        sha(prepared / "plan.json") == bound["plan_sha256"]
        and sha(prepared / "matrix.json") == bound["matrix_sha256"],
        "prepared plan changed",
    )
    value = json.loads((prepared / "plan.json").read_text())
    library = value["cell"]["expected_optimizer_environment"].get(
        "IFL_BF16_KERNEL_LIBRARY"
    )
    require(
        value
        == plan(
            value["fixture"], value["fixture_sha256"], value["cell"]["id"], library
        ),
        "plan/options changed",
    )
    require(
        json.loads((prepared / "matrix.json").read_text())
        == {"cells": [value["cell"]]},
        "matrix changed",
    )
    return value, bound


def capture(prepared, output):
    import torch
    from benchmarks.vla.cosmos_quality_policy import CosmosQualityPolicy
    from benchmarks.regression.user_e2e import request_hash
    from benchmarks.regression.reproduce import require_installed

    require(
        sys.flags.isolated and sys.dont_write_bytecode,
        "use run for isolated installed capture",
    )
    require_installed()
    require(
        json.loads(contained(output, "launch.json").read_text())["source_sha256"]
        == sha(__file__),
        "installed replay helper differs from launched source",
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    value, bound = loaded_plan(prepared)
    cell_id = value["cell"]["id"]
    entry = fixture_catalog(value["fixture"], value["fixture_sha256"])["cells"][cell_id]
    arrays = load_arrays(value["fixture"], entry)
    verify_inputs(arrays, entry)
    policy = CosmosQualityPolicy.from_matrix(
        Path(prepared) / "matrix.json",
        cell_id,
        matrix_sha256=bound["matrix_sha256"],
        output_dir=Path(output) / "requests",
    )
    comparisons = []
    try:
        for i, row in enumerate(entry["requests"]):
            if row["request_id"] == 0:
                policy.infer(
                    {
                        "reset": True,
                        "episode_id": row["episode_id"],
                        "prompt": row["prompt"],
                        "benchmark_seed": row["benchmark_seed"],
                        "max_policy_chunks": row["max_policy_chunks"],
                        "benchmark_identity_sha256": policy.identity_sha256,
                    }
                )
            request = observation(arrays, row, i)
            request.update(
                episode_id=row["episode_id"],
                request_id=row["request_id"],
                benchmark_identity_sha256=policy.identity_sha256,
            )
            response = policy.infer(request)
            require(
                response["request_seed"] == row["request_seed"],
                "replayed request seed changed",
            )
            action = response["action"]
            comparisons.append(
                {
                    "input_sha256": row["input_sha256"],
                    "historical_action_sha256": row["action_sha256"],
                    "replay_action_sha256": request_hash(action),
                    **action_comparison(action, arrays["actions"][i]),
                }
            )
    finally:
        policy.close()
    write(Path(output) / "action_comparisons.json", comparisons)
    write(
        Path(output) / "capture_complete.json",
        {
            "requests": len(comparisons),
            "episodes": 6,
            "plan_sha256": bound["plan_sha256"],
            "task_quality_certified": False,
            "files": {
                str(p.relative_to(output)): sha(p)
                for p in sorted(Path(output).rglob("*"))
                if p.is_file() and p.suffix in {".json", ".npz"}
            },
        },
    )


def run(prepared, output, *, timeout=3600, lock_path="/tmp/thor_gpu.lock"):
    import fcntl
    from benchmarks.regression.reproduce import child_environment

    value, bound = loaded_plan(prepared)
    output = Path(output).absolute()
    require(type(timeout) is int and 0 < timeout <= 14400, "invalid deadline")
    require(not output.exists(), "run output exists")
    output.mkdir(parents=True, exist_ok=False)
    environment = child_environment(value, value["cell"], bound)
    environment["TORCHDYNAMO_DISABLE"] = (
        "1" if value["cell"]["arm"] == "eager_native" else "0"
    )
    command = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        __name__ if __name__ != "__main__" else "benchmarks.regression.screen_replay",
        "capture",
        "--prepared",
        str(Path(prepared).resolve()),
        "--output",
        str(output),
    ]
    write(
        output / "launch.json",
        {
            "command": command,
            "source_sha256": sha(__file__),
            "timeout_seconds": timeout,
        },
    )
    try:
        with (
            Path(lock_path).open("a+") as lock,
            (output / "worker.log").open("xb") as log,
        ):
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            child = subprocess.Popen(
                command,
                cwd=output,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = child.wait(timeout=timeout)
            except BaseException:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
                raise
        require(code == 0, f"capture exited {code}; see worker.log")
        write(
            output / "completion.json",
            {
                "exit_code": code,
                "task_quality_certified": False,
                "capture_sha256": sha(output / "capture_complete.json"),
            },
        )
        return report(prepared, output)
    except BaseException:
        write(
            output / "failure.json",
            {"traceback": traceback.format_exc(), "task_quality_certified": False},
        )
        raise


def report(prepared, output):
    value, bound = loaded_plan(prepared)
    output = Path(output)
    terminal = json.loads(contained(output, "completion.json").read_text())
    require(
        terminal["exit_code"] == 0
        and terminal["capture_sha256"] == sha(output / "capture_complete.json"),
        "capture did not close",
    )
    done = json.loads(contained(output, "capture_complete.json").read_text())
    require(
        done["plan_sha256"] == bound["plan_sha256"]
        and done["requests"] == value["request_count"],
        "incomplete replay",
    )
    for name, expected in done["files"].items():
        require(
            sha(contained(output, name)) == expected,
            "completed replay evidence changed",
        )
    closed = json.loads(contained(output, "requests/closed.json").read_text())
    require(
        closed["failed"] is False
        and closed["episodes"] == 6
        and closed["requests"] == value["request_count"],
        "native ledger did not close",
    )
    receipts = [
        json.loads(contained(output, f"requests/request_{i:06d}.json").read_text())
        for i in range(value["request_count"])
    ]
    entry = fixture_catalog(value["fixture"], value["fixture_sha256"])["cells"][
        value["cell"]["id"]
    ]
    for actual, original in zip(receipts, entry["requests"]):
        require(
            actual["status"] == "passed"
            and actual["input_sha256"] == original["input_sha256"]
            and actual["request_seed"] == original["request_seed"],
            "replay input or seed mismatch",
        )
    comparisons = json.loads(contained(output, "action_comparisons.json").read_text())
    require(
        len(comparisons) == value["request_count"], "action comparison count changed"
    )
    result = {
        "schema": SCHEMA,
        "cell": value["cell"]["id"],
        "requests": len(receipts),
        "replay_p50_ms": statistics.median(
            r["predict_host_seconds"] * 1000 for r in receipts
        ),
        "historical_p50_ms": statistics.median(
            r["historical_predict_host_seconds"] * 1000 for r in entry["requests"]
        ),
        "bitwise_equal_actions": sum(r["bitwise_equal"] for r in comparisons),
        "max_abs_error": max(r["max_abs_error"] for r in comparisons),
        "scope": value["scope"],
        "task_quality_certified": False,
        "historical_outcomes_replayed": False,
    }
    path = output / "report.json"
    if path.exists():
        require(json.loads(path.read_text()) == result, "existing report differs")
    else:
        write(path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--latency", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for command in ("plan", "prepare"):
        p = sub.add_parser(command)
        p.add_argument("--fixture", type=Path, required=True)
        p.add_argument("--fixture-sha256", required=True)
        p.add_argument("--cell", choices=CELLS, required=True)
        p.add_argument("--bf16-library", type=Path)
        if command == "prepare":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--cache-dir")
    for command in ("run", "capture", "report"):
        p = sub.add_parser(command)
        p.add_argument("--prepared", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if command == "run":
            p.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_fixture(args.study, args.latency, args.output)
        print(
            json.dumps(
                {
                    "cells": list(result["cells"]),
                    "manifest_sha256": sha(args.output / "manifest.json"),
                }
            )
        )
    elif args.command in ("plan", "prepare"):
        if args.command == "plan":
            print(
                json.dumps(
                    plan(
                        args.fixture, args.fixture_sha256, args.cell, args.bf16_library
                    ),
                    indent=2,
                )
            )
        else:
            prepare(
                args.fixture,
                args.fixture_sha256,
                args.cell,
                args.output,
                args.bf16_library,
                args.cache_dir,
            )
    elif args.command == "run":
        print(
            json.dumps(run(args.prepared, args.output, timeout=args.timeout), indent=2)
        )
    elif args.command == "capture":
        capture(args.prepared, args.output)
    else:
        print(json.dumps(report(args.prepared, args.output), indent=2))


if __name__ == "__main__":
    main()

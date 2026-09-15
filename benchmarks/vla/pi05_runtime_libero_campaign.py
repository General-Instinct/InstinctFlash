"""Frozen 10-task × 2-seed qualification of the current pi05 Runtime.

Run one precision arm against an already loaded, identity-pinned server. Each
episode uses a fresh simulator process and explicitly resets the same server.
Failures are retained and stop the arm; they are never replaced with new seeds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .util import ConfigurationError, sha256_file, write_json_atomic


def frozen_jobs(document):
    scenes = document.get("scenes", {})
    jobs = []
    for task in range(10):
        for replicate in range(2):
            seed = task * 10000 + replicate
            key = f"libero_10/{task}/{seed}"
            scene = scenes.get(key)
            if not isinstance(scene, dict):
                raise ConfigurationError(f"missing frozen scene {key}")
            if (scene.get("task") != f"libero_10/{task}"
                    or scene.get("resolved_seed") != seed
                    or scene.get("requested_seed") != seed):
                raise ConfigurationError(f"scene identity mismatch: {key}")
            if not scene.get("prompt") or not scene.get("init_state"):
                raise ConfigurationError(f"incomplete frozen scene {key}")
            jobs.append({"task": task, "seed": seed, "key": key})
    if set(scenes) != {j["key"] for j in jobs}:
        raise ConfigurationError("expected exactly the frozen 10-task × 2-seed matrix")
    return jobs


def validate_identity(identity):
    expected = {
        "protocol": "pi05-public-libero-50-v1",
        "model_id": "lerobot/pi05_libero_finetuned_v044",
        "model_revision": "8e174154ef5f6c60a8da12ae99c303d8963138c1",
        "n_action_steps": 50, "action_dim": 7, "action_nfe": 10,
    }
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ConfigurationError("server identity is not the qualified 50-action protocol")
    if identity.get("precision") not in {"native", "fp8"}:
        raise ConfigurationError("explicit native/fp8 identity required")


def run(args):
    from .wan_va_libero_driver import sources

    scenes = json.loads(args.scenes.read_text())
    identity = json.loads(args.identity.read_text())
    jobs = frozen_jobs(scenes)
    validate_identity(identity)
    simulator_identity = sources(os.environ["LIBERO_ROOT"])
    # Pin imported simulation code/assets before starting the first episode.
    if scenes.get("sources") != {k: simulator_identity[k] for k in ("revision", "tree_sha256")}:
        raise ConfigurationError("frozen scene source/assets differ from the current simulator")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen_scene_path = args.output / "scenes.json"
    frozen_identity_path = args.output / "identity.json"
    write_json_atomic(frozen_scene_path, scenes)
    write_json_atomic(frozen_identity_path, identity)
    manifest_path = args.output / "campaign.json"
    manifest = {
        "protocol": "pi05-public-libero-50-campaign-v1",
        "status": "running", "identity": identity,
        "scene_sha256": sha256_file(frozen_scene_path),
        "simulator_identity": simulator_identity,
        "driver_sources": {p.name: sha256_file(p) for p in (
            Path(__file__), Path(__file__).with_name("pi05_runtime_libero_scene.py"),
            Path(__file__).with_name("remote_policy.py"),
            Path(__file__).with_name("wan_va_libero_driver.py"))},
        "scope": "20 paired-scene screening episodes per arm; no real-time or non-inferiority certificate",
        "jobs": [{**j, "status": "pending"} for j in jobs],
    }
    write_json_atomic(manifest_path, manifest)
    active = None
    try:
        for job in manifest["jobs"]:
            active = job
            stem = f"task{job['task']}-seed{job['seed']}"
            output = args.output / f"{stem}.json"
            job.update(status="running", result=output.name, log=f"{stem}.log")
            write_json_atomic(manifest_path, manifest)
            command = [sys.executable, "-m", "benchmarks.vla.pi05_runtime_libero_scene",
                       "--scenes", str(frozen_scene_path), "--identity", str(frozen_identity_path),
                       "--endpoint", args.endpoint, "--output", str(output),
                       "--task", str(job["task"]), "--seed", str(job["seed"])]
            with (args.output / job["log"]).open("x") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            job["exit_code"] = completed.returncode
            if completed.returncode:
                raise RuntimeError(f"episode process failed: {stem} ({completed.returncode})")
            result = json.loads(output.read_text())
            if (result.get("ok") is not True or result.get("identity") != identity
                    or result.get("simulator_identity") != simulator_identity
                    or result.get("scene_sha256") != manifest["scene_sha256"]
                    or result.get("task") != job["task"] or result.get("seed") != job["seed"]):
                raise ConfigurationError(f"episode result identity mismatch: {stem}")
            job.update(status="complete", success=result["success"], steps=result["steps"],
                       result_sha256=sha256_file(output),
                       actions_sha256=sha256_file(output.with_suffix(".npz")),
                       trace_sha256=sha256_file(output.with_suffix(".trace") / "trace.json"))
            write_json_atomic(manifest_path, manifest)
            print(identity["precision"], stem, job["success"], job["steps"], flush=True)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        if active is not None and active["status"] == "running":
            active["status"] = "failed"
        raise
    finally:
        write_json_atomic(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="ws://127.0.0.1:19051")
    args = parser.parse_args()
    args.scenes = args.scenes.resolve()
    args.identity = args.identity.resolve()
    args.output = args.output.resolve()
    run(args)


if __name__ == "__main__":
    main()

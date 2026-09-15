"""Crash-safe, resumable, shell-free execution of an immutable benchmark plan."""

from __future__ import annotations

import concurrent.futures
import threading
import fcntl
import importlib.metadata
import os
import platform
import re
import signal
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .plan import pipeline_digest, validate_plan
from .result import validate_result
from .util import (
    ConfigurationError,
    command_output,
    expand_command,
    load_json,
    sha256_file,
    sha256_json,
    write_json_atomic,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _environment_manifest(repo_root: Path, gpu: str | None) -> dict[str, Any]:
    packages = sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    git_revision = command_output(["git", "rev-parse", "HEAD"], repo_root)
    git_status = command_output(["git", "status", "--porcelain=v1"], repo_root)
    gpu_info = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        repo_root,
    )
    return {
        "captured_utc": _utc_now(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gpu_selection": gpu,
        "nvidia_smi": gpu_info.splitlines() if gpu_info else None,
        "git_revision": git_revision,
        "git_status": git_status.splitlines() if git_status else [],
        # importlib.metadata deliberately omits direct_url.json, where credentials can appear.
        "packages": packages,
    }


def _valid_existing(path: Path, job: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    result = load_json(path)
    validate_result(result, job)
    return True


def execute_plan(
    plan: dict[str, Any],
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    gpu: str | None = None,
    fail_fast: bool = False,
    paired_workers: int = 1,
) -> dict[str, Any]:
    validate_plan(plan)
    groups = _execution_groups(plan, paired_workers)
    current_pipeline = pipeline_digest()
    if plan["pipeline_sha256"] != current_pipeline:
        raise ConfigurationError(
            f"plan was built by pipeline {plan['pipeline_sha256']}, current pipeline is {current_pipeline}"
        )
    root = Path(output_dir).resolve()
    source_root = Path(repo_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".pipeline.lock"
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_stream.close()
        raise ConfigurationError(f"another benchmark process owns {lock_path}") from error

    gpu_lock_stream = None
    if gpu is not None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(gpu)):
            lock_stream.close()
            raise ConfigurationError("--gpu must name exactly one ordinal or GPU UUID")
        gpu_lock_path = Path(tempfile.gettempdir()) / f"instinctflash-vla-gpu-{gpu}.lock"
        gpu_lock_stream = gpu_lock_path.open("a+")
        try:
            fcntl.flock(gpu_lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            gpu_lock_stream.close()
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()
            raise ConfigurationError(f"GPU {gpu} is owned by another benchmark: {gpu_lock_path}") from error

    try:
        plan_path = root / "plan.json"
        if plan_path.exists():
            existing = load_json(plan_path)
            if existing.get("plan_id") != plan["plan_id"]:
                raise ConfigurationError(
                    f"{root} belongs to plan {existing.get('plan_id')}, not {plan['plan_id']}"
                )
        else:
            write_json_atomic(plan_path, plan)
        environment_path = root / "environment.json"
        if not environment_path.exists():
            write_json_atomic(environment_path, _run_environment(source_root, gpu, paired_workers))
        else:
            # Resume must run in the environment the evidence started in. Driver-side drift is
            # caught by the per-result environment_fingerprint; this closes the orchestrator
            # side: recapture and refuse when anything but the capture timestamp moved.
            recaptured = _run_environment(source_root, gpu, paired_workers)
            original = load_json(environment_path)
            drifted = sorted(
                key
                for key in set(original) | set(recaptured)
                if key != "captured_utc" and original.get(key) != recaptured.get(key)
            )
            if drifted:
                raise ConfigurationError(
                    f"environment drifted since this run started ({', '.join(drifted)}); "
                    f"resume in the original environment or start a fresh run directory"
                )
        environment_sha256 = sha256_json(load_json(environment_path))
        run_manifest = {
            "schema_version": 1,
            "plan_id": plan["plan_id"],
            "pipeline_sha256": plan["pipeline_sha256"],
            "registry_sha256": plan["registry_sha256"],
            "environment_sha256": environment_sha256,
        }
        run_manifest_path = root / "run_manifest.json"
        if run_manifest_path.exists():
            if load_json(run_manifest_path) != run_manifest:
                raise ConfigurationError(f"run manifest drifted: {run_manifest_path}")
        else:
            write_json_atomic(run_manifest_path, run_manifest)

        requests = root / "requests"
        results = root / "results"
        logs = root / "logs"
        failures = root / "failures"
        for directory in (requests, results, logs, failures):
            directory.mkdir(parents=True, exist_ok=True)

        completed = 0
        failed: list[dict[str, Any]] = []
        skipped = 0
        active = {}
        process_lock = threading.Lock()
        cancelled = threading.Event()
        def execute_one(position, job):
            result_path = results / f"{job['job_id']}.json"
            try:
                if _valid_existing(result_path, job):
                    return True, None
            except ConfigurationError as error:
                raise ConfigurationError(
                    f"refusing to overwrite stale/invalid existing result {result_path}: {error}"
                ) from error

            request_path = requests / f"{job['job_id']}.json"
            _archive_previous_attempt(root, job['job_id'])
            write_json_atomic(request_path, job)
            pending_path = results / f".{job['job_id']}.pending.json"
            if pending_path.exists():
                pending_path.unlink()
            driver = job["driver"]
            command = expand_command(driver["command"])
            executable = command[0]
            if os.path.sep not in executable and shutil.which(executable) is None:
                raise ConfigurationError(f"driver executable is not on PATH: {executable}")
            if os.path.sep in executable and not Path(executable).is_file():
                raise ConfigurationError(f"driver executable does not exist: {executable}")
            command += ["--request", str(request_path), "--output", str(pending_path)]
            environment = dict(os.environ)
            environment.update(driver["environment"])
            environment.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "PYTHONHASHSEED": "0",
                    "IFL_BENCH_PLAN_ID": plan["plan_id"],
                    "IFL_BENCH_JOB_ID": job["job_id"],
                }
            )
            if gpu is not None:
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = logs / f"{job['job_id']}.log"
            started = _utc_now()
            returncode = -1
            detail = ""
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"started_utc={started}\n")
                log.write(f"argv={command!r}\n")
                log.flush()
                try:
                    with process_lock:
                        if cancelled.is_set():
                            raise RuntimeError("benchmark execution cancelled")
                        process = subprocess.Popen(
                            command,
                            cwd=source_root,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            text=True,
                            start_new_session=True,
                        )
                        active[job["job_id"]] = process
                    try:
                        returncode = process.wait(timeout=int(driver["timeout_seconds"]))
                    except subprocess.TimeoutExpired:
                        _terminate_process_group(process)
                        returncode = 124
                        detail = f"timeout after {driver['timeout_seconds']} seconds"
                    else:
                        detail = f"exit {returncode}"
                except BaseException:
                    if "process" in locals() and process.poll() is None:
                        _terminate_process_group(process)
                    raise
                finally:
                    with process_lock:
                        active.pop(job["job_id"], None)
                log.write(f"\nfinished_utc={_utc_now()}\nreturncode={returncode}\n")

            try:
                if returncode != 0:
                    raise ConfigurationError(detail)
                if not pending_path.exists():
                    raise ConfigurationError("driver exited successfully without writing its output")
                result = load_json(pending_path)
                validate_result(result, job)
                os.replace(pending_path, result_path)
                # The previous error is retained in attempts; the current job is now complete.
                (failures / f"{job['job_id']}.json").unlink(missing_ok=True)
            except ConfigurationError as error:
                failure = {
                    "job_id": job["job_id"],
                    "position": position,
                    "returncode": returncode,
                    "error": str(error),
                    "log": str(log_path.relative_to(root)),
                }
                write_json_atomic(failures / f"{job['job_id']}.json", failure)
                return False, failure
            return False, None

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=paired_workers)
        try:
            for group in groups:
                futures = [pool.submit(execute_one, position, job) for position, job in group]
                for future in concurrent.futures.as_completed(futures):
                    resumed, failure = future.result()
                    if failure is None:
                        completed += 1
                        skipped += int(resumed)
                    else:
                        failed.append(failure)
                        failed.sort(key=lambda item: item["position"])
                    write_json_atomic(root / "progress.json", _progress(plan, completed, skipped, failed))
                # Both already-started arms finish; never start a later pair after fail-fast.
                if failed and fail_fast:
                    break
        except BaseException:
            with process_lock:
                cancelled.set()
                processes = list(active.values())
            for process in processes:
                _terminate_process_group(process)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        progress = _progress(plan, completed, skipped, failed)
        write_json_atomic(root / "progress.json", progress)
        return progress
    finally:
        if gpu_lock_stream is not None:
            fcntl.flock(gpu_lock_stream.fileno(), fcntl.LOCK_UN)
            gpu_lock_stream.close()
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def _progress(
    plan: dict[str, Any], completed: int, skipped: int, failed: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "updated_utc": _utc_now(),
        "expected": len(plan["jobs"]),
        "completed": completed,
        "resumed": skipped,
        "failed": failed,
        "finished": completed + len(failed) == len(plan["jobs"]),
        "progress_sha256": sha256_json(
            {"plan_id": plan["plan_id"], "completed": completed, "failed": failed}
        ),
    }


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate a driver and any policy servers it spawned, then escalate after five seconds."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_environment(source_root, gpu, paired_workers):
    manifest = _environment_manifest(source_root, gpu)
    # Preserve the legacy serial manifest while binding any parallel scheduling choice.
    if paired_workers != 1:
        manifest["paired_workers"] = paired_workers
    return manifest


def _archive_previous_attempt(root, job_id):
    """Retain interrupted/failed evidence before a resume rewrites per-job files."""
    paths = [root / 'requests' / f'{job_id}.json',
             root / 'logs' / f'{job_id}.log',
             root / 'failures' / f'{job_id}.json',
             root / 'results' / f'.{job_id}.pending.json']
    existing = [path for path in paths if path.is_file()]
    trace = root / 'results' / f'.{job_id}.pending.trace'
    if trace.is_dir():
        existing.extend(path for path in sorted(trace.rglob('*')) if path.is_file())
    if not existing and not trace.is_dir():
        return
    attempts = root / 'failures' / 'attempts' / job_id
    attempts.mkdir(parents=True, exist_ok=True)
    number = 1 + max((int(path.name) for path in attempts.iterdir()
                      if path.is_dir() and path.name.isdigit()), default=0)
    target = attempts / f'{number:04d}'
    target.mkdir()
    inventory = {}
    for path in existing:
        relative = path.relative_to(root)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        inventory[str(relative)] = sha256_file(destination)
    write_json_atomic(target / 'attempt.json', {
        'job_id': job_id, 'archived_utc': _utc_now(), 'files': inventory,
        'scope': 'Previous attempt before resume; not a completed policy outcome',
    })
    # Only remove the previous recording after its complete inventory is durable.
    # The fresh attempt must be able to create its own trace directory exclusively.
    if trace.is_dir():
        shutil.rmtree(trace)


def _execution_groups(plan, paired_workers):
    if type(paired_workers) is not int or paired_workers not in {1, 2}:
        raise ConfigurationError("paired_workers must be 1 or 2")
    jobs = list(enumerate(plan["jobs"]))
    if paired_workers == 1:
        return [[item] for item in jobs]
    groups = []
    seen = set()
    for item in jobs:
        request = item[1]["request"]
        if request["suite"]["kind"] != "closed_loop":
            raise ConfigurationError("paired concurrency is only for closed-loop quality runs")
        pair = request["pair_id"]
        if not groups or groups[-1][0][1]["request"]["pair_id"] != pair:
            if pair in seen:
                raise ConfigurationError("parallel pairs must be contiguous in the frozen plan")
            seen.add(pair)
            groups.append([])
        groups[-1].append(item)
    for group in groups:
        endpoints = [job["request"]["arm"]["operating_point"].get("remote", {}).get("endpoint") for _, job in group]
        if len(group) != 2 or any(not endpoint for endpoint in endpoints) or len(set(endpoints)) != 2:
            raise ConfigurationError("parallel pairs require exactly two independent remote endpoints")
    return groups

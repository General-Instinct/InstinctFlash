"""Preflight a plan without importing model frameworks or allocating a GPU."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .plan import pipeline_digest, validate_plan
from .registry import Registry
from .util import ConfigurationError, expand_command


def _hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    return hf_home / "hub"


def _repo_cache_path(repo_type: str, repo_id: str) -> Path:
    prefix = "models" if repo_type == "model" else "datasets"
    return _hub_cache() / f"{prefix}--{repo_id.replace('/', '--')}"


def inspect_plan(plan: dict[str, Any], registry: Registry, *, require_cached: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        validate_plan(plan)
    except ConfigurationError as error:
        return {"ok": False, "errors": [str(error)], "warnings": [], "coverage": {}}
    if plan["registry_sha256"] != registry.digest:
        errors.append(
            f"plan registry {plan['registry_sha256']} differs from current registry {registry.digest}"
        )
    current_pipeline = pipeline_digest()
    if plan["pipeline_sha256"] != current_pipeline:
        errors.append(
            f"plan pipeline {plan['pipeline_sha256']} differs from current pipeline {current_pipeline}"
        )

    commands = {}
    for arm in plan["arms"]:
        try:
            command = expand_command(arm["driver"]["command"])
        except ConfigurationError as error:
            errors.append(f"arm {arm['id']}: {error}")
            continue
        executable = command[0]
        exists = (
            Path(executable).is_file()
            if os.path.sep in executable
            else shutil.which(executable) is not None
        )
        if not exists:
            errors.append(f"arm {arm['id']}: driver executable not found: {executable}")
        commands[arm["id"]] = command
        if exists and arm["operating_point"].get("optimization") == "instinctflash_capture":
            try:
                probe = subprocess.check_output(
                    [executable, "-c", "import importlib.metadata as m; print(any(e.name == 'pi05' for e in m.entry_points(group='instinctflash.adapters')))"],
                    env={**os.environ, **arm['driver'].get('environment', {})},
                    text=True, stderr=subprocess.PIPE, timeout=15).strip()
                if probe != "True":
                    errors.append(f"arm {arm['id']}: pi05-iwm adapter entry point is missing from the driver interpreter")
            except (OSError, subprocess.SubprocessError) as error:
                errors.append(f"arm {arm['id']}: cannot inspect pi05 adapter installation: {error}")

    selected = [registry.models[model_id] for model_id in plan["selected_models"]]
    for model in selected:
        cache = _repo_cache_path("model", model["id"])
        snapshot = cache / "snapshots" / model["revision"]
        if not snapshot.is_dir():
            message = f"model not cached at locked revision: {model['id']}@{model['revision']}"
            (errors if require_cached else warnings).append(message)
        missing_env = [name for name in model["source_env"] if not os.environ.get(name)]
        if missing_env:
            warnings.append(
                f"{model['id']}: external source variable(s) not set: {', '.join(missing_env)}; "
                "a self-contained driver may still provide them"
            )

    # Per-arm checkpoint overrides are the weights an expensive treatment run will actually
    # load (a quantized arm's artifact, not the registry parent), so preflight them with the
    # same cached-at-locked-revision rule as the registry models.
    selected_ids = set(plan["selected_models"])
    hub_repo_id = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")
    for arm in plan["arms"]:
        overrides = arm["operating_point"].get("checkpoint_overrides", {})
        for model_id, override in sorted(overrides.items()):
            if model_id not in selected_ids:
                continue
            if Path(override["id"]).is_dir():
                continue                                   # an immutable local package, present
            if hub_repo_id.fullmatch(override["id"]):
                snapshot = (
                    _repo_cache_path("model", override["id"]) / "snapshots" / override["revision"]
                )
                if not snapshot.is_dir():
                    message = (
                        f"arm {arm['id']}: override checkpoint not cached at locked revision: "
                        f"{override['id']}@{override['revision']}"
                    )
                    (errors if require_cached else warnings).append(message)
                continue
            message = (
                f"arm {arm['id']}: override checkpoint directory does not exist: "
                f"{override['id']}"
            )
            (errors if require_cached else warnings).append(message)

    used_datasets = {job["request"]["dataset"]["id"] for job in plan["jobs"]}
    for dataset_id in sorted(used_datasets):
        dataset = registry.datasets[dataset_id]
        if dataset["kind"] == "synthetic":
            continue
        if dataset["kind"] == "simulator":
            roots = set()
            for job in plan["jobs"]:
                if job["request"]["dataset"]["id"] != dataset_id:
                    continue
                environment = job["driver"].get("environment", {})
                for name in ("LIBERO_ROOT", "ROBOTWIN_ROOT"):
                    value = environment.get(name) or os.environ.get(name)
                    if value:
                        roots.add(str(Path(value).expanduser()))
            if not roots:
                message = f"simulator {dataset_id}: no explicit local LIBERO_ROOT/ROBOTWIN_ROOT to inspect"
                (errors if require_cached else warnings).append(message)
            for root in sorted(roots):
                try:
                    revision = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
                        stderr=subprocess.PIPE, timeout=10).strip()
                    if revision != dataset["revision"]:
                        errors.append(f"simulator checkout {root} has revision {revision}, expected {dataset['revision']}")
                except (OSError, subprocess.SubprocessError) as error:
                    errors.append(f"cannot inspect simulator checkout {root}: {error}")
            warnings.append(f"simulator {dataset_id}: drivers additionally verify source/assets and frozen reset observations")
        else:
            cache = _repo_cache_path("dataset", dataset["source"])
            snapshot = cache / "snapshots" / dataset["revision"]
            if not snapshot.is_dir():
                message = f"dataset not cached at locked revision: {dataset['source']}@{dataset['revision']}"
                (errors if require_cached else warnings).append(message)
        for dependency in dataset.get("dependencies", ()):
            source = dependency["source"]
            if source.startswith(("http://", "https://")):
                warnings.append(
                    f"source checkout must be verified separately: {source}@{dependency['revision']}"
                )
                continue
            dependency_snapshot = (
                _repo_cache_path("dataset", source) / "snapshots" / dependency["revision"]
            )
            if not dependency_snapshot.is_dir():
                message = (
                    f"dataset dependency not cached at locked revision: "
                    f"{source}@{dependency['revision']}"
                )
                (errors if require_cached else warnings).append(message)

    coverage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for job in plan["jobs"]:
        request = job["request"]
        coverage[request["model_id"]][request["suite_id"]] += 1
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "commands": commands,
        "coverage": {model: dict(suites) for model, suites in sorted(coverage.items())},
    }

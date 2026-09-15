"""Compile a registry/profile/arm declaration into an immutable paired job plan."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Iterable

from .adapters import bind_adapter
from .registry import Registry
from .util import (
    ENV_ITEM,
    SECRET_NAME,
    ConfigurationError,
    load_json,
    require_keys,
    require_revision,
    sha256_json,
    validate_public_environment,
)


ROLES = {"control", "treatment"}
TIERS = {"BITEXACT", "NUMERIC", "BEHAVIORAL"}
HERE = Path(__file__).resolve().parent


def pipeline_digest() -> str:
    """Content identity of every shipped pipeline source, protocol, and configuration file."""
    digest = hashlib.sha256()
    paths = sorted(
        path for path in HERE.rglob("*")
        if path.is_file() and path.suffix in {".py", ".json", ".md"}
        and "__pycache__" not in path.parts
        and "runs" not in path.relative_to(HERE).parts
    )
    for path in paths:
        digest.update(path.relative_to(HERE).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_arms(path: str | Path) -> dict[str, Any]:
    arms = load_json(Path(path))
    validate_arms(arms)
    return arms


def validate_arms(spec: dict[str, Any]) -> None:
    require_keys(spec, ("schema_version", "control_arm", "arms"), "arms")
    if spec["schema_version"] != 1:
        raise ConfigurationError(f"arms: unsupported schema_version {spec['schema_version']!r}")
    if not isinstance(spec["arms"], list) or len(spec["arms"]) < 2:
        raise ConfigurationError("arms: at least one control and one treatment arm are required")
    identifiers = [arm.get("id") for arm in spec["arms"]]
    if None in identifiers or len(set(identifiers)) != len(identifiers):
        raise ConfigurationError("arms: ids must be non-empty and unique")
    if spec["control_arm"] not in identifiers:
        raise ConfigurationError("arms: control_arm does not name an arm")
    controls = 0
    for index, arm in enumerate(spec["arms"]):
        where = f"arms[{index}]"
        require_keys(arm, ("id", "role", "driver", "operating_point"), where)
        if arm["role"] not in ROLES:
            raise ConfigurationError(f"{where}: unsupported role {arm['role']!r}")
        if arm["role"] == "control":
            controls += 1
            if arm["id"] != spec["control_arm"]:
                raise ConfigurationError(f"{where}: only control_arm may have role=control")
        driver = arm["driver"]
        require_keys(
            driver, ("command", "environment", "timeout_seconds", "revision"), f"{where}.driver"
        )
        if not isinstance(driver["command"], list) or not driver["command"]:
            raise ConfigurationError(f"{where}.driver.command must be a non-empty argv list")
        if not all(isinstance(item, str) and item for item in driver["command"]):
            raise ConfigurationError(f"{where}.driver.command entries must be non-empty strings")
        for item in driver["command"]:
            match = ENV_ITEM.fullmatch(item)
            if match and SECRET_NAME.search(match.group(1)):
                raise ConfigurationError(
                    f"{where}.driver.command references secret-like variable {match.group(1)}; "
                    "driver commands are logged"
                )
        if not isinstance(driver["revision"], str) or not driver["revision"]:
            raise ConfigurationError(f"{where}.driver.revision must be a non-empty immutable identity")
        revision_ref = ENV_ITEM.fullmatch(driver["revision"])
        if revision_ref and SECRET_NAME.search(revision_ref.group(1)):
            raise ConfigurationError(
                f"{where}.driver.revision references secret-like variable {revision_ref.group(1)}"
            )
        if not isinstance(driver["environment"], dict):
            raise ConfigurationError(f"{where}.driver.environment must be an object")
        validate_public_environment(driver["environment"], f"{where}.driver.environment")
        if not isinstance(driver["timeout_seconds"], int) or driver["timeout_seconds"] <= 0:
            raise ConfigurationError(f"{where}.driver.timeout_seconds must be a positive integer")
        operating_point = arm["operating_point"]
        require_keys(operating_point, ("name", "tier"), f"{where}.operating_point")
        if operating_point["tier"] not in TIERS:
            raise ConfigurationError(f"{where}: unsupported tier {operating_point['tier']!r}")
        overrides = operating_point.get("checkpoint_overrides", {})
        if not isinstance(overrides, dict):
            raise ConfigurationError(f"{where}: checkpoint_overrides must be an object")
        for model_id, override in overrides.items():
            require_keys(override, ("id", "revision", "derived_from_revision"), f"{where}.{model_id}")
            require_revision(override["revision"], f"{where}.{model_id}")
            require_revision(override["derived_from_revision"], f"{where}.{model_id}.derived_from")
        if arm["role"] == "treatment":
            require_keys(arm, ("gates",), where)
            _validate_gates(arm["gates"], f"{where}.gates")
    if controls != 1:
        raise ConfigurationError(f"arms: expected exactly one control arm, found {controls}")


def _validate_gates(gates: dict[str, Any], where: str) -> None:
    require_keys(gates, ("performance", "action", "success"), where)
    if float(gates["performance"].get("min_speedup", 0.0)) <= 0.0:
        raise ConfigurationError(f"{where}.performance.min_speedup must be positive")
    action = gates["action"]
    if action.get("mode") not in {"registry", "bitexact", "numeric"}:
        raise ConfigurationError(f"{where}.action.mode must be registry, bitexact, or numeric")
    if action.get("mode") == "numeric":
        if float(action.get("max_abs", -1.0)) < 0.0:
            raise ConfigurationError(f"{where}.action.max_abs must be non-negative")
        cosine = float(action.get("min_cosine", -1.0))
        if not -1.0 <= cosine <= 1.0:
            raise ConfigurationError(f"{where}.action.min_cosine must be in [-1, 1]")
    success = gates["success"]
    margin = float(success.get("margin", 1.0))
    if not -1.0 < margin < 0.0:
        raise ConfigurationError(f"{where}.success.margin must be strictly between -1 and 0")
    if success.get("interval") not in {"wald_central95", "tango_one_sided95"}:
        raise ConfigurationError(f"{where}.success.interval is unsupported")
    if int(success.get("min_pairs", 0)) < 1:
        raise ConfigurationError(f"{where}.success.min_pairs must be positive")


def _count_for_suite(value: int | dict[str, int], kind: str, suite_id: str) -> int:
    count = value.get(suite_id, value.get(kind)) if isinstance(value, dict) else value
    if not isinstance(count, int) or count < 1:
        raise ConfigurationError(f"profile count for {kind} must be a positive integer")
    return count


def _eligible(model: dict[str, Any], suite: dict[str, Any]) -> bool:
    if model["backbone"] not in suite["compatible_backbones"]:
        return False
    constrained = suite.get("model_ids")
    return not constrained or model["id"] in constrained


def _selected_models(registry: Registry, selected: Iterable[str] | None) -> list[dict[str, Any]]:
    models = registry.models
    identifiers = sorted(selected) if selected is not None else sorted(registry.builtin_model_ids)
    unknown = sorted(set(identifiers) - set(models))
    if unknown:
        raise ConfigurationError(f"unknown selected model(s): {', '.join(unknown)}")
    return [models[identifier] for identifier in identifiers]


def _arm_checkpoint(model: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    override = arm["operating_point"].get("checkpoint_overrides", {}).get(model["id"])
    if override is None:
        checkpoint = {"id": model["id"], "revision": model["revision"], "derived_from_revision": model["revision"]}
        if "checkpoint_subdir" in arm["operating_point"]:
            checkpoint["subdir"] = arm["operating_point"]["checkpoint_subdir"]
        return checkpoint
    if override["derived_from_revision"] != model["revision"]:
        raise ConfigurationError(
            f"arm {arm['id']} checkpoint for {model['id']} derives from "
            f"{override['derived_from_revision']}, expected registry revision {model['revision']}"
        )
    return copy.deepcopy(override)


def _resolved_arms(spec: dict[str, Any]) -> list[dict[str, Any]]:
    arms = copy.deepcopy(spec["arms"])
    for arm in arms:
        revision = arm["driver"]["revision"]
        match = ENV_ITEM.fullmatch(revision)
        if match:
            import os

            name = match.group(1)
            if not os.environ.get(name):
                raise ConfigurationError(f"arm {arm['id']} requires driver revision variable {name}")
            arm["driver"]["revision"] = os.environ[name]
    return arms


def build_plan(
    registry: Registry,
    arms: dict[str, Any],
    profile_name: str,
    selected_models: Iterable[str] | None = None,
) -> dict[str, Any]:
    validate_arms(arms)
    if profile_name not in registry.profiles:
        raise ConfigurationError(f"unknown profile {profile_name!r}")
    profile = registry.profiles[profile_name]
    models = _selected_models(registry, selected_models)
    arm_entries = _resolved_arms(arms)
    suites = registry.suites
    jobs: list[dict[str, Any]] = []
    pair_count = 0

    for model in models:
        for suite_id in profile["suites"]:
            suite = suites[suite_id]
            if not _eligible(model, suite):
                continue
            task_limit = profile["limits"]["tasks"]
            tasks = suite["tasks"] if task_limit is None else suite["tasks"][: int(task_limit)]
            seeds_per_task = _count_for_suite(
                profile["limits"]["seeds_per_task"], suite["kind"], suite_id
            )
            repeats = _count_for_suite(profile["arm_repeats"], suite["kind"], suite_id)
            for task_index, task in enumerate(tasks):
                for seed_index in range(seeds_per_task):
                    requested_seed = int(suite["seed_base"]) + task_index * 10_000 + seed_index
                    for repeat in range(repeats):
                        pair_identity = {
                            "model_id": model["id"],
                            "suite_id": suite_id,
                            "task": task,
                            "requested_seed": requested_seed,
                            "repeat": repeat,
                        }
                        pair_id = sha256_json(pair_identity)[:24]
                        pair_count += 1
                        # Counterbalance arm order deterministically to avoid a fixed thermal/order bias.
                        ordered_arms = (
                            arm_entries
                            if int(pair_id[:2], 16) % 2 == 0
                            else list(reversed(arm_entries))
                        )
                        for arm in ordered_arms:
                            request = {
                                "schema_version": 1,
                                "pair_id": pair_id,
                                **pair_identity,
                                "model": {
                                    "registry_id": model["id"],
                                    "backbone": model["backbone"],
                                    "checkpoint": _arm_checkpoint(model, arm),
                                },
                                "dataset": copy.deepcopy(registry.datasets[suite["dataset"]]),
                                "suite": {
                                    "id": suite_id,
                                    "kind": suite["kind"],
                                    "seed_strategy": suite["seed_strategy"],
                                    "seed_max_attempts": int(suite.get("seed_max_attempts", 1)),
                                    "protocol": copy.deepcopy(suite.get("protocol", {})),
                                    "required_metrics": list(suite["required_metrics"]),
                                },
                                "arm": {
                                    "id": arm["id"],
                                    "role": arm["role"],
                                    "operating_point": copy.deepcopy(arm["operating_point"]),
                                },
                                "measurement": {
                                    "warmup": int(profile["latency"]["warmup"]),
                                    "iterations": int(profile["latency"]["iterations"]),
                                },
                            }
                            bind_adapter(request, arm["driver"])
                            request_hash = sha256_json(request)
                            jobs.append(
                                {
                                    "job_id": request_hash[:24],
                                    "request_sha256": request_hash,
                                    "request": request,
                                    "driver": copy.deepcopy(arm["driver"]),
                                }
                            )

    covered = {job["request"]["model_id"] for job in jobs}
    missing = sorted(model["id"] for model in models if model["id"] not in covered)
    if missing:
        raise ConfigurationError(f"profile {profile_name} covers no suites for: {', '.join(missing)}")
    body = {
        "schema_version": 1,
        "pipeline_sha256": pipeline_digest(),
        "registry_sha256": registry.digest,
        "profile": profile_name,
        "control_arm": arms["control_arm"],
        "arms": arm_entries,
        "selected_models": [model["id"] for model in models],
        "pair_count": pair_count,
        "job_count": len(jobs),
        "jobs": jobs,
    }
    body["plan_id"] = sha256_json(body)
    validate_plan(body)
    return body


def validate_plan(plan: dict[str, Any]) -> None:
    require_keys(
        plan,
        (
            "schema_version", "pipeline_sha256", "registry_sha256", "profile",
            "control_arm", "arms", "jobs", "plan_id",
        ),
        "plan",
    )
    if plan["schema_version"] != 1:
        raise ConfigurationError(f"plan: unsupported schema_version {plan['schema_version']!r}")
    unsigned = dict(plan)
    claimed = unsigned.pop("plan_id")
    actual = sha256_json(unsigned)
    if claimed != actual:
        raise ConfigurationError(f"plan: digest mismatch, claimed {claimed}, computed {actual}")
    job_ids = [job.get("job_id") for job in plan["jobs"]]
    if None in job_ids or len(set(job_ids)) != len(job_ids):
        raise ConfigurationError("plan: job ids must be present and unique")
    arm_ids = {arm["id"] for arm in plan["arms"]}
    arms = {arm["id"]: arm for arm in plan["arms"]}
    selected_models = set(plan.get("selected_models", ()))
    pairs: dict[str, set[str]] = {}
    for index, job in enumerate(plan["jobs"]):
        where = f"plan jobs[{index}]"
        require_keys(job, ("job_id", "request_sha256", "request", "driver"), where)
        if sha256_json(job["request"]) != job["request_sha256"]:
            raise ConfigurationError(f"{where}: request digest mismatch")
        if job["job_id"] != job["request_sha256"][:24]:
            raise ConfigurationError(f"{where}: job id is not derived from request digest")
        request = job["request"]
        require_keys(
            request,
            (
                "schema_version", "pair_id", "model_id", "suite_id", "task",
                "requested_seed", "repeat", "model", "dataset", "suite", "arm",
                "measurement",
            ),
            f"{where}.request",
        )
        if request["schema_version"] != 1:
            raise ConfigurationError(f"{where}: unsupported request schema")
        if request["arm"]["id"] not in arm_ids:
            raise ConfigurationError(f"{where}: unknown arm {request['arm']['id']!r}")
        arm = arms[request["arm"]["id"]]
        if request["arm"]["role"] != arm["role"]:
            raise ConfigurationError(f"{where}: request arm role disagrees with plan arm")
        if request["arm"]["operating_point"] != arm["operating_point"]:
            raise ConfigurationError(f"{where}: request operating point disagrees with plan arm")
        if job["driver"] != arm["driver"]:
            raise ConfigurationError(f"{where}: request driver disagrees with plan arm")
        if request["model_id"] not in selected_models:
            raise ConfigurationError(f"{where}: request model is not selected by the plan")
        if request["model"]["registry_id"] != request["model_id"]:
            raise ConfigurationError(f"{where}: model registry identity mismatch")
        pair_identity = {
            "model_id": request["model_id"],
            "suite_id": request["suite_id"],
            "task": request["task"],
            "requested_seed": request["requested_seed"],
            "repeat": request["repeat"],
        }
        if request["pair_id"] != sha256_json(pair_identity)[:24]:
            raise ConfigurationError(f"{where}: pair id is not derived from pair identity")
        pairs.setdefault(request["pair_id"], set()).add(request["arm"]["id"])
    incomplete = [pair for pair, present in pairs.items() if present != arm_ids]
    if incomplete:
        raise ConfigurationError(f"plan: {len(incomplete)} pair(s) do not contain every arm")

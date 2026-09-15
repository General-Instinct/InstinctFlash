"""Standard result contract shared by real model drivers and the pipeline."""

from __future__ import annotations

import math
import hashlib
import struct
import re
from typing import Any

from .util import ConfigurationError, require_keys, sha256_json


DIGEST = re.compile(r"[0-9a-f]{64}")


def validate_result(result: dict[str, Any], job: dict[str, Any]) -> None:
    where = f"result for job {job['job_id']}"
    require_keys(
        result,
        (
            "schema_version", "job_id", "request_sha256", "status", "resolved_seed",
            "metrics", "provenance",
        ),
        where,
    )
    if result["schema_version"] != 1:
        raise ConfigurationError(f"{where}: unsupported schema_version {result['schema_version']!r}")
    if result["job_id"] != job["job_id"]:
        raise ConfigurationError(f"{where}: job_id mismatch")
    if result["request_sha256"] != job["request_sha256"]:
        raise ConfigurationError(f"{where}: request digest mismatch")
    if result["status"] != "completed":
        raise ConfigurationError(f"{where}: driver status is not completed")
    request = job["request"]
    seed = result["resolved_seed"]
    requested = request["requested_seed"]
    if type(seed) is not int:
        raise ConfigurationError(f"{where}: resolved_seed must be an integer")
    if request["suite"]["seed_strategy"] == "fixed" and seed != requested:
        raise ConfigurationError(f"{where}: fixed seed changed from {requested} to {seed}")
    maximum = requested + request["suite"]["seed_max_attempts"]
    if not requested <= seed < maximum:
        raise ConfigurationError(f"{where}: resolved_seed {seed} is outside [{requested}, {maximum})")

    metrics = result["metrics"]
    if not isinstance(metrics, dict):
        raise ConfigurationError(f"{where}: metrics must be an object")
    for metric in request["suite"]["required_metrics"]:
        if metric not in metrics:
            raise ConfigurationError(f"{where}: missing required metric {metric!r}")
    if metrics.get("finite") is not True:
        raise ConfigurationError(f"{where}: finite must be true")
    if "success" in metrics and not isinstance(metrics["success"], bool):
        raise ConfigurationError(f"{where}: success must be boolean")
    if "latency_ms" in metrics:
        samples = metrics["latency_ms"]
        if not isinstance(samples, list) or not samples:
            raise ConfigurationError(f"{where}: latency_ms must be a non-empty list")
        if any(
            not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0
            for item in samples
        ):
            raise ConfigurationError(f"{where}: latency samples must be finite and positive")
    if "action_digest" in metrics and not DIGEST.fullmatch(str(metrics["action_digest"])):
        raise ConfigurationError(f"{where}: action_digest must be a lowercase SHA-256")
    if "action_values" in metrics:
        values = metrics["action_values"]
        if not isinstance(values, list) or not values:
            raise ConfigurationError(f"{where}: action_values must be a non-empty flat list")
        if any(not isinstance(item, (int, float)) or not math.isfinite(item) for item in values):
            raise ConfigurationError(f"{where}: action_values must contain only finite numbers")

    provenance = result["provenance"]
    require_keys(
        provenance,
        ("model_revision", "driver_revision", "environment_fingerprint", "synthetic"),
        f"{where}.provenance",
    )
    expected_revision = request["model"]["checkpoint"]["revision"]
    if provenance["model_revision"] != expected_revision:
        raise ConfigurationError(
            f"{where}: driver loaded model revision {provenance['model_revision']}, "
            f"expected {expected_revision}"
        )
    if not isinstance(provenance["driver_revision"], str) or not provenance["driver_revision"]:
        raise ConfigurationError(f"{where}: driver_revision must be a non-empty string")
    if provenance["driver_revision"] != job["driver"]["revision"]:
        raise ConfigurationError(
            f"{where}: driver revision {provenance['driver_revision']!r} does not match "
            f"planned revision {job['driver']['revision']!r}"
        )
    if not DIGEST.fullmatch(str(provenance["environment_fingerprint"])):
        raise ConfigurationError(f"{where}: environment_fingerprint must be a SHA-256")
    if not isinstance(provenance["synthetic"], bool):
        raise ConfigurationError(f"{where}: synthetic must be boolean")
    bridge = request["suite"].get("protocol", {}).get("bridge")
    _validate_controller_trace(metrics, bridge, where)
    if bridge in {"groot-libero-paused-v1", "wan-va-robotwin-paused-v1", "wan-va-libero-paused-v1", "pi05-libero-schedule-paused-v1", "lingbot_vla-joint-robotwin-paused-v1", "lingbot_vla_v2-joint-robotwin-paused-v1"}:
        point = request["arm"]["operating_point"]
        expected = point.get("scene_manifest", {}).get("sha256")
        scene = provenance.get("scene")
        if not expected or provenance.get("scene_manifest_sha256") != expected:
            raise ConfigurationError(f"{where}: frozen scene manifest identity mismatch")
        if not isinstance(scene, dict) or provenance.get("scene_sha256") != sha256_json(scene):
            raise ConfigurationError(f"{where}: missing or corrupt scene evidence")
        if (scene.get("resolved_seed"), scene.get("requested_seed"), scene.get("task"), scene.get("suite_id")) != (
                seed, requested, request["task"], request["suite"]["id"]):
            raise ConfigurationError(f"{where}: scene identity disagrees with job")
        if provenance.get("evaluation_mode") != "paused_simulation":
            raise ConfigurationError(f"{where}: unsupported evaluation mode")


def _validate_controller_trace(metrics, bridge, where):
    dimensions = {"groot-libero-paused-v1": 7, "pi05-libero-schedule-paused-v1": 7,
                  "lingbot_vla-joint-robotwin-paused-v1": 14,
                  "lingbot_vla_v2-joint-robotwin-paused-v1": 14,
                  "wan-va-robotwin-paused-v1": 16}
    if bridge not in dimensions or "action_values" not in metrics:
        return
    steps = metrics.get("executed_steps")
    values = metrics["action_values"]
    if type(steps) is not int or steps <= 0 or len(values) != steps * dimensions[bridge]:
        raise ConfigurationError(f"{where}: controller trace length disagrees with executed steps")
    digest = hashlib.sha256(b"".join(struct.pack("!d", value) for value in values)).hexdigest()
    if digest != metrics.get("action_digest"):
        raise ConfigurationError(f"{where}: controller trace bytes disagree with action digest")

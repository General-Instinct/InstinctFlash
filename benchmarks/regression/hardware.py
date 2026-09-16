"""Explicit benchmark hardware targets; planning never imports a GPU library."""
from __future__ import annotations

import math

TARGETS = {"jetson_thor": (11, 0), "rtx4090": (8, 9), "rtx5090": (12, 0)}
TARGET_DEVICE_NAMES = {"rtx4090": "NVIDIA GeForce RTX 4090", "rtx5090": "NVIDIA GeForce RTX 5090"}
DEFAULT_TARGET = "jetson_thor"


def target_record(name=DEFAULT_TARGET):
    if not isinstance(name, str) or name not in TARGETS:
        raise ValueError(f"unsupported benchmark target: {name!r}")
    return {"name": name, "capability": list(TARGETS[name])}


def bound_target(value=None):
    """Old matrices without a target belong only to the original Thor study."""
    if value is None:
        return target_record()
    if (not isinstance(value, dict) or set(value) != {"name", "capability"}
            or not isinstance(value.get("capability"), list)
            or any(type(part) is not int for part in value["capability"])
            or value != target_record(value.get("name"))):
        raise ValueError("benchmark target/capability binding differs")
    return target_record(value["name"])


def device_matches(target, capability, name):
    target = bound_target(target)
    return (isinstance(capability, (list, tuple))
            and all(type(part) is int for part in capability)
            and list(capability) == target["capability"]
            and (target["name"] not in TARGET_DEVICE_NAMES
                 or name == TARGET_DEVICE_NAMES[target["name"]]))


def probe_device(target, torch):
    """Read the actual selected device in the process that will run inference."""
    target = bound_target(target)
    props = torch.cuda.get_device_properties(0)
    capability = list(torch.cuda.get_device_capability(0))
    return {"target": target, "capability": capability, "name": str(props.name),
            "uuid": str(props.uuid), "total_memory_bytes": int(props.total_memory),
            "status": "passed" if device_matches(target, capability, str(props.name)) else "failed"}


def validate_device_receipt(receipt, target):
    """CPU validation of observed device fields, never evidence of a new GPU run."""
    target = bound_target(target)
    if (not isinstance(receipt, dict) or receipt.get("target") != target
            or receipt.get("status") != "passed"
            or not device_matches(target, receipt.get("capability", []), receipt.get("name"))
            or not isinstance(receipt.get("uuid"), str) or not receipt["uuid"]
            or type(receipt.get("total_memory_bytes")) is not int or receipt["total_memory_bytes"] <= 0):
        raise ValueError("actual GPU does not match the frozen benchmark target")


def positive_timeout(value, *, name="timeout", maximum=86400):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 < value <= maximum):
        raise ValueError(f"{name} must be finite and between zero and {maximum} seconds")
    return float(value)

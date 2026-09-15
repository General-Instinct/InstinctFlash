"""Synthetic contract driver used only for CI and pipeline development.

It deliberately marks every result ``synthetic=true``. The report command refuses to call such a
run reportable unless ``--allow-synthetic`` is passed, so this executable cannot be mistaken for a
model benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import random
import struct
import sys
from pathlib import Path

from .util import load_json, sha256_json, write_json_atomic


DRIVER_REVISION = "reference-driver-v1"


def _stable_seed(request: dict) -> int:
    seed = request["requested_seed"]
    if request["suite"]["seed_strategy"] == "fixed":
        return seed
    maximum = seed + request["suite"]["seed_max_attempts"]
    while seed < maximum:
        marker = hashlib.sha256(f"{request['task']}:{seed}".encode()).digest()[0]
        if marker % 13:
            return seed
        seed += 1
    raise RuntimeError("synthetic stable-seed resolver exhausted")


def _actions(request: dict, seed: int) -> list[float]:
    # Stable input identity, not arm or repeat identity: the CI treatment is bit-exact and
    # independently repeated jobs exercise the repeatability gate on the same observation.
    identity = (
        request["model_id"], request["suite_id"], request["task"],
        request["requested_seed"], seed,
    )
    rng = random.Random(repr(identity))
    offset = float(request["arm"]["operating_point"].get("synthetic_action_offset", 0.0))
    return [rng.uniform(-1.0, 1.0) + offset for _ in range(16)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = load_json(args.request)
    request = job["request"]
    seed = _stable_seed(request)
    actions = _actions(request, seed)
    digest = hashlib.sha256(b"".join(struct.pack("!d", item) for item in actions)).hexdigest()
    role = request["arm"]["role"]
    base = 10.0 if role == "control" else 5.0
    iterations = request["measurement"]["iterations"]
    latency = [base + index * 0.01 for index in range(iterations)]
    success = hashlib.sha256(f"{request['task']}:{seed}".encode()).digest()[0] % 5 != 0
    fingerprint = sha256_json(
        {"python": sys.version, "platform": platform.platform(), "driver": DRIVER_REVISION}
    )
    metrics = {
        "action_digest": digest,
        "action_values": actions,
        "finite": True,
        "latency_ms": latency,
        "success": success,
    }
    write_json_atomic(
        args.output,
        {
            "schema_version": 1,
            "job_id": job["job_id"],
            "request_sha256": job["request_sha256"],
            "status": "completed",
            "resolved_seed": seed,
            "metrics": metrics,
            "provenance": {
                "model_revision": request["model"]["checkpoint"]["revision"],
                "driver_revision": job["driver"]["revision"],
                "environment_fingerprint": fingerprint,
                "synthetic": True,
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

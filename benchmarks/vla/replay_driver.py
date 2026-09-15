"""Replay recorded closed-loop episode outcomes through the pipeline's gate path.

This driver executes NO model and NO simulator. It exists to validate the closed-loop verdict
path end to end — plan, runner, result validation, pairing, and the frozen
``instinctflash.verify.certify`` call in the report stage — by feeding it episode outcomes that
an already-certified campaign produced (for example the V2 M3 RoboTwin 50x10 paired run). The
report's certificate must then be byte-for-byte what a direct ``certify()`` call on the same
outcomes produces; anything else is a pipeline bug.

Because no model ran, every result is marked ``synthetic=true``: a replay validates the
gate path and is NEVER fresh benchmark evidence. Reports containing it stay non-reportable
without ``--allow-synthetic``, exactly like the CI reference driver.

Episode files are the campaign JSONLs: one object per episode with at least ``task``,
``ep_index`` and ``success``. The plan's ``requested_seed`` follows
``seed_base + task_index * 10_000 + seed_index``; ``--seed-base`` names the suite's declared
seed base so the driver can recover ``seed_index`` and replay episode ``ep_index == seed_index``
of the request's task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.vla.util import (  # noqa: E402
    ConfigurationError,
    load_json,
    sha256_file,
    sha256_json,
    write_json_atomic,
)


DRIVER_REVISION = "replay-driver-v1"


def load_episodes(path: Path) -> dict[tuple[str, int], dict]:
    episodes: dict[tuple[str, int], dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (str(record["task"]), int(record["ep_index"]))
            if key in episodes:
                raise ConfigurationError(f"{path}:{line_number}: duplicate episode {key}")
            episodes[key] = record
    if not episodes:
        raise ConfigurationError(f"{path}: no episodes")
    return episodes


def replay(job: dict, episodes_path: Path, seed_base: int) -> dict:
    request = job["request"]
    if request["suite"]["kind"] != "closed_loop":
        raise ConfigurationError(
            f"the replay driver serves recorded closed-loop outcomes only, not "
            f"{request['suite']['kind']!r}"
        )
    episodes = load_episodes(episodes_path)
    requested = int(request["requested_seed"])
    seed_index = (requested - seed_base) % 10_000
    key = (str(request["task"]), seed_index)
    record = episodes.get(key)
    if record is None:
        raise ConfigurationError(
            f"{episodes_path} has no recorded episode for task {key[0]!r} ep_index {key[1]} "
            f"(requested_seed {requested}, seed_base {seed_base})"
        )
    fingerprint = sha256_json(
        {"driver": DRIVER_REVISION, "episodes_sha256": sha256_file(episodes_path)}
    )
    return {
        "schema_version": 1,
        "job_id": job["job_id"],
        "request_sha256": job["request_sha256"],
        "status": "completed",
        "resolved_seed": requested,
        "metrics": {
            "success": bool(record["success"]),
            "action_digest": sha256_json(record),
            "finite": True,
        },
        "provenance": {
            "model_revision": request["model"]["checkpoint"]["revision"],
            "driver_revision": DRIVER_REVISION,
            "environment_fingerprint": fingerprint,
            "synthetic": True,                       # no model ran; never fresh evidence
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = load_json(args.request)
    result = replay(job, args.episodes, args.seed_base)
    write_json_atomic(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The closed-loop gate path, validated end to end with recorded episode outcomes.

The replay driver feeds already-measured closed-loop episode outcomes through the real plan,
runner, result validation, pairing, and report stages. The deciding certificate must be
byte-for-byte what a direct call to the frozen ``instinctflash.verify.certify`` produces on the
same outcomes — the pipeline may add its report envelope, never touch the statistics.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.vla.plan import build_plan, load_arms  # noqa: E402
from benchmarks.vla.registry import Registry, load_registry  # noqa: E402
from benchmarks.vla.report import build_report  # noqa: E402
from benchmarks.vla.runner import execute_plan  # noqa: E402
from benchmarks.vla.util import canonical_json, sha256_json  # noqa: E402
from instinctflash.verify.certify import Outcome, certify  # noqa: E402
from tests.run_tests import run_module_tests  # noqa: E402


ARMS = ROOT / "benchmarks" / "vla" / "config" / "arms.ci.json"
TASKS = ("adjust_bottle", "beat_block_hammer")           # first two robotwin50_easy tasks


def _replay_registry() -> Registry:
    registry = load_registry()
    raw = copy.deepcopy(registry.raw)
    raw["profiles"]["replay_test"] = {
        "suites": ["robotwin50_easy"],
        "limits": {"tasks": 2, "seeds_per_task": {"closed_loop": 2}},
        "latency": {"warmup": 1, "iterations": 1},
        "arm_repeats": {"closed_loop": 1},
    }
    return Registry(raw=raw, digest=sha256_json(raw), path=registry.path)


def _write_episodes(path: Path, successes: dict[tuple[str, int], bool], arm: str) -> None:
    lines = []
    for (task, ep_index), success in sorted(successes.items()):
        lines.append(json.dumps({
            "episode_id": f"{task}/{ep_index}",
            "task": task,
            "ep_index": ep_index,
            "seed": 100000 + ep_index,               # the campaign's own seeds, not the plan's
            "arm": arm,
            "success": success,
        }))
    path.write_text("\n".join(lines) + "\n")


def _replay_arms(control_episodes: Path, treatment_episodes: Path) -> dict:
    arms = load_arms(ARMS)
    for arm, episodes in zip(arms["arms"], (control_episodes, treatment_episodes)):
        arm["driver"] = {
            "command": [
                sys.executable, "-m", "benchmarks.vla.replay_driver",
                "--episodes", str(episodes), "--seed-base", "50100",
            ],
            "revision": "replay-driver-v1",
            "environment": {},
            "timeout_seconds": 60,
        }
    treatment = next(arm for arm in arms["arms"] if arm["role"] == "treatment")
    treatment["gates"]["success"] = {
        "margin": -0.05, "interval": "tango_one_sided95", "min_pairs": 4,
    }
    return arms


def test_replayed_outcomes_reproduce_the_frozen_certify_certificate_byte_for_byte() -> None:
    registry = _replay_registry()
    stock = {(TASKS[0], 0): True, (TASKS[0], 1): True, (TASKS[1], 0): True, (TASKS[1], 1): False}
    engine = {(TASKS[0], 0): True, (TASKS[0], 1): False, (TASKS[1], 0): True, (TASKS[1], 1): True}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        control_path = root / "stock.jsonl"
        treatment_path = root / "engine.jsonl"
        _write_episodes(control_path, stock, "stock")
        _write_episodes(treatment_path, engine, "engine")
        arms = _replay_arms(control_path, treatment_path)
        plan = build_plan(
            registry, arms, "replay_test", ["robbyant/lingbot-vla-v2-6b-robotwin"]
        )
        assert plan["job_count"] == 8                # 2 tasks x 2 seeds x 2 arms
        progress = execute_plan(plan, root / "run", repo_root=ROOT)
        assert progress["completed"] == progress["expected"], progress["failed"]
        report = build_report(root / "run", registry, allow_synthetic=True)
        assert report["complete"] is True
        assert report["synthetic"] is True           # a replay is never fresh evidence

        certificates = report["comparisons"][0]["closed_loop"]
        assert len(certificates) == 1
        reported = {
            key: value for key, value in certificates[0].items()
            if key not in ("model_id", "suite_id")
        }

        # The frozen call, reconstructed exactly as the report stage preregisters it: pair ids
        # as episode identity, resolved seeds, request-hash arm identities.
        jobs = {job["job_id"]: job for job in plan["jobs"]}
        teacher, student = [], []
        control_hashes, treatment_hashes = [], []
        for job_id, job in jobs.items():
            request = job["request"]
            result = json.loads((root / "run" / "results" / f"{job_id}.json").read_text())
            outcome = Outcome(
                request["pair_id"], result["resolved_seed"], request["task"],
                result["metrics"]["success"],
            )
            if request["arm"]["role"] == "control":
                teacher.append(outcome)
                control_hashes.append(job["request_sha256"])
            else:
                student.append(outcome)
                treatment_hashes.append(job["request_sha256"])
        direct = certify(
            teacher, student, margin=-0.05,
            teacher_hash=sha256_json(sorted(control_hashes)),
            student_hash=sha256_json(sorted(treatment_hashes)),
            harness="benchmarks.vla", recipe="accelerated",
            seeds="explicit in immutable plan", min_pairs=4,
            interval="tango_one_sided95", fail_on_task_collapse=True,
        )
        assert canonical_json(reported) == canonical_json(json.loads(direct.to_json()))
        assert reported["n_pairs"] == 4
        assert reported["discordant"] == [1, 1]
        # n=4 cannot decide a -5pp margin; the honest verdict is insufficient evidence, and the
        # report refuses to call the run's gates passed.
        assert reported["verdict"].startswith("FAIL (insufficient evidence)")
        assert report["gates_passed"] is False


def test_replay_driver_refuses_non_closed_loop_and_unknown_episodes() -> None:
    from benchmarks.vla.replay_driver import replay
    from benchmarks.vla.util import ConfigurationError

    with tempfile.TemporaryDirectory() as temporary:
        episodes = Path(temporary) / "episodes.jsonl"
        _write_episodes(episodes, {(TASKS[0], 0): True}, "stock")
        job = {
            "job_id": "0" * 24,
            "request_sha256": "0" * 64,
            "request": {
                "task": TASKS[0],
                "requested_seed": 50100,
                "suite": {"kind": "latency"},
                "model": {"checkpoint": {"revision": "0" * 40}},
            },
        }
        try:
            replay(job, episodes, 50100)
        except ConfigurationError as error:
            assert "closed-loop" in str(error)
        else:
            raise AssertionError("a latency request was replayed")
        job["request"]["suite"]["kind"] = "closed_loop"
        value = replay(job, episodes, 50100)
        assert value["metrics"]["success"] is True
        assert value["provenance"]["synthetic"] is True
        job["request"]["requested_seed"] = 50101     # ep_index 1 does not exist
        try:
            replay(job, episodes, 50100)
        except ConfigurationError as error:
            assert "no recorded episode" in str(error)
        else:
            raise AssertionError("a missing episode was fabricated")


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))

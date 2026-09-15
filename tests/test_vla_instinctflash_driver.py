"""Contract-shape and refusal invariants for the production InstinctFlash benchmark driver.

These tests never load a model: the family arms are stubbed on the CPU, and the assertions are
about the parts that must hold everywhere — the canonical action convention, the emitted result
contract, and the loud refusals for anything the driver cannot honestly serve.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmarks.vla.instinctflash_driver as driver  # noqa: E402
from benchmarks.vla.plan import load_arms  # noqa: E402
from benchmarks.vla.registry import load_registry  # noqa: E402
from benchmarks.vla.result import validate_result  # noqa: E402
from benchmarks.vla.util import ConfigurationError, sha256_json  # noqa: E402
from tests.run_tests import run_module_tests  # noqa: E402


class _StubArm:
    """Deterministic fake family arm: the action is a function of prompt + episode resets."""

    def __init__(self):
        self.episodes = 0
        self.prompt = ""
        self.chunk_resets = 0

    def new_episode(self, prompt: str) -> None:
        self.episodes += 1
        self.prompt = prompt

    def reset_chunk(self) -> None:
        self.chunk_resets += 1

    def predict(self, observation) -> np.ndarray:
        base = float(len(self.prompt))
        return np.asarray([base, base + 0.5, float(self.episodes)], dtype=np.float64)

    def close(self) -> None:
        pass


def _job(suite_kind: str, task: str, *, required: list[str]) -> dict:
    request = {
        "schema_version": 1,
        "pair_id": "0" * 24,
        "model_id": "lerobot/pi05_base",
        "suite_id": "model_contract" if suite_kind == "contract" else "single_gpu_latency",
        "task": task,
        "requested_seed": 10100,
        "repeat": 0,
        "model": {
            "registry_id": "lerobot/pi05_base",
            "backbone": "pi05",
            "checkpoint": {
                "id": "lerobot/pi05_base",
                "revision": "b" * 40,
                "derived_from_revision": "b" * 40,
            },
        },
        "dataset": {"id": "synthetic_contract_v1"},
        "suite": {
            "id": "model_contract" if suite_kind == "contract" else "single_gpu_latency",
            "kind": suite_kind,
            "seed_strategy": "fixed",
            "seed_max_attempts": 1,
            "protocol": {},
            "required_metrics": required,
        },
        "arm": {
            "id": "runtime_default",
            "role": "treatment",
            "operating_point": {"name": "runtime_default", "tier": "NUMERIC"},
        },
        "measurement": {"warmup": 1, "iterations": 3},
    }
    digest = sha256_json(request)
    return {
        "job_id": digest[:24],
        "request_sha256": digest,
        "request": request,
        "driver": {
            "command": ["python", "driver"],
            "revision": "stub-driver-revision",
            "environment": {},
            "timeout_seconds": 60,
        },
    }


class _Stubbed:
    """Scoped monkeypatching of the driver's model-touching seams."""

    def __enter__(self):
        self._saved = {
            name: getattr(driver, name)
            for name in (
                "build_arm", "make_observation", "environment_fingerprint",
                "seed_everything", "driver_revision",
            )
        }
        driver.build_arm = lambda request: _StubArm()
        driver.make_observation = lambda request, seed, prompt: {"seed": seed, "prompt": prompt}
        driver.environment_fingerprint = (
            lambda backbone: hashlib.sha256(f"stub:{backbone}".encode()).hexdigest()
        )
        driver.seed_everything = lambda seed: None
        driver.driver_revision = lambda: "stub-driver-revision"
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(driver, name, value)


def test_canonical_action_convention_matches_the_documented_contract() -> None:
    values = np.asarray([0.25, -1.5, 3.0], dtype=np.float64)
    packed = b"".join(struct.pack("!d", item) for item in values.tolist())
    assert driver.action_digest(values) == hashlib.sha256(packed).hexdigest()
    # dict outputs flatten by sorted key; torch-free inputs pass straight through
    flat = driver.flatten_action({"b": [3.0, 4.0], "a": [[1.0], [2.0]]})
    assert flat.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert flat.dtype == np.float64


def test_contract_jobs_emit_the_full_result_contract() -> None:
    with _Stubbed():
        for task in driver.CONTRACT_TASKS:
            job = _job("contract", task, required=["action_digest", "action_values", "finite"])
            result = driver.run_job(job)
            validate_result(result, job)
            assert result["provenance"]["synthetic"] is False
            assert result["resolved_seed"] == job["request"]["requested_seed"]
            values = result["metrics"]["action_values"]
            assert values and all(isinstance(item, float) for item in values)


def test_changed_prompt_and_reset_replay_report_the_second_prediction() -> None:
    with _Stubbed():
        switched = driver.run_job(
            _job("contract", "changed_prompt", required=["action_digest", "action_values", "finite"])
        )
        # the stub encodes prompt length + episode count; the reported action must be the
        # post-switch prompt-B prediction (episode 2), not the first one
        prompt_b = driver.PROMPTS["pi05"][1]
        assert switched["metrics"]["action_values"] == [
            float(len(prompt_b)), float(len(prompt_b)) + 0.5, 2.0,
        ]
        replay = driver.run_job(
            _job("contract", "reset_replay", required=["action_digest", "action_values", "finite"])
        )
        assert replay["metrics"]["action_values"][2] == 2.0


def test_latency_jobs_emit_samples_and_reset_the_chunk_each_iteration() -> None:
    with _Stubbed():
        job = _job("latency", "full_policy_chunk", required=["latency_ms", "action_digest", "finite"])
        result = driver.run_job(job)
        validate_result(result, job)
        samples = result["metrics"]["latency_ms"]
        assert len(samples) == job["request"]["measurement"]["iterations"]
        assert all(item > 0 for item in samples)


def test_unsupported_suites_backbones_and_operating_points_are_refused() -> None:
    with _Stubbed():
        job = _job("contract", "seeded_prompt_a", required=["finite"])
        job["request"]["suite"]["kind"] = "closed_loop"
        try:
            driver.run_job(job)
        except driver.DriverRefusal as error:
            assert "simulator" in str(error)
        else:
            raise AssertionError("closed_loop request was not refused")

    with _Stubbed():
        job = _job("contract", "not_a_task", required=["finite"])
        try:
            driver.run_job(job)
        except driver.DriverRefusal as error:
            assert "unknown contract task" in str(error)
        else:
            raise AssertionError("unknown contract task was not refused")

    # build_arm refusals run unstubbed: they must fire before any model import
    bad_point = _job("contract", "seeded_prompt_a", required=["finite"])
    bad_point["request"]["arm"]["operating_point"]["name"] = "mystery"
    try:
        driver.build_arm(bad_point["request"])
    except driver.DriverRefusal as error:
        assert "operating point" in str(error)
    else:
        raise AssertionError("unknown operating point was not refused")

    bad_backbone = _job("contract", "seeded_prompt_a", required=["finite"])
    bad_backbone["request"]["model"]["backbone"] = "wan_va"
    try:
        driver.build_arm(bad_backbone["request"])
    except driver.DriverRefusal as error:
        assert "not supported" in str(error)
    else:
        raise AssertionError("unsupported backbone was not refused")


def test_dirty_checkout_and_planned_revision_mismatch_are_refused() -> None:
    with _Stubbed():
        driver.driver_revision = lambda: "abc123-dirty"
        try:
            driver.run_job(_job("contract", "seeded_prompt_a", required=["finite"]))
        except driver.DriverRefusal as error:
            assert "dirty" in str(error)
        else:
            raise AssertionError("dirty-tree measurement was not refused")
    with _Stubbed():
        driver.driver_revision = lambda: "not-the-planned-revision"
        try:
            driver.run_job(_job("contract", "seeded_prompt_a", required=["finite"]))
        except driver.DriverRefusal as error:
            assert "planned" in str(error)
        else:
            raise AssertionError("revision mismatch was not refused")


def test_observation_builders_are_seed_deterministic_and_prompt_carrying() -> None:
    request = _job("contract", "seeded_prompt_a", required=["finite"])["request"]
    request["model"]["backbone"] = "lingbot_vla_v2"
    first = driver.make_observation(request, 7, "prompt")
    second = driver.make_observation(request, 7, "prompt")
    third = driver.make_observation(request, 8, "prompt")
    for key in driver.LINGBOT_CAMERAS:
        assert np.array_equal(first[key], second[key])
        assert not np.array_equal(first[key], third[key])
    assert first["prompt"] == first["task"] == "prompt"
    assert first["observation.state"].shape == (14,)

    request["model"]["backbone"] = "groot_n17"
    groot = driver.make_observation(request, 7, "prompt")
    assert set(groot) == {"video", "state"}
    assert groot["video"]["exterior_image_1_left"].shape == (1, 2, 256, 256, 3)

    request["model"]["backbone"] = "pi05"
    pi05 = driver.make_observation(request, 7, "prompt")
    # geometry comes from the checkpoint's declaration (pi05_base: three 224 cams + 32 state)
    assert pi05["observation.state"].shape == (32,)
    assert pi05["observation.images.base_0_rgb"].shape == (3, 224, 224)
    assert pi05["prompt"] == "prompt"


def test_shipped_arms_manifest_and_serving_profile_plan_every_family() -> None:
    import os

    from benchmarks.vla.plan import build_plan

    registry = load_registry()
    saved = {
        name: os.environ.get(name)
        for name in ("IFL_BENCH_FAMILY_PYTHON", "IFL_BENCH_IFL_DRIVER", "IFL_BENCH_DRIVER_REVISION")
    }
    os.environ["IFL_BENCH_FAMILY_PYTHON"] = sys.executable
    os.environ["IFL_BENCH_IFL_DRIVER"] = str(
        ROOT / "benchmarks" / "vla" / "instinctflash_driver.py"
    )
    os.environ["IFL_BENCH_DRIVER_REVISION"] = "0" * 40
    try:
        arms = load_arms(ROOT / "benchmarks" / "vla" / "config" / "arms.instinctflash.json")
        for model_id in (
            "lerobot/pi05_libero_finetuned_v044",
            "nvidia/GR00T-N1.7-3B",
            "robbyant/lingbot-vla-4b-posttrain-robotwin",
            "robbyant/lingbot-vla-v2-6b-robotwin",
        ):
            plan = build_plan(registry, arms, "serving", [model_id])
            suites = {job["request"]["suite_id"] for job in plan["jobs"]}
            assert suites == {"model_contract", "single_gpu_latency"}
            # 4 contract tasks x 2 repeats x 2 arms + 1 latency pair
            assert plan["job_count"] == 18
            points = {job["request"]["arm"]["operating_point"]["name"] for job in plan["jobs"]}
            assert points == {"stock_upstream", "runtime_default"}
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_snapshot_resolution_refuses_missing_and_moved_revisions() -> None:
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        saved = os.environ.get("HF_HUB_CACHE")
        os.environ["HF_HUB_CACHE"] = temporary
        try:
            try:
                driver.resolve_snapshot("example/model", "a" * 40)
            except driver.DriverRefusal as error:
                assert "not cached" in str(error)
            else:
                raise AssertionError("missing snapshot was not refused")
            repo = Path(temporary) / "models--example--model"
            (repo / "snapshots" / ("a" * 40)).mkdir(parents=True)
            assert driver.resolve_snapshot("example/model", "a" * 40).is_dir()
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text("b" * 40)
            try:
                driver.resolve_snapshot("example/model", "a" * 40)
            except driver.DriverRefusal as error:
                assert "refs/main" in str(error)
            else:
                raise AssertionError("moved refs/main was not refused")
        finally:
            if saved is None:
                os.environ.pop("HF_HUB_CACHE", None)
            else:
                os.environ["HF_HUB_CACHE"] = saved


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))

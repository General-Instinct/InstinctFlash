"""Production invariants for the manifest-driven VLA benchmark pipeline."""

from __future__ import annotations

import ast
import copy
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.vla.doctor import inspect_plan  # noqa: E402
from benchmarks.vla.plan import (  # noqa: E402
    build_plan,
    load_arms,
    pipeline_digest,
    validate_arms,
    validate_plan,
)
from benchmarks.vla.registry import Registry, load_registry  # noqa: E402
from benchmarks.vla.report import build_report  # noqa: E402
from benchmarks.vla.runner import execute_plan  # noqa: E402
from benchmarks.vla.util import ConfigurationError, load_json, sha256_json, write_json_atomic  # noqa: E402
from instinctflash.descriptors.known import KNOWN_DECLARATIONS  # noqa: E402
from tests.run_tests import run_module_tests  # noqa: E402


ARMS = ROOT / "benchmarks" / "vla" / "config" / "arms.ci.json"


def _arms() -> dict:
    value = load_arms(ARMS)
    for arm in value["arms"]:
        arm["driver"]["command"] = [sys.executable, "-m", "benchmarks.vla.reference_driver"]
    return value


def _unit_registry() -> Registry:
    registry = load_registry()
    raw = copy.deepcopy(registry.raw)
    raw["profiles"]["unit"] = {
        "suites": ["model_contract", "single_gpu_latency"],
        "limits": {
            "tasks": 1,
            "seeds_per_task": {"contract": 1, "latency": 1},
        },
        "latency": {"warmup": 1, "iterations": 3},
        "arm_repeats": {"contract": 2, "latency": 1},
    }
    return Registry(raw=raw, digest=sha256_json(raw), path=registry.path)


def _adapter_null_envelope() -> float:
    """The V2 adapter's measured NULL_ENVELOPE, read without importing its torch module."""
    source = ROOT / "examples" / "lingbot_vla_v2" / "lingbot_vla_v2_iwm" / "static_capture.py"
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "NULL_ENVELOPE"
            for target in node.targets
        ):
            return float(ast.literal_eval(node.value))
    raise AssertionError(f"NULL_ENVELOPE not found in {source}")


def test_registry_tracks_every_builtin_checkpoint() -> None:
    registry = load_registry()
    assert registry.builtin_model_ids == set(KNOWN_DECLARATIONS)
    for model_id, declaration in KNOWN_DECLARATIONS.items():
        assert registry.models[model_id]["backbone"] == declaration["execution"]["backbone"]
    # The V2 numeric envelope is the adapter's measured constant, not a rounded copy: a
    # hand-copied 0.051 slightly loosened the acceptance envelope and nothing pinned it.
    envelope = registry.models["robbyant/lingbot-vla-v2-6b-robotwin"]["repeatability_max_abs"]
    assert envelope == _adapter_null_envelope()


def test_libero_long_registry_entry_matches_the_existing_pinned_adapter() -> None:
    registry = load_registry()
    model_id = "robbyant/lingbot-va-posttrain-libero-long"
    adapter_manifest = load_json(ROOT / "benchmarks" / "vla" / "config" / "adapters.json")
    declared = [checkpoint for adapter in adapter_manifest["adapters"]
                for checkpoint in adapter["checkpoints"] if checkpoint["id"] == model_id]
    assert len(declared) == 1
    model = registry.models[model_id]
    assert model["revision"] == declared[0]["revision"] == "0e89d1e753019988aba484e8da2dc0810e264d9f"
    assert model["backbone"] == "wan_va" and model["builtin"] is True
    assert model["quality_certified"] is False
    # Adding catalog/contract coverage does not silently enroll this LIBERO
    # checkpoint in the separately declared RoboTwin evaluation tasks.
    assert all(model_id not in suite.get("model_ids", ())
               for suite in registry.suites.values() if suite["id"].startswith("robotwin"))


def test_every_builtin_model_has_contract_and_latency_coverage() -> None:
    registry = load_registry()
    plan = build_plan(registry, _arms(), "smoke")
    coverage = {}
    for job in plan["jobs"]:
        request = job["request"]
        coverage.setdefault(request["model_id"], set()).add(request["suite_id"])
    assert set(coverage) == registry.builtin_model_ids
    for suites in coverage.values():
        assert {"model_contract", "single_gpu_latency"} <= suites


def test_plan_is_deterministic_paired_and_counterbalanced() -> None:
    registry = load_registry()
    arms = _arms()
    selected = ["lerobot/pi05_base", "lerobot/pi05_libero_finetuned_v044"]
    first = build_plan(registry, arms, "smoke", selected)
    second = build_plan(registry, arms, "smoke", reversed(selected))
    assert first == second
    validate_plan(first)
    by_pair = {}
    for job in first["jobs"]:
        request = job["request"]
        by_pair.setdefault(request["pair_id"], []).append(request["arm"]["id"])
    assert all(set(values) == {"upstream", "accelerated"} for values in by_pair.values())
    assert any(values[0] == "upstream" for values in by_pair.values())
    assert any(values[0] == "accelerated" for values in by_pair.values())


def test_ignored_run_artifacts_do_not_change_pipeline_identity() -> None:
    before = pipeline_digest()
    runs = ROOT / "benchmarks" / "vla" / "runs"
    created = not runs.exists()
    runs.mkdir(exist_ok=True)
    artifact = runs / "identity-test.json"
    artifact.write_text("{}\n")
    try:
        assert pipeline_digest() == before
    finally:
        artifact.unlink()
        if created:
            runs.rmdir()


def test_quantized_override_must_name_the_locked_parent() -> None:
    registry = load_registry()
    arms = _arms()
    treatment = next(arm for arm in arms["arms"] if arm["role"] == "treatment")
    treatment["operating_point"]["checkpoint_overrides"] = {
        "lerobot/pi05_base": {
            "id": "example/pi05-quantized",
            "revision": "1" * 40,
            "derived_from_revision": "2" * 40,
        }
    }
    validate_arms(arms)
    try:
        build_plan(registry, arms, "smoke", ["lerobot/pi05_base"])
    except ConfigurationError as error:
        assert "derives from" in str(error)
    else:
        raise AssertionError("mismatched quantization parent was accepted")


def test_secret_values_cannot_enter_committed_arm_manifests() -> None:
    arms = _arms()
    arms["arms"][0]["driver"]["environment"]["HF_TOKEN"] = "must-not-land"
    try:
        validate_arms(arms)
    except ConfigurationError as error:
        assert "looks secret" in str(error)
    else:
        raise AssertionError("secret-bearing arm manifest was accepted")
    arms = _arms()
    arms["arms"][0]["driver"]["command"] = ["${PRIVATE_KEY}"]
    try:
        validate_arms(arms)
    except ConfigurationError as error:
        assert "commands are logged" in str(error)
    else:
        raise AssertionError("secret-bearing driver argv was accepted")


def test_runner_rejects_a_plan_from_different_pipeline_bytes() -> None:
    registry = _unit_registry()
    plan = build_plan(registry, _arms(), "unit", ["lerobot/pi05_base"])
    plan["pipeline_sha256"] = "0" * 64
    unsigned = dict(plan)
    unsigned.pop("plan_id")
    plan["plan_id"] = sha256_json(unsigned)
    validate_plan(plan)
    with tempfile.TemporaryDirectory() as temporary:
        try:
            execute_plan(plan, Path(temporary) / "run", repo_root=ROOT)
        except ConfigurationError as error:
            assert "current pipeline" in str(error)
        else:
            raise AssertionError("stale pipeline plan was executed")


def test_plan_validation_recomputes_pair_identity() -> None:
    registry = _unit_registry()
    plan = build_plan(registry, _arms(), "unit", ["lerobot/pi05_base"])
    job = plan["jobs"][0]
    job["request"]["pair_id"] = "0" * 24
    job["request_sha256"] = sha256_json(job["request"])
    job["job_id"] = job["request_sha256"][:24]
    unsigned = dict(plan)
    unsigned.pop("plan_id")
    plan["plan_id"] = sha256_json(unsigned)
    try:
        validate_plan(plan)
    except ConfigurationError as error:
        assert "pair id" in str(error)
    else:
        raise AssertionError("plan accepted a rehashed but internally false pair identity")


def test_screening_protocol_cannot_be_reportable_even_when_all_gates_pass() -> None:
    registry = _unit_registry()
    for suite in registry.raw["suites"]:
        suite.setdefault("protocol", {})["screening"] = True
    registry = Registry(registry.raw, sha256_json(registry.raw), registry.path)
    plan = build_plan(registry, _arms(), "unit", ["lerobot/pi05_base"])
    with tempfile.TemporaryDirectory() as temporary:
        run_root = Path(temporary) / "run"
        execution = execute_plan(plan, run_root, repo_root=ROOT)
        assert not execution["failed"]
        report = build_report(run_root, registry, allow_synthetic=True)
        assert report["complete"] and report["gates_passed"]
        assert report["screening"] and not report["reportable"]


def test_runner_resume_repeatability_and_synthetic_refusal() -> None:
    registry = _unit_registry()
    plan = build_plan(registry, _arms(), "unit", ["lerobot/pi05_base"])
    with tempfile.TemporaryDirectory() as temporary:
        run_root = Path(temporary) / "run"
        first = execute_plan(plan, run_root, repo_root=ROOT)
        assert first["completed"] == first["expected"]
        assert not first["failed"]
        second = execute_plan(plan, run_root, repo_root=ROOT)
        assert second["resumed"] == second["expected"]
        refused = build_report(run_root, registry)
        assert refused["synthetic"] is True
        assert refused["reportable"] is False
        report = build_report(run_root, registry, allow_synthetic=True)
        assert report["reportable"] is True
        assert report["gates_passed"] is True
        assert report["repeatability"]
        assert all(item["verdict"] == "PASS" for item in report["repeatability"])
        assert all(item["verdict"] == "PASS" for item in report["environment_consistency"])
        environment_path = run_root / "environment.json"
        environment = load_json(environment_path)
        environment["python_version"] = "tampered"
        write_json_atomic(environment_path, environment)
        try:
            build_report(run_root, registry, allow_synthetic=True)
        except ConfigurationError as error:
            assert "run_manifest.json" in str(error)
        else:
            raise AssertionError("tampered environment evidence was accepted")


#: A driver that emits ONLY its suite's required_metrics -- the documented contract minimum
#: (DRIVER_CONTRACT.md: "Only metrics listed in required_metrics are mandatory for a job").
#: Action values are a function of pair identity, not arm, so numeric comparisons see zero delta.
REQUIRED_METRICS_ONLY_DRIVER = '''
import argparse, hashlib, json, random, struct
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
job = json.loads(args.request.read_text())
request = job["request"]
required = request["suite"]["required_metrics"]
identity = (request["model_id"], request["suite_id"], request["task"], request["requested_seed"])
rng = random.Random(repr(identity))
actions = [rng.uniform(-1.0, 1.0) for _ in range(16)]
metrics = {}
if "finite" in required:
    metrics["finite"] = True
if "action_digest" in required:
    packed = b"".join(struct.pack("!d", item) for item in actions)
    metrics["action_digest"] = hashlib.sha256(packed).hexdigest()
if "action_values" in required:
    metrics["action_values"] = actions
if "latency_ms" in required:
    base = 10.0 if request["arm"]["role"] == "control" else 5.0
    metrics["latency_ms"] = [base] * request["measurement"]["iterations"]
args.output.write_text(json.dumps({
    "schema_version": 1,
    "job_id": job["job_id"],
    "request_sha256": job["request_sha256"],
    "status": "completed",
    "resolved_seed": request["requested_seed"],
    "metrics": metrics,
    "provenance": {
        "model_revision": request["model"]["checkpoint"]["revision"],
        "driver_revision": job["driver"]["revision"],
        "environment_fingerprint": hashlib.sha256(b"required-metrics-only").hexdigest(),
        "synthetic": True,
    },
}))
'''


def test_numeric_envelope_model_is_reportable_with_required_metrics_only_driver() -> None:
    # Regression for the numeric action gate: it used to demand action_values from every
    # contract/latency/open_loop pair, but the latency suite preregisters only
    # latency_ms/action_digest/finite -- so every NUMERIC-tier run (registry mode on a
    # numeric_envelope model) was permanently INCOMPLETE with a contract-minimum driver.
    # The value comparison is scoped to suites that list action_values in required_metrics.
    registry = _unit_registry()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        driver = root / "required_metrics_only_driver.py"
        driver.write_text(REQUIRED_METRICS_ONLY_DRIVER)
        arms = _arms()
        for arm in arms["arms"]:
            arm["driver"] = {
                "command": [sys.executable, str(driver)],
                "revision": "required-metrics-only-v1",
                "environment": {},
                "timeout_seconds": 60,
            }
        treatment = next(arm for arm in arms["arms"] if arm["role"] == "treatment")
        treatment["gates"]["action"] = {"mode": "registry"}
        plan = build_plan(registry, arms, "unit", ["robbyant/lingbot-vla-v2-6b-robotwin"])
        progress = execute_plan(plan, root / "run", repo_root=ROOT)
        assert progress["completed"] == progress["expected"], progress["failed"]
        report = build_report(root / "run", registry, allow_synthetic=True)
        gates = {
            suite["suite_id"]: suite["action"]
            for comparison in report["comparisons"]
            for model in comparison["models"]
            for suite in model["suites"]
        }
        assert gates["single_gpu_latency"]["mode"] == "numeric"
        assert gates["single_gpu_latency"]["verdict"] == "PASS", gates["single_gpu_latency"]
        assert gates["model_contract"]["verdict"] == "PASS", gates["model_contract"]
        assert report["gates_passed"] is True
        assert report["reportable"] is True


def test_doctor_require_cached_preflights_checkpoint_overrides() -> None:
    # A quantized arm's weights are what the treatment actually loads; doctor used to check
    # only the registry parents, so an expensive run could start with the override missing.
    registry = load_registry()
    arms = _arms()
    parent = registry.models["lerobot/pi05_base"]["revision"]
    treatment = next(arm for arm in arms["arms"] if arm["role"] == "treatment")
    treatment["operating_point"]["checkpoint_overrides"] = {
        "lerobot/pi05_base": {
            "id": "example/pi05-quantized-not-cached",
            "revision": "1" * 40,
            "derived_from_revision": parent,
        }
    }
    plan = build_plan(registry, arms, "smoke", ["lerobot/pi05_base"])
    strict = inspect_plan(plan, registry, require_cached=True)
    assert strict["ok"] is False
    assert any("override checkpoint not cached" in error for error in strict["errors"])
    relaxed = inspect_plan(plan, registry, require_cached=False)
    assert any("override checkpoint not cached" in warning for warning in relaxed["warnings"])
    assert not any("override checkpoint" in error for error in relaxed["errors"])

    # A local-directory override is checked for existence rather than against the Hub cache.
    with tempfile.TemporaryDirectory() as temporary:
        treatment["operating_point"]["checkpoint_overrides"]["lerobot/pi05_base"]["id"] = str(
            Path(temporary) / "missing-package"
        )
        plan = build_plan(registry, arms, "smoke", ["lerobot/pi05_base"])
        strict = inspect_plan(plan, registry, require_cached=True)
        assert any("override checkpoint directory" in error for error in strict["errors"])


def test_runner_refuses_resume_after_orchestrator_environment_drift() -> None:
    import benchmarks.vla.runner as runner_module

    registry = _unit_registry()
    plan = build_plan(registry, _arms(), "unit", ["lerobot/pi05_base"])
    with tempfile.TemporaryDirectory() as temporary:
        run_root = Path(temporary) / "run"
        first = execute_plan(plan, run_root, repo_root=ROOT)
        assert first["completed"] == first["expected"]
        original = runner_module._environment_manifest

        def drifted(repo_root, gpu):
            manifest = original(repo_root, gpu)
            manifest["python_version"] = "drifted-orchestrator"
            return manifest

        runner_module._environment_manifest = drifted
        try:
            execute_plan(plan, run_root, repo_root=ROOT)
        except ConfigurationError as error:
            assert "environment drifted" in str(error)
            assert "python_version" in str(error)
        else:
            raise AssertionError("resume in a drifted orchestrator environment was accepted")
        finally:
            runner_module._environment_manifest = original


def test_missing_digests_are_not_bitexact_agreement() -> None:
    # With a custom registry whose suites drop action_digest from required_metrics, a
    # contract-minimum driver emits no digests at all. None == None must read as missing
    # evidence (FAIL), never as bit-exact agreement, in both the paired gate and repeatability.
    registry = load_registry()
    raw = copy.deepcopy(registry.raw)
    for suite in raw["suites"]:
        if suite["id"] == "model_contract":
            suite["required_metrics"] = ["finite"]
    raw["profiles"]["unit_digestless"] = {
        "suites": ["model_contract"],
        "limits": {"tasks": 1, "seeds_per_task": {"contract": 1}},
        "latency": {"warmup": 1, "iterations": 1},
        "arm_repeats": {"contract": 2},
    }
    digestless = Registry(raw=raw, digest=sha256_json(raw), path=registry.path)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        driver = root / "required_metrics_only_driver.py"
        driver.write_text(REQUIRED_METRICS_ONLY_DRIVER)
        arms = _arms()
        for arm in arms["arms"]:
            arm["driver"] = {
                "command": [sys.executable, str(driver)],
                "revision": "required-metrics-only-v1",
                "environment": {},
                "timeout_seconds": 60,
            }
        plan = build_plan(digestless, arms, "unit_digestless", ["lerobot/pi05_base"])
        progress = execute_plan(plan, root / "run", repo_root=ROOT)
        assert progress["completed"] == progress["expected"], progress["failed"]
        report = build_report(root / "run", digestless, allow_synthetic=True)
        actions = [
            model["action"]
            for comparison in report["comparisons"]
            for model in comparison["models"]
        ]
        assert actions and all(gate["verdict"] == "FAIL" for gate in actions)
        assert all(gate["digest_matches"] == 0 for gate in actions)
        assert report["repeatability"]
        assert all(check["verdict"] == "FAIL" for check in report["repeatability"])
        assert report["gates_passed"] is False


def test_doctor_reports_an_unresolvable_driver_without_running_it() -> None:
    registry = load_registry()
    arms = _arms()
    arms["arms"][1]["driver"]["command"] = ["/definitely/missing/driver"]
    plan = build_plan(registry, arms, "smoke", ["lerobot/pi05_base"])
    diagnosis = inspect_plan(plan, registry)
    assert diagnosis["ok"] is False
    assert any("driver executable" in error for error in diagnosis["errors"])


def test_timeout_terminates_the_driver_process_group() -> None:
    registry = load_registry()
    raw = copy.deepcopy(registry.raw)
    raw["profiles"]["timeout_test"] = {
        "suites": ["model_contract"],
        "limits": {"tasks": 1, "seeds_per_task": {"contract": 1}},
        "latency": {"warmup": 1, "iterations": 1},
        "arm_repeats": {"contract": 1},
    }
    test_registry = Registry(raw=raw, digest=sha256_json(raw), path=registry.path)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        pid_file = root / "child.pid"
        driver = root / "timeout_driver.py"
        driver.write_text(
            "import os, subprocess, time\n"
            "from pathlib import Path\n"
            "child = subprocess.Popen(['sleep', '30'])\n"
            "Path(os.environ['PID_FILE']).write_text(str(child.pid))\n"
            "time.sleep(30)\n"
        )
        arms = _arms()
        for arm in arms["arms"]:
            arm["driver"] = {
                "command": [sys.executable, str(driver)],
                "revision": "timeout-test-v1",
                "environment": {"PID_FILE": str(pid_file)},
                "timeout_seconds": 1,
            }
        plan = build_plan(test_registry, arms, "timeout_test", ["lerobot/pi05_base"])
        progress = execute_plan(plan, root / "run", repo_root=ROOT, fail_fast=True)
        assert progress["failed"]
        assert progress["failed"][0]["returncode"] == 124
        child_pid = int(pid_file.read_text())
        alive = True
        for _ in range(20):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                alive = False
                break
            time.sleep(0.05)
        if alive:
            os.kill(child_pid, 9)
        assert alive is False


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))

"""CPU-only tests for the preregistered pi0.5 paired outcome pipeline."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "pi05_vla"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLE))

from emit_tf32_closed_loop_outcomes import emit
from certify_tf32_closed_loop import main as certify_main
from instinctflash.verify.certify import (
    ONE_SIDED_95_Z,
    NotCertifiable,
    Outcome,
    _tango_paired_score_bounds,
    certify,
    load_jsonl,
)

#: The preregistered decision rule. certify()'s DEFAULT stays the repo-wide Wald central-95
#: certificate; this suite exercises the opt-in mode the closed-loop gate declares.
TANGO = dict(interval="tango_one_sided95")
from run_tf32_closed_loop import (
    DEFAULT_REVISION,
    EpisodeSeedSchedule,
    _configure_fp32_policy,
    _null_projection,
    _precision_assertion,
    _prepare_manifest,
    _task_rows_complete,
    _worker_command,
)


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    for name in (
        "config.json", "model.safetensors", "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (path / name).write_text(f"fixture:{name}\n")
    return path


def _raw(
    path: Path, *, flip_one: bool = False, run_id: str = "unit-run",
    arm: str = "control_fp32",
):
    with path.open("w") as output:
        for task_index in range(10):
            task = f"task_{task_index:02d}"
            for episode in range(50):
                seed = 1000 + 50 * task_index + episode
                success = not (flip_one and task_index == 0 and episode == 0)
                row = {
                    "run_id": run_id, "arm": arm, "suite": "libero_spatial",
                    "task_id": task_index, "task": task, "episode_index": episode,
                    "seed": seed, "init_state_id": episode, "policy_seed": seed,
                    "checkpoint_revision": DEFAULT_REVISION, "success": success,
                }
                output.write(json.dumps(row) + "\n")


def _null_controls(run_id: str) -> dict:
    results = {}
    for arm, precision, installed, tier, digest in (
        (
            "control_fp32",
            {"float32_matmul_precision": "highest", "allow_tf32": False},
            ["loop_constant_hoist", "graph_capture_static_kv"],
            "BITEXACT", "a" * 64,
        ),
        (
            "treatment_tf32",
            {"float32_matmul_precision": "high", "allow_tf32": True},
            ["loop_constant_hoist", "graph_capture_static_kv", "pi05_tf32_numeric"],
            "NUMERIC", "b" * 64,
        ),
    ):
        repeat = {
            "rows": [{
                "arm": arm, "suite": "libero_spatial", "task_id": 0,
                "task": "task_00", "episode_index": 0, "seed": 1000,
                "init_state_id": 0, "policy_seed": 1000, "success": True,
                "checkpoint_revision": DEFAULT_REVISION,
            }],
            "precision": precision, "installed": installed,
            "action_trace_sha256": digest, "plan_tier": tier,
        }
        results[arm] = {"passed": True, "repeat_0": repeat, "repeat_1": repeat}
    return {
        "run_id": run_id, "completed_utc": "2026-08-26T00:00:00+00:00",
        "protocol": "task 0 episode 0 repeated in two fresh processes per arm",
        "results": results, "passed": True,
    }


def test_emitter_and_certifier_run_the_preregistered_500_pairs():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        control_raw, treatment_raw = root / "control.raw", root / "treatment.raw"
        control, treatment = root / "control.jsonl", root / "treatment.jsonl"
        _raw(control_raw)
        _raw(treatment_raw, flip_one=True, arm="treatment_tf32")
        assert emit(control_raw, control, arm="control_fp32") == 500
        assert emit(treatment_raw, treatment, arm="treatment_tf32") == 500
        certificate = certify(
            load_jsonl(control), load_jsonl(treatment), margin=-0.05, min_pairs=500,
            harness="unit schema gate", **TANGO,
        )
        assert certificate.n_pairs == 500
        assert certificate.margin_declared == -0.05
        assert certificate.passed
        assert certificate.discordant == (1, 0)


def test_tango_gate_is_one_sided_95_and_mcnemar_is_descriptive_only():
    assert ONE_SIDED_95_Z == 1.6448536269514722
    teacher = [
        # Sixteen teacher-only wins make equality implausible, but the one-sided Tango lower bound
        # still clears the separately declared -5 point non-inferiority margin.
        *[Outcome(f"ep{i}", i, "task", True) for i in range(16)],
        *[Outcome(f"ep{i}", i, "task", True) for i in range(16, 500)],
    ]
    student = [
        *[Outcome(f"ep{i}", i, "task", False) for i in range(16)],
        *[Outcome(f"ep{i}", i, "task", True) for i in range(16, 500)],
    ]
    certificate = certify(teacher, student, margin=-0.05, min_pairs=500, **TANGO)
    assert certificate.passed
    assert certificate.lower_confidence_bound > -0.05
    assert certificate.p_value < 0.05
    assert "descriptive" not in certificate.p_value_kind  # role is explicit in rendered certificate
    assert "equality" in certificate.p_value_kind
    assert "one-sided 95%" in certificate.ci_method

    # One additional teacher-only loss crosses the score-bound decision boundary.
    student[16] = Outcome("ep16", 16, "task", False)
    failed = certify(teacher, student, margin=-0.05, min_pairs=500, **TANGO)
    assert not failed.passed
    assert failed.lower_confidence_bound <= -0.05


def test_zero_discordance_tango_bound_is_non_degenerate_and_task_collapse_is_a_hard_gate():
    identical = [(True, True)] * 500
    lower, upper = _tango_paired_score_bounds(identical)
    assert -0.05 < lower < 0 < upper < 0.05

    teacher, student = [], []
    for index in range(500):
        task = "task_00" if index < 50 else "task_01" if index < 100 else "task_02"
        teacher_success = index < 50 or index >= 100
        student_success = 50 <= index
        teacher.append(Outcome(f"ep{index}", index, task, teacher_success))
        student.append(Outcome(f"ep{index}", index, task, student_success))
    primary_only = certify(teacher, student, margin=-0.05, min_pairs=500, **TANGO)
    assert primary_only.passed and primary_only.delta == 0
    gated = certify(
        teacher, student, margin=-0.05, min_pairs=500, fail_on_task_collapse=True, **TANGO
    )
    assert not gated.passed
    assert gated.collapsed_tasks == ("task_00",)
    assert "secondary task-collapse gate" in gated.verdict


def test_jsonl_loader_refuses_truthy_non_boolean_success():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "outcomes.jsonl"
        path.write_text(json.dumps({
            "episode_id": "ep", "seed": 1, "task": "task", "success": "false",
        }) + "\n")
        try:
            load_jsonl(path)
        except NotCertifiable as exc:
            assert "JSON true/false" in str(exc)
        else:
            raise AssertionError("string success was coerced to true")


def test_incomplete_or_unpaired_arms_are_refused():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw, emitted = root / "raw", root / "emitted"
        raw.write_text(json.dumps({
            "run_id": "unit-run", "arm": "control_fp32", "suite": "libero_spatial",
            "task_id": 0, "task": "task_00", "episode_index": 0, "seed": 1000,
            "init_state_id": 0, "policy_seed": 1000,
            "checkpoint_revision": DEFAULT_REVISION, "success": True,
        }) + "\n")
        try:
            emit(raw, emitted, arm="control_fp32")
        except ValueError as exc:
            assert "10 tasks x 50" in str(exc)
        else:
            raise AssertionError("incomplete arm was accepted")

        emit(raw, emitted, arm="control_fp32", allow_partial=True)
        other = root / "other"
        other.write_text(json.dumps({
            "episode_id": "different", "seed": 1000, "task": "one", "success": True,
        }) + "\n")
        try:
            certify(load_jsonl(emitted), load_jsonl(other), margin=-0.05)
        except NotCertifiable as exc:
            assert "no (episode, seed) pairs" in str(exc)
        else:
            raise AssertionError("unpaired arms were accepted")


def test_runner_reseeds_policy_at_every_episode_boundary():
    class FakeCuda:
        def __init__(self):
            self.seeds = []
        def manual_seed_all(self, seed):
            self.seeds.append(seed)

    class FakeTorch:
        def __init__(self):
            self.seeds = []
            self.cuda = FakeCuda()
        def manual_seed(self, seed):
            self.seeds.append(seed)

    class Policy:
        def __init__(self):
            self.resets = 0
        def reset(self):
            self.resets += 1

    torch, policy = FakeTorch(), Policy()
    schedule = EpisodeSeedSchedule(torch, policy, 1000)
    for _ in range(3):
        policy.reset()
    assert schedule.seeds == [1000, 1001, 1002]
    assert torch.seeds == [1000, 1001, 1002]
    assert torch.cuda.seeds == [1000, 1001, 1002]
    assert policy.resets == 3


def test_runner_asserts_the_only_precision_difference():
    def fake(precision, allow):
        return SimpleNamespace(
            get_float32_matmul_precision=lambda: precision,
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=allow))
            ),
        )

    assert _precision_assertion(fake("highest", False), "control_fp32") == {
        "float32_matmul_precision": "highest", "allow_tf32": False,
    }
    assert _precision_assertion(fake("high", True), "treatment_tf32") == {
        "float32_matmul_precision": "high", "allow_tf32": True,
    }
    try:
        _precision_assertion(fake("high", True), "control_fp32")
    except RuntimeError as exc:
        assert "precision contract" in str(exc)
    else:
        raise AssertionError("control accepted TF32")


def test_runner_overrides_published_bfloat16_config_to_real_fp32_gemms():
    config = SimpleNamespace(
        pretrained_path=None, pretrained_revision="main", compile_model=True,
        device="cpu", dtype="bfloat16",
    )
    checkpoint = Path("/checkpoint/v044")
    _configure_fp32_policy(config, checkpoint)
    assert config.pretrained_path == checkpoint
    assert config.pretrained_revision is None
    assert not config.compile_model
    assert config.device == "cuda"
    assert config.dtype == "float32"


def test_null_control_compares_action_trace_not_only_success():
    base = {
        "rows": [{
            "arm": "control_fp32", "suite": "libero_spatial", "task_id": 0,
            "task": "task_00", "episode_index": 0, "seed": 1000,
            "init_state_id": 0, "policy_seed": 1000, "success": True,
            "checkpoint_revision": DEFAULT_REVISION,
        }],
        "precision": {"float32_matmul_precision": "highest", "allow_tf32": False},
        "installed": ["graph_capture_static_kv"],
        "action_trace_sha256": "one",
        "plan_tier": "BITEXACT",
    }
    changed = json.loads(json.dumps(base))
    changed["action_trace_sha256"] = "two"
    assert _null_projection(base) != _null_projection(changed)


def test_manifest_precedes_outcomes_and_resume_requires_exact_schedule():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = _checkpoint(root / "checkpoint")
        output = root / "run"
        manifest = _prepare_manifest(output, checkpoint, DEFAULT_REVISION, [0], 2)
        rows = [
            {
                "run_id": manifest["run_id"], "arm": "control_fp32",
                "task_id": 0, "task": "task_00", "episode_index": index,
                "seed": 1000 + index, "success": True,
            }
            for index in range(2)
        ]
        assert _task_rows_complete(
            rows, run_id=manifest["run_id"], arm="control_fp32",
            task_id=0, episodes=2, seed_base=1000,
        )
        rows[1]["seed"] = 1000
        assert not _task_rows_complete(
            rows, run_id=manifest["run_id"], arm="control_fp32",
            task_id=0, episodes=2, seed_base=1000,
        )
        resumed = _prepare_manifest(output, checkpoint, DEFAULT_REVISION, [0], 2)
        assert resumed == manifest
        try:
            _prepare_manifest(output, checkpoint, DEFAULT_REVISION, [0], 3)
        except RuntimeError as exc:
            assert "does not match" in str(exc)
        else:
            raise AssertionError("resume changed the preregistered episode count")


def test_worker_command_is_an_explicit_isolated_arm_process():
    args = SimpleNamespace(python="/venv/bin/python", episodes=2)
    command = _worker_command(
        args, arm="treatment_tf32", task_id=4, checkpoint=Path("/ckpt"),
        manifest=Path("/run/manifest.json"), run_id="run", result=Path("/run/result.json"),
    )
    assert command[0] == "/venv/bin/python"
    assert command[command.index("--worker-arm") + 1] == "treatment_tf32"
    assert command[command.index("--worker-task-id") + 1] == "4"
    assert command[command.index("--worker-episodes") + 1] == "2"


def test_certifier_takes_predeclared_hashes_from_manifest():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = _checkpoint(root / "checkpoint")
        run = root / "run"
        manifest = _prepare_manifest(run, checkpoint, DEFAULT_REVISION, list(range(10)), 50)
        (run / "null_controls.json").write_text(
            json.dumps(_null_controls(manifest["run_id"])) + "\n"
        )
        control_raw, treatment_raw = run / "control.raw", run / "treatment.raw"
        control, treatment = run / "control.jsonl", run / "treatment.jsonl"
        _raw(control_raw, run_id=manifest["run_id"], arm="control_fp32")
        _raw(treatment_raw, flip_one=True, run_id=manifest["run_id"], arm="treatment_tf32")
        emit(control_raw, control, arm="control_fp32")
        emit(treatment_raw, treatment, arm="treatment_tf32")
        certificate = run / "certificate.json"
        assert certify_main([
            "--control", str(control), "--treatment", str(treatment),
            "--run-manifest", str(run / "paired_run_manifest.json"),
            "--output", str(certificate),
        ]) == 0
        payload = json.loads(certificate.read_text())
        assert payload["teacher_hash"] == manifest["identities"]["control_hash"]
        assert payload["student_hash"] == manifest["identities"]["treatment_hash"]
        assert payload["run_manifest"]["run_id"] == manifest["run_id"]
        assert payload["lower_confidence_level"] == 0.95
        assert payload["alpha"] == 0.05
        assert "one-sided 95%" in payload["ci_method"]

        # A historical two-field stub must not satisfy the closed-loop reproducibility gate.
        null_path = run / "null_controls.json"
        valid_null = null_path.read_text()
        null_path.write_text(json.dumps({
            "run_id": manifest["run_id"], "passed": True,
        }) + "\n")
        assert certify_main([
            "--control", str(control), "--treatment", str(treatment),
            "--run-manifest", str(run / "paired_run_manifest.json"),
            "--output", str(certificate),
        ]) == 2
        null_path.write_text(valid_null)

        # Final certification re-hashes the model and processor snapshot, not merely its revision label.
        model = checkpoint / "model.safetensors"
        original_model = model.read_text()
        model.write_text(original_model + "tampered")
        assert certify_main([
            "--control", str(control), "--treatment", str(treatment),
            "--run-manifest", str(run / "paired_run_manifest.json"),
            "--output", str(certificate),
        ]) == 2
        model.write_text(original_model)

        manifest_path = run / "paired_run_manifest.json"
        manifest_text = manifest_path.read_text()
        changed_manifest = json.loads(manifest_text)
        changed_manifest["source_files"]["instinctflash/verify/certify.py"] = "0" * 64
        manifest_path.write_text(json.dumps(changed_manifest) + "\n")
        assert certify_main([
            "--control", str(control), "--treatment", str(treatment),
            "--run-manifest", str(manifest_path), "--output", str(certificate),
        ]) == 2
        manifest_path.write_text(manifest_text)

        # The normalized certificate input must retain the exact policy seed and initial-state schedule.
        treatment_text = treatment.read_text()
        treatment_rows = [json.loads(line) for line in treatment_text.splitlines()]
        treatment_rows[0]["policy_seed"] += 1
        treatment.write_text("".join(json.dumps(row) + "\n" for row in treatment_rows))
        assert certify_main([
            "--control", str(control), "--treatment", str(treatment),
            "--run-manifest", str(run / "paired_run_manifest.json"),
            "--output", str(certificate),
        ]) == 2


if __name__ == "__main__":
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))

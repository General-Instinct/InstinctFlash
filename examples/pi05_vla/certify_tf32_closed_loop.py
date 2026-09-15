#!/usr/bin/env python3
"""Certify preregistered paired closed-loop outcomes for pi0.5 FP32 vs TF32.

This script does not simulate episodes. It refuses unpaired/incomplete JSONL and delegates the
statistics to ``instinctflash.verify.certify``. Generate both arms with the same task, seed and
initial-state schedule described in ``tf32_closed_loop_preregistration.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from instinctflash.verify.certify import NotCertifiable, certify, load_jsonl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "tf32_closed_loop_preregistration.json"
ARMS = ("control_fp32", "treatment_tf32")
REQUIRED_SOURCE_FILES = {
    "examples/pi05_vla/run_tf32_closed_loop.py",
    "examples/pi05_vla/tf32_closed_loop_preregistration.json",
    "examples/pi05_vla/emit_tf32_closed_loop_outcomes.py",
    "examples/pi05_vla/certify_tf32_closed_loop.py",
    "instinctflash/verify/certify.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise NotCertifiable(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NotCertifiable(f"{label} {path} must contain a JSON object")
    return value


def _validate_source_hashes(manifest: dict) -> None:
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict) or not REQUIRED_SOURCE_FILES <= set(source_files):
        missing = sorted(REQUIRED_SOURCE_FILES - set(source_files or {}))
        raise NotCertifiable(f"run manifest lacks required source fingerprints: {missing}")
    root = ROOT.resolve()
    for relative, expected_hash in source_files.items():
        if not isinstance(relative, str) or not _is_sha256(expected_hash):
            raise NotCertifiable("run manifest contains an invalid source fingerprint")
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise NotCertifiable(f"source fingerprint escapes repository root: {relative!r}") from exc
        if not source.is_file() or _sha256(source) != expected_hash:
            raise NotCertifiable(f"source changed after manifest lock: {relative}")


def _validate_checkpoint_hashes(manifest: dict) -> None:
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise NotCertifiable("run manifest lacks checkpoint identity")
    checkpoint_path = checkpoint.get("path")
    files = checkpoint.get("files")
    if not isinstance(checkpoint_path, str) or not isinstance(files, dict) or not files:
        raise NotCertifiable("run manifest lacks checkpoint path/file fingerprints")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_dir() or "model.safetensors" not in files:
        raise NotCertifiable("checkpoint directory/model.safetensors is missing")
    actual_names = {
        item.name for item in path.iterdir()
        if item.is_file() and item.suffix in {".json", ".safetensors", ".model"}
    }
    if actual_names != set(files):
        raise NotCertifiable("checkpoint semantic file set changed after manifest lock")
    for name, metadata in files.items():
        if Path(name).name != name or not isinstance(metadata, dict):
            raise NotCertifiable(f"invalid checkpoint fingerprint entry {name!r}")
        expected_bytes, expected_hash = metadata.get("bytes"), metadata.get("sha256")
        if (
            not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool)
            or expected_bytes < 0 or not _is_sha256(expected_hash)
        ):
            raise NotCertifiable(f"invalid checkpoint fingerprint metadata for {name!r}")
        item = path / name
        if item.stat().st_size != expected_bytes or _sha256(item) != expected_hash:
            raise NotCertifiable(f"checkpoint file changed after manifest lock: {name}")

    identities = manifest.get("identities")
    if not isinstance(identities, dict):
        raise NotCertifiable("run manifest lacks operating-point identities")
    for arm_key in ("control", "treatment"):
        document = identities.get(arm_key)
        digest = identities.get(f"{arm_key}_hash")
        if not isinstance(document, dict) or digest != f"sha256:{_canonical_sha256(document)}":
            raise NotCertifiable(f"manifest {arm_key} identity hash is not self-consistent")
        if (
            document.get("checkpoint_path") != checkpoint_path
            or document.get("checkpoint_revision") != checkpoint.get("revision")
            or document.get("checkpoint_files") != files
        ):
            raise NotCertifiable(f"manifest {arm_key} identity is not bound to checkpoint files")


def _validate_null_controls(path: Path, manifest: dict, declaration: dict) -> None:
    null_path = path.parent / "null_controls.json"
    null_controls = _load_json(null_path, "null controls")
    run_id = manifest["run_id"]
    if (
        null_controls.get("run_id") != run_id
        or null_controls.get("passed") is not True
        or not isinstance(null_controls.get("completed_utc"), str)
        or null_controls.get("protocol")
        != "task 0 episode 0 repeated in two fresh processes per arm"
    ):
        raise NotCertifiable("closed-loop null-control header is incomplete or mismatched")
    results = null_controls.get("results")
    if not isinstance(results, dict) or set(results) != set(ARMS):
        raise NotCertifiable("null controls must contain exactly both preregistered arms")
    expected_precision = {
        "control_fp32": {"float32_matmul_precision": "highest", "allow_tf32": False},
        "treatment_tf32": {"float32_matmul_precision": "high", "allow_tf32": True},
    }
    expected_installed = {
        "control_fp32": ["loop_constant_hoist", "graph_capture_static_kv"],
        "treatment_tf32": [
            "loop_constant_hoist", "graph_capture_static_kv", "pi05_tf32_numeric"
        ],
    }
    expected_tier = {"control_fp32": "BITEXACT", "treatment_tf32": "NUMERIC"}
    expected_row_keys = {
        "arm", "suite", "task_id", "task", "episode_index", "seed", "init_state_id",
        "policy_seed", "success", "checkpoint_revision",
    }
    for arm in ARMS:
        result = results[arm]
        if not isinstance(result, dict) or result.get("passed") is not True:
            raise NotCertifiable(f"{arm} null control did not pass")
        left, right = result.get("repeat_0"), result.get("repeat_1")
        if not isinstance(left, dict) or left != right:
            raise NotCertifiable(f"{arm} closed-loop null repeats are missing or disagree")
        if (
            left.get("precision") != expected_precision[arm]
            or left.get("installed") != expected_installed[arm]
            or left.get("plan_tier") != expected_tier[arm]
            or not _is_sha256(left.get("action_trace_sha256"))
        ):
            raise NotCertifiable(f"{arm} null control has wrong precision/stack/action trace")
        rows = left.get("rows")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise NotCertifiable(f"{arm} null control must contain exactly one episode row")
        row = rows[0]
        if set(row) != expected_row_keys or not isinstance(row.get("success"), bool):
            raise NotCertifiable(f"{arm} null-control row has an invalid schema")
        expected = {
            "arm": arm, "suite": "libero_spatial", "task_id": 0, "task": "task_00",
            "episode_index": 0, "seed": int(declaration["seed_base"]), "init_state_id": 0,
            "policy_seed": int(declaration["seed_base"]),
            "checkpoint_revision": declaration["checkpoint"]["revision"],
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise NotCertifiable(f"{arm} null-control row is not the preregistered episode")


def _validate_outcome_file(
    path: Path, *, arm: str, run_id: str, declaration: dict
) -> None:
    rows: list[dict] = []
    try:
        with path.open() as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise NotCertifiable(f"{path}:{line_no} outcome must be a JSON object")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise NotCertifiable(f"cannot read outcome file {path}: {exc}") from exc

    exact_pairs = int(declaration["exact_pairs"])
    if len(rows) != exact_pairs:
        raise NotCertifiable(f"{path} has {len(rows)} rows; exactly {exact_pairs} are required")
    expected_keys = {
        (int(task_id), episode)
        for task_id in declaration["task_ids"]
        for episode in range(int(declaration["episodes_per_task"]))
    }
    observed: set[tuple[int, int]] = set()
    for line_no, row in enumerate(rows, 1):
        integer_names = (
            "task_id", "episode_index", "seed", "init_state_id", "policy_seed"
        )
        if any(
            not isinstance(row.get(name), int) or isinstance(row.get(name), bool)
            for name in integer_names
        ) or not isinstance(row.get("success"), bool):
            raise NotCertifiable(f"{path}:{line_no} has non-strict integer/boolean fields")
        task_id, episode = row["task_id"], row["episode_index"]
        key = (task_id, episode)
        if key not in expected_keys or key in observed:
            raise NotCertifiable(f"{path}:{line_no} has unexpected/duplicate episode {key}")
        observed.add(key)
        task = f"task_{task_id:02d}"
        seed = (
            int(declaration["seed_base"])
            + int(declaration["seed_stride"]) * task_id
            + episode
        )
        expected = {
            "run_id": run_id, "arm": arm, "suite": "libero_spatial", "task": task,
            "episode_id": f"libero_spatial/{task}/{episode:03d}", "seed": seed,
            "init_state_id": episode, "policy_seed": seed,
            "checkpoint_revision": declaration["checkpoint"]["revision"],
        }
        if any(row.get(name) != value for name, value in expected.items()):
            raise NotCertifiable(f"{path}:{line_no} is not bound to the preregistered schedule")
    if observed != expected_keys:
        raise NotCertifiable(f"{path} does not contain the exact preregistered episode set")


def _validate_manifest(path: Path, declaration: dict, control: Path, treatment: Path) -> dict:
    manifest = _load_json(path, "run manifest")
    if not manifest.get("declared_before_first_outcome"):
        raise NotCertifiable("run manifest was not locked before the first outcome")
    if manifest.get("preregistration_sha256") != _sha256(PREREG):
        raise NotCertifiable("run manifest names a different preregistration revision")
    expected_checkpoint = declaration["checkpoint"]
    observed_checkpoint = manifest.get("checkpoint", {})
    for key in ("repo", "revision"):
        if observed_checkpoint.get(key) != expected_checkpoint.get(key):
            raise NotCertifiable(f"run manifest checkpoint {key} disagrees with preregistration")
    protocol = manifest.get("protocol", {})
    expected_protocol = {
        "suite": "libero_spatial",
        "tasks": declaration["task_ids"],
        "episodes_per_task": declaration["episodes_per_task"],
        "batch_size": 1,
        "seed_base": declaration["seed_base"],
        "seed_stride": declaration["seed_stride"],
        "seed_rule": declaration["seed_rule"],
        "init_states": True,
        "hard_reset": True,
        "use_async_envs": False,
        "max_parallel_tasks": 1,
        "observation_resolution": [256, 256],
        "arm_order": declaration["arm_order"],
    }
    wrong = {
        key: (protocol.get(key), value)
        for key, value in expected_protocol.items()
        if protocol.get(key) != value
    }
    if wrong:
        raise NotCertifiable(f"run manifest is not the preregistered 500-pair protocol: {wrong}")
    identities = manifest.get("identities", {})
    for key in ("control_hash", "treatment_hash"):
        value = identities.get(key)
        if not isinstance(value, str) or not value.startswith("sha256:"):
            raise NotCertifiable(f"run manifest lacks a predeclared {key}")
    run_id = manifest.get("run_id")
    if not run_id or not manifest.get("declared_utc"):
        raise NotCertifiable("run manifest lacks run_id or declaration time")
    _validate_source_hashes(manifest)
    _validate_checkpoint_hashes(manifest)
    _validate_null_controls(path, manifest, declaration)
    _validate_outcome_file(control, arm="control_fp32", run_id=run_id, declaration=declaration)
    _validate_outcome_file(
        treatment, arm="treatment_tf32", run_id=run_id, declaration=declaration
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True, help="FP32 episodes.jsonl")
    parser.add_argument("--treatment", required=True, help="TF32 episodes.jsonl")
    parser.add_argument("--output", required=True, help="certificate JSON path")
    parser.add_argument(
        "--run-manifest", required=True,
        help="immutable paired_run_manifest.json written before the first episode",
    )
    args = parser.parse_args(argv)

    declaration = json.loads(PREREG.read_text())
    if not declaration.get("declared_before_paired_run"):
        raise SystemExit("preregistration does not assert it preceded the paired run")
    control_path, treatment_path = Path(args.control), Path(args.treatment)
    try:
        manifest = _validate_manifest(
            Path(args.run_manifest), declaration, control_path, treatment_path
        )
        identities = manifest["identities"]
        cert = certify(
            load_jsonl(str(control_path)), load_jsonl(str(treatment_path)),
            margin=float(declaration["margin"]),
            min_pairs=int(declaration["min_pairs"]),
            teacher_hash=identities["control_hash"],
            student_hash=identities["treatment_hash"],
            harness="LIBERO paired JSONL; exact (episode_id, seed) pairing",
            recipe="pi05 TF32 GEMMs + static-KV CUDA Graph operating point",
            seeds=str(declaration["seed_rule"]),
            # The preregistered decision rule, selected EXPLICITLY: certify()'s default stays
            # the repo-wide Wald central-95 certificate, and the Tango one-sided 95% bound is
            # an opt-in mode with its own field names (see verify/certify.py's migration note).
            interval="tango_one_sided95",
            fail_on_task_collapse=bool(
                declaration["secondary_gates"]["task_collapse"]["enabled"]
            ),
        )
    except NotCertifiable as exc:
        print(f"NOT CERTIFIABLE: {exc}")
        return 2
    output = Path(args.output)
    payload = json.loads(cert.to_json())
    payload["run_manifest"] = {
        "path": str(Path(args.run_manifest)),
        "sha256": _sha256(Path(args.run_manifest)),
        "run_id": manifest["run_id"],
        "declared_utc": manifest["declared_utc"],
    }
    payload["outcome_sha256"] = {
        "control": _sha256(control_path),
        "treatment": _sha256(treatment_path),
    }
    payload["null_controls_sha256"] = _sha256(Path(args.run_manifest).parent / "null_controls.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(cert)
    print(cert.per_task_table())
    print(f"wrote {output}")
    return 0 if cert.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

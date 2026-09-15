"""Prospective Cosmos/RoboLab pairing and fail-closed success reports.

This module imports neither Isaac Sim nor a model. A manifest is not a simulator
qualification or a certificate. Its formal loss margin has no default: an
explicit approval and exact deployment bindings must precede the episode records.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from instinctflash.verify.certify import (
    ONE_SIDED_95_Z, Outcome, _tango_paired_score_bounds, certify,
)

ROBOLAB_REVISION = "9db0aaf09d9fe5d4f37b168320788258c7012463"
FAMILIES = ("edge", "nano")
ARMS = ("baseline", "candidate")
CHECKPOINTS = {
    "edge": {"model_id": "nvidia/Cosmos3-Edge-Policy-DROID",
             "revision": "f9fddb427fd7cff0bef7c73791be36f0ab52d81e"},
    "nano": {"model_id": "nvidia/Cosmos3-Nano-Policy-DROID",
             "revision": "6706d7680581c255ff61e0f3bb49d90eac55c79e"},
}
SMOKE_TASKS = ("BananaInBowlTask", "RubiksCubeAndBananaTask")
SOURCE_PATHS = (
    "robolab/constants.py", "robolab/core/task/task.py",
    "robolab/core/environments/runtime.py", "robolab/core/environments/env.py",
    "robolab/core/logging/results.py", "robolab/eval/runner.py",
    "robolab/eval/episode.py", "robolab/eval/base_client.py",
    "robolab/registrations/droid/auto_env_registrations_jointpos.py",
    "robolab/registrations/droid/camera_presets.py",
    "policies/cosmos3/run.py", "policies/cosmos3/client.py",
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
NATIVE_RENDERER_PAIRING = "native_renderer_physical_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, low, high, name):
    _require(type(value) is int and low <= value <= high, f"Invalid {name}")
    return value


def _sha(value, name):
    _require(isinstance(value, str) and SHA256.fullmatch(value), f"Invalid {name}")
    return value


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], timeout=30)


def _source(root, relative):
    data = (root / relative).read_bytes()
    _require(data == _git(root, "show", f"{ROBOLAB_REVISION}:{relative}"),
             f"Working source differs from pinned RoboLab: {relative}")
    return data, {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _assignments(node):
    result = {}
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            result[stmt.target.id] = stmt.value
    return result


def build_task_inventory(robolab_root):
    """Read all 120 pinned task definitions without importing simulator code."""
    root = Path(robolab_root).resolve()
    _require(_git(root, "rev-parse", "HEAD").decode().strip() == ROBOLAB_REVISION,
             "RoboLab checkout is not the pinned revision")
    paths = _git(root, "ls-tree", "-r", "--name-only", ROBOLAB_REVISION,
                 "robolab/tasks/benchmark").decode().splitlines()
    _require(len(paths) == 120 and all(p.endswith(".py") for p in paths),
             "Pinned RoboLab-120 task inventory changed")
    tasks = []
    for relative in paths:
        source, identity = _source(root, relative)
        tree = ast.parse(source)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
                   and any(isinstance(b, ast.Name) and b.id == "Task" for b in n.bases)]
        _require(len(classes) == 1, f"Expected exactly one Task subclass in {relative}")
        node = classes[0]
        values = _assignments(node)
        duration = ast.literal_eval(values["episode_length_s"])
        instruction = ast.literal_eval(values["instruction"])
        instruction = instruction["default"] if isinstance(instruction, dict) else instruction
        scene = values["scene"]
        _require(isinstance(scene, ast.Call) and isinstance(scene.func, ast.Name)
                 and scene.func.id == "import_scene", f"Unexpected scene expression: {relative}")
        tasks.append({"task_id": node.name, "source_path": relative,
                      "source_sha256": identity["sha256"], "source_bytes": len(source),
                      "scene": ast.literal_eval(scene.args[0]),
                      "episode_length_s": duration, "instruction_default": instruction})
    sources = [_source(root, relative)[1] for relative in SOURCE_PATHS]
    value = {"schema_version": 1, "kind": "robolab120_static_task_inventory",
             "robolab_revision": ROBOLAB_REVISION, "tasks": sorted(tasks, key=lambda t: t["task_id"]),
             "sources": sources, "task_count": 120, "simulator_executed": False,
             "scope": "Pinned task/source inventory only; assets, renderer and runtime remain unqualified"}
    _validate_inventory(value)
    return value


def _validate_inventory(inventory):
    _require(inventory.get("schema_version") == 1
             and inventory.get("kind") == "robolab120_static_task_inventory"
             and inventory.get("robolab_revision") == ROBOLAB_REVISION,
             "Invalid pinned task inventory")
    tasks = inventory.get("tasks")
    _require(isinstance(tasks, list) and len(tasks) == inventory.get("task_count") == 120,
             "Inventory must contain all 120 tasks")
    names, paths = [], []
    for row in tasks:
        name = row.get("task_id")
        _require(isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*Task", name),
                 "Invalid task identity")
        relative = row.get("source_path")
        _require(isinstance(relative, str) and relative.startswith("robolab/tasks/benchmark/")
                 and Path(relative).name == relative.removeprefix("robolab/tasks/benchmark/")
                 and relative.endswith(".py"), "Invalid task source path")
        _sha(row.get("source_sha256"), "task source digest")
        _integer(row.get("source_bytes"), 1, 10_000_000, "task source size")
        _integer(row.get("episode_length_s"), 1, 600, "task duration")
        _require(isinstance(row.get("instruction_default"), str)
                 and row["instruction_default"].strip(), "Missing default instruction")
        names.append(name)
        paths.append(relative)
    _require(names == sorted(set(names)) and len(set(paths)) == 120,
             "Task inventory is duplicate or unsorted")
    _require(set(SMOKE_TASKS) <= set(names), "Smoke tasks absent from inventory")
    sources = inventory.get("sources", [])
    _require([row.get("path") for row in sources] == list(SOURCE_PATHS),
             "Missing pinned RoboLab contract sources")
    for row in sources:
        _sha(row.get("sha256"), "RoboLab source digest")


def _bindings(value):
    if value is None:
        return {family: {arm: None for arm in ARMS} for family in FAMILIES}
    _require(isinstance(value, dict) and set(value) == set(FAMILIES), "Need both family bindings")
    for family in FAMILIES:
        _require(isinstance(value[family], dict) and set(value[family]) == set(ARMS),
                 "Need baseline and candidate bindings")
        for binding in value[family].values():
            if binding is not None:
                _sha(binding, "execution binding")
    return {family: dict(value[family]) for family in FAMILIES}


def build_protocol(inventory, *, stage="smoke", episodes_per_task=None,
                   success_margin=None, approval_reference=None, report_only=False,
                   execution_bindings=None, pairing_mode=None):
    """Freeze episode identities; supplied approval becomes part of every record's hash."""
    _validate_inventory(inventory)
    _require(stage in ("smoke", "formal"), "Unknown stage")
    _require(type(report_only) is bool, "report_only must be boolean")
    _require(pairing_mode in (None, NATIVE_RENDERER_PAIRING), "Unknown renderer pairing mode")
    count = (1 if stage == "smoke" else 10) if episodes_per_task is None else episodes_per_task
    _integer(count, 1 if stage == "smoke" else 10, 1000, "episodes per task")
    if success_margin is not None:
        _require(type(success_margin) in (float, int) and math.isfinite(success_margin)
                 and -1 < success_margin <= 0, "Success margin must be a finite loss in (-1, 0]")
        _require(isinstance(approval_reference, str) and approval_reference.strip(),
                 "A margin requires its prospective user approval reference")
        _require(not report_only, "Report-only cannot carry a certification margin")
    else:
        _require(approval_reference is None, "Approval reference has no declared margin")
    approval = {"mode": "report_only" if report_only else "pending" if success_margin is None else "certify",
                "success_margin": success_margin, "approval_reference": approval_reference,
                "interval": "tango_one_sided95", "task_collapse_gate": False}
    chosen = [t for t in inventory["tasks"] if stage == "formal" or t["task_id"] in SMOKE_TASKS]
    indices = {t["task_id"]: i for i, t in enumerate(inventory["tasks"])}
    # Disjoint uint32 spaces. Reserve 1024 episode slots per task and 1024
    # model-query seeds per episode; the largest native task requires < 1024.
    namespace = 0x10000000 if stage == "smoke" else 0x50000000
    episodes = []
    for task in chosen:
        task_index = indices[task["task_id"]]
        steps = task["episode_length_s"] * 15
        chunks = math.ceil(steps / 32)
        _require(chunks <= 1024, "Task exceeds reserved policy query range")
        for index in range(count):
            episodes.append({"pair_id": f"{stage}/{task['task_id']}/{index:04d}",
                             "task_id": task["task_id"], "episode_index": index,
                             "scene_seed": namespace + task_index * 1024 + index,
                             "model_seed_base": namespace + (task_index + 1) * 1048576 + index * 1024,
                             "max_episode_steps": steps, "max_policy_chunks": chunks})
    protocol = {"schema_version": 2, "kind": "cosmos_robolab_paired_quality_v2", "stage": stage,
            "status": "prospective_protocol_not_evidence", "inventory": inventory,
            "inventory_sha256": digest(inventory), "episodes_per_task": count,
            "task_ids": [t["task_id"] for t in chosen], "episodes": episodes,
            "expected_pairs_per_family": len(episodes), "expected_records": len(episodes) * 4,
            "families": dict(CHECKPOINTS), "arms": list(ARMS), "acceptance": approval,
            "execution_bindings": _bindings(execution_bindings),
            "contract": {"sampler": "unipc", "steps": 4, "guidance": 3.0, "shift": 5.0,
                         "precision": "native", "action_padding": "native",
                         "instruction_type": "default", "num_envs": 1,
                         "control_hz": 15, "executed_action_horizon": 32,
                         "gripper_binarization": ">0.5", "native_initial_resets": 2,
                         "seed_rule": "model_seed_base + zero-based policy chunk index",
                         "paired_scene_state_required": True,
                         "baseline": "upstream eager", "candidate": "declared NUMERIC Runtime"},
            "scope": "Per-family paired success non-inferiority on this fixed task panel; no real-world or realtime claim"}
    if pairing_mode is not None:
        protocol.update(schema_version=3, kind="cosmos_robolab_paired_quality_v3")
        protocol["contract"].update(
            pairing_mode=pairing_mode,
            initial_observation_policy="retain_complete_native_observation_for_each_arm",
            paired_observation_fields="exact_nonvisual_values_and_image_keys_dtypes_shapes",
            semantic_scene_identity="session_renderproduct_view_picking_id_v1",
            renderer_intervention="none_native_two_resets_and_render_preset",
        )
    return protocol


def validate_protocol(protocol):
    _require(isinstance(protocol, dict), "Protocol must be an object")
    _require(type(protocol.get("schema_version")) is int and protocol["schema_version"] in (2, 3),
             "Schema 1 drafts carried an unapproved task-collapse gate; freeze a new schema 2 or 3 protocol")
    pairing_mode = protocol.get("contract", {}).get("pairing_mode")
    _require((protocol["schema_version"] == 2 and pairing_mode is None)
             or (protocol["schema_version"] == 3 and pairing_mode == NATIVE_RENDERER_PAIRING),
             "Renderer pairing mode must be explicitly bound by schema 3")
    a = protocol.get("acceptance", {})
    rebuilt = build_protocol(protocol.get("inventory", {}), stage=protocol.get("stage"),
                             episodes_per_task=protocol.get("episodes_per_task"),
                             success_margin=a.get("success_margin"),
                             approval_reference=a.get("approval_reference"),
                             report_only=a.get("mode") == "report_only",
                             execution_bindings=protocol.get("execution_bindings"),
                             pairing_mode=pairing_mode)
    _require(protocol == rebuilt, "Protocol fields, episode seeds or expected coverage were altered")
    return digest(protocol)


def model_seed(episode, chunk_index):
    _integer(chunk_index, 0, episode["max_policy_chunks"] - 1, "policy chunk index")
    return _integer(episode["model_seed_base"] + chunk_index, 0, 2**32 - 1, "model seed")


def _validate_record(record, protocol, expected, protocol_hash):
    _require(isinstance(record, dict), "Episode record must be an object")
    family, arm = record.get("family"), record.get("arm")
    _require(family in FAMILIES and arm in ARMS, "Unknown family or arm")
    pair_id = record.get("pair_id")
    _require(pair_id in expected, "Unexpected episode identity")
    episode = expected[pair_id]
    for field in ("task_id", "episode_index", "scene_seed"):
        _require(type(record.get(field)) is type(episode[field]) and record[field] == episode[field],
                 f"Episode {field} differs from manifest")
    _require(record.get("protocol_sha256") == protocol_hash, "Record belongs to another protocol")
    _require(record.get("robolab_revision") == ROBOLAB_REVISION, "RoboLab revision differs")
    _require(record.get("checkpoint_revision") == CHECKPOINTS[family]["revision"],
             "Checkpoint revision differs")
    _require(record.get("status") == "completed" and type(record.get("success")) is bool,
             "Incomplete/non-boolean simulator outcome")
    for field in ("process_exit_code", "renderer_process_exit_code"):
        if field in record:
            _require(type(record[field]) is int and record[field] == 0,
                     "A failed or unverified renderer process cannot become a completed outcome")
    _require(not record.get("cleanup_errors")
             and record.get("native_app_close") not in ("failed", "not_created"),
             "A renderer cleanup failure cannot become a completed outcome")
    _integer(record.get("reset_count"), 2, 2, "native initial reset count")
    steps = _integer(record.get("executed_steps"), 1, episode["max_episode_steps"], "executed steps")
    chunks = _integer(record.get("generated_chunks"), 1, episode["max_policy_chunks"], "generated chunks")
    _require(chunks == math.ceil(steps / 32), "Policy queries do not match executed 32-action chunks")
    seeds = record.get("model_seeds")
    _require(isinstance(seeds, list) and all(type(s) is int for s in seeds)
             and seeds == [model_seed(episode, i) for i in range(chunks)],
             "Missing, duplicated or mismatched policy query seeds")
    for field in ("initial_state_sha256", "initial_observation_sha256", "scene_config_sha256", "asset_inventory_sha256",
                  "simulator_fingerprint_sha256", "action_trace_sha256", "execution_binding_sha256"):
        _sha(record.get(field), field)
    if protocol["schema_version"] == 3:
        _require(record.get("pairing_mode") == NATIVE_RENDERER_PAIRING,
                 "Record lacks the prospectively declared native renderer pairing mode")
        for field in ("initial_nonvisual_observation_sha256", "initial_image_schema_sha256",
                      "raw_scene_config_sha256", "render_product_identity_sha256",
                      "renderer_process_completion_sha256", "collector_source_sha256"):
            _sha(record.get(field), field)
        _require(type(record.get("renderer_process_exit_code")) is int
                 and record["renderer_process_exit_code"] == 0,
                 "Native renderer records require independently collected process exit 0")
    binding = protocol["execution_bindings"][family][arm]
    if binding is not None:
        _require(record["execution_binding_sha256"] == binding, "Actual deployment binding differs")
    return family, arm, pair_id


def build_report(protocol, records):
    """Reject partial/mismatched runs; pending approval can never return a certificate."""
    protocol_hash = validate_protocol(protocol)
    expected = {row["pair_id"]: row for row in protocol["episodes"]}
    _require(expected, "Empty task suite")
    _require(isinstance(records, list) and records, "No episode records")
    indexed = {}
    for record in records:
        key = _validate_record(record, protocol, expected, protocol_hash)
        _require(key not in indexed, f"Duplicate episode record: {key}")
        indexed[key] = record
    keys = {(family, arm, pair_id) for family in FAMILIES for arm in ARMS for pair_id in expected}
    _require(set(indexed) == keys, f"Incomplete paired coverage: {len(indexed)}/{len(keys)} records")
    _require(len({r["simulator_fingerprint_sha256"] for r in records}) == 1,
             "Mixed simulator stacks or asset inventories")
    for pair_id in expected:
        paired = [indexed[(family, arm, pair_id)] for family in FAMILIES for arm in ARMS]
        paired_fields = ["initial_state_sha256", "scene_config_sha256", "asset_inventory_sha256"]
        if protocol["schema_version"] == 3:
            paired_fields.extend(("initial_nonvisual_observation_sha256", "initial_image_schema_sha256"))
        else:
            paired_fields.append("initial_observation_sha256")
        for field in paired_fields:
            _require(len({r[field] for r in paired}) == 1,
                     f"Paired {field} differs: {pair_id}")
    for family in FAMILIES:
        for arm in ARMS:
            _require(len({indexed[(family, arm, p)]["execution_binding_sha256"] for p in expected}) == 1,
                     "Mixed deployments within one arm")
    pending = []
    if protocol["stage"] != "formal":
        pending.append("smoke episodes are not a formal certificate")
    if protocol["acceptance"]["mode"] != "certify":
        pending.append("no prospectively approved success margin")
    if any(value is None for pair in protocol["execution_bindings"].values() for value in pair.values()):
        pending.append("exact deployments were not bound before capture")
    families = {}
    for family in FAMILIES:
        control, treatment = [], []
        for pair_id, episode in expected.items():
            for arm, outcomes in (("baseline", control), ("candidate", treatment)):
                row = indexed[(family, arm, pair_id)]
                outcomes.append(Outcome(pair_id, episode["scene_seed"], episode["task_id"], row["success"]))
        pairs = [(a.success, b.success) for a, b in zip(control, treatment)]
        low, high = _tango_paired_score_bounds(pairs, z=ONE_SIDED_95_Z)
        baseline_success = sum(a for a, _ in pairs) / len(pairs)
        candidate_success = sum(b for _, b in pairs) / len(pairs)
        per_task = {}
        for a, b in zip(control, treatment):
            task = per_task.setdefault(a.task, {"pairs": 0, "baseline_successes": 0, "candidate_successes": 0})
            task["pairs"] += 1
            task["baseline_successes"] += a.success
            task["candidate_successes"] += b.success
        for task in per_task.values():
            task["baseline_success"] = task["baseline_successes"] / task["pairs"]
            task["candidate_success"] = task["candidate_successes"] / task["pairs"]
            task["delta"] = task["candidate_success"] - task["baseline_success"]
        row = {"pairs": len(pairs), "baseline_success": baseline_success,
               "candidate_success": candidate_success, "delta": candidate_success - baseline_success,
               "lower_confidence_bound": low, "upper_confidence_bound": high,
               "interval": "Tango paired score one-sided 95% bounds (together central90)",
               "task_quality_validated": False, "certificate": None, "per_task": per_task,
               "task_collapse_gate": False,
               "collapsed_tasks_descriptive": sorted(t for t, counts in per_task.items()
                    if counts["baseline_successes"] > 0 and counts["candidate_successes"] == 0)}
        if not pending:
            certificate = certify(control, treatment,
                                  margin=protocol["acceptance"]["success_margin"],
                                  teacher_hash=protocol["execution_bindings"][family]["baseline"],
                                  student_hash=protocol["execution_bindings"][family]["candidate"],
                                  harness="benchmarks.vla.robolab_protocol", recipe=protocol_hash,
                                  seeds="explicit scene seeds and per-chunk model seeds in protocol",
                                  min_pairs=len(expected), interval="tango_one_sided95",
                                  fail_on_task_collapse=False)
            row["certificate"] = json.loads(certificate.to_json())
            row["task_quality_validated"] = certificate.passed
        families[family] = row
    passed = not pending and all(row["task_quality_validated"] for row in families.values())
    report = {"schema_version": 1, "status": "NOT_CERTIFIABLE" if pending else "PASS" if passed else "FAIL",
            "task_quality_validated": passed, "protocol_sha256": protocol_hash,
            "records_sha256": digest(sorted(records, key=lambda r: (r["family"], r["arm"], r["pair_id"]))),
            "records": len(records), "families": families, "pending": pending,
            "scope": protocol["scope"],
            "limits": ["Nominal per-family one-sided 95% bounds; no simultaneous family-wise guarantee",
                       f"Relative success on the fixed {len(protocol['task_ids'])}-task panel; not absolute policy quality",
                       "Repeated episodes share fixed task strata; bounds do not certify a population of unseen tasks",
                       "Provenance hashes must be backed by actual simulator and deployed-code evidence"]}
    if protocol["schema_version"] == 3:
        report["pairing_mode"] = NATIVE_RENDERER_PAIRING
        report["limits"].append(
            "Native rendering may vary across arms; original observations and view identifiers are retained. "
            "Pairing requires exact physical/camera state, nonvisual observations, image schema and semantic scene. "
            "This is a task-success comparison, not pixel equivalence or isolated action-error attribution."
        )
    return report


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--robolab-root", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--inventory", required=True)
    plan.add_argument("--stage", choices=("smoke", "formal"), required=True)
    plan.add_argument("--episodes-per-task", type=int)
    plan.add_argument("--success-margin", type=float)
    plan.add_argument("--approval-reference")
    plan.add_argument("--report-only", action="store_true")
    plan.add_argument("--execution-bindings")
    plan.add_argument("--pairing-mode", choices=(NATIVE_RENDERER_PAIRING,),
                      help="Explicit schema 3 native-renderer pairing; default retains schema 2 byte matching")
    report = commands.add_parser("report")
    report.add_argument("--protocol", required=True)
    report.add_argument("--records", required=True, help="One strict episode record per JSONL line")
    for command in (inventory, plan, report):
        command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(Path(path).read_text())
    if args.command == "inventory":
        value = build_task_inventory(args.robolab_root)
    elif args.command == "plan":
        value = build_protocol(read(args.inventory), stage=args.stage,
                               episodes_per_task=args.episodes_per_task, success_margin=args.success_margin,
                               approval_reference=args.approval_reference, report_only=args.report_only,
                               execution_bindings=read(args.execution_bindings) if args.execution_bindings else None,
                               pairing_mode=args.pairing_mode)
    else:
        records = [json.loads(line) for line in Path(args.records).read_text().splitlines() if line.strip()]
        value = build_report(read(args.protocol), records)
    _write_new(args.output, value)
    if args.command == "report":
        return 0 if value["status"] == "PASS" else 2 if value["status"] == "NOT_CERTIFIABLE" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Finite formal controller with durable completed renderer and Thor staging archives.

No launch occurs on import or --prepare. The scientific manifest is never rebuilt
or sampled here. Existing records are never resumed, skipped, or retried.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import shutil
import subprocess
import sys
import time
import traceback
from urllib.parse import urlparse


WRAPPER_SHA = "79807ef881e85211e1720e93bbb34859a2ed3b3c01d6d8c08ddd3f62602acf3f"
AUDITOR_SHA = "452b939461e269e73c89334a026a3082416510f3a30d838d55fe44a53d63de10"
CELLS = ("edge-eager_native", "edge-runtime_selected", "nano-eager_native", "nano-runtime_selected")
BATCH_SIZE = 8
LEGACY_CONTROLLER_SHA = "29c99649018ccab2201cdf77c2a52bff6d63adf0873ba971483ead69c9edeb6e"
RETAINED_CELL = "edge-eager_native"
RETAINED_PAIR = "formal/AnimalsInBinTask/0000"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_directories(path, *, exclusive=False):
    """Persist every newly created directory entry through its existing ancestor."""
    path = Path(path)
    missing, current = [], path
    while not current.exists():
        missing.append(current)
        current = current.parent
    require(current.is_dir(), "existing archive ancestor is not a directory")
    path.mkdir(parents=True, exist_ok=not exclusive)
    for created in reversed(missing):
        sync_directory(created)
        sync_directory(created.parent)


def durable(path, value, *, exclusive=True):
    path = Path(path)
    durable_directories(path.parent)
    target = path if exclusive else path.with_suffix(path.suffix + ".tmp")
    with target.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        target.replace(path)
    sync_directory(path.parent)


def load_module(path, expected, name):
    path = Path(path)
    require(sha(path) == expected, f"bound CPU source differs: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def batches(protocol):
    episodes = protocol["episodes"]
    require(len(episodes) == 1200 and len({row["pair_id"] for row in episodes}) == 1200,
            "formal execution requires the exact 1,200 unique prospective pairs")
    for row in episodes:
        require(row["pair_id"] == f"formal/{row['task_id']}/{row['episode_index']:04d}"
                and all(part not in {"", ".", ".."} for part in row["pair_id"].split("/")),
                "formal pair identity is not an exact safe task/index key")
    return [episodes[i:i+BATCH_SIZE] for i in range(0, len(episodes), BATCH_SIZE)]


def segment_episodes(protocol, batch_index, cell):
    require(cell in CELLS and type(batch_index) is int and 0 <= batch_index < 150,
            "unknown continuation segment")
    batch = batches(protocol)[batch_index]
    if batch_index == 0 and cell == RETAINED_CELL:
        require(batch[0]["pair_id"] == RETAINED_PAIR, "adopted first pair differs")
        return batch[1:]
    return batch


def adoption_manifest(config):
    reference = config["continuation_adoption"]
    require(sha(reference["local"]) == reference["sha256"], "completed episode adoption manifest changed")
    value = read(reference["local"])
    require(value.get("schema_version") == 1 and value.get("kind") == "formal_completed_first_episode_adoption_v1"
            and value.get("status") == "passed_preserved_native_only"
            and value.get("protocol_sha256") == config["protocol"]["canonical_sha256"]
            and value.get("protocol_file_sha256") == config["protocol"]["sha256"]
            and value.get("original_run_id") == "paired_formal_v1", "unknown or changed completed episode adoption")
    retained = value["retained"]
    require({key: retained.get(key) for key in ("cell", "pair_id", "collector_source_sha256", "success",
                                              "executed_steps", "generated_chunks", "closed_server_episode_count")} == {
        "cell": RETAINED_CELL, "pair_id": RETAINED_PAIR, "collector_source_sha256": LEGACY_CONTROLLER_SHA,
        "success": False, "executed_steps": 1350, "generated_chunks": 43, "closed_server_episode_count": 1},
        "only the exact preserved native first outcome may be adopted")
    require(value["continuation"] == {
        "first_server_job_id": "batches/0000/servers/" + RETAINED_CELL,
        "first_server_episode_ids": [f"formal/AnimalsInBinTask/{index:04d}" for index in range(1, 8)],
        "new_episode_runs": 4799, "new_server_runs": 600, "aggregate_expected_records": 4800},
        "the exact first1+7 continuation cannot be resampled or shortened")
    return value


def validate_adoption(config, protocol, identities, helper, auditor, archive, api):
    """Independently replay the retained native outcome before any new transport."""
    manifest = adoption_manifest(config)
    refs = [manifest[key] for key in ("original_config", "original_controller_source", "diagnostic_audit",
                                     "local_inventory", "preserved_tree_verification")]
    require(set(manifest["remote_inventories"]) == {"renderer", "thor_closed_server"},
            "adoption must bind both retained native trees")
    for pair in manifest["remote_inventories"].values():
        require(isinstance(pair, list) and len(pair) == 2, "two preserved remote inventories are required")
        refs.extend(pair)
    for ref in refs:
        require(Path(ref["local"]).is_absolute() and sha(ref["local"]) == ref["sha256"],
                "preserved completed-episode evidence changed")
    require(manifest["original_controller_source"]["sha256"] == LEGACY_CONTROLLER_SHA,
            "retained record cannot be relabeled under a new collector")
    original = Path(manifest["original_study_root"])
    require(original.is_absolute() and original.is_dir() and not original.is_symlink(),
            "explicit retained original study is missing")
    previous_config = read(manifest["original_config"]["local"])
    require(previous_config["run_id"] == manifest["original_run_id"] and read(original / "config.json") == previous_config
            and read(original / "protocol.json") == protocol and read(original / "identities.json") == identities,
            "retained original study changed protocol, identities or configuration")
    require(read(original / "source.json") == {
        "config_sha256": manifest["original_config"]["sha256"], "controller_sha256": LEGACY_CONTROLLER_SHA,
        "transport_auditor_sha256": AUDITOR_SHA, "wrapper_sha256": WRAPPER_SHA},
        "retained actual controller source receipt differs")
    for host in ("renderer", "thor"):
        old, new = Path(previous_config[host]["output_root"]), Path(config[host]["output_root"])
        require(not new.is_relative_to(old) and not old.is_relative_to(new),
                "new staging roots must exclude all retained original staging")
    old = load_module(manifest["original_controller_source"]["local"], LEGACY_CONTROLLER_SHA,
                      "formal_retained_original_controller")
    old_protocol, old_ids, old_helper, old_auditor, old_archive, old_api = old.load_inputs(
        previous_config, manifest["original_config"]["sha256"], for_execution=True)
    require(old_protocol == protocol and old_ids == identities, "old and continued scientific contracts differ")
    episode = protocol["episodes"][0]
    renderer_spec = old.specifications(previous_config, old_helper, 0, RETAINED_CELL, episode)
    server_spec = old.specifications(previous_config, old_helper, 0, RETAINED_CELL)
    directory = original / renderer_spec["job_id"]
    renderer, snapshot = directory / "renderer", directory / "thor_snapshot"
    final = original / server_spec["job_id"] / "final_server"
    record_path = directory / "collected_record.json"
    require(sha(record_path) == manifest["retained"]["collected_record_sha256"], "retained native record bytes changed")
    controller = old.FormalController(previous_config, protocol, identities, old_helper, old_auditor,
                                    old_archive, old_api, None, original)
    original_durable = old.durable
    try:
        def existing_only(path, value, **kwargs):
            require(read(path) == value, "retained original anchor changed")
        old.durable = existing_only
        completion = read(renderer / "completion.json")
        record = controller.collect_record(directory, completion, RETAINED_CELL, episode)
    finally:
        old.durable = original_durable
    require(record == read(record_path) and record["success"] is False and record["executed_steps"] == 1350
            and record["generated_chunks"] == 43 and record["reset_count"] == 2
            and record["collector_source_sha256"] == LEGACY_CONTROLLER_SHA,
            "retained native fields, outcome or old collector proof changed")
    auditor.validate_job_identity(renderer, renderer_spec, read(directory / "launch.json"))
    controller.validate_owned_kit_log(renderer, renderer_spec)
    controller.validate_server_evidence(snapshot, server_spec, RETAINED_CELL)
    controller.validate_server_evidence(final, server_spec, RETAINED_CELL, closed=True)
    instruction = next(row["instruction_default"] for row in protocol["inventory"]["tasks"]
                       if row["task_id"] == episode["task_id"])
    transport = auditor.audit_transport(directory, record, episode, identities[RETAINED_CELL], 0, 0, instruction)
    require(read(directory / "local_episode_audit.json") == dict(
        transport, status="passed", task_quality_certified=False, source_sha256=LEGACY_CONTROLLER_SHA,
        transport_auditor_sha256=AUDITOR_SHA), "retained complete native transport changed")
    closed = read(final / "completion.json")
    require(closed["status"] == "stopped" and closed["stop_reason"] == "SIGTERM" and closed["exit_code"] == 0
            and not any(closed.get(key) for key in ("forced_kill", "cleanup_forced_kill", "owned_processes_still_live")),
            "retained one-stream server did not stop cleanly")
    ledger = final / "supervisor" / RETAINED_CELL / "requests"
    require(read(ledger / "closed.json") == {"failed": False, "requests": 43, "episodes": 1,
                                            "benchmark_identity_sha256": identities[RETAINED_CELL]["benchmark_identity_sha256"]},
            "retained server is not the exact single completed stream")
    auditor.validate_ledger_files(ledger, 43, 1, closed=True)
    before = snapshot / "supervisor" / RETAINED_CELL / "requests"
    require(all(sha(path) == sha(ledger / path.name) for path in before.iterdir()),
            "retained request ledger changed before its clean close")
    supervisor = read(final / "supervisor/completion.json")
    require(supervisor["scripts_sha256"] == {
        Path(previous_config["thor"]["supervisor"]).name: previous_config["thor"]["bound_files"][previous_config["thor"]["supervisor"]],
        Path(previous_config["thor"]["worker"]).name: previous_config["thor"]["bound_files"][previous_config["thor"]["worker"]]},
        "retained actual supervisor/worker source hashes differ")
    worker_command = supervisor["jobs"][0]["command"]
    require(worker_command[1:3] == ["-I", previous_config["thor"]["worker"]]
            and worker_command[worker_command.index("--mode") + 1] == "serve"
            and worker_command[worker_command.index("--cell-id") + 1] == RETAINED_CELL,
            "retained isolated native worker command differs")
    measured_ref = previous_config["measured_receipts"][RETAINED_CELL]
    expected_guard = {"measured_receipt_sha256": measured_ref["sha256"],
                      "verified_files": read(measured_ref["local"])["sources"],
                      "expected_identity_file_sha256": previous_config["identities"][RETAINED_CELL]["sha256"]}
    require(supervisor["jobs"][0]["source_guard_before"] == supervisor["jobs"][0]["source_guard_after"] == expected_guard
            and supervisor["installation_sha256"] == previous_config["thor"]["installation_sha256"],
            "retained original native installation guards changed")
    diagnostic = read(manifest["diagnostic_audit"]["local"])
    require(diagnostic["status"] == "passed_preserved_native_and_one_stream_close_diagnostic_only"
            and diagnostic["collected_record_sha256"] == sha(record_path) and diagnostic["transport"] == transport
            and diagnostic["native_external_completion_sha256"] == sha(renderer / "completion.json")
            and diagnostic["closed_server_completion_sha256"] == sha(final / "completion.json")
            and diagnostic["closed_ledger_sha256"] == sha(ledger / "closed.json")
            and diagnostic["benchmark_identity_sha256"] == identities[RETAINED_CELL]["benchmark_identity_sha256"]
            and diagnostic["local_inventory"]["sha256"] == manifest["local_inventory"]["sha256"]
            and diagnostic["archive_admitted"] is False and diagnostic["reclamation_occurred"] is False,
            "retained audit does not bind the actual outcome and failed archive gate")
    proven_hosts = set()
    for ref in diagnostic["actual_external_completion_rpcs"]:
        require(sha(ref["path"]) == ref["sha256"], "retained external completion RPC bytes changed")
        rpc = read(ref["path"])
        host = ref["host"]
        expected_output = renderer_spec["output"] if host == "renderer" else server_spec["output"]
        expected_completion = completion if host == "renderer" else closed
        require(host in {"renderer", "thor"} and rpc["returncode"] == 0
                and rpc["command"][:-1] == ["ssh", *previous_config[host]["ssh"]["options"], previous_config[host]["ssh"]["target"]]
                and shlex.split(rpc["command"][-1]) == [previous_config[host]["control_python"], previous_config[host]["runner"],
                                                        "--remote-status", expected_output]
                and json.loads(rpc["stdout"])["completion"] == expected_completion,
                "retained actual external native completion differs")
        proven_hosts.add(host)
    require(proven_hosts == {"renderer", "thor"}, "retained native and server external completion proof missing")
    failure = diagnostic["archive_failure"]
    require(sha(failure["path"]) == failure["sha256"]
            and json.loads(read(failure["path"])["stdout"]) == {
                "error": "cannot prove absence of live job handles", "error_type": "ValueError",
                "reclaimed": False, "status": "failed"}
            and read(original / "completion.json")["status"] == "failed"
            and read(original / "completion.json")["completed_records"] == 0
            and not (original / "records.json").exists(), "original archive failure was relabeled or extended")
    inventory = read(manifest["local_inventory"]["local"])
    verification = read(manifest["preserved_tree_verification"]["local"])
    require(verification["status"] == "passed_preserved_native_tree_verification_only"
            and verification["frozen_local_inventory_sha256"] == manifest["local_inventory"]["sha256"]
            and verification["old_global_handle_gate_passed"] is False
            and verification["original_archive_failure_preserved"] is True,
            "retained tree verification cannot invent archive admission")
    trees = {"renderer": renderer, "thor_episode_snapshot": snapshot, "thor_closed_server": final}
    for name, tree in trees.items():
        entries = archive.scan_tree(tree)
        require(entries == inventory[name]["entries"] and archive.object_sha(entries) == inventory[name]["entries_sha256"],
                "retained whole local native tree changed")
        if name == "thor_episode_snapshot":
            continue
        remote_expected = renderer_spec["output"] if name == "renderer" else server_spec["output"]
        checked = verification["trees"][name]
        require(checked["local_tree"] == str(tree) and checked["remote_tree"] == remote_expected
                and checked["entries_sha256"] == inventory[name]["entries_sha256"]
                and checked["original_remote_tree_retained"] is True and checked["reclamation"] is False,
                "preserved native tree verification binds another location")
        for index, ref in enumerate(manifest["remote_inventories"][name]):
            remote = read(ref["local"])
            require(ref["sha256"] == checked[("first" if index == 0 else "second") + "_remote_inventory_sha256"]
                    and remote["status"] == "read_only_inventory" and remote["remote_tree"] == remote_expected
                    and remote["entries"] == entries and remote["entries_sha256"] == inventory[name]["entries_sha256"]
                    and remote["archive_admission"] is False and remote["reclamation"] is False
                    and remote["global_handle_absence_claim"] is False,
                    "two remote native inventories do not bind the identical retained bytes")
    anchor = read(original / "anchors" / (RETAINED_PAIR + ".json"))
    require(anchor == read(directory / "native_anchor.json"), "retained golden/native anchor differs")
    return {"manifest": manifest, "record": record, "anchor": anchor, "original_study_root": original,
            "trees": trees, "transport": transport, "inventory_sha256": manifest["local_inventory"]["sha256"]}


def specifications(config, helper, batch_index, cell, episode=None):
    """Only job/anchor paths change; native worker/driver argument semantics stay frozen."""
    prefix = f"batches/{batch_index:04d}"
    if episode is None:
        spec = helper.server_spec(config, cell)
        new_id = f"{prefix}/servers/{cell}"
    else:
        require((cell, episode["pair_id"]) != (RETAINED_CELL, RETAINED_PAIR),
                "the adopted completed episode must never be launched again")
        spec, old_anchor = helper.episode_spec(config, cell, episode)
        new_id = f"{prefix}/episodes/{cell}/{episode['pair_id']}"
    old_output = spec["output"]
    host = "thor" if episode is None else "renderer"
    new_output = f"{config[host]['output_root']}/{new_id}"
    command = []
    for argument in spec["command"]:
        if argument == old_output or argument.startswith(old_output + "/"):
            argument = new_output + argument[len(old_output):]
        if episode is not None and argument == old_anchor:
            argument = f"{config['renderer']['output_root']}/anchors/{episode['pair_id']}.json"
        if episode is not None and argument == "--kit_args=--/log/file={job_output}/kit.log":
            require(not any(character.isspace() or character in "\"'" for character in new_output),
                    "owned Kit log path must remain one native argument")
            argument = "--kit_args=--/log/file=" + new_output + "/kit.log"
        command.append(argument)
    return dict(spec, job_id=new_id, output=new_output, command=command)


def ownership(config, protocol, helper, config_sha):
    jobs = [specifications(config, helper, i, cell, episode)
            for i, batch in enumerate(batches(protocol)) for cell in CELLS
            for episode in segment_episodes(protocol, i, cell)]
    return {"schema_version": 1, "kind": "formal_episode_run_ownership_v1", "stage": "formal",
            "run_id": config["run_id"], "run_root": config["renderer"]["output_root"],
            "owner_uid": config["renderer"]["owner_uid"], "run_root_mode": 0o700,
            "continuation_adoption_source": {"path": config["continuation_adoption"]["renderer"],
                                             "sha256": config["continuation_adoption"]["sha256"]},
            "adopted_anchor_sha256": sha(Path(adoption_manifest(config)["original_study_root"]) / "anchors" / (RETAINED_PAIR + ".json")),
            "protocol_sha256": config["protocol"]["canonical_sha256"], "config_sha256": config_sha,
            "wrapper_source": {"path": config["renderer"]["runner"], "sha256": WRAPPER_SHA},
            "archive_helper_sha256": config["archive"]["helper_sha256"],
            "reclamation_authorized": config["archive"]["reclaim_completed_renderer_episodes"],
            "episode_jobs": [spec["job_id"] for spec in jobs],
            "episode_spec_sha256": {spec["job_id"]: hashlib.sha256(encoded(spec)).hexdigest() for spec in jobs}}


def server_ownership(config, protocol, helper, config_sha):
    """Bind the exceptional first seven and all 599 ordinary eight-stream segments."""
    jobs, plans = [], {}
    for index, batch in enumerate(batches(protocol)):
        for cell in CELLS:
            spec = specifications(config, helper, index, cell)
            jobs.append(spec)
            identity = read(config["identities"][cell]["local"])
            plans[spec["job_id"]] = {
                "worker_cell": cell, "episode_ids": [row["pair_id"] for row in segment_episodes(protocol, index, cell)],
                "expected_episode_count": len(segment_episodes(protocol, index, cell)),
                **({"adopted_pair_id": RETAINED_PAIR} if index == 0 and cell == RETAINED_CELL else {}),
                "identity_source": {"path": config["identities"][cell]["thor"],
                                    "sha256": config["identities"][cell]["sha256"]},
                "measured_receipt_source": {"path": config["measured_receipts"][cell]["thor"],
                                            "sha256": config["measured_receipts"][cell]["sha256"]},
                "benchmark_identity_sha256": identity["benchmark_identity_sha256"]}
    thor = config["thor"]
    return {"schema_version": 1, "kind": "formal_server_run_ownership_v1", "stage": "formal",
            "run_id": config["run_id"], "run_root": thor["output_root"],
            "owner_uid": thor["owner_uid"], "run_root_mode": 0o700,
            "continuation_adoption_source": {"path": config["continuation_adoption"]["thor"],
                                             "sha256": config["continuation_adoption"]["sha256"]},
            "protocol_sha256": config["protocol"]["canonical_sha256"], "config_sha256": config_sha,
            "wrapper_source": {"path": thor["runner"], "sha256": WRAPPER_SHA},
            "supervisor_source": {"path": thor["supervisor"], "sha256": thor["bound_files"][thor["supervisor"]]},
            "worker_source": {"path": thor["worker"], "sha256": thor["bound_files"][thor["worker"]]},
            "installation_source": {"path": thor["installation"], "sha256": thor["installation_sha256"]},
            "archive_helper_sha256": config["archive"]["helper_sha256"],
            "reclamation_authorized": config["archive"]["reclaim_completed_thor_servers"],
            "server_jobs": [spec["job_id"] for spec in jobs],
            "server_spec_sha256": {spec["job_id"]: hashlib.sha256(encoded(spec)).hexdigest() for spec in jobs},
            "server_batches": plans}


def load_inputs(config, config_sha, *, for_execution):
    require(config["schema_version"] == 1 and config["kind"] == "cosmos_robolab_formal_controller_v4",
            "unknown explicit formal controller configuration")
    require(config["controller_sha256"] == sha(__file__), "formal controller source differs from frozen config")
    root = Path(config["local_flash_root"])
    here = Path(__file__).resolve().parent
    helper = load_module(here / "run_paired_smoke_v2.py", WRAPPER_SHA, "formal_frozen_remote_wrapper")
    auditor = load_module(here / "audit_paired_smoke_v1.py", AUDITOR_SHA, "formal_frozen_transport_auditor")
    archive = load_module(config["archive"]["helper_local"], config["archive"]["helper_sha256"], "formal_archive_helper")
    sys.path.insert(0, str(root))
    api = load_module(root / "benchmarks/vla/robolab_protocol.py", config["protocol_validator_sha256"], "formal_pinned_protocol")
    protocol = read(config["protocol"]["local"])
    require(sha(config["protocol"]["local"]) == config["protocol"]["sha256"]
            and api.digest(protocol) == config["protocol"]["canonical_sha256"], "formal manifest bytes/digest differ")
    api.validate_protocol(protocol)
    require(protocol["stage"] == "formal" and protocol["schema_version"] == 3
            and protocol["contract"]["pairing_mode"] == "native_renderer_physical_v1"
            and protocol["expected_records"] == 4800 and protocol["episodes_per_task"] == 10
            and len(protocol["task_ids"]) == 120 and protocol["acceptance"]["mode"] == "certify"
            and protocol["acceptance"]["success_margin"] == -0.05
            and protocol["acceptance"]["interval"] == "tango_one_sided95",
            "formal execution cannot change fixed coverage or approved 5pp/95% acceptance")
    batches(protocol)
    require(set(config["identities"]) == set(CELLS), "exactly four pre-collected deployment identities are required")
    identities = {}
    for cell in CELLS:
        declaration = config["identities"][cell]
        require(sha(declaration["local"]) == declaration["sha256"], "frozen policy identity file differs")
        identity = read(declaration["local"])
        family, route = cell.split("-", 1)
        arm = "baseline" if route == "eager_native" else "candidate"
        require(identity["benchmark_identity"]["cell_id"] == cell
                and identity["benchmark_identity_sha256"] == auditor.digest(identity["benchmark_identity"])
                == protocol["execution_bindings"][family][arm], "four identities differ from prospective formal bindings")
        identities[cell] = identity
    require(set(config["measured_receipts"]) == set(CELLS), "four exact measured installation receipts are required")
    for cell in CELLS:
        declaration = config["measured_receipts"][cell]
        require(sha(declaration["local"]) == declaration["sha256"]
                and declaration["thor"] == config["thor"]["bundle"].rstrip("/") + "/measured_receipts/" + cell + ".json"
                and config["thor"]["bound_files"].get(declaration["thor"]) == declaration["sha256"],
                "native measured source receipt must be explicitly hash-bound at its actual bundle path")
        measured = read(declaration["local"])
        identity = identities[cell]["benchmark_identity"]
        require(measured["cell_id"] == cell and measured["model_id"] == identity["model_id"]
                and measured["revision"] == identity["revision"]
                and measured["runtime_kwargs"] == identity["runtime_kwargs"]
                and measured["optimizer_environment"] == identity["optimizer_environment"],
                "measured installation receipt differs from the prospective selected policy")
    require(type(config["archive"]["reclaim_completed_renderer_episodes"]) is bool, "explicit reclamation selection required")
    require(type(config["archive"]["reclaim_completed_thor_servers"]) is bool, "explicit Thor reclamation selection required")
    renderer = config["renderer"]
    endpoint = urlparse(renderer["endpoint"])
    require(endpoint.scheme == "ws" and endpoint.hostname in {"127.0.0.1", "localhost"}
            and endpoint.port == config["thor"]["port"], "private renderer relay endpoint must match the Thor server port")
    protected = {"--protocol", "--pair-id", "--family", "--arm", "--robolab-root", "--endpoint", "--identity",
                 "--asset-manifest", "--renderer-compat-manifest", "--output", "--anchor", "--device", "--selector",
                 "--build-result", "--flash-root", "--bootstrap-receipt", "--entry-script", "--episode-driver", "--rpc-timeout"}
    require(isinstance(renderer["launcher_args"], list)
            and all(isinstance(argument, str) and argument.split("=", 1)[0] not in protected
                    for argument in renderer["launcher_args"]), "launcher arguments cannot override protected native fields or GPU0")
    owned_log_args = ["--kit_args=--/log/file={job_output}/kit.log"]
    require(renderer["launcher_args"] in ([], owned_log_args),
            "formal Kit arguments allow only the single exact owned-log token; native abbreviations are forbidden")
    require(not for_execution or renderer["launcher_args"] == owned_log_args,
            "formal execution requires the original unsuppressed Kit log inside its owned job")
    kit_args = [index for index, argument in enumerate(renderer["launcher_args"])
                if argument.split("=", 1)[0] == "--kit_args"]
    if kit_args:
        index = kit_args[0]
        require(len(kit_args) == 1
                and renderer["launcher_args"][index] == "--kit_args=--/log/file={job_output}/kit.log"
                and not any(character.isspace() or character in "\"'" for character in renderer["output_root"]),
                "formal Kit arguments may only route the original log into the owned job")
    require(all("{job_output}" not in argument or argument == "--kit_args=--/log/file={job_output}/kit.log"
                for argument in renderer["launcher_args"]), "unknown owned-log placeholder")
    require(all(type(config[host]["owner_uid"]) is int and config[host]["owner_uid"] >= 0
                for host in ("thor", "renderer")), "both private run roots require explicit actual owner UIDs")
    adoption_manifest(config)
    for host in ("thor", "renderer"):
        require(config[host]["bound_files"].get(config["continuation_adoption"][host])
                == config["continuation_adoption"]["sha256"], "both remote owners must bind the exact adoption source")
    require("CUDA_VISIBLE_DEVICES" in renderer["environment"] and renderer["environment"]["CUDA_VISIBLE_DEVICES"] is None,
            "renderer CUDA_VISIBLE_DEVICES must remain explicitly unset")
    require(isinstance(renderer["launch_prefix"], list) and all(isinstance(argument, str) for argument in renderer["launch_prefix"])
            and isinstance(renderer["launch_prefix_sources"], list), "renderer launch prefix and sources must be explicit lists")
    for path in renderer["launch_prefix_sources"]:
        require(path in renderer["bound_files"], "renderer launch prefix source must be remotely hash-bound")
    for host in ("thor", "renderer"):
        entry = config[host]
        require(Path(entry["output_root"]).is_absolute()
                and Path(entry["output_root"]) != Path("/") and ".." not in Path(entry["output_root"]).parts
                and Path(entry["runner"]).is_absolute()
                and entry["bound_files"].get(entry["runner"]) == WRAPPER_SHA, "explicit fresh remote root/immutable wrapper required")
        require(isinstance(entry["ssh"]["options"], list)
                and all(isinstance(option, str) for option in entry["ssh"]["options"])
                and isinstance(entry["ssh"]["target"], str) and entry["ssh"]["target"], "explicit SSH destination/options required")
        for cell in CELLS:
            require(entry["bound_files"].get(config["identities"][cell][host]) == config["identities"][cell]["sha256"],
                    f"remote deployment identity must be hash-bound: {host}/{cell}")
        required_sources = ("supervisor", "worker", "installation") if host == "thor" else (
            "bootstrap", "selector", "build_result", "asset_manifest", "compat_manifest")
        for key in required_sources:
            require(entry[key] in entry["bound_files"], f"remote execution source must be hash-bound: {host}/{key}")
    require(config["thor"]["bound_files"][config["thor"]["installation"]] == config["thor"]["installation_sha256"],
            "native supervisor and wrapper must bind the same installation manifest")
    require(renderer["bound_files"].get(config["protocol"]["renderer"]) == config["protocol"]["sha256"],
            "remote formal protocol must be hash-bound")
    for relative in ("benchmarks/vla/robolab_driver.py", "benchmarks/vla/robolab_assets.py", "instinctflash/verify/certify.py"):
        remote_path = renderer["flash_root"].rstrip("/") + "/" + relative
        require(remote_path in renderer["bound_files"], "native driver and certificate math sources must be hash-bound")
        if relative == "instinctflash/verify/certify.py":
            require(sha(root / relative) == renderer["bound_files"][remote_path]
                    and sha(sys.modules[api.certify.__module__].__file__) == renderer["bound_files"][remote_path],
                    "actually imported certificate math source differs from frozen execution")
    for host in ("thor", "renderer"):
        require(config[host]["bound_files"].get(config["archive"]["helper_" + host]) == config["archive"]["helper_sha256"],
                "both remote immutable job guards must bind archive helper source")
    for host, key in (("renderer", "ownership_renderer"), ("thor", "server_ownership_thor")):
        path = Path(config["archive"][key])
        require(path.is_absolute() and ".." not in path.parts
                and not path.is_relative_to(config[host]["output_root"]),
                "prospective ownership must be outside the fresh remote run root")
    limits = config["limits"]
    minimum_free = config["capacity"]["minimum_free_bytes"]
    require(set(minimum_free) == {"thor", "renderer"}
            and all(type(value) is int and value >= 3 * (1 << 29) for value in minimum_free.values()),
            "both remote hosts require at least 1.5GiB configured free-space headroom")
    require(all(type(limits[name]) in (int, float) and 0 < limits[name] <= 172800
                for name in ("server_start_seconds", "episode_seconds", "server_seconds", "copy_seconds",
                             "grace_seconds", "control_seconds", "rpc_seconds", "poll_seconds")), "finite limits required")
    control_calls = 10 if config["archive"]["reclaim_completed_renderer_episodes"] else 8
    episode_budget = (limits["episode_seconds"] + limits["grace_seconds"] + 2 * limits["copy_seconds"]
                      + control_calls * limits["control_seconds"])
    require(limits["server_seconds"] >= limits["server_start_seconds"] + BATCH_SIZE * episode_budget + limits["grace_seconds"],
            "server lifetime does not cover eight bounded episodes, copies, archive RPCs, startup and cleanup")
    require(type(config["thor"]["worker_grace_seconds"]) in (int, float)
            and math.isfinite(config["thor"]["worker_grace_seconds"]) and config["thor"]["worker_grace_seconds"] > 0
            and limits["grace_seconds"] >= config["thor"]["worker_grace_seconds"] + 30,
            "outer cleanup bound must leave room after native worker grace")
    validate_adoption(config, protocol, identities, helper, auditor, archive, api)
    if for_execution:
        require(config.get("template_only") is False, "formal template is not an execution admission")
        admission_ref = config["admission"]
        require(sha(admission_ref["local"]) == admission_ref["sha256"], "formal admission file differs")
        admission = read(admission_ref["local"])
        require(admission.get("kind") == "formal_native_execution_admission_v1" and admission.get("status") == "passed"
                and admission.get("protocol_sha256") == api.digest(protocol)
                and admission.get("four_smoke_cells_complete") is True
                and admission.get("native_renderer_formal_root_qualified") is True
                and admission.get("formal_asset_manifest_sha256") == config["renderer"]["bound_files"][config["renderer"]["asset_manifest"]]
                and admission.get("renderer_compat_manifest_sha256") == config["renderer"]["bound_files"][config["renderer"]["compat_manifest"]],
                "actual smoke and separate formal-root native admission are required")
        require(admission.get("bootstrap_sha256") == renderer["bound_files"][renderer["bootstrap"]]
                and admission.get("driver_sha256") == renderer["bound_files"][renderer["flash_root"].rstrip("/")
                                                                              + "/benchmarks/vla/robolab_driver.py"],
                "formal renderer qualification must bind the actually launched bootstrap and native driver")
        for ref in admission["evidence"]:
            require(sha(ref["local"]) == ref["sha256"], "formal prerequisite evidence changed")
        require(admission["evidence"], "formal admission lacks bound actual evidence")
        require(admission.get("continuation_adoption_sha256") == config["continuation_adoption"]["sha256"]
                and admission.get("private_renderer_guard_qualified") is True,
                "continuation requires explicit adopted evidence and actual private renderer guard admission")
        expected = ownership(config, protocol, helper, config_sha)
        require(read(config["archive"]["ownership_local"]) == expected, "formal ownership differs from exact prospective jobs/config")
        expected_server = server_ownership(config, protocol, helper, config_sha)
        require(read(config["archive"]["server_ownership_local"]) == expected_server,
                "formal Thor ownership differs from exact prospective batches/config")
    return protocol, identities, helper, auditor, archive, api


class FormalTransport:
    """Bounded one-attempt SSH calls; successful response JSON is retained."""

    def __init__(self, config, output, helper):
        self.config, self.output, self.index = config, Path(output), 0
        self.base = helper.SSHTransport(config, output / "legacy_copy_control")

    def invoke(self, host, executable, arguments, payload=None):
        entry = self.config[host]
        command = ["ssh", *entry["ssh"]["options"], entry["ssh"]["target"],
                   shlex.join([entry["control_python"], executable, *arguments])]
        self.index += 1
        path = self.output / "control" / f"{self.index:07d}-{host}.json"
        try:
            result = subprocess.run(command, input=None if payload is None else encoded(payload).decode(),
                                    capture_output=True, text=True, timeout=self.config["limits"]["control_seconds"])
        except subprocess.TimeoutExpired as error:
            def text(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value
            durable(path, {"command": command, "timeout": True, "timeout_seconds": error.timeout,
                           "stdout": text(error.stdout), "stderr": text(error.stderr), "automatic_retry": False})
            raise
        durable(path, {"command": command, "returncode": result.returncode,
                       "stdout": result.stdout, "stderr": result.stderr})
        require(result.returncode == 0, "formal control RPC failed; no retry")
        return json.loads(result.stdout)

    def call(self, host, args, payload=None):
        return self.invoke(host, self.config[host]["runner"], args, payload)

    def prepare(self, host, path):
        return self.call(host, ["--remote-prepare", str(path)])

    def spawn(self, host, spec):
        return self.call(host, ["--remote-spawn"], spec)

    def status(self, host, output):
        return self.call(host, ["--remote-status", output])

    def stop(self, host, output):
        return self.call(host, ["--remote-stop", output])

    def read_json(self, host, path):
        return self.call(host, ["--remote-read-json", path])

    def copy(self, host, source, destination):
        self.base.copy(host, source, destination)

    def archive(self, operation, job_id, receipt, *, inventory=None, ack=None, payload=None, host="renderer"):
        require(host in {"renderer", "thor"}, "explicit archive host required")
        archive = self.config["archive"]
        local_key, remote_key = (("ownership_local", "ownership_renderer") if host == "renderer"
                                 else ("server_ownership_local", "server_ownership_thor"))
        args = [operation, "--run-root", self.config[host]["output_root"], "--job-id", job_id,
                "--ownership", archive[remote_key], "--ownership-sha256", sha(archive[local_key]),
                "--receipt", receipt]
        if inventory:
            args += ["--inventory", inventory["receipt"], "--inventory-sha256", inventory["receipt_sha256"]]
        if ack:
            args += ["--archive-ack", ack["receipt"], "--archive-ack-sha256", ack["receipt_sha256"], "--allow-reclaim"]
        return self.invoke(host, archive["helper_" + host], args, payload)

    def prepare_private_root(self, host):
        archive = self.config["archive"]
        root = self.config[host]["output_root"]
        local, remote = (("server_ownership_local", "server_ownership_thor") if host == "thor"
                         else ("ownership_local", "ownership_renderer"))
        return self.invoke(host, archive["helper_" + host], [
            "prepare-run", "--run-root", root, "--ownership", archive[remote],
            "--ownership-sha256", sha(archive[local]),
            "--receipt", root + "/archive_receipts/private_root.json"])

    def install_adopted_anchor(self, anchor):
        archive = self.config["archive"]
        root = self.config["renderer"]["output_root"]
        return self.invoke("renderer", archive["helper_renderer"], [
            "adopt-anchor", "--run-root", root, "--ownership", archive["ownership_renderer"],
            "--ownership-sha256", sha(archive["ownership_local"]),
            "--receipt", root + "/archive_receipts/adopted_anchor.json"], anchor)

    def free_space(self, host):
        return self.invoke(host, self.config["archive"]["helper_" + host], [
            "free-space", "--path", self.config[host]["output_root"], "--minimum-free-bytes",
            str(self.config["capacity"]["minimum_free_bytes"][host])])


class FormalController:
    def __init__(self, config, protocol, identities, helper, auditor, archive, api, transport, output):
        self.config, self.protocol, self.identities = config, protocol, identities
        self.helper, self.auditor, self.archive, self.api = helper, auditor, archive, api
        self.transport, self.output = transport, Path(output)
        self.records, self.server, self.renderer = [], None, None
        self.server_lifetimes = {}
        self.state = {"status": "prepared", "expected_records": 4800, "completed_records": 0,
                      "automatic_retry": False, "formal_follow_on": False, "task_quality_validated": False}

    def save(self):
        durable(self.output / "live_status.json", self.state, exclusive=False)

    def wait(self, host, spec, seconds):
        deadline = time.monotonic() + seconds
        while True:
            status = self.transport.status(host, spec["output"])
            if status["completion"] is not None:
                return status["completion"]
            require(status["live"], "owned native job exited without completion")
            require(time.monotonic() < deadline, "finite native job deadline exceeded")
            time.sleep(min(self.config["limits"]["poll_seconds"], max(0, deadline - time.monotonic())))

    def ready(self, spec, cell):
        deadline = time.monotonic() + self.config["limits"]["server_start_seconds"]
        while True:
            status = self.transport.status("thor", spec["output"])
            require(status["completion"] is None and status["live"], "owned server exited during startup")
            receipt = self.transport.read_json("thor", spec["output"] + f"/supervisor/{cell}/ready.json")
            if receipt is not None:
                require(receipt["status"] == "listening" and receipt["metadata"] == self.identities[cell]
                        and receipt["host"] == "127.0.0.1" and receipt["port"] == self.config["thor"]["port"],
                        "actual server readiness or deployment identity differs")
                return receipt
            require(time.monotonic() < deadline, "finite server readiness deadline exceeded")
            time.sleep(self.config["limits"]["poll_seconds"])

    def stop(self, host, spec):
        self.transport.stop(host, spec["output"])
        return self.wait(host, spec, self.config["limits"]["grace_seconds"] + self.config["limits"]["control_seconds"])

    def validate_server_evidence(self, snapshot, server, cell, *, closed=False):
        """Bind copied wrapper, supervisor and worker lifetimes to the original launch."""
        launch = read(self.output / server["job_id"] / "launch.json")
        self.auditor.validate_job_identity(snapshot, server, launch)
        completion_path = snapshot / "completion.json"
        require(closed or not completion_path.exists(), "server completed before its episode evidence was collected")
        outer = read(completion_path if closed else snapshot / "live_status.json")
        require(outer["job_id"] == server["job_id"] and outer["command"] == server["command"]
                and outer.get("automatic_retry") is False
                and all(outer["wrapper"][key] == launch["process"][key]
                        for key in ("pid", "start_ticks", "session_id", "group_id")), "copied Thor wrapper lifetime differs")
        require(closed or outer["status"] == "running", "Thor snapshot is not an active owned server")
        supervisor = read(snapshot / "supervisor" / ("completion.json" if closed else "live_status.json"))
        require(supervisor.get("mode") == "serve" and supervisor.get("automatic_retry") is False
                and supervisor["pid"] == outer["child"]["pid"] and len(supervisor["jobs"]) == 1,
                "Thor supervisor does not belong to the launched wrapper")
        worker = supervisor["jobs"][0]
        require(worker["cell_id"] == cell and type(worker["pid"]) is int and worker["pid"] > 0
                and worker["output"] == server["output"] + "/supervisor/" + cell
                and not worker.get("forced_kill") and not supervisor.get("cleanup_forced_kill"),
                "native server worker identity or cleanup differs")
        require(read(snapshot / "supervisor" / cell / "ready.json")["metadata"] == self.identities[cell]
                and read(snapshot / "supervisor" / cell / "requests/identity.json") == self.identities[cell],
                "copied native server deployment identity differs")
        lifetime = {"supervisor_process": {key: outer["child"][key]
                                           for key in ("pid", "start_ticks", "session_id", "group_id")},
                    "supervisor_started": supervisor["started"], "worker_pid": worker["pid"], "worker_started": worker["started"]}
        previous = self.server_lifetimes.setdefault(server["job_id"], lifetime)
        require(previous == lifetime, "Thor supervisor or worker restarted during the batch")
        require(not list((snapshot / "supervisor" / cell).glob("connection-*-failure.json")), "server recorded a failed connection")
        if closed:
            worker_path = snapshot / "supervisor" / cell / "completion.json"
            worker_completion = read(worker_path)
            require(supervisor["status"] == "passed" and worker["status"] == "passed"
                    and type(worker["exit_code"]) is int and worker["exit_code"] == 0
                    and worker.get("supervisor_stop_reason") in {None, "SIGTERM"}
                    and worker["worker_completion_sha256"] == sha(worker_path)
                    and worker["benchmark_identity_sha256"] == self.identities[cell]["benchmark_identity_sha256"]
                    and worker_completion["status"] == "passed" and worker_completion["metadata"] == self.identities[cell],
                    "native supervisor/worker closure is not a clean identity-bound completion")
        else:
            require(supervisor["status"] == "running" and worker["status"] == "running", "server worker ended during its batch")

    def collect_record(self, directory, completion, cell, episode):
        renderer = directory / "renderer"
        require(read(renderer / "completion.json") == completion and completion["status"] == "passed"
                and type(completion.get("exit_code")) is int and completion["exit_code"] == 0
                and not any(completion.get(key) for key in ("forced_kill", "cleanup_forced_kill", "stop_reason", "owned_processes_still_live")),
                "copied native completion differs from observed clean exit0")
        native = read(renderer / "episode" / "result.json")
        require(not self.auditor.PROOF_FIELDS.intersection(native), "native result predeclares collector proof")
        record = dict(native, renderer_process_exit_code=0,
                      renderer_process_completion_sha256=sha(renderer / "completion.json"), collector_source_sha256=sha(__file__))
        indexed = {row["pair_id"]: row for row in self.protocol["episodes"]}
        family, route = cell.split("-", 1)
        arm = "baseline" if route == "eager_native" else "candidate"
        require(self.api._validate_record(record, self.protocol, indexed, self.api.digest(self.protocol))
                == (family, arm, episode["pair_id"]), "native result belongs to another formal pair")
        require(record["native_app_close"] in {"pending_external_exit_check", "returned"}
                and not record.get("cleanup_errors") and not record.get("renderer_compatibility_error"), "native cleanup failed")
        native_outcome = record.get("native_outcome")
        require(isinstance(native_outcome, dict) and type(native_outcome.get("env_id")) is int
                and native_outcome["env_id"] == 0 and type(native_outcome.get("step")) is int
                and native_outcome["step"] == record["executed_steps"]
                and 3 <= record["executed_steps"] <= episode["max_episode_steps"]
                and type(native_outcome.get("success")) is bool and native_outcome["success"] == record["success"],
                "formal result differs from the native single-env terminal outcome")
        require(record["evaluation_mode"] == "paused_simulation" and record.get("task_quality_validated") is False,
                "native result cannot predeclare a quality certificate or change timing semantics")
        out = renderer / "episode"
        anchor = read(directory / "native_anchor.json")
        initial = read(out / "initial_state.json")
        require({field: record[field] for field in self.helper.anchor_fields(self.protocol)} == anchor
                and initial["binding"] == anchor and self.auditor.digest(initial["state"]) == anchor["initial_state_sha256"],
                "formal physical/camera anchor differs")
        for filename, field in (("simulator_fingerprint.json", "simulator_fingerprint_sha256"),
                                ("scene_config_after_reset.json", "scene_config_sha256"),
                                ("verified_asset_inventory.json", "asset_inventory_sha256")):
            require(self.auditor.digest(read(out / filename)) == record[field], "formal native source/scene/assets binding differs")
        fingerprint = read(out / "simulator_fingerprint.json")
        renderer_config = self.config["renderer"]
        bound = renderer_config["bound_files"]
        flash = renderer_config["flash_root"].rstrip("/")
        require(fingerprint["source"] == {"revision": self.protocol["inventory"]["robolab_revision"],
                                           "driver_sha256": bound[flash + "/benchmarks/vla/robolab_driver.py"],
                                           "asset_driver_sha256": bound[flash + "/benchmarks/vla/robolab_assets.py"]},
                "actual native driver/source differs from the frozen formal execution")
        bootstrap = read(renderer / "bootstrap.json")
        require(bootstrap["bootstrap_sha256"] == bound[renderer_config["bootstrap"]]
                and bootstrap["entry_script"] is None and bootstrap["entry_script_sha256"] is None
                and bootstrap["episode_driver"] is True, "actual bootstrap source or selected native entry point differs")
        require(self.auditor.digest(read(out / "simulator_source_inventory.json")) == fingerprint["installed_simulator_source_sha256"],
                "actual renderer source inventory differs from the simulator fingerprint")
        state = initial["state"]
        require(self.auditor.array(state["episode_length_buf"]).tolist() == [0]
                and self.auditor.array(state["frozen_envs"]).tolist() == [False] and state["has_stepped"] is False,
                "native initial two-reset state is not fresh")
        diagnostics = read(out / "native_subtask_status.json")
        require(isinstance(diagnostics, list) and len(diagnostics) == record["executed_steps"]
                and all(isinstance(row, list) and len(row) == 1 and isinstance(row[0], dict) for row in diagnostics),
                "native per-step single-env diagnostic coverage differs")
        self.helper.validate_physical_observation_evidence(out, record)
        golden = self.output / "anchors" / (episode["pair_id"] + ".json")
        if cell == "edge-eager_native":
            durable(golden, anchor)
        else:
            require(read(golden) == anchor, "full pair_id anchor differs between formal arms")
        return record

    def collect_episode(self, batch_index, cell, episode, ordinal, request_offset, server):
        spec = specifications(self.config, self.helper, batch_index, cell, episode)
        directory = self.output / spec["job_id"]
        durable_directories(directory, exclusive=True)
        self.ensure_capacity("renderer", directory)
        self.renderer = spec
        durable(directory / "launch.json", self.transport.spawn("renderer", spec))
        completion = self.wait("renderer", spec, self.config["limits"]["episode_seconds"] + self.config["limits"]["grace_seconds"])
        # Even failed attempts are copied before failing; they cannot be reclaimed.
        self.transport.copy("renderer", spec["output"], directory / "renderer")
        self.transport.copy("thor", server["output"], directory / "thor_snapshot")
        self.renderer = None
        self.validate_owned_kit_log(directory / "renderer", spec)
        anchor_path = f"{self.config['renderer']['output_root']}/anchors/{episode['pair_id']}.json"
        anchor = self.transport.read_json("renderer", anchor_path)
        durable(directory / "native_anchor.json", anchor)
        self.auditor.validate_job_identity(directory / "renderer", spec, read(directory / "launch.json"))
        self.validate_server_evidence(directory / "thor_snapshot", server, cell)
        record = self.collect_record(directory, completion, cell, episode)
        instruction = next(row["instruction_default"] for row in self.protocol["inventory"]["tasks"] if row["task_id"] == episode["task_id"])
        transport_audit = self.auditor.audit_transport(directory, record, episode, self.identities[cell],
                                                      request_offset, ordinal, instruction)
        durable(directory / "local_episode_audit.json", dict(transport_audit, status="passed", task_quality_certified=False,
                                                             source_sha256=sha(__file__), transport_auditor_sha256=AUDITOR_SHA))
        durable(directory / "collected_record.json", record)
        receipt_dir = f"{self.config['renderer']['output_root']}/archive_receipts/{spec['job_id']}"
        self.transport.prepare("renderer", receipt_dir)
        inventory_ref = self.transport.archive("inventory", spec["job_id"], receipt_dir + "/inventory.json")
        inventory = self.transport.read_json("renderer", inventory_ref["receipt"])
        durable(directory / "remote_inventory.json", inventory)
        require(sha(directory / "remote_inventory.json") == inventory_ref["receipt_sha256"], "remote immutable inventory receipt bytes differ")
        owner = read(self.config["archive"]["ownership_local"])
        self.archive._validate_inventory_binding(
            inventory, owner, {"completion_sha256": sha(directory / "renderer/completion.json"),
                               "spec_sha256": sha(directory / "renderer/spec.json"),
                               "launch_sha256": sha(directory / "renderer/launch.json"),
                               "native_result_sha256": sha(directory / "renderer/episode/result.json")},
            Path(spec["output"]), spec["job_id"], sha(self.config["archive"]["ownership_local"]))
        local_entries = self.archive.scan_tree(directory / "renderer")
        require(local_entries == inventory["entries"] and self.archive.object_sha(local_entries) == inventory["entries_sha256"],
                "copied whole native job differs from remote immutable inventory")
        durable(directory / "local_archive_inventory.json", {"entries": local_entries, "entries_sha256": inventory["entries_sha256"]})
        # Flush every copied byte and directory before attesting local durability.
        for path in (directory / "renderer").rglob("*"):
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            elif path.is_dir() and not path.is_symlink():
                sync_directory(path)
        sync_directory(directory / "renderer")
        durable_entries = self.archive.scan_tree(directory / "renderer")
        require(durable_entries == local_entries
                and self.archive.object_sha(durable_entries) == inventory["entries_sha256"],
                "copied native job changed while establishing local archive durability")
        ack = {"schema_version": 1, "kind": "formal_episode_archive_ack_v1", "status": "passed", "durable": True,
               "run_id": self.config["run_id"], "run_root": self.config["renderer"]["output_root"], "job_id": spec["job_id"],
               "tree_root": spec["output"], "ownership_sha256": sha(self.config["archive"]["ownership_local"]),
               "inventory_sha256": inventory_ref["receipt_sha256"], "entries_sha256": inventory["entries_sha256"],
               "completion_sha256": sha(directory / "renderer/completion.json"), "spec_sha256": sha(directory / "renderer/spec.json"),
               "local_archive_path": str((directory / "renderer").resolve()),
               "local_archive_inventory_sha256": sha(directory / "local_archive_inventory.json"),
               "local_audit_sha256": sha(directory / "local_episode_audit.json"),
               "collected_record_sha256": sha(directory / "collected_record.json"),
               "owner_uid": self.config["renderer"]["owner_uid"], "handle_visibility_scope": "same_effective_uid_only",
               "other_uid_handles_excluded": True}
        durable(directory / "archive_ack.json", ack)
        self.records.append(record)
        durable(self.output / "records.json", self.records, exclusive=False)
        self.state["completed_records"] = len(self.records)
        self.state["new_completed_records"] = len(self.records) - 1
        self.save()
        if self.config["archive"]["reclaim_completed_renderer_episodes"]:
            ack_ref = self.transport.archive("ack", spec["job_id"], receipt_dir + "/ack.json", inventory=inventory_ref, payload=ack)
            require(ack_ref["receipt_sha256"] == sha(directory / "archive_ack.json"), "remote archive acknowledgment bytes differ")
            reclaimed = self.transport.archive("reclaim", spec["job_id"], receipt_dir + "/reclaimed.json", inventory=inventory_ref, ack=ack_ref)
            durable(directory / "reclamation_receipt.json", reclaimed)
        require(self.transport.status("thor", server["output"])["completion"] is None, "native server ended before its batch completed")
        return record

    def validate_owned_kit_log(self, renderer, spec):
        selected = "--kit_args=--/log/file={job_output}/kit.log" in self.config["renderer"]["launcher_args"]
        if not selected:
            return
        path = renderer / "kit.log"
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                "selected original Kit log is missing, empty or not a regular owned file")
        expected = spec["output"] + "/kit.log"
        observed = []
        for line in (renderer / "worker.log").read_text(errors="replace").splitlines():
            if "Logging to file:" in line:
                observed.append(line.split("Logging to file:", 1)[1].strip())
        require(observed == [expected], "native Kit logging receipt differs from the exact owned job path")

    def close_cell(self, server, cell, batch_index, cell_records):
        completion = self.stop("thor", server)
        final = self.output / f"batches/{batch_index:04d}/servers/{cell}/final_server"
        self.transport.copy("thor", server["output"], final)
        self.server = None
        self.validate_server_evidence(final, server, cell, closed=True)
        require(read(final / "completion.json") == completion and completion["status"] in {"passed", "stopped"}
                and type(completion["exit_code"]) is int and completion["exit_code"] == 0
                and not any(completion.get(k) for k in ("forced_kill", "cleanup_forced_kill", "owned_processes_still_live"))
                and completion.get("stop_reason") in {None, "SIGTERM"}
                and read(final / "supervisor/completion.json")["status"] == "passed", "owned batch server failed clean shutdown")
        expected_batch = segment_episodes(self.protocol, batch_index, cell)
        expected_count = len(expected_batch)
        if len(cell_records) != expected_count:
            return
        ledger = final / "supervisor" / cell / "requests"
        total = sum(row["generated_chunks"] for row in cell_records)
        require(read(ledger / "closed.json") == {"failed": False, "requests": total, "episodes": expected_count,
                                                 "benchmark_identity_sha256": self.identities[cell]["benchmark_identity_sha256"]},
                "batch server did not close its exact declared reset streams")
        self.auditor.validate_ledger_files(ledger, total, expected_count, closed=True)
        require([row["pair_id"] for row in cell_records] == [row["pair_id"] for row in expected_batch],
                "closed server records do not cover the exact ordered formal batch")
        replayed, request_offset = [], 0
        for row in cell_records:
            spec = specifications(self.config, self.helper, batch_index, cell, row)
            directory = self.output / spec["job_id"]
            snapshot = directory / "thor_snapshot"
            self.validate_server_evidence(snapshot, server, cell)
            before = snapshot / "supervisor" / cell / "requests"
            require(all(sha(path) == sha(ledger / path.name) for path in before.iterdir()), "batch server immutable request ledger changed")
            require(read(directory / "collected_record.json") == row, "durable batch record changed before server archive")
            ordinal = len(replayed)
            episode = expected_batch[ordinal]
            instruction = next(item["instruction_default"] for item in self.protocol["inventory"]["tasks"]
                               if item["task_id"] == episode["task_id"])
            transport_audit = self.auditor.audit_transport(directory, row, episode, self.identities[cell],
                                                          request_offset, ordinal, instruction)
            require(read(directory / "local_episode_audit.json") == dict(
                transport_audit, status="passed", task_quality_certified=False,
                source_sha256=sha(__file__), transport_auditor_sha256=AUDITOR_SHA),
                "native transport replay differs from the durable collected episode audit")
            replayed.append({"pair_id": row["pair_id"], "collected_record_sha256": sha(directory / "collected_record.json"),
                             "local_episode_audit_sha256": sha(directory / "local_episode_audit.json"),
                             "generated_chunks": row["generated_chunks"]})
            request_offset += row["generated_chunks"]
        audit = {"schema_version": 1, "kind": "formal_closed_server_local_audit_v1", "status": "passed",
                 "cell": cell, "job_id": server["job_id"], "episode_ids": [row["pair_id"] for row in cell_records],
                 "completed_episode_count": expected_count, "request_count": total, "episodes": replayed,
                 "one_owned_process_lifetime": True, "immutable_request_ledger_unchanged": True,
                 "external_exit_code": 0, "collector_source_sha256": sha(__file__),
                 "transport_auditor_sha256": AUDITOR_SHA, "task_quality_certified": False}
        self.archive_closed_server(server, cell, cell_records, final, audit)

    def archive_closed_server(self, server, cell, cell_records, final, audit):
        """Only fully completed, explicitly declared segments reach this boundary."""
        expected_count = len(segment_episodes(self.protocol, int(server["job_id"].split("/")[1]), cell))
        require(len(cell_records) == expected_count and audit["status"] == "passed"
                and audit["completed_episode_count"] == expected_count
                and audit["episode_ids"] == [row["pair_id"] for row in cell_records],
                "server archival requires every declared independently replayed record")
        directory = final.parent
        durable(directory / "collected_records.json", cell_records)
        durable(directory / "local_server_audit.json", audit)
        receipt_dir = f"{self.config['thor']['output_root']}/archive_receipts/{server['job_id']}"
        self.transport.prepare("thor", receipt_dir)
        inventory_ref = self.transport.archive("inventory", server["job_id"], receipt_dir + "/inventory.json", host="thor")
        inventory = self.transport.read_json("thor", inventory_ref["receipt"])
        durable(directory / "remote_inventory.json", inventory)
        require(sha(directory / "remote_inventory.json") == inventory_ref["receipt_sha256"],
                "Thor immutable inventory receipt bytes differ")
        owner_path = self.config["archive"]["server_ownership_local"]
        owner = read(owner_path)
        bindings = {"completion_sha256": sha(final / "completion.json"), "spec_sha256": sha(final / "spec.json"),
                    "launch_sha256": sha(final / "launch.json"),
                    "supervisor_completion_sha256": sha(final / "supervisor/completion.json"),
                    "worker_completion_sha256": sha(final / "supervisor" / cell / "completion.json"),
                    "closed_ledger_sha256": sha(final / "supervisor" / cell / "requests/closed.json")}
        self.archive._validate_inventory_binding(inventory, owner, bindings, Path(server["output"]),
                                                 server["job_id"], sha(owner_path))
        local_entries = self.archive.scan_tree(final)
        require(local_entries == inventory["entries"]
                and self.archive.object_sha(local_entries) == inventory["entries_sha256"],
                "copied whole Thor server differs from remote immutable inventory")
        durable(directory / "local_archive_inventory.json", {"entries": local_entries, "entries_sha256": inventory["entries_sha256"]})
        for path in final.rglob("*"):
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            elif path.is_dir() and not path.is_symlink():
                sync_directory(path)
        sync_directory(final)
        sync_directory(directory)
        durable_entries = self.archive.scan_tree(final)
        require(durable_entries == local_entries
                and self.archive.object_sha(durable_entries) == inventory["entries_sha256"],
                "copied Thor server changed while establishing local archive durability")
        ack = {"schema_version": 1, "kind": "formal_server_archive_ack_v1", "status": "passed", "durable": True,
               "run_id": self.config["run_id"], "run_root": self.config["thor"]["output_root"], "job_id": server["job_id"],
               "tree_root": server["output"], "ownership_sha256": sha(owner_path),
               "inventory_sha256": inventory_ref["receipt_sha256"], "entries_sha256": inventory["entries_sha256"],
               **bindings, "local_archive_path": str(final.resolve()),
               "local_archive_inventory_sha256": sha(directory / "local_archive_inventory.json"),
               "local_audit_sha256": sha(directory / "local_server_audit.json"),
               "collected_records_sha256": sha(directory / "collected_records.json"),
               "episode_ids": [row["pair_id"] for row in cell_records], "completed_episode_count": expected_count,
               "owner_uid": self.config["thor"]["owner_uid"], "handle_visibility_scope": "same_effective_uid_only",
               "other_uid_handles_excluded": True}
        durable(directory / "archive_ack.json", ack)
        if self.config["archive"]["reclaim_completed_thor_servers"]:
            ack_ref = self.transport.archive("ack", server["job_id"], receipt_dir + "/ack.json",
                                             inventory=inventory_ref, payload=ack, host="thor")
            require(ack_ref["receipt_sha256"] == sha(directory / "archive_ack.json"), "Thor archive acknowledgment bytes differ")
            reclaimed_ref = self.transport.archive("reclaim", server["job_id"], receipt_dir + "/reclaimed.json",
                                                   inventory=inventory_ref, ack=ack_ref, host="thor")
            durable(directory / "reclamation_response.json", reclaimed_ref)
            require(reclaimed_ref["status"] == "reclaimed" and reclaimed_ref["reclaimed"] is True
                    and reclaimed_ref["job_id"] == server["job_id"], "Thor staging reclamation did not complete")
            receipt = self.transport.read_json("thor", reclaimed_ref["receipt"])
            durable(directory / "reclamation_receipt.json", receipt)
            require(sha(directory / "reclamation_receipt.json") == reclaimed_ref["receipt_sha256"]
                    and receipt["status"] == "reclaimed" and receipt["job_id"] == server["job_id"]
                    and receipt["entries_sha256"] == inventory["entries_sha256"]
                    and receipt["archive_ack_sha256"] == sha(directory / "archive_ack.json"),
                    "persisted Thor reclamation proof differs")
        self.state["closed_servers"] = self.state.get("closed_servers", 0) + 1
        self.state["archived_servers"] = self.state.get("archived_servers", 0) + 1
        self.state["reclaimed_servers"] = self.state.get("reclaimed_servers", 0) + int(
            self.config["archive"]["reclaim_completed_thor_servers"])
        self.save()

    def finalize_cell(self, server, cell, batch_index, records):
        """Keep the policy endpoint alive until an interrupted renderer is stopped."""
        try:
            if self.renderer is not None:
                renderer = self.renderer
                self.stop("renderer", renderer)
                self.transport.copy("renderer", renderer["output"],
                                    self.output / "failure_evidence" / renderer["job_id"])
                self.renderer = None
        finally:
            self.close_cell(server, cell, batch_index, records)

    def import_adopted_evidence(self):
        adopted = validate_adoption(self.config, self.protocol, self.identities, self.helper,
                                    self.auditor, self.archive, self.api)
        original = adopted["original_study_root"]
        require(not self.output.resolve().is_relative_to(original.resolve())
                and not original.resolve().is_relative_to(self.output.resolve()),
                "continued local output must exclude the immutable retained study")
        before = self.archive.scan_tree(original)
        target = self.output / "adopted_legacy" / adopted["manifest"]["original_run_id"]
        durable_directories(target.parent, exclusive=True)
        shutil.copytree(original, target, symlinks=True)
        require(self.archive.scan_tree(target) == before, "retained full local study copy differs")
        for path in target.rglob("*"):
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            elif path.is_dir() and not path.is_symlink():
                sync_directory(path)
        sync_directory(target)
        sync_directory(target.parent)
        require(self.archive.scan_tree(target) == before == self.archive.scan_tree(original),
                "retained study changed during durable import and second full rehash")
        durable(self.output / "adopted_legacy_inventory.json", {
            "entries": before, "entries_sha256": self.archive.object_sha(before),
            "original_study_root": str(original), "copied_study_root": str(target)})
        durable(self.output / "continuation_adoption.json", adopted["manifest"])
        golden = self.output / "anchors" / (RETAINED_PAIR + ".json")
        durable(golden, adopted["anchor"])
        require(sha(golden) == sha(original / "anchors" / (RETAINED_PAIR + ".json")),
                "imported golden anchor bytes differ")
        self.adopted_anchor = adopted["anchor"]
        self.records = [adopted["record"]]
        durable(self.output / "records.json", self.records)
        durable(self.output / "adopted_episode_import.json", {
            "schema_version": 1, "kind": "formal_completed_episode_local_import_v1", "status": "passed",
            "continuation_adoption_sha256": self.config["continuation_adoption"]["sha256"],
            "original_collector_source_sha256": LEGACY_CONTROLLER_SHA,
            "collected_record_sha256": adopted["manifest"]["retained"]["collected_record_sha256"],
            "full_import_inventory_sha256": sha(self.output / "adopted_legacy_inventory.json"),
            "anchor_sha256": sha(golden), "retained_episode_count": 1, "retained_server_streams": 1,
            "retained_native_outcome": adopted["record"]["native_outcome"],
            "old_local_and_remote_staging_preserved": True, "old_archive_admission": False,
            "durable": True, "double_full_byte_rehash": True, "replayed_native_episodes": 0,
            "task_quality_validated": False})
        self.state.update(completed_records=1, adopted_records=1, new_completed_records=0,
                          retained_closed_server_lifetimes=1)
        self.save()

    def prepare_hosts(self):
        for host, kind, owner_key in (("thor", "server", "server_ownership_local"),
                                      ("renderer", "episode", "ownership_local")):
            response = self.transport.prepare_private_root(host)
            durable(self.output / (host + "_private_root_response.json"), response)
            root = self.config[host]["output_root"]
            require(response["status"] == "prepared" and response["run_root"] == root
                    and response["receipt"] == root + "/archive_receipts/private_root.json",
                    "new private formal staging root was not prepared")
            receipt = self.transport.read_json(host, response["receipt"])
            receipt_path = self.output / (host + "_private_root.json")
            durable(receipt_path, receipt)
            expected = {"schema_version": 1, "kind": "formal_" + kind + "_private_root_v1", "status": "prepared",
                        "run_id": self.config["run_id"], "run_root": root, "run_root_mode": 0o700,
                        "owner_uid": self.config[host]["owner_uid"], "handle_visibility_scope": "same_effective_uid_only",
                        "other_uid_handles_excluded": True,
                        "ownership_sha256": sha(self.config["archive"][owner_key]),
                        "archive_helper_sha256": self.config["archive"]["helper_sha256"]}
            require(sha(receipt_path) == response["receipt_sha256"]
                    and all(receipt.get(key) == value for key, value in expected.items()),
                    "private root preparation receipt differs")
        response = self.transport.install_adopted_anchor(self.adopted_anchor)
        durable(self.output / "adopted_anchor_response.json", response)
        root = self.config["renderer"]["output_root"]
        expected_anchor = root + "/anchors/" + RETAINED_PAIR + ".json"
        anchor_sha = sha(self.output / "anchors" / (RETAINED_PAIR + ".json"))
        require(response["status"] == "adopted_anchor" and response["anchor"] == expected_anchor
                and response["anchor_sha256"] == anchor_sha
                and response["receipt"] == root + "/archive_receipts/adopted_anchor.json",
                "exact retained remote anchor was not imported")
        receipt = self.transport.read_json("renderer", response["receipt"])
        durable(self.output / "adopted_anchor_receipt.json", receipt)
        require(sha(self.output / "adopted_anchor_receipt.json") == response["receipt_sha256"]
                and receipt["kind"] == "formal_preserved_anchor_adoption_v1"
                and receipt["status"] == "adopted_anchor" and receipt["run_id"] == self.config["run_id"]
                and receipt["run_root"] == root and receipt["pair_id"] == RETAINED_PAIR
                and receipt["ownership_sha256"] == sha(self.config["archive"]["ownership_local"])
                and receipt["continuation_adoption_sha256"] == self.config["continuation_adoption"]["sha256"]
                and receipt["archive_helper_sha256"] == self.config["archive"]["helper_sha256"]
                and receipt["anchor"] == expected_anchor and receipt["anchor_sha256"] == anchor_sha
                and receipt["owner_uid"] == self.config["renderer"]["owner_uid"]
                and receipt["handle_visibility_scope"] == "same_effective_uid_only"
                and receipt["other_uid_handles_excluded"] is True,
                "retained anchor adoption receipt differs")
        require(self.transport.read_json("renderer", expected_anchor) == self.adopted_anchor,
                "actual remote retained native anchor differs before continuation")

    def ensure_capacity(self, host, directory):
        receipt = self.transport.free_space(host)
        durable(directory / "capacity.json", receipt)
        minimum = self.config["capacity"]["minimum_free_bytes"][host]
        require(receipt["path"] == self.config[host]["output_root"]
                and receipt["minimum_free_bytes"] == minimum
                and receipt["helper_source_sha256"] == self.config["archive"]["helper_sha256"]
                and type(receipt["available_bytes"]) is int
                and receipt["status"] == "passed" and receipt["available_bytes"] >= minimum,
                "remote free-space headroom is insufficient or its read-only receipt differs; no job launched")

    def run(self):
        self.save()
        try:
            self.import_adopted_evidence()
            self.prepare_hosts()
            for batch_index, batch in enumerate(batches(self.protocol)):
                for cell in CELLS:
                    server = specifications(self.config, self.helper, batch_index, cell)
                    root = self.output / server["job_id"]
                    durable_directories(root, exclusive=True)
                    self.ensure_capacity("thor", root)
                    self.server = server
                    durable(root / "launch.json", self.transport.spawn("thor", server))
                    records = []
                    try:
                        durable(root / "ready.json", self.ready(server, cell))
                        for ordinal, episode in enumerate(segment_episodes(self.protocol, batch_index, cell)):
                            self.state.update(status="running_episode", batch=batch_index, cell=cell, pair_id=episode["pair_id"])
                            self.save()
                            records.append(self.collect_episode(batch_index, cell, episode, ordinal,
                                                                sum(row["generated_chunks"] for row in records), server))
                    finally:
                        self.finalize_cell(server, cell, batch_index, records)
            require(len(self.records) == 4800 and self.state.get("adopted_records") == 1
                    and self.state.get("new_completed_records") == 4799,
                    "continued capture did not retain the exact original plus4799 prescribed records")
            require(sum((row["family"], row["arm"], row["pair_id"]) == ("edge", "baseline", RETAINED_PAIR)
                        for row in self.records) == 1, "adopted outcome was duplicated or omitted")
            require(self.state.get("closed_servers") == 600 and self.state.get("archived_servers") == 600
                    and self.state.get("reclaimed_servers") == 600 * int(self.config["archive"]["reclaim_completed_thor_servers"]),
                    "formal capture lacks complete durable server archives or selected reclamations")
            report = self.api.build_report(self.protocol, self.records)
            durable(self.output / "formal_report.json", report)
            self.state.update(status="completed_formal_capture", physical_server_lifetimes_including_retained=601,
                              task_quality_validated=report["task_quality_validated"])
        except BaseException:
            self.state.update(status="failed", error=traceback.format_exc())
            raise
        finally:
            for host, spec in (("renderer", self.renderer), ("thor", self.server)):
                if spec is not None:
                    try:
                        self.stop(host, spec)
                        self.transport.copy(host, spec["output"], self.output / "failure_evidence" / host)
                    except BaseException:
                        self.state.setdefault("cleanup_errors", []).append(traceback.format_exc())
                        self.state["status"] = "failed"
            self.state["ended"] = time.time()
            self.save()
            durable(self.output / "completion.json", self.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sha(args.config) == args.config_sha256, "explicit frozen config SHA required")
    config = read(args.config)
    protocol, identities, helper, auditor, archive, api = load_inputs(config, args.config_sha256, for_execution=args.execute)
    durable_directories(args.output, exclusive=True)
    durable(args.output / "config.json", config)
    durable(args.output / "protocol.json", protocol)
    durable(args.output / "identities.json", identities)
    durable(args.output / "source.json", {"controller_sha256": sha(__file__), "wrapper_sha256": WRAPPER_SHA,
                                        "transport_auditor_sha256": AUDITOR_SHA, "config_sha256": args.config_sha256})
    if args.prepare:
        durable(args.output / "ownership.json", ownership(config, protocol, helper, args.config_sha256))
        durable(args.output / "server_ownership.json", server_ownership(config, protocol, helper, args.config_sha256))
        durable(args.output / "prepared.json", {"status": "prepared_only", "batches": 150, "servers": 600,
                                               "episodes": 4799, "adopted_episodes": 1, "aggregate_records": 4800,
                                               "first_segment_episodes": 7, "launched": False, "task_quality_validated": False})
        return

    def interrupted(number, _frame):
        raise InterruptedError(f"formal controller received {signal.Signals(number).name}")
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    FormalController(config, protocol, identities, helper, auditor, archive, api,
                     FormalTransport(config, args.output, helper), args.output).run()


if __name__ == "__main__":
    main()

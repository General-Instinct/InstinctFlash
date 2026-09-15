"""Inventory and explicitly reclaim completed, owned formal staging jobs.

Python 3.10+ standard library only. This does not launch, signal, or retry jobs.
The controller supplies a SHA-bound ownership receipt and exact episode allowlist.
Successful inventory and reclamation receipts are exclusive, fsynced JSON files
outside the job tree. Reclamation additionally creates an exclusive intent receipt;
an interrupted deletion is never automatically retried.

Ownership kind ``formal_episode_run_ownership_v1`` requires schema_version=1,
stage="formal", run_id, run_root, protocol_sha256, config_sha256, wrapper_source
{path, sha256}, archive_helper_sha256, reclamation_authorized, episode_jobs,
and episode_spec_sha256 (an exact job_id to prospective spec-file SHA256 map).
An archive acknowledgment of kind ``formal_episode_archive_ack_v1`` requires
schema_version=1, status="passed", durable=true, the matching run_id/run_root,
job_id/tree_root, ownership_sha256, inventory_sha256, entries_sha256,
completion_sha256, spec_sha256, plus local_archive_path and SHA256s for
local_archive_inventory, local_audit, and collected_record. The remote helper
checks this controller attestation; it cannot independently fsync a remote
machine's local archive. The controller must issue it only after durable copy,
complete entry verification, native episode audit, and collected-record fsync.

V2 also supports kind ``formal_server_run_ownership_v1`` with server_jobs,
server_spec_sha256, supervisor_source, worker_source, and server_batches instead
of episode_jobs/episode_spec_sha256. Each server_batches[job_id] contains
worker_cell, expected_episode_count, ordered episode_ids, identity_source {path, sha256}, and
benchmark_identity_sha256. Source descriptors use {path, sha256}. Server receipt
kinds are formal_server_inventory_v1, formal_server_archive_ack_v1, and
formal_server_reclamation_v1. Server inventory binds supervisor_completion_sha256,
worker_completion_sha256, and closed_ledger_sha256 instead of native_result_sha256.
Server acknowledgments include those three bindings, ordered episode_ids,
completed_episode_count, and collected_records_sha256 instead of
collected_record_sha256. A server acknowledgment requires the controller's
durable final transport audit; this stdlib helper checks persisted
ledger identities, file hashes, sources, and clean closure, not NumPy geometry
or the renderer's observation/action parity. No v1 file is imported or changed.
Both episode and server owners bind owner_uid=current effective UID and
run_root_mode=448 (0700). Their roots are created exclusively by prepare-run;
every job path/entry must belong to that UID. The run root's existing parent
must also belong to that UID and have no group/other write bits.
All inventories, acknowledgments and reclamations declare handle_visibility_scope as
same_effective_uid_only and other_uid_handles_excluded=true; privileged and
other-UID handles are outside this expressly limited guard. Unreadable same-UID
processes remain fatal. Existing public V1 renderer roots are not admitted or
changed by this version. prepare-run accepts
--run-root, --ownership, --ownership-sha256 and --receipt, without --job-id.
Ordinary servers retain eight episodes. A SHA-bound continuation_adoption_source
can authorize only the original first Edge eager server's seven remaining pairs,
while preserving all 600 server jobs and the already completed failure record.
adopt-anchor accepts the prepare-run options and bounded exact original JSON
bytes on stdin. It exclusively creates anchors/formal/AnimalsInBinTask/0000.json
and archive_receipts/adopted_anchor.json immediately after private preparation.
The read-only free-space operation accepts --path and --minimum-free-bytes;
passed and insufficient_space both exit0, so the controller can persist its
sample before stopping. Invalid inputs exit1. It never creates a receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _hex(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), name + " must be a SHA256")
    return value


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _token(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns, value.st_nlink)


def file_sha(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), "expected a regular file")
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 8 << 20)
            if not block:
                break
            digest.update(block)
        require(_token(before) == _token(os.fstat(fd)), "file changed while hashing")
        require(_token(before) == _token(os.stat(path, follow_symlinks=False)), "file replaced while hashing")
        return digest.hexdigest()
    finally:
        os.close(fd)


def absolute_path(value, *, directory=False, exists=True):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "path must be absolute without parent traversal")
    require(path != Path("/"), "filesystem root is never an allowed archive path")
    for item in [path, *path.parents]:
        require(not item.is_symlink(), "symlink path components are forbidden")
    require(path.resolve(strict=exists) == path, "path must be canonical")
    if exists:
        require(path.is_dir() if directory else path.is_file(), "archive path type differs")
    return path


def load_bound(path, expected_sha):
    path = absolute_path(path)
    _hex(expected_sha, "receipt digest")
    require(file_sha(path) == expected_sha, "receipt SHA256 differs")
    value = json.loads(path.read_text())
    require(file_sha(path) == expected_sha, "receipt changed while reading")
    require(isinstance(value, dict), "receipt must be an object")
    return value


def _relative_job(value):
    require(isinstance(value, str) and value and not value.startswith("/")
            and all(part not in {"", ".", ".."} for part in value.split("/")), "invalid relative job ID")
    return value


def _outside(path, tree, *, absent=False):
    path = absolute_path(path, exists=not absent)
    require(not path.is_relative_to(tree), "archive receipts must live outside the job tree")
    if absent:
        require(not path.exists(), "receipt already exists; no retry or overwrite")
        absolute_path(path.parent, directory=True)
    return path


def _receipt_path(path, tree, run_root):
    path = _outside(path, tree, absent=True)
    require(path.is_relative_to(Path(run_root) / "archive_receipts"), "receipt must use the owned archive_receipts tree")
    return path


def mounted_paths():
    result = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        require(len(fields) >= 6, "unreadable mount boundary evidence")
        result.append(Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4])))
    return result


def _no_mounts(tree):
    require(not any(path.is_relative_to(tree) for path in mounted_paths()), "job tree contains a mount boundary")


def write_once_raw(path, payload):
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(path, value):
    write_once_raw(path, _bytes(value))


def process_identity(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {"pid": int(pid), "start_ticks": fields[19], "state": fields[0],
            "parent_pid": int(fields[1]), "group_id": int(fields[2]), "session_id": int(fields[3])}


def current_processes():
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            identity = process_identity(int(path.name))
            if identity is not None:
                result.append(identity)
    return result


def _same_uid_proc_paths(uid, proc_root=Path("/proc")):
    """/proc/status identifies effective UID, including non-dumpable processes."""
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = next(line.split() for line in (proc / "status").read_text().splitlines()
                          if line.startswith("Uid:"))
            effective_uid = int(fields[2])
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # Unknown UID is not proof that a process is outside the scoped UID.
            raise ValueError("cannot establish process effective UID for scoped handle visibility") from None
        if effective_uid == uid:
            yield proc


def same_uid_processes():
    result = []
    for proc in _same_uid_proc_paths(os.geteuid()):
        identity = process_identity(int(proc.name))
        if identity is not None:
            result.append(identity)
    return result


def _handles_for_processes(tree, processes):
    """Fail closed on unreadable live processes; never overlook active mappings."""
    tree = Path(tree)
    found = []

    def check(pid, kind, target):
        target = target.removesuffix(" (deleted)")
        if target.startswith("/") and Path(target).is_relative_to(tree):
            found.append({"pid": pid, "kind": kind, "path": target})

    for proc in processes:
        if not proc.name.isdigit():
            continue
        identity = process_identity(int(proc.name))
        if identity is None or identity["state"] == "Z":
            continue
        try:
            paths = [proc / name for name in ("cwd", "root", "exe")]
            paths.extend((proc / "fd").iterdir())
            for path in paths:
                try:
                    check(identity["pid"], str(path.relative_to(proc)), os.readlink(path))
                except (FileNotFoundError, ProcessLookupError):
                    continue
            for line in (proc / "maps").read_text().splitlines():
                parts = line.split(maxsplit=5)
                if len(parts) == 6:
                    target = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), parts[5])
                    check(identity["pid"], "mapping", target)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            require(process_identity(identity["pid"]) is None, "cannot prove absence of live job handles")
    return found


def live_handles(tree):
    return _handles_for_processes(tree, Path("/proc").iterdir())


def same_uid_live_handles(tree):
    return _handles_for_processes(tree, _same_uid_proc_paths(os.geteuid()))


def _process_key(value):
    require(isinstance(value, dict) and type(value.get("pid")) is int and value["pid"] > 0
            and isinstance(value.get("start_ticks"), str) and value["start_ticks"].isdigit(),
            "invalid recorded process identity")
    return value["pid"], value["start_ticks"]


def _job_type(owner):
    kinds = {"formal_episode_run_ownership_v1": "episode", "formal_server_run_ownership_v1": "server"}
    require(owner.get("kind") in kinds, "unknown formal run ownership kind")
    return kinds[owner["kind"]]


def _private_owner(owner):
    require(type(owner.get("owner_uid")) is int and owner["owner_uid"] == os.geteuid()
            and type(owner.get("run_root_mode")) is int and owner["run_root_mode"] == 0o700,
            "ownership must bind the current effective UID and private run root mode0700")


FIRST_SERVER = "batches/0000/servers/edge-eager_native"
ADOPTED_PAIR = "formal/AnimalsInBinTask/0000"
CONTINUATION_PAIRS = [f"formal/AnimalsInBinTask/{index:04d}" for index in range(1, 8)]
FORMAL_CELLS = {"edge-eager_native", "edge-runtime_selected", "nano-eager_native", "nano-runtime_selected"}


def _continuation_header(owner):
    """Only the specifically preserved first record can shorten one native stream."""
    source = owner.get("continuation_adoption_source")
    if source is None:
        return None
    require(isinstance(source, dict), "continuation adoption source must be explicit")
    path = _outside(source.get("path", ""), Path(owner["run_root"]))
    header = load_bound(path, _hex(source.get("sha256"), "continuation adoption source"))
    require(type(header.get("schema_version")) is int and header["schema_version"] == 1
            and header.get("kind") == "formal_completed_first_episode_adoption_v1"
            and header.get("status") == "passed_preserved_native_only"
            and header.get("original_run_id") == "paired_formal_v1"
            and header.get("protocol_sha256") == owner["protocol_sha256"]
            and header.get("protocol_file_sha256") == "7e445daf605c1eba60ae46236448878b6967600701f63dbb431d91b2e91cc336",
            "continuation adoption header or unchanged formal protocol differs")
    retained = header.get("retained", {})
    require(isinstance(retained, dict) and retained.get("cell") == "edge-eager_native"
            and retained.get("pair_id") == ADOPTED_PAIR and retained.get("success") is False
            and retained.get("collector_source_sha256") == "29c99649018ccab2201cdf77c2a52bff6d63adf0873ba971483ead69c9edeb6e"
            and all(type(retained.get(key)) is int and retained[key] == value
                    for key, value in {"executed_steps": 1350, "generated_chunks": 43,
                                       "closed_server_episode_count": 1}.items()),
            "adoption must retain the exact original completed first episode")
    _hex(retained.get("collected_record_sha256"), "adopted original record")
    continuation = header.get("continuation", {})
    require(isinstance(continuation, dict) and continuation.get("first_server_job_id") == FIRST_SERVER
            and continuation.get("first_server_episode_ids") == CONTINUATION_PAIRS
            and all(type(continuation.get(key)) is int and continuation[key] == value
                    for key, value in {"new_episode_runs": 4799, "new_server_runs": 600,
                                       "aggregate_expected_records": 4800}.items()),
            "continuation must retain all original formal coverage")
    return header


def _batch_episode_count(owner, job, batch, adoption):
    count = batch.get("expected_episode_count")
    require(type(count) is int and count in {7, 8}, "explicit exact native episode count is required")
    if count == 7:
        require(adoption is not None and job == FIRST_SERVER
                and batch.get("worker_cell") == "edge-eager_native"
                and batch.get("adopted_pair_id") == ADOPTED_PAIR
                and batch.get("episode_ids") == CONTINUATION_PAIRS,
                "only the bound first seven-episode continuation is authorized")
    else:
        require("adopted_pair_id" not in batch and not (adoption is not None and job == FIRST_SERVER),
                "ordinary eight-episode servers cannot adopt or replay the preserved first pair")
    return count


def _server_batches(owner, jobs):
    _private_owner(owner)
    adoption = _continuation_header(owner)
    if adoption is not None:
        expected_jobs = {f"batches/{index:04d}/servers/{cell}" for index in range(150) for cell in FORMAL_CELLS}
        require(set(jobs) == expected_jobs, "continuation ownership must retain all 600 exact server jobs")
    batches = owner.get("server_batches")
    require(isinstance(batches, dict) and set(batches) == set(jobs), "prospective server batch allowlist differs")
    for job, batch in batches.items():
        require(isinstance(batch, dict) and batch.get("worker_cell") in FORMAL_CELLS
                and re.fullmatch(r"batches/[0-9]{4}/servers/" + re.escape(batch["worker_cell"]), job),
                "server job is not an exact formal batch/cell path")
        episodes = batch.get("episode_ids")
        count = _batch_episode_count(owner, job, batch, adoption)
        require(isinstance(episodes, list) and len(episodes) == count and len(set(episodes)) == count
                and all(isinstance(item, str) and re.fullmatch(r"formal/[^/]+/[0-9]{4}", item)
                        and item.split("/")[1] not in {".", ".."} for item in episodes),
                "server ownership requires the exact ordered formal episode count")
        _hex(batch.get("benchmark_identity_sha256"), "prospective server identity")
        for source in (batch.get("identity_source"), batch.get("measured_receipt_source"),
                       owner.get("supervisor_source"), owner.get("worker_source"), owner.get("installation_source")):
            require(isinstance(source, dict), "prospective server source is missing")
            _hex(source.get("sha256"), "prospective server source")
            path = Path(source.get("path", ""))
            require(path.is_absolute() and path != Path("/") and ".." not in path.parts,
                    "prospective server source path is invalid")
    return batches


def _private_owned_entry(path, uid, *, protected=False):
    value = path.stat(follow_symlinks=False)
    require(value.st_uid == uid and (not protected or not stat.S_IMODE(value.st_mode) & 0o022),
            "evidence path has another owner or group/other write access")


def _private_server_tree(root, tree, owner):
    """Private path ownership for either a renderer episode or a Thor server."""
    _private_owner(owner)
    _private_owned_entry(root.parent, owner["owner_uid"], protected=True)
    require(root.stat().st_uid == owner["owner_uid"] and stat.S_IMODE(root.stat().st_mode) == 0o700,
            "run root must be owned by the bound UID with mode0700")
    for path in [tree, *tree.parents]:
        if path.is_relative_to(root):
            _private_owned_entry(path, owner["owner_uid"])
    for path in tree.rglob("*"):
        _private_owned_entry(path, owner["owner_uid"])


def _server_visibility(owner):
    return {"handle_visibility_scope": "same_effective_uid_only", "owner_uid": owner["owner_uid"],
            "other_uid_handles_excluded": True}


def _json_file(path):
    path = absolute_path(path)
    before = file_sha(path)
    result = json.loads(path.read_text())
    require(file_sha(path) == before, "native JSON changed while reading")
    require(isinstance(result, dict), "native JSON evidence must be an object")
    return result


def _option(command, name):
    require(isinstance(command, list) and all(isinstance(item, str) for item in command)
            and command.count(name) == 1 and command.index(name) + 1 < len(command)
            and not any(item.startswith(name + "=") for item in command), "ambiguous native command option: " + name)
    return command[command.index(name) + 1]


def _measured_installation_guard(tree, owner, batch, identity, spec):
    """Reconstruct exactly native supervisor verify(): measured sources plus its selected binary.

    Installation guards and imported policy source inventories have distinct scopes;
    neither may silently replace the other.
    """
    refs = (batch["measured_receipt_source"], owner["installation_source"])
    for source in refs:
        path = _outside(source["path"], tree)
        require(spec["bound_files"].get(str(path)) == source["sha256"] and file_sha(path) == source["sha256"],
                "measured installation guard source is not prospectively bound or changed")
    measured = load_bound(batch["measured_receipt_source"]["path"], batch["measured_receipt_source"]["sha256"])
    installation = load_bound(owner["installation_source"]["path"], owner["installation_source"]["sha256"])
    policy = identity["benchmark_identity"]
    require(measured.get("cell_id") == batch["worker_cell"]
            and isinstance(measured.get("sources"), dict) and measured["sources"]
            and isinstance(policy.get("optimizer_environment"), dict)
            and measured.get("optimizer_environment") == policy["optimizer_environment"],
            "measured installation source scope or optimizer selection differs")
    checked = dict(measured["sources"])
    library = policy["optimizer_environment"].get("IFL_BF16_KERNEL_LIBRARY")
    libraries = policy.get("native_libraries", {})
    if library:
        digest = _hex(installation.get("native_sha256", {}).get(Path(library).name), "selected measured native library")
        require(libraries.get("IFL_BF16_KERNEL_LIBRARY") == {"path": library, "sha256": digest},
                "selected native library differs between installation and policy identity")
        checked[library] = digest
    else:
        require("IFL_BF16_KERNEL_LIBRARY" not in libraries, "unselected native library in policy identity")
    imported = policy.get("source_inventory")
    require(isinstance(imported, dict) and imported, "prospective imported policy source inventory missing")
    for path, digest in checked.items():
        require(file_sha(_outside(path, tree)) == _hex(digest, "measured guarded source"),
                "actual measured guarded source changed")
        require(path not in imported or imported[path] == digest,
                "common installation/imported policy source hash differs")
    return {"measured_receipt_sha256": batch["measured_receipt_source"]["sha256"],
            "verified_files": checked, "expected_identity_file_sha256": batch["identity_source"]["sha256"]}


def _validate_server(tree, owner, job_id, spec, completion):
    batch = owner["server_batches"][job_id]
    adoption = _continuation_header(owner)
    episode_count = _batch_episode_count(owner, job_id, batch, adoption)
    if adoption is not None:
        source = owner["continuation_adoption_source"]
        require(spec.get("bound_files", {}).get(source["path"]) == source["sha256"],
                "native server spec does not bind the continuation adoption manifest")
    cell = batch["worker_cell"]
    bound = spec.get("bound_files", {})
    for source in (owner["supervisor_source"], owner["worker_source"], batch["identity_source"]):
        path = _outside(source["path"], tree)
        require(bound.get(str(path)) == source["sha256"] and file_sha(path) == source["sha256"],
                "prospective native server source differs")
    identity = load_bound(batch["identity_source"]["path"], batch["identity_source"]["sha256"])
    identity_sha = batch["benchmark_identity_sha256"]
    require(identity.get("benchmark_identity_sha256") == identity_sha
            and object_sha(identity.get("benchmark_identity")) == identity_sha
            and identity["benchmark_identity"].get("cell_id") == cell,
            "prospective native server deployment identity differs")
    command = spec["command"]
    require(len(command) >= 2 and command[1] == owner["supervisor_source"]["path"]
            and _option(command, "--mode") == "serve" and _option(command, "--cells") == cell
            and _option(command, "--output") == str(tree / "supervisor")
            and _option(command, "--expected-identity") == batch["identity_source"]["path"],
            "prospective server specification does not select the bound native supervisor")
    supervisor_path = tree / "supervisor/completion.json"
    supervisor = _json_file(supervisor_path)
    require(supervisor.get("schema_version") == 1 and supervisor.get("kind") == "finite-thor-worker-supervisor"
            and supervisor.get("mode") == "serve" and supervisor.get("status") == "passed"
            and supervisor.get("automatic_retry") is False
            and type(supervisor.get("pid")) is int and supervisor["pid"] == completion["child"]["pid"]
            and not any(supervisor.get(key) for key in ("cleanup_forced_kill", "forced_kill", "error", "cleanup_errors")),
            "native supervisor lacks clean completion and exact owned PID")
    require(supervisor.get("scripts_sha256") == {
        Path(owner["supervisor_source"]["path"]).name: owner["supervisor_source"]["sha256"],
        Path(owner["worker_source"]["path"]).name: owner["worker_source"]["sha256"]},
        "native supervisor source receipt differs")
    require(isinstance(supervisor.get("jobs"), list) and len(supervisor["jobs"]) == 1,
            "server must contain exactly one native worker")
    worker = supervisor["jobs"][0]
    worker_root = tree / "supervisor" / cell
    worker_path = worker_root / "completion.json"
    native = _json_file(worker_path)
    require(worker.get("cell_id") == cell and worker.get("output") == str(worker_root)
            and worker.get("status") == "passed" and type(worker.get("exit_code")) is int and worker["exit_code"] == 0
            and worker.get("supervisor_stop_reason") in {None, "SIGTERM"}
            and not any(worker.get(key) for key in ("forced_kill", "cleanup_forced_kill", "error", "cleanup_errors"))
            and worker.get("worker_completion_sha256") == file_sha(worker_path)
            and worker.get("benchmark_identity_sha256") == identity_sha, "native worker lacks clean bound external exit0")
    worker_identities = [row for row in completion["owned_processes"] if row["pid"] == worker.get("pid")]
    require(type(worker.get("pid")) is int and len(worker_identities) == 1
            and worker_identities[0].get("parent_pid") == supervisor["pid"]
            and worker["pid"] not in {completion["wrapper"]["pid"], supervisor["pid"]},
            "native worker PID/start evidence is absent from owned descendants")
    worker_command = worker.get("command", [])
    require(len(worker_command) >= 3 and worker_command[1:3] == ["-I", owner["worker_source"]["path"]]
            and _option(worker_command, "--mode") == "serve" and _option(worker_command, "--cell-id") == cell
            and _option(worker_command, "--output") == str(worker_root)
            and _option(worker_command, "--expected-identity") == batch["identity_source"]["path"],
            "native worker command/source selection differs")
    guard = worker.get("source_guard_before")
    expected_guard = _measured_installation_guard(tree, owner, batch, identity, spec)
    expected_sources = identity["benchmark_identity"].get("source_inventory")
    require(isinstance(expected_sources, dict) and expected_sources
            and guard == worker.get("source_guard_after") == expected_guard
            and supervisor.get("installation_sha256") == owner["installation_source"]["sha256"],
            "native server source guards differ across its lifetime or prospective identity")
    _hex(guard.get("measured_receipt_sha256"), "native measured source receipt")
    require(native.get("schema_version") == 1 and native.get("kind") == "finite-thor-policy-worker"
            and native.get("mode") == "serve" and native.get("cell_id") == cell and native.get("status") == "passed"
            and native.get("automatic_retry") is False and native.get("stop_reason") in {None, "SIGTERM"}
            and native.get("bootstrap_sha256") == owner["worker_source"]["sha256"]
            and native.get("expected_identity_file_sha256") == batch["identity_source"]["sha256"]
            and native.get("metadata") == identity
            and not any(native.get(key) for key in ("error", "cleanup_errors", "forced_kill")),
            "native worker completion/source/identity differs")
    require(not list(worker_root.glob("connection-*-failure.json")), "native server recorded connection failure")
    require(_json_file(worker_root / "identity.json") == identity
            and _json_file(worker_root / "ready.json").get("metadata") == identity,
            "native worker readiness and completed identity differ")
    ledger = worker_root / "requests"
    closed_path = ledger / "closed.json"
    closed = _json_file(closed_path)
    requests = closed.get("requests")
    require(type(requests) is int and requests >= episode_count and type(closed.get("episodes")) is int
            and closed == {"failed": False, "requests": requests, "episodes": episode_count, "benchmark_identity_sha256": identity_sha}
            and closed.get("failed") is False and native.get("closed") == closed,
            "native server did not close exactly its authorized successful episode streams")
    ledger_files = list(ledger.iterdir())
    require(len(ledger_files) == 2 + 2 * episode_count + 4 * requests,
            "native closed ledger request count differs from persisted files")
    names = {"identity.json", "closed.json"}
    names.update(f"episode_{i:06d}{suffix}" for i in range(episode_count) for suffix in (".intent.json", ".json"))
    names.update(f"request_{i:06d}{suffix}" for i in range(requests)
                 for suffix in (".intent.json", ".json", ".npz", ".sources.json"))
    require({path.name for path in ledger_files} == names and all(path.is_file() and not path.is_symlink()
                                                               for path in ledger_files),
            "native closed ledger has missing, extra, or incomplete request/reset files")
    require(_json_file(ledger / "identity.json") == identity, "closed ledger identity differs")
    resets = []
    for index, episode_id in enumerate(batch["episode_ids"]):
        reset = _json_file(ledger / f"episode_{index:06d}.json")
        require(reset == _json_file(ledger / f"episode_{index:06d}.intent.json")
                and reset.get("episode_id") == episode_id and reset.get("reset") is True
                and reset.get("benchmark_identity_sha256") == identity_sha
                and type(reset.get("benchmark_seed")) is int and reset["benchmark_seed"] >= 0
                and type(reset.get("max_policy_chunks")) is int and reset["max_policy_chunks"] > 0
                and isinstance(reset.get("prompt"), str) and reset["prompt"],
                "native closed ledger reset differs from ordered prospective episodes")
        resets.append(reset)
    episode_index, chunk_index = 0, 0
    for index in range(requests):
        stem = f"request_{index:06d}"
        request = _json_file(ledger / (stem + ".json"))
        if request.get("episode_id") != batch["episode_ids"][episode_index]:
            require(chunk_index > 0 and episode_index < episode_count - 1, "request stream skips or repeats an episode")
            episode_index, chunk_index = episode_index + 1, 0
        reset = resets[episode_index]
        intent = _json_file(ledger / (stem + ".intent.json"))
        expected = {"episode_id": reset["episode_id"], "request_id": chunk_index,
                    "request_seed": reset["benchmark_seed"] + chunk_index,
                    "input_sha256": _hex(request.get("input_sha256"), "request input"),
                    "benchmark_identity_sha256": identity_sha}
        require(intent == expected and all(request.get(key) == value for key, value in expected.items())
                and type(request.get("request_id")) is int and type(request.get("request_seed")) is int
                and request.get("status") == "passed" and request.get("task_quality_certified") is False
                and chunk_index < reset["max_policy_chunks"], "native request receipt/intent/order differs")
        require(request.get("action_file") == stem + ".npz"
                and request.get("action_file_sha256") == file_sha(ledger / (stem + ".npz"))
                and request.get("source_inventory_file") == stem + ".sources.json"
                and request.get("source_inventory_sha256") == file_sha(ledger / (stem + ".sources.json")),
                "native persisted action or source inventory file hash differs")
        sources = _json_file(ledger / (stem + ".sources.json"))
        require(all(sources.get(path) == value for path, value in expected_sources.items()),
                "native request source differs from prospective server deployment")
        chunk_index += 1
    require(episode_index == episode_count - 1 and chunk_index > 0,
            "native closed ledger lacks requests for all authorized episodes")
    return {"supervisor_completion_sha256": file_sha(supervisor_path),
            "worker_completion_sha256": file_sha(worker_path), "closed_ledger_sha256": file_sha(closed_path)}


def validate_job(run_root, job_id, ownership_path, ownership_sha, *, process_snapshot=None, handle_probe=None):
    root = absolute_path(run_root, directory=True)
    ownership = load_bound(ownership_path, ownership_sha)
    job_type = _job_type(ownership)
    _private_owner(ownership)
    if ownership.get("continuation_adoption_source") is not None:
        _continuation_header(ownership)
    require(ownership.get("schema_version") == 1
            and ownership.get("stage") == "formal" and ownership.get("run_root") == str(root)
            and isinstance(ownership.get("run_id"), str) and ownership["run_id"], "formal run ownership differs")
    for key in ("protocol_sha256", "config_sha256", "archive_helper_sha256"):
        _hex(ownership.get(key), key)
    require(ownership["archive_helper_sha256"] == file_sha(Path(__file__).resolve()), "archive helper source differs")
    require(type(ownership.get("reclamation_authorized")) is bool, "ownership lacks explicit cleanup preference")
    jobs = ownership.get(job_type + "_jobs")
    require(isinstance(jobs, list) and jobs and len(jobs) == len(set(jobs)), "invalid formal job allowlist")
    for item in jobs:
        _relative_job(item)
    planned_specs = ownership.get(job_type + "_spec_sha256")
    require(isinstance(planned_specs, dict) and set(planned_specs) == set(jobs), "prospective formal spec allowlist differs")
    for value in planned_specs.values():
        _hex(value, "prospective episode spec")
    if job_type == "server":
        _server_batches(ownership, jobs)
    job_id = _relative_job(job_id)
    require(job_id in jobs, "job is not an explicitly owned formal " + job_type)
    tree = absolute_path(root / job_id, directory=True)
    require(tree != root and tree.is_relative_to(root), "job must be strictly inside its owned run")
    _private_server_tree(root, tree, ownership)
    _no_mounts(tree)
    require(not any(other != job_id and (root / other).is_relative_to(tree) for other in jobs),
            "episode job is an ancestor of another owned job")
    _outside(ownership_path, tree)
    wrapper = ownership.get("wrapper_source", {})
    wrapper_path = _outside(wrapper.get("path", ""), tree)
    require(file_sha(wrapper_path) == _hex(wrapper.get("sha256"), "wrapper source"), "frozen wrapper source differs")
    spec_path, launch_path, completion_path = (tree / name for name in ("spec.json", "launch.json", "completion.json"))
    for path in (spec_path, launch_path, completion_path):
        absolute_path(path)
    spec, launch, completion = (json.loads(path.read_text()) for path in (spec_path, launch_path, completion_path))
    if ownership.get("continuation_adoption_source") is not None:
        adoption_source = ownership["continuation_adoption_source"]
        require(spec.get("bound_files", {}).get(adoption_source["path"]) == adoption_source["sha256"],
                "native job spec does not bind the continuation adoption manifest")
    require(file_sha(spec_path) == planned_specs[job_id], "job specification differs from prospective ownership")
    require(spec.get("schema_version") == 1 and spec.get("job_id") == job_id and spec.get("output") == str(tree)
            and spec.get("runner_sha256") == wrapper["sha256"], "wrapper specification ownership differs")
    require(launch.get("job_id") == job_id and launch.get("output") == str(tree)
            and launch.get("runner_sha256") == wrapper["sha256"] and launch.get("spec_sha256") == file_sha(spec_path),
            "launch does not bind the exact owned specification")
    require(completion.get("job_id") == job_id and completion.get("command") == spec.get("command")
            and isinstance(spec.get("command"), list) and spec["command"], "completion command/job differs")
    clean_status = (completion.get("status") == "passed" and not completion.get("stop_reason"))
    if job_type == "server":
        clean_status = (completion.get("status") in {"passed", "stopped"}
                        and completion.get("stop_reason") in {None, "SIGTERM"})
    require(clean_status and type(completion.get("exit_code")) is int
            and completion["exit_code"] == 0 and completion.get("automatic_retry") is False
            and not any(completion.get(key) for key in ("forced_kill", "cleanup_forced_kill",
                                                       "error", "cleanup_errors", "owned_processes_still_live")),
            "completed job lacks a clean external exit0")
    require(_process_key(launch.get("process")) == _process_key(completion.get("wrapper")), "wrapper PID/start identity differs")
    for key in ("group_id", "session_id"):
        require(type(launch["process"].get(key)) is int and launch["process"][key] == completion["wrapper"].get(key),
                "wrapper process group/session differs")
    recorded = [completion["wrapper"], completion.get("child")]
    require(isinstance(completion.get("owned_processes"), list), "owned descendant evidence is missing")
    recorded.extend(completion["owned_processes"])
    keys = {_process_key(row) for row in recorded}
    require(_process_key(completion["child"]) in {_process_key(row) for row in completion["owned_processes"]},
            "direct child is absent from owned process evidence")
    if process_snapshot is None:
        process_snapshot = same_uid_processes
    if handle_probe is None:
        handle_probe = same_uid_live_handles
    require(not any(_process_key(row) in keys and row.get("state") != "Z" for row in process_snapshot()),
            "owned wrapper or descendant remains live")
    require(not handle_probe(tree), "job has live file, directory or memory-mapping handles")
    bindings = {"completion_sha256": file_sha(completion_path), "spec_sha256": file_sha(spec_path),
                "launch_sha256": file_sha(launch_path)}
    if job_type == "server":
        return tree, ownership, dict(bindings, **_validate_server(tree, ownership, job_id, spec, completion))
    result_path = tree / "episode" / "result.json"
    absolute_path(result_path)
    result = json.loads(result_path.read_text())
    require(result.get("status") == "completed" and type(result.get("success")) is bool
            and result.get("protocol_sha256") == ownership["protocol_sha256"]
            and not result.get("cleanup_errors")
            and result.get("native_app_close") in {"returned", "pending_external_exit_check"},
            "native episode is incomplete or belongs to another protocol")
    return tree, ownership, dict(bindings, native_result_sha256=file_sha(result_path))


def scan_tree(tree):
    tree = absolute_path(tree, directory=True)
    _no_mounts(tree)
    device = tree.stat().st_dev
    entries = []

    def visit(directory):
        before = directory.stat(follow_symlinks=False)
        require(stat.S_ISDIR(before.st_mode) and before.st_dev == device, "directory boundary changed during inventory")
        for path in sorted(directory.iterdir()):
            value = path.stat(follow_symlinks=False)
            require(value.st_dev == device, "file crosses the job filesystem boundary")
            row = {"path": str(path.relative_to(tree)), "mode": stat.S_IMODE(value.st_mode)}
            if stat.S_ISLNK(value.st_mode):
                target = os.readlink(path)
                resolved = path.resolve(strict=True)
                require(resolved.is_relative_to(tree), "symlink escapes the inventoried job")
                row.update(type="symlink", target=target)
            elif stat.S_ISREG(value.st_mode):
                require(value.st_nlink == 1, "multiply linked files cannot establish exclusive staging ownership")
                row.update(type="file", bytes=value.st_size, sha256=file_sha(path))
            elif stat.S_ISDIR(value.st_mode):
                row.update(type="directory")
                visit(path)
            else:
                raise ValueError("special file in job evidence")
            require(_token(value) == _token(path.stat(follow_symlinks=False)), "entry changed during inventory")
            entries.append(row)
        require(_token(before) == _token(directory.stat(follow_symlinks=False)), "directory changed during inventory")

    visit(tree)
    return sorted(entries, key=lambda row: row["path"])


def inventory_job(*, run_root, job_id, ownership, ownership_sha256, receipt,
                  process_snapshot=None, handle_probe=None):
    tree, owner, bindings = validate_job(run_root, job_id, ownership, ownership_sha256,
                                        process_snapshot=process_snapshot, handle_probe=handle_probe)
    receipt = _receipt_path(receipt, tree, run_root)
    before = tree.stat()
    entries = scan_tree(tree)
    tree2, owner2, bindings2 = validate_job(run_root, job_id, ownership, ownership_sha256,
                                           process_snapshot=process_snapshot, handle_probe=handle_probe)
    require(tree == tree2 and owner == owner2 and bindings == bindings2 and _token(before) == _token(tree.stat()),
            "completed job changed during inventory")
    value = {"schema_version": 1, "kind": "formal_" + _job_type(owner) + "_inventory_v1", "status": "inventoried",
             "created": time.time(), "run_id": owner["run_id"], "run_root": str(Path(run_root)), "job_id": job_id,
             "tree_root": str(tree), "tree_identity": {"device": before.st_dev, "inode": before.st_ino},
             "ownership_sha256": ownership_sha256, "archive_helper_sha256": file_sha(Path(__file__).resolve()),
             "entries": entries, "entries_sha256": object_sha(entries), **bindings}
    value.update(_server_visibility(owner))
    write_once(receipt, value)
    return {"status": "inventoried", "receipt": str(receipt), "receipt_sha256": file_sha(receipt),
            "job_id": job_id, "files": sum(row["type"] == "file" for row in entries),
            "bytes": sum(row.get("bytes", 0) for row in entries), "reclaimed": False}


def _validate_archive_ack(ack, saved, owner, bindings, tree, job_id, ownership_sha, inventory_sha):
    job_type = _job_type(owner)
    require(isinstance(ack, dict) and ack.get("schema_version") == 1
            and ack.get("kind") == "formal_" + job_type + "_archive_ack_v1"
            and ack.get("status") == "passed" and ack.get("durable") is True,
            "durable local archive acknowledgment is required")
    expected = {"run_id": owner["run_id"], "run_root": owner["run_root"], "job_id": job_id,
                "tree_root": str(tree), "ownership_sha256": ownership_sha,
                "completion_sha256": bindings["completion_sha256"], "spec_sha256": bindings["spec_sha256"],
                "inventory_sha256": inventory_sha, "entries_sha256": saved.get("entries_sha256")}
    expected.update(_server_visibility(owner))
    if job_type == "server":
        expected.update({key: bindings[key] for key in ("supervisor_completion_sha256", "worker_completion_sha256",
                                                       "closed_ledger_sha256")})
        expected.update(episode_ids=owner["server_batches"][job_id]["episode_ids"],
                        completed_episode_count=owner["server_batches"][job_id]["expected_episode_count"])
        require(type(ack.get("completed_episode_count")) is int, "server archive requires its exact audited episode count")
    require(all(ack.get(key) == value for key, value in expected.items()), "local archive acknowledgment binding differs")
    archive_path = Path(ack.get("local_archive_path", ""))
    require(archive_path.is_absolute() and archive_path != Path("/") and ".." not in archive_path.parts,
            "local durable archive path must be explicit and absolute")
    collected = "collected_records_sha256" if job_type == "server" else "collected_record_sha256"
    for key in ("local_archive_inventory_sha256", "local_audit_sha256", collected):
        _hex(ack.get(key), key)


def _validate_inventory_binding(saved, owner, bindings, tree, job_id, ownership_sha):
    require(saved.get("schema_version") == 1 and saved.get("kind") == "formal_" + _job_type(owner) + "_inventory_v1"
            and saved.get("status") == "inventoried", "not a completed inventory receipt")
    expected = {"run_id": owner["run_id"], "run_root": owner["run_root"], "job_id": job_id,
                "tree_root": str(tree), "ownership_sha256": ownership_sha, **bindings}
    expected.update(_server_visibility(owner))
    require(all(saved.get(key) == value for key, value in expected.items())
            and saved.get("archive_helper_sha256") == owner["archive_helper_sha256"], "inventory job/source binding differs")
    require(isinstance(saved.get("entries"), list) and object_sha(saved["entries"]) == saved.get("entries_sha256"),
            "inventory entries digest differs")
    return expected


def acknowledge_job(*, run_root, job_id, ownership, ownership_sha256, inventory, inventory_sha256,
                    receipt, acknowledgment_bytes, process_snapshot=None, handle_probe=None):
    tree, owner, bindings = validate_job(run_root, job_id, ownership, ownership_sha256,
                                        process_snapshot=process_snapshot, handle_probe=handle_probe)
    receipt = _receipt_path(receipt, tree, run_root)
    saved = load_bound(_outside(inventory, tree), inventory_sha256)
    _validate_inventory_binding(saved, owner, bindings, tree, job_id, ownership_sha256)
    require(isinstance(acknowledgment_bytes, bytes) and 0 < len(acknowledgment_bytes) <= (1 << 20),
            "archive acknowledgment must be bounded JSON bytes")
    ack = json.loads(acknowledgment_bytes)
    _validate_archive_ack(ack, saved, owner, bindings, tree, job_id, ownership_sha256, inventory_sha256)
    require(scan_tree(tree) == saved["entries"], "staging job changed before archive acknowledgment")
    current = tree.stat()
    require(saved.get("tree_identity") == {"device": current.st_dev, "inode": current.st_ino}, "inventoried job root was replaced")
    write_once_raw(receipt, acknowledgment_bytes)
    return {"status": "acknowledged", "receipt": str(receipt), "receipt_sha256": file_sha(receipt),
            "job_id": job_id, "reclaimed": False}


def _remove_anchored(tree, expected_identity):
    """Unlink only this opened tree, without following links (also on Python 3.10)."""
    _no_mounts(tree)
    parent = os.open(tree.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    root = None
    try:
        root = os.open(tree.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        value = os.fstat(root)
        require({"device": value.st_dev, "inode": value.st_ino} == expected_identity, "job root replaced before removal")

        def clear(fd):
            for name in os.listdir(fd):
                entry = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(entry.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        require(_token(entry) == _token(os.fstat(child)), "directory replaced during removal")
                        clear(child)
                    finally:
                        os.close(child)
                    os.rmdir(name, dir_fd=fd)
                else:
                    os.unlink(name, dir_fd=fd)
            os.fsync(fd)

        clear(root)
        current = os.stat(tree.name, dir_fd=parent, follow_symlinks=False)
        require((current.st_dev, current.st_ino) == (value.st_dev, value.st_ino), "root moved during removal")
        os.rmdir(tree.name, dir_fd=parent)
        os.fsync(parent)
    finally:
        if root is not None:
            os.close(root)
        os.close(parent)


def prepare_run(*, run_root, ownership, ownership_sha256, receipt):
    """Create a new private episode/server root; never chmod or reuse existing paths."""
    root = absolute_path(run_root, exists=False)
    require(not root.exists(), "run root already exists; no reuse or retry")
    parent = absolute_path(root.parent, directory=True)
    owner = load_bound(ownership, ownership_sha256)
    job_type = _job_type(owner)
    _private_owner(owner)
    require(owner.get("schema_version") == 1
            and owner.get("stage") == "formal" and owner.get("run_root") == str(root)
            and isinstance(owner.get("run_id"), str) and owner["run_id"]
            and type(owner.get("reclamation_authorized")) is bool,
            "private root preparation requires explicit formal ownership")
    for key in ("protocol_sha256", "config_sha256", "archive_helper_sha256"):
        _hex(owner.get(key), key)
    require(owner["archive_helper_sha256"] == file_sha(Path(__file__).resolve()), "archive helper source differs")
    jobs = owner.get(job_type + "_jobs")
    require(isinstance(jobs, list) and jobs and len(jobs) == len(set(jobs)), "invalid formal job allowlist")
    for job in jobs:
        _relative_job(job)
    planned = owner.get(job_type + "_spec_sha256")
    require(isinstance(planned, dict) and set(planned) == set(jobs), "prospective job spec allowlist differs")
    for digest in planned.values():
        _hex(digest, "prospective server spec")
    if job_type == "server":
        _server_batches(owner, jobs)
    _private_owned_entry(parent, owner["owner_uid"], protected=True)
    require(not Path(ownership).is_relative_to(root), "ownership must be staged outside the new run root")
    checked_sources = {}
    sources = [owner.get("wrapper_source", {})]
    if job_type == "server":
        sources.extend((owner["supervisor_source"], owner["worker_source"], owner["installation_source"],
                        *(batch["identity_source"] for batch in owner["server_batches"].values()),
                        *(batch["measured_receipt_source"] for batch in owner["server_batches"].values())))
    if owner.get("continuation_adoption_source") is not None:
        _continuation_header(owner)
        sources.append(owner["continuation_adoption_source"])
    for source in sources:
        path = _outside(source.get("path", ""), root)
        expected = _hex(source.get("sha256"), "prospective native source")
        if path not in checked_sources:
            checked_sources[path] = file_sha(path)
        require(checked_sources[path] == expected,
                "prospective native source differs before private root creation")
    receipt = absolute_path(receipt, exists=False)
    require(receipt.parent == root / "archive_receipts" and not receipt.exists(),
            "private preparation receipt must be directly inside new archive_receipts")
    # The parent is already owned and protected. Preserve a partial root on error.
    root.mkdir(mode=0o700)
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "umask did not create required private mode0700")
    receipts = root / "archive_receipts"
    receipts.mkdir(mode=0o700)
    value = {"schema_version": 1, "kind": "formal_" + job_type + "_private_root_v1", "status": "prepared",
             "run_id": owner["run_id"], "run_root": str(root), "run_root_mode": 0o700,
             "ownership_sha256": ownership_sha256, "archive_helper_sha256": owner["archive_helper_sha256"],
             "created": time.time(), "reclaimed": False, **_server_visibility(owner)}
    write_once(receipt, value)
    for path in (root, parent):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {"status": "prepared", "run_root": str(root), "receipt": str(receipt),
            "receipt_sha256": file_sha(receipt), "reclaimed": False}


def free_space(*, path, minimum_free_bytes):
    """Sample bytes available to this UID without reserving or modifying storage."""
    require(type(minimum_free_bytes) is int and minimum_free_bytes >= 0,
            "minimum_free_bytes must be a nonnegative integer")
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts and path.is_dir()
            and path.resolve(strict=True) == path and not any(part.is_symlink() for part in [path, *path.parents]),
            "capacity path must be an existing canonical absolute directory")
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        sample = os.fstatvfs(fd)
        after = path.stat(follow_symlinks=False)
        require((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
                "capacity directory identity changed during sampling")
        require(sample.f_frsize > 0 and sample.f_bavail >= 0 and sample.f_blocks >= 0,
                "filesystem returned invalid available-space counters")
        available, total = sample.f_bavail * sample.f_frsize, sample.f_blocks * sample.f_frsize
        return {"status": "passed" if available >= minimum_free_bytes else "insufficient_space", "path": str(path),
                "minimum_free_bytes": minimum_free_bytes, "available_bytes": available, "total_bytes": total,
                "device": before.st_dev, "sampled": time.time(), "helper_source_sha256": file_sha(Path(__file__).resolve())}
    finally:
        os.close(fd)


def adopt_anchor(*, run_root, ownership, ownership_sha256, receipt, anchor_bytes):
    """Copy only the exact preserved first physical anchor into a fresh private run."""
    root = absolute_path(run_root, directory=True)
    owner = load_bound(ownership, ownership_sha256)
    require(owner.get("schema_version") == 1 and _job_type(owner) == "episode"
            and owner.get("stage") == "formal" and owner.get("run_root") == str(root),
            "anchor adoption requires exact private renderer ownership")
    _private_owner(owner)
    _private_server_tree(root, root, owner)
    require(owner.get("archive_helper_sha256") == file_sha(Path(__file__).resolve()), "archive helper source differs")
    header = _continuation_header(owner)
    require(header is not None, "preserved anchor requires the explicit first-record adoption manifest")
    require(isinstance(anchor_bytes, bytes) and 0 < len(anchor_bytes) <= (1 << 20), "anchor must be bounded JSON bytes")
    expected_sha = _hex(owner.get("adopted_anchor_sha256"), "preserved anchor")
    require(hashlib.sha256(anchor_bytes).hexdigest() == expected_sha and isinstance(json.loads(anchor_bytes), dict),
            "anchor bytes differ from the independently bound original anchor")
    prepared_path = root / "archive_receipts/private_root.json"
    prepared = _json_file(prepared_path)
    expected = {"schema_version": 1, "kind": "formal_episode_private_root_v1", "status": "prepared",
                "run_id": owner["run_id"], "run_root": str(root), "run_root_mode": 0o700,
                "ownership_sha256": ownership_sha256, "archive_helper_sha256": owner["archive_helper_sha256"],
                **_server_visibility(owner)}
    require(all(prepared.get(key) == value for key, value in expected.items()),
            "anchor adoption lacks the exact fresh private-root preparation")
    require({path.name for path in root.iterdir()} == {"archive_receipts"}
            and {path.name for path in (root / "archive_receipts").iterdir()} == {"private_root.json"},
            "anchor may be adopted only once before any native job or other root activity")
    receipt = absolute_path(receipt, exists=False)
    require(receipt == root / "archive_receipts/adopted_anchor.json" and not receipt.exists(),
            "anchor receipt path is fixed and exclusive")
    parent = root
    for name in ("anchors", "formal", "AnimalsInBinTask"):
        parent = parent / name
        parent.mkdir(mode=0o700)
        _private_owned_entry(parent, owner["owner_uid"], protected=True)
    anchor = parent / "0000.json"
    write_once_raw(anchor, anchor_bytes)
    require(file_sha(anchor) == expected_sha and load_bound(ownership, ownership_sha256) == owner
            and _continuation_header(owner) == header, "anchor adoption source or exact copied bytes changed")
    value = {"schema_version": 1, "kind": "formal_preserved_anchor_adoption_v1", "status": "adopted_anchor",
             "run_id": owner["run_id"], "run_root": str(root), "pair_id": ADOPTED_PAIR,
             "anchor": str(anchor), "anchor_sha256": expected_sha, "ownership_sha256": ownership_sha256,
             "continuation_adoption_sha256": owner["continuation_adoption_source"]["sha256"],
             "archive_helper_sha256": owner["archive_helper_sha256"], "created": time.time(),
             **_server_visibility(owner)}
    write_once(receipt, value)
    for directory in (parent, parent.parent, parent.parent.parent, root):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {"status": "adopted_anchor", "anchor": str(anchor), "anchor_sha256": expected_sha,
            "receipt": str(receipt), "receipt_sha256": file_sha(receipt)}


def reclaim_job(*, run_root, job_id, ownership, ownership_sha256, inventory, inventory_sha256,
                archive_ack, archive_ack_sha256, receipt, allow_reclaim=False,
                process_snapshot=None, handle_probe=None):
    require(allow_reclaim is True, "reclamation is disabled unless explicitly requested")
    tree, owner, bindings = validate_job(run_root, job_id, ownership, ownership_sha256,
                                        process_snapshot=process_snapshot, handle_probe=handle_probe)
    require(owner["reclamation_authorized"] is True, "run ownership does not authorize staging reclamation")
    receipt = _receipt_path(receipt, tree, run_root)
    intent_path = _receipt_path(str(receipt) + ".intent.json", tree, run_root)
    saved = load_bound(_outside(inventory, tree), inventory_sha256)
    ack = load_bound(_outside(archive_ack, tree), archive_ack_sha256)
    expected = _validate_inventory_binding(saved, owner, bindings, tree, job_id, ownership_sha256)
    _validate_archive_ack(ack, saved, owner, bindings, tree, job_id, ownership_sha256, inventory_sha256)
    entries = scan_tree(tree)
    require(entries == saved.get("entries") and object_sha(entries) == saved.get("entries_sha256"),
            "staging job changed after inventory")
    value = tree.stat()
    require(saved.get("tree_identity") == {"device": value.st_dev, "inode": value.st_ino}, "inventoried job root was replaced")
    _, owner_after, bindings_after = validate_job(run_root, job_id, ownership, ownership_sha256,
                                                 process_snapshot=process_snapshot, handle_probe=handle_probe)
    require(owner_after == owner and bindings_after == bindings and scan_tree(tree) == entries,
            "job changed during reclamation validation")
    evidence = {"schema_version": 1, "kind": "formal_" + _job_type(owner) + "_reclamation_v1", **expected,
                "inventory_sha256": inventory_sha256, "archive_ack_sha256": archive_ack_sha256,
                "archive_helper_sha256": owner["archive_helper_sha256"], "entries_sha256": saved["entries_sha256"]}
    write_once(intent_path, dict(evidence, status="reclamation_intent", created=time.time()))
    _remove_anchored(tree, saved["tree_identity"])
    write_once(receipt, dict(evidence, status="reclaimed", ended=time.time()))
    return {"status": "reclaimed", "receipt": str(receipt), "receipt_sha256": file_sha(receipt),
            "job_id": job_id, "reclaimed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    capacity = sub.add_parser("free-space")
    capacity.add_argument("--path", required=True)
    capacity.add_argument("--minimum-free-bytes", required=True, type=int)
    for name in ("prepare-run", "adopt-anchor"):
        prepare = sub.add_parser(name)
        for option in ("run-root", "ownership", "ownership-sha256", "receipt"):
            prepare.add_argument("--" + option, required=True)
    for name in ("inventory", "ack", "reclaim"):
        item = sub.add_parser(name)
        for option in ("run-root", "job-id", "ownership", "ownership-sha256", "receipt"):
            item.add_argument("--" + option, required=True)
        if name in {"ack", "reclaim"}:
            for option in ("inventory", "inventory-sha256"):
                item.add_argument("--" + option, required=True)
        if name == "reclaim":
            for option in ("archive-ack", "archive-ack-sha256"):
                item.add_argument("--" + option, required=True)
            item.add_argument("--allow-reclaim", action="store_true")
    options = vars(parser.parse_args())
    operation = options.pop("operation")
    try:
        if operation == "ack":
            options["acknowledgment_bytes"] = sys.stdin.buffer.read((1 << 20) + 1)
        if operation == "adopt-anchor":
            options["anchor_bytes"] = sys.stdin.buffer.read((1 << 20) + 1)
        result = {"inventory": inventory_job, "ack": acknowledge_job,
                  "reclaim": reclaim_job, "prepare-run": prepare_run, "adopt-anchor": adopt_anchor,
                  "free-space": free_space}[operation](**options)
    except Exception as error:
        result = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                  "reclaimed": None if operation == "reclaim" else False}
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())

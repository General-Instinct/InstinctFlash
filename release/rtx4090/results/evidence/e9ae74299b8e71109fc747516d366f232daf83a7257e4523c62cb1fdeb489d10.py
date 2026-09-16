"""Render source-bound RTX4090 receipts; never import a model or contact a device.

Input is an explicit SHA256 manifest of deployment profiles, cohort contexts,
independent audits and their complete raw archives. NumPy only reads saved actions.
The primary sample convention matches audit_family_archive_v1.py: 20 stateless
predictions, or 12 steady history predictions excluding each episode's first call.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

FAMILIES = ("va", "vla4", "vla2", "pi05", "groot", "edge", "nano", "dreamzero")
REQUIRED_MODES = {family: {"native", "fp8"} for family in FAMILIES}
REQUIRED_MODES["va"] |= {"2v4a-native", "2v4a-fp8"}
for _family in ("vla2", "edge", "nano"):
    REQUIRED_MODES[_family].add("numeric")
REQUIRED_MODES["dreamzero"] |= {"dynamic-native", "dynamic-fp8"}
BASE = "instinctflash.rtx_archive_independent_audit.v1"
COMPLETE = "instinctflash.complete_pipeline_independent_audit.v1"
VA_REDUCED = "instinctflash.va_native_2v4a_archive_independent_audit.v1"
MODE = "instinctflash.mode_supplement_archive_independent_audit.v1"
TARGET = {"name": "rtx4090", "capability": [8, 9]}
INPUT_V1 = "instinctflash.rtx4090.results_input.v1"
INPUT_V2 = "instinctflash.rtx4090.results_input.v2"
INPUT_V3 = "instinctflash.rtx4090.results_input.v3"
CAPACITY_REPORT_SHA = "76b9fa956affc17951fb7a9fb77ff79a34d8edc8ab07fed90d2dc3a10a59ae80"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_bound(base, spec):
    require(valid_sha(spec["sha256"]), "Invalid manifest SHA256")
    data = (base / spec["path"]).read_bytes()
    require(sha(data) == spec["sha256"], "Manifest file hash differs: " + Path(spec["path"]).name)
    return data


class Archive:
    def __init__(self, data):
        self.sha256 = sha(data)
        self.data = data
        self.files = {}
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            seen = set()
            total = 0
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                require(not path.is_absolute() and ".." not in path.parts and "\\" not in member.name,
                        "Unsafe archive path")
                require(member.name not in seen, "Duplicate archive path")
                seen.add(member.name)
                require(member.isfile() or member.isdir(), "Archive link or device is forbidden")
                total += member.size
                require(total <= 2 * 1024**3 and member.size <= 256 * 1024**2, "Oversized evidence archive")
                if member.isfile():
                    self.files[member.name] = archive.extractfile(member).read()
        self.by_sha = {}
        self.objects = {}
        for name, content in self.files.items():
            self.by_sha.setdefault(sha(content), []).append(name)
            if name.endswith(".json"):
                self.objects[name] = json.loads(content)
        inventory = self.objects.get("provenance/archive_manifest.json")
        if inventory:
            members = inventory.get("files", inventory.get("members"))
            require(members is not None, "Archive manifest inventory missing")
            if isinstance(members, list):
                members = {item.get("path", item.get("name")): item for item in members}
            require(set(members) == set(self.files) - {"provenance/archive_manifest.json"},
                    "Archive manifest does not cover every file")
            for name, item in members.items():
                require(item["sha256"] == sha(self.files[name]), "Archive member hash differs")
                require(item.get("bytes", item.get("size")) == len(self.files[name]),
                        "Archive member size differs")

    def ref(self, name):
        return {"archive_sha256": self.sha256, "member": name, "sha256": sha(self.files[name])}


def normalized_sources(sources):
    require(isinstance(sources, dict) and bool(sources), "Loaded source inventory is missing")
    result = {}
    for path, checksum in sources.items():
        require(valid_sha(checksum), "Invalid loaded source checksum")
        if "/site-packages/" in path:
            name = path.split("/site-packages/", 1)[1]
        elif "/benchmarks/" in path:
            name = "benchmarks/" + path.split("/benchmarks/", 1)[1]
        else:
            # Preserve uniqueness without publishing a task-specific directory.
            name = "other/" + sha(path.encode())[:16] + "/" + PurePosixPath(path).name
        require(name not in result or result[name] == checksum, "Conflicting source inventory")
        result[name] = checksum
    return dict(sorted(result.items()))


def options(value):
    return {"device": "cuda:0", "precision": "native", "step_cache": "checkpoint",
            "tier_ceiling": "bitexact", **value}


def public_policy(value):
    if value is None:
        return None
    keys = ("category", "changed_schedule", "nfe", "precision", "schedule_options", "step_cache",
            "tier_ceiling", "transform_tier")
    return {key: value[key] for key in keys if key in value}


def mode_for(receipt, profile):
    if receipt["arm"] == "eager_native":
        return "pytorch"
    matches = [name for name, mode in profile["execution_modes"].items()
               if options(receipt["runtime_kwargs"]) == options(mode["runtime_kwargs"])
               and receipt["optimizer_environment"] == mode["environment"]
               and receipt["effective_schedule"] == mode["effective_schedule"]]
    require(len(matches) == 1, "Raw runtime recipe does not identify one declared mode")
    return matches[0]


def loaded_numeric_environment(receipt):
    observed = receipt['numeric_environment']
    require(set(observed) == {'matmul_tf32', 'cudnn_tf32', 'cudnn_benchmark'}
            and all(type(value) is bool for value in observed.values()),
            'Raw loaded numerical flags are incomplete or invalid')
    return dict(observed)


def gpu_uuid(value):
    require(isinstance(value, str), "Actual GPU UUID is missing")
    normalized = value.removeprefix("GPU-").lower()
    require(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", normalized) is not None,
            "Invalid actual GPU UUID")
    return normalized


def load_contexts(manifest, base, evidence):
    if manifest["schema"] == INPUT_V1:
        require("machine_contexts" not in manifest, "Ambiguous legacy and cohort context declarations")
        specs, legacy = {"legacy": manifest["machine_context"]}, True
    else:
        require(manifest["schema"] in (INPUT_V2, INPUT_V3) and "machine_context" not in manifest,
                "Cohort input must not declare a global machine context")
        specs, legacy = manifest["machine_contexts"], False
    require(isinstance(specs, dict) and bool(specs), "At least one actual machine context is required")
    contexts = {}
    for identifier, spec in specs.items():
        require(isinstance(identifier, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", identifier),
                "Invalid cohort identifier")
        data = load_bound(base, spec)
        obj = json.loads(data)
        hardware = machine_summary(obj)
        require(obj["gpu_query_fields"] == "index,uuid,name,memory.total,pci.bus_id,pcie.link.gen.max,pcie.link.width.max,power.limit,driver_version",
                "Unknown machine-context GPU inventory layout")
        devices = list(csv.reader(io.StringIO(obj["gpu_csv"]), skipinitialspace=True))
        uuids = [gpu_uuid(row[1]) for row in devices]
        require(len(uuids) == len(set(uuids)), "Duplicate GPU UUID in machine context")
        contexts[identifier] = {"machine_context_sha256": sha(data), "hardware": hardware,
                                "cgroup": cgroup_summary(obj),
                                "gpu_uuids": sorted(uuids), "families": set(), "torch_versions": set(),
                                "source_commits": set(), "cell_ids": set()}
        evidence.setdefault(sha(data), (".json", data))
    return contexts, legacy


def check_context_hardware(hardware, context):
    require(hardware.get("status") == "passed" and hardware.get("target") == TARGET
            and hardware.get("capability") == [8, 9]
            and hardware.get("name") == "NVIDIA GeForce RTX 4090",
            "Actual receipt hardware admission is incomplete")
    identity = gpu_uuid(hardware.get("uuid"))
    require(identity in context["gpu_uuids"], "Actual receipt GPU does not belong to its declared cohort")
    require(type(hardware.get("total_memory_bytes")) is int and hardware["total_memory_bytes"] > 0,
            "Actual receipt GPU memory is missing")
    return identity


def bind_cohort(pair, identifier, context):
    row, raw = pair
    actual_gpu = check_context_hardware(raw["receipt"]["hardware"], context)
    row.update(cohort=identifier, machine_context_sha256=context["machine_context_sha256"],
               measured_gpu_uuid=actual_gpu,
               receipt_gpu_memory_bytes=raw["receipt"]["hardware"]["total_memory_bytes"])
    context["families"].add(row["family"])
    context["torch_versions"].add(row["torch"])
    context["cell_ids"].add(row["id"])
    if row["source_commit"]:
        context["source_commits"].add(row["source_commit"])


def raw_cell(archive, name, metric, profile):
    receipt = archive.objects[name]
    family = profile["id"]
    require(receipt.get("ok") is True and receipt["family"] == family, "Failed or wrong-family raw receipt")
    require(receipt["target"] == TARGET and "4090" in receipt["device"], "Receipt is not actual RTX4090")
    require({"model_id": receipt["model_id"], "revision": receipt["revision"]} == profile["checkpoint"],
            "Original checkpoint identity differs from target profile")
    require(receipt["observed_nfe_before"] == receipt["observed_nfe_after"], "Observed schedule changed")
    for key in ("matrix_sha256", "input_archive_sha256"):
        require(receipt[key] in archive.by_sha, "Raw matrix or recorded input bytes missing")
    require(receipt["precision"] == metric["precision"], "Precision differs from audit")
    expected_schedule = metric.get("schedule", metric.get("effective_schedule"))
    if expected_schedule is not None:
        require(receipt["effective_schedule"] == expected_schedule, "Schedule differs from audit")
    elif "nfe" in metric:
        require(receipt["effective_schedule"]["nfe"] == metric["nfe"], "NFE differs from audit")
    for key in ("runtime_kwargs", "optimizer_environment"):
        if key in metric:
            require(receipt[key] == metric[key], "Mode options differ from audit")
    if "source_inventory" in metric:
        require(receipt["sources"] == metric["source_inventory"], "Mode source inventory differs")
    actions_name = name[:-5] + ".npz"
    require(actions_name in archive.files and sha(archive.files[actions_name]) == receipt["actions_sha256"],
            "Raw actions missing or corrupted")
    with np.load(io.BytesIO(archive.files[actions_name]), allow_pickle=False) as loaded:
        actions = loaded["actions"].copy()
    cases, calls = receipt["cases"], receipt["calls"]
    history = family in ("va", "dreamzero")
    expected_count, expected_samples = (21, 12) if history else (25, 20)
    require(len(cases) == len(calls) == actions.shape[0] == expected_count, "Incomplete raw prediction coverage")
    require(actions.dtype.kind == "f" and np.isfinite(actions).all(), "Nonfinite action evidence")
    selected = []
    for index, (case, call) in enumerate(zip(cases, calls)):
        require(case["i"] == call["i"] == index, "Nonsequential prediction evidence")
        require(list(actions.shape[1:]) == call["shape"], "Action shape differs from raw call")
        require(math.isfinite(call["ms"]) and call["ms"] > 0, "Invalid raw latency")
        if case["phase"] == "measured" and (not history or case["cycle"] > 0):
            selected.append(call["ms"])
    require(len(selected) == metric["primary_samples"] == expected_samples, "Primary sample coverage differs")
    p50 = statistics.median(selected)
    require(p50 == metric["p50_ms"], "Audit p50 differs from raw timed calls")
    if "receipt_sha256" in metric:
        require(sha(archive.files[name]) == metric["receipt_sha256"], "Mode receipt hash differs")
    else:
        require(any(isinstance(obj, dict) and isinstance(obj.get("cells"), list)
                    and any(isinstance(cell, dict) and cell.get("id") == receipt["cell_id"]
                            and isinstance(cell.get("receipt"), dict)
                            and cell["receipt"].get("sha256") == sha(archive.files[name])
                            for cell in obj["cells"]) for obj in archive.objects.values()),
                "Raw receipt is not bound by its archived validated report")
    if "actions_sha256" in metric:
        require(receipt["actions_sha256"] == metric["actions_sha256"], "Mode action hash differs")
    sources = normalized_sources(receipt["sources"])
    mode = mode_for(receipt, profile)
    row = {"family": family, "cell": receipt["cell_id"], "mode": mode, "p50_ms": p50,
           "precision": receipt["precision"], "primary_samples": len(selected), "total_predictions": len(calls),
           "effective_schedule": receipt["effective_schedule"], "runtime_kwargs": receipt["runtime_kwargs"],
           "optimizer_environment": receipt["optimizer_environment"], "checkpoint": profile["checkpoint"],
           "torch": receipt["torch"], "numeric_environment": loaded_numeric_environment(receipt), "execution_policy": public_policy(receipt.get("execution_policy")),
           "fixture_sha256": receipt["input_archive_sha256"], "matrix_sha256": receipt["matrix_sha256"],
           "source_inventory": sources, "source_inventory_sha256": sha(encoded(sources)),
           "receipt": archive.ref(name), "actions": archive.ref(actions_name), "serving": []}
    row["id"] = sha(encoded([family, mode, archive.sha256, sha(archive.files[name])]))
    queue = receipt.get("queue_drain")
    if family == "pi05":
        require(queue and len(queue["calls"]) == 51 and queue["reset_between_calls"] is False,
                "Pi05 action-queue evidence is incomplete")
        queue_name = name[:-5] + ".queue.npz"
        require(sha(archive.files[queue_name]) == queue["actions_sha256"], "Queue action checksum differs")
        with np.load(io.BytesIO(archive.files[queue_name]), allow_pickle=False) as loaded:
            queued = loaded["actions"].copy()
        require(queued.shape == (51, *actions.shape[1:]) and np.isfinite(queued).all(), "Invalid queue actions")
        row["queue_actions"] = archive.ref(queue_name)
    private = {"receipt": receipt, "actions": actions, "primary_ms": selected, "archive": archive}
    if family == "pi05":
        private["queue"] = queued
    return row, private


def compare(left, right, expected):
    lrow, lraw = left
    rrow, rraw = right
    require(lrow["cohort"] == rrow["cohort"]
            and lrow["machine_context_sha256"] == rrow["machine_context_sha256"]
            and lrow["measured_gpu_uuid"] == rrow["measured_gpu_uuid"]
            and lrow["torch"] == rrow["torch"],
            "Paired speed comparison must use the same cohort, GPU and Torch build")
    require(lrow["checkpoint"] == rrow["checkpoint"], "Compared checkpoint differs")
    require(lraw["receipt"]["cases"] == rraw["receipt"]["cases"], "Compared requests/history/feedback differ")
    require(lraw["receipt"].get("guidance") == rraw["receipt"].get("guidance"), "Compared guidance differs")
    same = lrow["effective_schedule"] == rrow["effective_schedule"]
    ratio = lrow["p50_ms"] / rrow["p50_ms"]
    require(ratio == expected.get("speedup", expected.get("ratio_of_p50")), "Comparison ratio differs")
    if "same_sampling_policy" in expected:
        require(same is expected["same_sampling_policy"], "Same-schedule comparison label is incorrect")
    a, b = lraw["actions"], rraw["actions"]
    require(a.shape == b.shape, "Compared action shapes differ")
    delta = a.astype(np.float64) - b.astype(np.float64)
    empirical = {"exact_bytes": a.dtype == b.dtype and a.tobytes() == b.tobytes(),
                 "exact_values": bool(np.array_equal(a, b)), "max_abs": float(np.abs(delta).max()),
                 "rmse": float(np.sqrt(np.mean(delta * delta)))}
    require({"exact_bytes", "max_abs", "rmse"} <= set(expected["actions"]), "Independent action comparison missing")
    for key, value in expected["actions"].items():
        if key in empirical:
            actual = empirical[key]
            require(actual == value if isinstance(actual, bool) else math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-15),
                    "Empirical action comparison differs: " + key)
    paired = statistics.median(a / b for a, b in zip(lraw["primary_ms"], rraw["primary_ms"]))
    if "median_paired_ratio" in expected:
        require(paired == expected["median_paired_ratio"], "Median paired ratio differs")
    result = {"baseline": lrow["id"], "candidate": rrow["id"], "same_sampling_policy": same,
              "cohort": lrow["cohort"], "machine_context_sha256": lrow["machine_context_sha256"],
              "comparison_scope": "whole-runtime comparison",
              "ratio_of_p50": ratio, "median_paired_ratio": paired, "actions": empirical}
    if "queue" in lraw:
        require(lraw["receipt"]["queue_drain"]["cases"] == rraw["receipt"]["queue_drain"]["cases"],
                "Paired queue requests differ")
        qa, qb = lraw["queue"], rraw["queue"]
        delta = qa.astype(np.float64) - qb.astype(np.float64)
        result["queue"] = {"exact_bytes": qa.dtype == qb.dtype and qa.tobytes() == qb.tobytes(),
                           "max_abs": float(np.abs(delta).max()), "rmse": float(np.sqrt(np.mean(delta * delta)))}
        for key, value in result["queue"].items():
            require(value == expected["queue"][key] if isinstance(value, bool) else
                    math.isclose(value, expected["queue"][key], rel_tol=1e-12, abs_tol=1e-15),
                    "Empirical queue comparison differs")
    return result


def serving_for(row, archives, context):
    found = {}
    for archive in archives:
        for name, obj in archive.objects.items():
            if not isinstance(obj, dict) or obj.get("schema") != "instinctflash.serving_smoke.v1":
                continue
            policy = obj.get("metadata", {}).get("execution_policy", {})
            if (obj.get("cell_id") != row["cell"] or obj.get("checkpoint") != row["checkpoint"]
                    or policy.get("precision") != row["precision"]
                    or policy.get("nfe") != row["effective_schedule"]["nfe"]
                    or public_policy(policy) != row["execution_policy"]
                    or obj.get("fixture_sha256") != row["fixture_sha256"]):
                continue
            require(obj["status"] == "passed" and obj["server_exit_code"] == 0, "Serving failed")
            require(obj["target"] == TARGET and obj["hardware"]["capability"] == [8, 9], "Serving GPU differs")
            require(check_context_hardware(obj["hardware"], context) == row["measured_gpu_uuid"],
                    "Serving GPU differs from the paired API cohort")
            require(len(obj["calls"]) == 6 and [(v["episode"], v["cycle"]) for v in obj["calls"]]
                    == [(e, c) for e in range(2) for c in range(3)], "Serving reset/history coverage is incomplete")
            checksum = obj["action_archive_sha256"]
            require(checksum in archive.by_sha, "Serving raw actions missing or corrupted")
            actions_name = archive.by_sha[checksum][0]
            with np.load(io.BytesIO(archive.files[actions_name]), allow_pickle=False) as loaded:
                actions = loaded["actions"]
                require(actions.shape[0] == 6 and np.isfinite(actions).all(), "Invalid serving actions")
            ref = archive.ref(name)
            found[ref["sha256"]] = {"receipt": ref, "actions": archive.ref(actions_name), "calls": 6,
                                   "cohort": row["cohort"], "machine_context_sha256": row["machine_context_sha256"],
                                   "episodes": 2, "latency_benchmark": False,
                                   "request_signatures_sha256": sha(encoded([
                                       [v["episode"], v["cycle"], v["request_sha256"]] for v in obj["calls"]]))}
    return list(found.values())


def ingest(entry, base, profiles, all_evidence, contexts, legacy):
    cohort_id = entry.get("cohort", "legacy" if legacy else None)
    require(isinstance(cohort_id, str) and cohort_id in contexts,
            "Every audit requires a known, explicitly bound cohort")
    require(not legacy or cohort_id == "legacy", "Legacy single-context input cannot override audit cohorts")
    context = contexts[cohort_id]
    audit_data = load_bound(base, entry["audit"])
    audit = json.loads(audit_data)
    require(audit.get("passed") is True, "Independent audit has not passed")
    require(audit["schema"] in (BASE, COMPLETE, VA_REDUCED, MODE), "Unsupported independent audit schema")
    family = audit.get("family", "va" if audit["schema"] == VA_REDUCED else None)
    require(family in FAMILIES, "Unknown model family")
    archives = [Archive(load_bound(base, spec)) for spec in entry["archives"]]
    archive_map = {a.sha256: a for a in archives}
    require(audit["archive_sha256"] in archive_map, "Audit raw archive missing")
    for key in ("parent_archive_sha256",):
        if key in audit:
            require(audit[key] in archive_map, "Audit parent raw archive missing")
    # Outer completion and source attestations must resolve to actual archived bytes.
    for key in ("source_completion_sha256", "parent_completion_sha256"):
        if key in audit:
            require(any(audit[key] in a.by_sha for a in archives), "Outer completion bytes missing")
    for mapping in ("outer_provenance_hashes", "outer_completion_sha256_bindings"):
        for checksum in audit.get(mapping, {}).values():
            require(any(checksum in a.by_sha for a in archives), "Outer provenance bytes missing")
    if audit.get("source_commit"):
        require(any(isinstance(o, dict) and o.get("source_commit") == audit["source_commit"]
                    for a in archives for o in a.objects.values()), "Source commit is not bound by raw provenance")
    for archive in archives:
        all_evidence.setdefault(archive.sha256, (".tar.gz", archive.data))
    all_evidence.setdefault(sha(audit_data), (".json", audit_data))
    rows = {}
    capture = audit.get("capture", audit)
    schema = audit["schema"]
    if schema == MODE:
        specs = list(audit["parent_baselines"].items()) + [("@candidate", audit["candidate"])]
    elif isinstance(capture["metrics"], dict):
        specs = list(capture["metrics"].items())
    else:
        specs = [(m["cell"], m) for m in capture["metrics"]]
    for key, metric in specs:
        identifier = metric.get("cell", key)
        expected_archive = audit["archive_sha256"]
        if schema == MODE and key != "@candidate":
            expected_archive = audit["parent_archive_sha256"]
        if schema == VA_REDUCED and identifier == "va-2v4a-fp8":
            expected_archive = audit["parent_archive_sha256"]
        archive = archive_map[expected_archive]
        candidates = [name for name, obj in archive.objects.items() if isinstance(obj, dict)
                      and obj.get("cell_id") == identifier and obj.get("family") == family and "sources" in obj
                      and ("receipt_sha256" not in metric or sha(archive.files[name]) == metric["receipt_sha256"])]
        require(len(candidates) == 1, "Expected one raw cell receipt for " + identifier)
        pair = raw_cell(archive, candidates[0], metric, profiles[family])
        pair[0]["audit_sha256"] = sha(audit_data)
        pair[0]["source_commit"] = audit.get("source_commit")
        bind_cohort(pair, cohort_id, context)
        if pair[0]["mode"] != "pytorch":
            pair[0]["serving"] = serving_for(pair[0], [archive] + [a for a in archives if a is not archive], context)
        if schema == MODE and key == "@candidate":
            require(pair[0]["mode"] == audit["mode"] == metric["mode"], "Mode identity differs from raw recipe")
        rows[key] = pair
    if schema == MODE:
        comparisons = [compare(rows[key], rows["@candidate"], item) for key, item in audit["comparisons"].items()]
    else:
        comparisons = [compare(rows[item["baseline"]], rows[item["candidate"]], item)
                       for item in capture["comparisons"]]
    return [pair[0] for pair in rows.values()], comparisons, family if schema in (BASE, COMPLETE) else None


def admit_capacity_scope(manifest, base, profiles, profile_data, evidence):
    if manifest["schema"] != INPUT_V3:
        require("requested_scope" not in manifest and "capacity_exclusions" not in manifest,
                "Capacity exclusions require explicit scoped input v3")
        return list(FAMILIES), [], None
    scope = manifest["requested_scope"]
    requested = scope["families"]
    require(isinstance(requested, list) and requested and all(isinstance(f, str) for f in requested)
            and len(requested) == len(set(requested)) and set(requested) <= set(FAMILIES),
            "Invalid requested family scope")
    entries = manifest["capacity_exclusions"]
    require(isinstance(entries, list) and len(entries) == 1,
            "Only the reviewed single Nano family exclusion is admitted")
    entry = entries[0]
    report_data = load_bound(base, entry["report"])
    require(sha(report_data) == CAPACITY_REPORT_SHA,
            "Unreviewed capacity report; prepare an additive reviewed exclusion")
    report = json.loads(report_data)
    require(entry["family"] == report["family"] == "nano"
            and report["schema"] == "instinctflash.capacity_exclusion.v1"
            and report["deployment_target"] == "rtx4090"
            and report["scope"] == "entire_family_on_bound_host"
            and report["status"] == "memory_capacity_excluded_not_tested",
            "Wrong capacity exclusion scope")
    require(set(report["excluded_modes"]) == REQUIRED_MODES["nano"]
            and set(report["excluded_serving_modes"]) == REQUIRED_MODES["nano"]
            and set(requested) == set(FAMILIES) - {"nano"}, "Requested scope differs from reviewed exclusion")
    for key in ("tested", "passed", "GPU_used", "observed_OOM", "unsupported",
                "changes_to_runtime_or_model", "qualification_evidence"):
        require(report[key] is False, "Capacity exclusion cannot claim a measurement or failure")
    change = scope["change_source"]
    require(change == {"kind": "user_instruction_in_active_task",
                       "text": report["scope_authorization"]["user_instruction"]},
            "Scope change must retain the actual user instruction source")
    require(report["scope_authorization"]["native_impossibility_claimed"] is False,
            "Native capacity exclusion is not a proof of inevitable OOM")
    hardware, frozen = report["hardware"], report["frozen_inputs"]
    refs = {"identity": hardware["identity_evidence"], "memory": hardware["memory_evidence"],
            "bootstrap": frozen["bootstrap_profile"], "profile": frozen["deployment_catalog"],
            "source_review": frozen["source_review"], "tensor_headers": frozen["tensor_header_evidence"]}
    require(set(entry["support"]) == set(refs), "Capacity supporting evidence is incomplete")
    objects, support = {}, {}
    for role, expected in refs.items():
        spec = entry["support"][role]
        data = load_bound(base, spec)
        require(sha(data) == expected["sha256"], "Capacity supporting evidence binding differs")
        objects[role] = json.loads(data)
        support[role] = {"sha256": sha(data), "path": "evidence/" + sha(data) + ".json"}
        evidence.setdefault(sha(data), (".json", data))
    require(sha(profile_data) == frozen["deployment_catalog"]["sha256"]
            and frozen["checkpoint"] == profiles["nano"]["checkpoint"]
            and frozen["execution_modes"] == profiles["nano"]["execution_modes"],
            "Capacity evidence applies to a different original checkpoint/profile")
    identity, memory = objects["identity"], objects["memory"]
    devices = list(csv.reader(io.StringIO(identity["gpus"]), skipinitialspace=True))
    require(identity["hostname"] == hardware["hostname"] and identity["boot_id"] == hardware["boot_id"]
            and len(devices) == 1 and gpu_uuid(devices[0][1]) == gpu_uuid(hardware["gpu_uuid"])
            and devices[0][2] == "NVIDIA GeForce RTX 4090" and devices[0][3] == hardware["driver"],
            "Capacity hardware identity differs")
    limit = int(memory[hardware["memory_limit_source_field"]])
    require(limit == hardware["memory_limit_bytes"] > 0 and hardware["cgroup_version"] == 1,
            "Capacity memory limit is not bound to the actual cgroup observation")
    headers, review = objects["tensor_headers"], objects["source_review"]
    require(headers["asset_set_sha256"] == review["asset_set_sha256"] == frozen["original_asset_set_sha256"]
            and review["source_commit"] == frozen["source_commit"]
            and review["header_receipt_sha256"] == support["tensor_headers"]["sha256"],
            "Static tensor/source evidence differs")
    transformer = [row for row in headers["tensor_headers"] if row["file"].startswith("transformer/")]
    require(all(row["dtype"] == "BF16" and math.prod(row["shape"]) * 2 == row["bytes"]
                for row in transformer), "Original BF16 tensor header arithmetic differs")
    heads = [row for row in transformer if row["key"] == "lm_head.weight"]
    require(len(heads) == 1, "Pinned elided head identity differs")
    native = sum(row["bytes"] for row in transformer) - heads[0]["bytes"]
    projections = [row for row in transformer if re.fullmatch(
        r"layers\.\d+\.(?:mlp(?:_moe_gen)?\.(?:down|gate|up)_proj|self_attn\.(?:to_[qkv]|add_[qkv]_proj))\.weight",
        row["key"])]
    packed = sum(math.prod(row["shape"]) for row in projections)
    basis = report["capacity_basis"]
    require(len(projections) == 432 and native == basis["native_CPU_parameter_bytes"]
            == basis["FP8_retained_native_parameter_bytes"]
            and packed == basis["FP8_additional_packed_weight_bytes"]
            and native + packed == basis["FP8_live_weight_payload_lower_bound_bytes"] > limit
            and native + packed - limit == basis["FP8_lower_bound_over_memory_limit_bytes"]
            and limit - native == basis["native_remaining_for_all_other_memory_bytes"],
            "Static capacity bounds do not reproduce the admitted evidence")
    evidence.setdefault(sha(report_data), (".json", report_data))
    exclusion = {"family": "nano", "status": "capacity_excluded_not_tested", "tested": False,
                 "passed": False, "observed_OOM": False, "unsupported": False,
                 "excluded_modes": sorted(REQUIRED_MODES["nano"]),
                 "report": {"sha256": sha(report_data), "path": "evidence/" + sha(report_data) + ".json"},
                 "support": support, "gpu_uuid": gpu_uuid(hardware["gpu_uuid"]),
                 "driver": hardware["driver"], "memory_limit_bytes": limit,
                 "native_CPU_parameter_bytes": native, "native_fit_measured": False,
                 "FP8_live_weight_payload_lower_bound_bytes": native + packed,
                 "scope_authorization": report["scope_authorization"]}
    return requested, [exclusion], change


def apply_requested_scope(result, requested, exclusions, change, profiles):
    excluded = {item["family"] for item in exclusions}
    tested = {row["family"] for row in result["cells"]}
    require(not tested & excluded, "A measured family cannot be hidden by capacity exclusion")
    require(tested <= set(requested), "Measured cells fall outside the explicitly requested scope")
    pending = [family for family in requested if not result["coverage"][family]["complete"]]
    mode_count = sum(len(REQUIRED_MODES[family]) for family in requested)
    required_cells = mode_count + len(requested) + int("va" in requested)
    actual_modes = sum(row["mode"] != "pytorch" for row in result["cells"])
    require(not result["full_qualification"] or
            (actual_modes == sum(map(len, REQUIRED_MODES.values()))
             and len(result["cells"]) == sum(map(len, REQUIRED_MODES.values())) + len(FAMILIES) + 1),
            "Full catalog API/mode count differs")
    scope_missing = [item for item in result["missing"] if item.split("/", 1)[0] in requested]
    if not pending and (actual_modes != mode_count or len(result["cells"]) != required_cells):
        scope_missing.append("requested scope API/mode count differs")
    complete = not scope_missing
    result.update(requested_scope_qualification=complete, requested_scope_missing=scope_missing,
                  capacity_exclusions=exclusions, scope_change_source=change,
                  scope_coverage={"catalog_families": list(FAMILIES), "requested_families": requested,
                                  "tested_families": sorted(tested), "excluded_families": sorted(excluded),
                                  "pending_requested_families": pending,
                                  "completed_requested_families": sorted(set(requested) - set(pending)),
                                  "catalog_required_runtime_modes": sum(map(len, REQUIRED_MODES.values())),
                                  "requested_required_runtime_modes": mode_count, "tested_runtime_modes": actual_modes,
                                  "catalog_required_api_cells": sum(map(len, REQUIRED_MODES.values())) + len(FAMILIES) + 1,
                                  "requested_required_api_cells": required_cells, "tested_api_cells": len(result["cells"])})
    main = {row["family"]: row for row in result["main_table"]}
    result["catalog_table"] = []
    for family in FAMILIES:
        row = main.get(family)
        if row:
            result["catalog_table"].append({**row, "status": "measured", "tested": True})
        else:
            result["catalog_table"].append({"family": family, "label": profiles[family]["label"],
                                             "status": "capacity_excluded_not_tested" if family in excluded else "pending",
                                             "tested": False, "pytorch_p50_ms": None, "instinctflash_p50_ms": None,
                                             "comparison": None, "mode": None})


def cgroup_summary(obj):
    v2 = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/cpu.max")
    memory_v1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    cpu_v1 = ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct")
    has_v2 = any(path in obj for path in v2)
    has_v1 = memory_v1 in obj or any(root + "/cpu.cfs_quota_us" in obj or
                                    root + "/cpu.cfs_period_us" in obj for root in cpu_v1)
    require(has_v2 != has_v1, "Missing or ambiguous cgroup context")
    if has_v2:
        require(all(path in obj for path in v2), "Incomplete cgroup v2 context")
        memory = obj[v2[0]]
        quota, period = obj[v2[1]].split()
        limit = None if memory == "max" else int(memory)
        quota = None if quota == "max" else int(quota)
        observed = {path: obj[path] for path in v2}
        version = 2
    else:
        require(memory_v1 in obj, "Incomplete cgroup v1 memory context")
        roots = [root for root in cpu_v1 if root + "/cpu.cfs_quota_us" in obj or
                 root + "/cpu.cfs_period_us" in obj]
        require(roots, "Incomplete cgroup v1 CPU context")
        observed = {memory_v1: obj[memory_v1]}
        pairs = []
        for root in roots:
            keys = (root + "/cpu.cfs_quota_us", root + "/cpu.cfs_period_us")
            require(all(path in obj for path in keys), "Incomplete cgroup v1 CPU context")
            pairs.append(tuple(int(obj[path]) for path in keys))
            observed.update({path: obj[path] for path in keys})
        require(len(set(pairs)) == 1, "Conflicting cgroup v1 CPU aliases")
        quota, period = pairs[0]
        quota = None if quota == -1 else quota
        limit, version = int(obj[memory_v1]), 1
    period = int(period)
    require((limit is None or limit > 0) and period > 0 and (quota is None or quota > 0),
            "Invalid cgroup limit values")
    return {"version": version, "memory_limit_bytes": limit,
            "cpu_quota": None if quota is None else quota / period, "observed_paths": observed}


def machine_summary(obj):
    require(obj["gpu_query_exit"] == 0, "Hardware inventory failed")
    devices = list(csv.reader(io.StringIO(obj["gpu_csv"]), skipinitialspace=True))
    require(devices and all("4090" in row[2] for row in devices), "Hardware inventory is not RTX4090")
    variants = sorted({tuple(row[2:4] + row[5:]) for row in devices})
    require(len(variants) == 1, "Heterogeneous hardware inventory requires separate reports")
    gpu, memory, pcie_gen, pcie_width, power, driver = variants[0]
    physical = int(obj["host_memory_total"].split()[0]) * 1024
    cgroup = cgroup_summary(obj)
    effective = min(physical, cgroup["memory_limit_bytes"]) if cgroup["memory_limit_bytes"] is not None else physical
    return {"gpu": gpu, "gpu_memory_mib": int(memory), "gpus_per_model_run": 1,
            "available_device_count": len(devices), "pcie_max_generation": int(pcie_gen),
            "pcie_max_width": int(pcie_width), "power_limit_w": float(power), "driver": driver,
            "platform": obj["platform"], "physical_host_memory_bytes": physical,
            "effective_cgroup_memory_bytes": effective, "logical_cpu_count": obj["logical_cpu_count"],
            "cpu_affinity_count": obj["cpu_affinity_count"],
            "cgroup_cpu_quota": cgroup["cpu_quota"],
            "minimum_host_memory_measured": False, "peak_host_memory_measured": False}


def assemble(rows, comparisons, profiles, main_families):
    unique = {}
    for row in rows:
        previous = unique.get(row["id"])
        if previous:
            require(all(previous[k] == row[k] for k in ("p50_ms", "mode", "receipt", "actions")),
                    "Conflicting duplicate row")
            previous["audit_sha256s"] = sorted(set(previous["audit_sha256s"] + [row["audit_sha256"]]))
            if not previous["source_commit"]:
                previous["source_commit"] = row["source_commit"]
        else:
            unique[row["id"]] = {**row, "audit_sha256s": [row["audit_sha256"]]}
    by_family_mode = {}
    for row in unique.values():
        if row["mode"] != "pytorch":
            key = (row["family"], row["mode"])
            require(key not in by_family_mode, "Multiple captures for one mode require an explicit curated input manifest")
            by_family_mode[key] = row
    seen = set()
    paired = []
    for comparison in comparisons:
        key = (comparison["baseline"], comparison["candidate"])
        if key not in seen:
            seen.add(key)
            paired.append(comparison)
    main = []
    optional = []
    missing = []
    coverage = {}
    for family in FAMILIES:
        native_schedule = profiles[family]["execution_modes"]["native"]["effective_schedule"]
        baselines = [row for row in unique.values() if row["family"] == family and row["mode"] == "pytorch"
                     and row["effective_schedule"] == native_schedule]
        require(len(baselines) <= 1, "Multiple checkpoint-schedule baselines")
        candidate_pairs = []
        family_missing = []
        if family not in main_families:
            family_missing.append("missing independent main pipeline audit")
        for mode in sorted(REQUIRED_MODES[family]):
            row = by_family_mode.get((family, mode))
            if row is None:
                family_missing.append(mode + ": missing audited measurement")
                continue
            links = [c for c in paired if c["candidate"] == row["id"]
                     and unique[c["baseline"]]["mode"] == "pytorch"]
            if not links:
                family_missing.append(mode + ": missing paired PyTorch comparison")
            if mode.startswith("2v4a-") and not any(c["same_sampling_policy"] for c in links):
                family_missing.append(mode + ": missing same-schedule PyTorch 2V/4A comparison")
            if not row["serving"]:
                family_missing.append(mode + ": missing raw WebSocket/reset evidence")
            for link in links:
                baseline = unique[link["baseline"]]
                record = {"family": family, "label": profiles[family]["label"], "mode": mode,
                          "pytorch_p50_ms": baseline["p50_ms"], "instinctflash_p50_ms": row["p50_ms"],
                          "comparison": link, "baseline": baseline["id"], "candidate": row["id"],
                          "baseline_schedule": baseline["effective_schedule"],
                          "candidate_schedule": row["effective_schedule"], "torch": row["torch"]}
                if baselines and baseline["id"] == baselines[0]["id"] and row["effective_schedule"] == native_schedule:
                    candidate_pairs.append(record)
                elif (family, mode) == ("va", "2v4a-fp8") and not link["same_sampling_policy"]:
                    # Preserve this historical cross-schedule comparison in JSON only.
                    continue
                else:
                    optional.append(record)
        if not baselines:
            family_missing.append("missing original-checkpoint-schedule PyTorch baseline")
        if candidate_pairs:
            main.append(min(candidate_pairs, key=lambda value: (value["instinctflash_p50_ms"], value["mode"])))
        if family == "va" and not any(x["family"] == "va" and x["mode"] == "2v4a-native"
                                      and x["comparison"]["same_sampling_policy"] for x in optional):
            family_missing.append("missing actual PyTorch 2V/4A baseline")
        coverage[family] = {"required_modes": sorted(REQUIRED_MODES[family]), "complete": not family_missing,
                            "missing": family_missing}
        missing.extend(family + "/" + value for value in family_missing)
    return {"full_qualification": not missing, "coverage": coverage, "missing": missing,
            "main_table": main, "optional_table": optional, "cells": list(unique.values()), "comparisons": paired}


def csv_table(rows, headers):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return ".. csv-table::\n   :header: " + ", ".join('"' + h + '"' for h in headers) + "\n\n" + "".join(
        "   " + line + "\n" for line in stream.getvalue().splitlines()) + "\n"


def rst_report(result):
    multiple = len(result["cohorts"]) > 1
    status = "Complete requested pipeline and mode coverage" if result["full_qualification"] else "Partial results; full qualification is pending"
    if result.get("capacity_exclusions"):
        status = ("Complete requested scope; full-eight qualification remains incomplete" if result["requested_scope_qualification"]
                  else "Partial requested-scope results; full-eight qualification remains incomplete")
    text = "RTX 4090 measured results\n=========================\n\n" + status + ".\n\n"
    text += ("Each model run used one NVIDIA GeForce RTX 4090. Latencies are median public Runtime prediction time "
             "in milliseconds; the main table selects the fastest audited runtime at the original checkpoint schedule. "
             "History models use steady history predictions; setup and WebSocket round trips are outside these medians.\n\n")
    if multiple:
        text += ("Measurements span separate host/environment cohorts. Each speedup compares paired runs within "
                 "its own cohort; this table is not a comparison between hosts. The cohort column and environment "
                 "records below identify the driver, allocation and measured Torch builds.\n\n")
    display = result["catalog_table"] if result.get("capacity_exclusions") else result["main_table"]
    table = []
    for row in display:
        label = [row["label"], *([row.get("cohort", "—")] if multiple else [])]
        if row.get("status") in ("capacity_excluded_not_tested", "pending"):
            table.append(label + ["—", "Not measured: capacity excluded" if row["status"] == "capacity_excluded_not_tested" else "Pending"])
        else:
            table.append(label + [f'{row["pytorch_p50_ms"]:.2f}',
                                  f'{row["instinctflash_p50_ms"]:.2f} ({row["comparison"]["ratio_of_p50"]:.2f}×; {row["mode"]})'])
    text += csv_table(table, ["Model", *(["Cohort"] if multiple else []), "PyTorch p50 (ms)", "InstinctFlash p50 (ms)"])
    for item in result.get("capacity_exclusions", []):
        text += (f'Cosmos3 Nano was not tested on the capacity-assessed node (driver {item["driver"]}): '
                 f'its cgroup host-memory limit was {item["memory_limit_bytes"] / 10**9:.2f} GB. '
                 f'The current FP8 loading path requires at least {item["FP8_live_weight_payload_lower_bound_bytes"] / 10**9:.2f} GB '
                 f'of original and packed CPU weights. Native CPU weights require {item["native_CPU_parameter_bytes"] / 10**9:.2f} GB; '
                 'the remaining margin was not validated. The family was excluded without an OOM trial under the requested scope. '
                 'This is not an observed OOM or an unsupported-GPU result; no excluded mode counts as tested or passed. '
                 'The exact hardware, original checkpoint, static tensor evidence and scope instruction are bound in results.json.\n\n')
    if result["optional_table"]:
        text += "Changed computation\n-------------------\n\n"
        text += ("These modes do not replace the main checkpoint-schedule measurements. VA 2V/4A uses its measured "
                 "PyTorch 2V/4A baseline. Dynamic-cache comparisons are labeled when their PyTorch baseline uses "
                 "the checkpoint cache schedule. Equal declared adaptive policy can produce different actual "
                 "compute/reuse decisions; these are whole-runtime comparisons. Original per-call traces remain "
                 "in the raw receipts. Native precision alone does not establish byte equality.\n\n")
        text += csv_table([[r["label"] + " / " + r["mode"], *([r["cohort"]] if multiple else []), f'{r["pytorch_p50_ms"]:.2f}',
                            f'{r["instinctflash_p50_ms"]:.2f} ({r["comparison"]["ratio_of_p50"]:.2f}×)',
                            "same declared sampling policy" if r["comparison"]["same_sampling_policy"] else "checkpoint baseline; changed computation"]
                           for r in result["optional_table"]], ["Mode", *(["Cohort"] if multiple else []), "PyTorch p50 (ms)", "InstinctFlash p50 (ms)", "Comparison"])
    for identifier, cohort in result["cohorts"].items():
        hardware = cohort["hardware"]
        text += (f'Cohort ``{identifier}``: {", ".join(cohort["families"])}. '
             f'Tested PyTorch builds: {", ".join(cohort["torch_versions"])}. '
             f'Host inventory: {hardware["available_device_count"]} GPU' +
             ('s' if hardware["available_device_count"] != 1 else '') + '; each model used one. ' +
             f'GPU memory from the host inventory: {hardware["gpu_memory_mib"]} MiB; '
             f'driver {hardware["driver"]}; PCIe maximum Gen{hardware["pcie_max_generation"]} x{hardware["pcie_max_width"]}; '
             f'power limit {hardware["power_limit_w"]:g} W. Effective cgroup RAM allocation: '
             f'{hardware["effective_cgroup_memory_bytes"]} bytes ({hardware["effective_cgroup_memory_bytes"] / 1024**3:.2f} GiB); '
             f'physical host RAM: {hardware["physical_host_memory_bytes"] / 1024**3:.2f} GiB. '
             f'CPU quota: {hardware["cgroup_cpu_quota"]}; logical CPUs: {hardware["logical_cpu_count"]}. '
             "This records this cohort's tested allocation; minimum and peak host RAM were not measured.\n\n")
    if result["missing"]:
        pending = result.get("scope_coverage", {}).get("pending_requested_families")
        if pending is None:
            pending = [family for family, item in result["coverage"].items() if not item["complete"]]
        if pending:
            text += "Pending requested pipeline/mode coverage: " + ", ".join(pending) + ". See ``results.json`` for the exact missing evidence.\n\n"
    text += ("Action differences are empirical comparisons on the recorded inputs. These results do not validate robot task "
             "quality. ``results.json`` records every admitted cell, exact schedule, loaded-source hashes, raw receipt/action "
             "member hashes and archive hashes. The adjacent ``evidence/`` directory contains the original, unmodified bytes.\n")
    return text


def render(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    require(not output.exists(), "Output must be a new directory")
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    require(manifest["schema"] in (INPUT_V1, INPUT_V2, INPUT_V3), "Wrong input schema")
    require(manifest["target"] == "rtx4090", "Wrong target")
    base = manifest_path.parent
    profile_data = load_bound(base, manifest["target_profiles"])
    profile_document = json.loads(profile_data)
    require(profile_document["target"] == "rtx4090", "Target profiles are not for RTX4090")
    profiles = {p["id"]: p for p in profile_document["models"]}
    require(set(profiles) == set(FAMILIES), "Target profile must contain all eight families")
    for family in FAMILIES:
        require(REQUIRED_MODES[family] <= set(profiles[family]["execution_modes"]), "Required target mode missing")
    evidence = {sha(profile_data): (".json", profile_data)}
    requested, exclusions, scope_change = admit_capacity_scope(manifest, base, profiles, profile_data, evidence)
    contexts, legacy = load_contexts(manifest, base, evidence)
    rows, comparisons, main_families = [], [], set()
    for entry in manifest["audits"]:
        added, pairs, main_family = ingest(entry, base, profiles, evidence, contexts, legacy)
        rows.extend(added)
        comparisons.extend(pairs)
        if main_family:
            require(main_family not in main_families, "Duplicate independent main pipeline audit")
            main_families.add(main_family)
    row_contexts = {}
    for row in rows:
        binding = (row["cohort"], row["machine_context_sha256"], row["measured_gpu_uuid"])
        require(row["id"] not in row_contexts or row_contexts[row["id"]] == binding,
                "The same measured cell was assigned conflicting cohort contexts")
        row_contexts[row["id"]] = binding
    require(all(context["cell_ids"] for context in contexts.values()),
            "Declared cohort has no actual audited cells; future placeholders are forbidden")
    result = assemble(rows, comparisons, profiles, main_families)
    cells_by_id = {cell["id"]: cell for cell in result["cells"]}
    for row in result["main_table"] + result["optional_table"]:
        candidate = cells_by_id[row["candidate"]]
        row.update(cohort=candidate["cohort"], machine_context_sha256=candidate["machine_context_sha256"])
    apply_requested_scope(result, requested, exclusions, scope_change, profiles)
    for cell in result["cells"]:
        source_data = encoded(cell["source_inventory"])
        checksum = sha(source_data)
        evidence.setdefault(checksum, (".json", source_data))
        cell["source_inventory"] = {"sha256": checksum, "path": "evidence/" + checksum + ".json"}
    renderer_data = Path(__file__).read_bytes()
    evidence[sha(renderer_data)] = (".py", renderer_data)
    evidence[sha(raw_manifest)] = (".json", raw_manifest)
    replay = json.loads(raw_manifest)
    specs = ([replay["machine_context"]] if legacy else list(replay["machine_contexts"].values())) + [replay["target_profiles"]]
    for entry in replay["audits"]:
        specs.extend([entry["audit"], *entry["archives"]])
    for entry in replay.get("capacity_exclusions", []):
        specs.extend([entry["report"], *entry["support"].values()])
    for spec in specs:
        spec["path"] = "evidence/" + spec["sha256"] + evidence[spec["sha256"]][0]
    replay_data = json.dumps(replay, indent=2, sort_keys=True).encode() + b"\n"
    public_contexts = {key: {**{name: value for name, value in context.items() if not isinstance(value, set)},
                             "families": sorted(context["families"]), "torch_versions": sorted(context["torch_versions"]),
                             "source_commits": sorted(context["source_commits"]), "api_cell_count": len(context["cell_ids"])}
                       for key, context in sorted(contexts.items())}
    single = next(iter(public_contexts.values())) if len(public_contexts) == 1 else None
    result.update({"schema": "instinctflash.rtx4090.public_results.v3", "target": TARGET,
                   "scope": "Audited saved public API/WebSocket and requested mode coverage; not task-quality certification",
                   "cohorts": public_contexts, "single_hardware_context": single is not None,
                   "hardware": single["hardware"] if single else None, "input_manifest_sha256": sha(raw_manifest),
                   "renderer_source_sha256": sha(renderer_data),
                   "reproduce_manifest_sha256": sha(replay_data),
                   "target_profiles_sha256": sha(profile_data),
                   "machine_context_sha256": single["machine_context_sha256"] if single else None,
                   "machine_context_sha256s": {key: value["machine_context_sha256"] for key, value in public_contexts.items()},
                   "task_quality_validated": False, "minimum_host_memory_measured": False,
                   "evidence": [{"sha256": checksum, "bytes": len(data), "path": "evidence/" + checksum + extension}
                                for checksum, (extension, data) in sorted(evidence.items())]})
    # Finish all admission before creating output; never rewrite an existing result.
    output.mkdir(parents=True)
    (output / "evidence").mkdir()
    for checksum, (extension, data) in evidence.items():
        (output / "evidence" / (checksum + extension)).write_bytes(data)
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "results.rst").write_text(rst_report(result))
    (output / "render_results.py").write_bytes(renderer_data)
    (output / "reproduce_manifest.json").write_bytes(replay_data)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument("--require-full", action="store_true", help="Require original full-eight/23-mode/32-API coverage")
    gate.add_argument("--require-requested", action="store_true", help="Require all requested coverage with only reviewed capacity exclusions")
    args = parser.parse_args(argv)
    try:
        result = render(args.manifest, args.output)
    except (ValueError, KeyError, OSError, tarfile.TarError) as error:
        print("Evidence admission failed: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"full_qualification": result["full_qualification"],
                      "requested_scope_qualification": result["requested_scope_qualification"], "main_rows": len(result["main_table"]),
                      "optional_rows": len(result["optional_table"]), "missing": result["missing"]}))
    incomplete = ((args.require_full and not result["full_qualification"])
                  or (args.require_requested and not result["requested_scope_qualification"]))
    return 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())

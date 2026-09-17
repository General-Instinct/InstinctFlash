"""Portable RTX5090 saved-results reader; relative SHA references, no models or network."""
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

FAMILIES=("va","vla4","vla2","pi05","groot","edge","nano","dreamzero")
REQUIRED_MODES={f:{"native","fp8"} for f in FAMILIES}
REQUIRED_MODES["va"]|={"2v4a-native","2v4a-fp8"}
for _f in ("vla2","edge","nano"):REQUIRED_MODES[_f].add("numeric")
REQUIRED_MODES["dreamzero"]|={"dynamic-native","dynamic-fp8"}
TARGET={"name":"rtx5090","capability":[12,0]}
INPUT_SCHEMA="instinctflash.rtx5090.results_input.v1"
NUMERIC_SCHEMA="instinctflash.rtx5090_numeric_independent_audit.v1"
VA_SCHEMA="instinctflash.rtx5090_va_2v4a_independent_audit.v1"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

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

def check_context_hardware(hardware, context):
    require(hardware.get("status") == "passed" and hardware.get("target") == TARGET
            and hardware.get("capability") == [12, 0]
            and hardware.get("name") == "NVIDIA GeForce RTX 5090",
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
    require(receipt["target"] == TARGET and receipt["device"] == "NVIDIA GeForce RTX 5090", "Receipt is not actual RTX5090")
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
                            for cell in obj["cells"]) for obj in archive.objects.values())
                or any(isinstance(obj, dict) and obj.get("status") == "passed" and obj.get("mode") == "numeric"
                       and obj.get("cell_id") == receipt["cell_id"] and obj.get("receipt_sha256") == sha(archive.files[name])
                       for obj in archive.objects.values()),
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
    require(lrow["numeric_environment"] == rrow["numeric_environment"], "Paired native numerical settings differ")
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
            require(obj["target"] == TARGET and obj["hardware"]["capability"] == [12, 0], "Serving GPU differs")
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
    text = "RTX 5090 measured results\n=========================\n\n" + status + ".\n\n"
    text += ("Each model run used one NVIDIA GeForce RTX 5090. Latencies are median public Runtime prediction time "
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
        basis=item["capacity_basis"]
        text += (f'{item["label"]} was not tested on the bound host with '
                 f'a {item["memory_limit_bytes"] / 10**9:.2f} GB cgroup allocation. '
                 f'Original plus packed weight payload totals {basis["combined_payload_bytes"] / 10**9:.2f} GB. '
                 'Mapped or reclaimable pages mean this arithmetic is not a proved nonreclaimable-RAM lower bound. '
                 'Reliable native cold-load headroom was not established. All four modes were excluded under '
                 'the requested skip policy without an OOM trial; none counts as tested or passed. '
                 'This is not an unsupported-GPU finding or a minimum-RAM measurement. '
                 'The unchanged exclusion and exact catalog mapping are included.\n\n')
    if result["optional_table"]:
        text += "Additional declared schedules\n-----------------------------\n\n"
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
                 f'Actual GPU UUID: {hardware["uuid"]}; '
                 f'device-reported GPU memory: {hardware["total_memory_bytes"]} bytes. '
                 f'Measured Torch builds: {", ".join(cohort["torch_versions"])}. '
                 'The original saved device receipt is included. This receipt establishes neither '
                 'a host-RAM minimum nor a whole-machine memory peak.\n\n')
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

def load_bound(base, spec):
    require(isinstance(spec,dict) and valid_sha(spec.get('sha256')), 'Invalid relative SHA reference')
    name=spec.get('path');require(isinstance(name,str),'Missing relative input path')
    rel=PurePosixPath(name)
    require(name and not rel.is_absolute() and '..' not in rel.parts and '\\' not in name and ':' not in name,
            'Input reference must be a plain relative path')
    path=(base/name).resolve(strict=True)
    require(path.is_relative_to(base.resolve()),'Input reference escapes its evidence directory')
    data=path.read_bytes();require(sha(data)==spec['sha256'],'Input byte hash differs: '+rel.name)
    return data


def ingest(entry, base, profiles, evidence, contexts):
    family=entry['family'];kind=entry['kind'];require(family in FAMILIES and kind in ('primary','numeric','va_2v4a'),'Unknown family/evidence kind')
    cohort=entry['cohort'];require(cohort in contexts,'Unknown actual cohort');context=contexts[cohort]
    def bound(spec, extension):
        data=load_bound(base,spec);evidence.setdefault(sha(data),(extension,data));return data
    data=bound(entry['audit'],'.json');report=json.loads(data)
    schema=('instinctflash.rtx5090_'+family+'_independent_audit.v1' if kind=='primary' else NUMERIC_SCHEMA if kind=='numeric' else VA_SCHEMA)
    require(report.get('passed') is True and report['schema']==schema and report['family']==family
            and report['target']==TARGET and report['source_commit']==context['source_commit']
            and gpu_uuid(report['GPU_uuid']) in context['gpu_uuids'],'Wrong actual independent audit/source/cohort')
    require(kind!='numeric' or family in ('vla2','edge','nano'),'Unsupported numeric family')
    require(kind!='va_2v4a' or family=='va','Wrong reduced VA family')
    archives=[Archive(bound(spec,'.tar.gz')) for spec in entry['archives']]
    archive_map={a.sha256:a for a in archives};require(len(archive_map)==len(archives),'Duplicate explicit archive reference')
    archive=archive_map[report['archive_sha256']]
    def terminal(a,r):
        inventory=a.objects.get('provenance/archive_manifest.json');require(inventory is not None,'Mandatory original member inventory missing')
        require(inventory['source_commit']==r['source_commit'] and inventory['source_completion_sha256']==r['completion_sha256'],'Actual terminal/source manifest differs')
        require(r['completion_sha256'] in a.by_sha and r['config_sha256'] in a.by_sha,'Actual source/config terminal bytes missing')
        terminal_member=PurePosixPath(inventory['source_run']).name+'/completion.json'
        require(terminal_member in a.objects and sha(a.files[terminal_member])==r['completion_sha256'],
                'The original source-run completion member is missing')
        done=a.objects[terminal_member];require(done['passed'] is True and done['lease_released'] is True
                and done['owned_processes_terminal'] is True and done['source_commit']==r['source_commit']
                and done['config_sha256']==r['config_sha256'] and done['worker_pid']==r['worker']['pid']
                and done['worker_start_ticks']==r['worker']['start_ticks'] and r['worker']['exited'] is True,
                'Actual complete process/lease/source boundary differs')
    terminal(archive,report)
    parent_report=parent_archive=None
    if kind!='primary':
        parent_data=bound(entry['parent_audit'],'.json');parent_report=json.loads(parent_data)
        key='parent_independent_audit_sha256' if kind=='va_2v4a' else 'parent_audit_sha256'
        require(sha(parent_data)==report[key] and parent_report['passed'] is True
                and parent_report['schema']=='instinctflash.rtx5090_'+family+'_independent_audit.v1'
                and parent_report['family']==family and parent_report['source_commit']==report['source_commit']
                and parent_report['GPU_uuid']==report['GPU_uuid'],'Supplement does not bind its actual qualified parent')
        require(parent_report['archive_sha256']==report['parent_archive_sha256'],'Parent original archive differs')
        parent_archive=archive_map[report['parent_archive_sha256']];terminal(parent_archive,parent_report)
        require(report['primary_baselines_recaptured'] is False,'Supplement recaptured original baselines')
    require(set(archive_map)=={report['archive_sha256'],*([report['parent_archive_sha256']] if kind!='primary' else [])},'Uncurated extra archive input')
    def cell(a,identifier,metric,audit):
        names=[n for n,o in a.objects.items() if isinstance(o,dict) and o.get('family')==family and o.get('cell_id')==identifier and 'sources' in o]
        require(len(names)==1,'Missing/ambiguous complete raw cell '+identifier)
        pair=raw_cell(a,names[0],metric,profiles[family]);row=pair[0]
        row.update(audit_sha256=sha(encoded(audit)),source_commit=audit['source_commit'])
        # Bind the original audit bytes, never the normalized JSON representation.
        row['audit_sha256']=sha(data) if audit is report else sha(parent_data)
        bind_cohort(pair,cohort,context)
        if row['mode']!='pytorch':
            row['serving']=serving_for(row,[a],context)
            expected=[w for w in audit['websockets'] if w['cell']==identifier]
            require(len(expected)==len(row['serving'])==1,'Exactly one independently audited serving process required per mode')
            w=expected[0];actual=row['serving'][0]
            require(actual['receipt']['sha256']==w['receipt_sha256'] and actual['actions']['sha256']==w['action_archive_sha256']
                    and w['calls']==6 and w['episodes']==2 and w['finite_actions'] is True,'Raw serving differs from independent audit')
            with np.load(io.BytesIO(a.files[actual['actions']['member']]),allow_pickle=False) as saved:
                require(list(saved['actions'].shape)==w['action_shape'],'Full serving action shape differs from audit')
        return pair
    pairs={}
    if kind=='numeric':
        for identifier,metric in parent_report['metrics'].items():pairs[identifier]=cell(parent_archive,identifier,metric,parent_report)
        pairs['@candidate']=cell(archive,family+'-runtime_selected',report['metrics'],report)
        require(pairs['@candidate'][0]['mode']==report['mode']=='numeric','Actual NUMERIC recipe differs')
        comparisons=[]
        require({c['baseline'] for c in report['comparisons']}==set(parent_report['metrics']),'NUMERIC does not compare every original parent arm')
        for c in report['comparisons']:
            require(c['candidate']==family+'-numeric-supplement/runtime_selected','Wrong supplemental comparison identity')
            comparisons.append(compare(pairs[c['baseline']],pairs['@candidate'],c))
    else:
        for identifier,metric in report['metrics'].items():pairs[identifier]=cell(archive,identifier,metric,report)
        require(len(pairs)==3,'Original primary or reduced VA audit must have three actual cells')
        if kind=='va_2v4a':
            require(set(pairs)=={'va-eager_native-2v4a','va-2v4a-native','va-2v4a-fp8'}
                    and report['actual_new_native_baseline']=='va-eager_native-2v4a','Reduced VA needs its actual new native baseline')
        comparisons=[compare(pairs[c['baseline']],pairs[c['candidate']],c) for c in report['comparisons']]
    return [p[0] for p in pairs.values()],comparisons,family if kind=='primary' else None


def admit_requested_scope(manifest, base, profiles, evidence):
    if 'requested_scope' not in manifest:
        return list(FAMILIES), [], None
    scope=manifest['requested_scope']
    requested=[f for f in FAMILIES if f!='dreamzero']
    require(set(scope)=={'families','capacity_exclusion'} and scope['families']==requested,
            'Only the reviewed whole-DreamZero-family scope may be excluded')
    entry=scope['capacity_exclusion']
    require(set(entry)=={'report','catalog_mapping','catalog_at_exclusion'},'Exact capacity references required')
    def original(role, expected):
        data=load_bound(base,entry[role]);require(sha(data)==expected,'Unreviewed capacity '+role)
        evidence.setdefault(sha(data),('.json',data))
        return json.loads(data),{'path':'evidence/'+sha(data)+'.json','sha256':sha(data)}
    report,report_ref=original('report','04a58d4e9dbb59bb8e9ba79bc7acc1324477881cdc1e4f50be2a8aacfd925390')
    mapping,mapping_ref=original('catalog_mapping','217270d68e9c9786db6e69b9d6ab869beba6a8543a5fab535d9cef636a7aaabd')
    catalog,catalog_ref=original('catalog_at_exclusion',mapping['catalog']['sha256'])
    require(report['schema']=='instinctflash.capacity_exclusion.v1'
            and report['deployment_target']==mapping['target']=='rtx5090'
            and report['family']==mapping['family']=='dreamzero'
            and report['status']=='capacity_excluded_not_tested'
            and report['scope']==mapping['original_scope']=='entire_family_on_bound_54GB_host',
            'Wrong original capacity scope')
    require(mapping['schema']=='instinctflash.private_capacity_catalog_mapping.v1'
            and mapping['status']=='catalog_mapping_only'
            and mapping['original_exclusion']['sha256']==report_ref['sha256']
            and mapping['original_named_modes']==report['excluded_modes']==['native','fp8','fp8-dynamic']
            and mapping['mode_aliases']=={'fp8-dynamic':'dynamic-fp8'}
            and set(mapping['catalog_excluded_modes'])==REQUIRED_MODES['dreamzero'],
            'Exact original mode mapping differs')
    for obj,fields in ((report,('tested','passed','GPU_used','observed_OOM','unsupported','changes_to_model_loader','qualification_evidence')),
                       (mapping,('tested','passed','GPU_used','observed_OOM','unsupported','native_impossibility_claimed',
                                 'nonreclaimable_RAM_lower_bound_claimed','new_model_or_capacity_measurement'))):
        require(all(obj[k] is False for k in fields),'Exclusion cannot claim a model result, OOM or absolute RAM bound')
    require(mapping['scope_authorization_unchanged']==report['scope_authorization']
            and report['scope_authorization']['native_impossibility_claimed'] is False
            and report['source_commit']==mapping['source_commit_as_recorded_in_original']=='2f38d095fc2a222a7b51ace7219e4cbeed4d7123',
            'Original source or whole-family decision changed')
    require(catalog['target']=='rtx5090','Capacity original catalog is another target')
    originals=[r for r in catalog['models'] if r['id']=='dreamzero'];require(len(originals)==1,'Original DreamZero declaration missing')
    before=originals[0];current=profiles['dreamzero']
    require(before['checkpoint']==current['checkpoint'] and set(before['execution_modes'])==set(current['execution_modes'])==REQUIRED_MODES['dreamzero'],
            'Excluded checkpoint or mode set changed')
    keys=('runtime_kwargs','environment','native_requirements','effective_schedule','schedule_changed')
    for mode in REQUIRED_MODES['dreamzero']:
        require(all(before['execution_modes'][mode][key]==current['execution_modes'][mode][key] for key in keys),
                'Excluded computation contract changed')
    basis=report['capacity_basis'];limit=report['memory_limit_bytes']
    require(limit==53999996928 and basis['original_bf16_payload_bytes']==45848344232
            and basis['additional_fp8_payload_bytes']==8808038400
            and basis['combined_payload_bytes']==basis['original_bf16_payload_bytes']+basis['additional_fp8_payload_bytes']
            and basis['combined_payload_over_limit_bytes']==basis['combined_payload_bytes']-limit
            and basis['absolute_nonreclaimable_RAM_lower_bound_claimed'] is False and bool(basis['mmap_caveat']),
            'Original conservative payload arithmetic/caveat differs')
    require(report['hardware_evidence']['sha256']=='cc601711706b2f6445beeb56578b01ca3aa1882cd46598af5a882959d8eaf2f0'
            and report['static_review']['sha256']=='dd881f97dcef4ae789777cb178c6a4ad45de3b8c27664280a92ac0483ec08ec5',
            'Original hardware/static source references changed')
    exclusion={'family':'dreamzero','label':current['label'],'status':'capacity_excluded_not_tested',
               'tested':False,'passed':False,'observed_OOM':False,'unsupported':False,
               'excluded_modes':sorted(REQUIRED_MODES['dreamzero']),'checkpoint':current['checkpoint'],
               'report':report_ref,'catalog_mapping':mapping_ref,'catalog_at_exclusion':catalog_ref,
               'source_commit_as_recorded_in_original':report['source_commit'],'memory_limit_bytes':limit,
               'capacity_basis':basis,'scope_authorization':report['scope_authorization'],
               'supporting_original_SHA_refs':{role:report[role]['sha256'] for role in ('hardware_evidence','static_review')},
               'supporting_source_bytes_revalidated_by_this_reader':False,
               'scope':'Exact approved original exclusion and mode-mapping bytes; no new capacity measurement or host/static-source re-audit.'}
    change={'kind':'user_instruction_in_active_task','text':report['scope_authorization']['user_instruction']}
    return requested,[exclusion],change


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


def render(manifest_path, output):
    manifest_path=Path(manifest_path).resolve(strict=True);base=manifest_path.parent;output=Path(output)
    require(not output.exists(),'Output must be a new directory')
    raw_manifest=manifest_path.read_bytes();manifest=json.loads(raw_manifest)
    require(manifest['schema']==INPUT_SCHEMA and manifest['target']=='rtx5090','Wrong explicit target manifest')
    profile_data=load_bound(base,manifest['target_profiles']);document=json.loads(profile_data)
    require(document['target']=='rtx5090','Wrong target profiles');profiles={r['id']:r for r in document['models']}
    require(set(profiles)==set(FAMILIES),'All eight original target declarations must remain visible')
    for family in FAMILIES:require(REQUIRED_MODES[family]<=set(profiles[family]['execution_modes']),'Missing original target mode')
    evidence={sha(profile_data):('.json',profile_data)};contexts={}
    for identifier,spec in manifest['cohorts'].items():
        require(re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',identifier) is not None,'Invalid cohort ID')
        raw=load_bound(base,spec['device_receipt']);hardware=json.loads(raw)
        require(re.fullmatch(r'[0-9a-f]{40}',spec['source_commit']) is not None,'Actual source commit missing')
        context={'machine_context_sha256':sha(raw),'hardware':hardware,'source_commit':spec['source_commit'],
                 'gpu_uuids':[gpu_uuid(hardware['uuid'])],'families':set(),'torch_versions':set(),'source_commits':set(),'cell_ids':set()}
        check_context_hardware(hardware,context);contexts[identifier]=context;evidence[sha(raw)]=('.json',raw)
    require(contexts,'Actual cohort context required')
    rows=[];comparisons=[];main_families=set()
    for entry in manifest['audits']:
        added,pairs,family=ingest(entry,base,profiles,evidence,contexts);rows.extend(added);comparisons.extend(pairs)
        if family:require(family not in main_families,'Duplicate independent primary family');main_families.add(family)
    require(all(c['cell_ids'] for c in contexts.values()),'Future cohort placeholders are forbidden')
    for spec in manifest.get('provenance',[]):
        raw=load_bound(base,spec);suffix='.py' if spec['path'].endswith('.py') else '.rst' if spec['path'].endswith('.rst') else '.json'
        evidence.setdefault(sha(raw),(suffix,raw))
    result=assemble(rows,comparisons,profiles,main_families)
    requested,exclusions,change=admit_requested_scope(manifest,base,profiles,evidence)
    by_id={r['id']:r for r in result['cells']}
    for row in result['main_table']+result['optional_table']:
        candidate=by_id[row['candidate']];row.update(cohort=candidate['cohort'],machine_context_sha256=candidate['machine_context_sha256'])
    apply_requested_scope(result,requested,exclusions,change,profiles)
    for row in result['cells']:
        raw=encoded(row['source_inventory']);checksum=sha(raw);evidence.setdefault(checksum,('.json',raw));row['source_inventory']={'sha256':checksum,'path':'evidence/'+checksum+'.json'}
    renderer_data=Path(__file__).read_bytes();evidence[sha(renderer_data)]=('.py',renderer_data)
    replay=json.loads(raw_manifest)
    def relative_refs(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                checksum=value['sha256'];require(checksum in evidence,'Unadmitted manifest reference');value['path']='evidence/'+checksum+evidence[checksum][0]
            else:
                for child in value.values():relative_refs(child)
        elif isinstance(value,list):
            for child in value:relative_refs(child)
    relative_refs(replay);replay_data=json.dumps(replay,indent=2,sort_keys=True).encode()+b'\n'
    public_contexts={k:{**{n:v for n,v in c.items() if not isinstance(v,set)},'families':sorted(c['families']),
                       'torch_versions':sorted(c['torch_versions']),'source_commits':sorted(c['source_commits']),'api_cell_count':len(c['cell_ids'])} for k,c in contexts.items()}
    result.update(schema='instinctflash.rtx5090.public_results.v1',target=TARGET,cohorts=public_contexts,
        input_manifest_sha256=sha(raw_manifest),renderer_source_sha256=sha(renderer_data),reproduce_manifest_sha256=sha(replay_data),
        target_profiles_sha256=sha(profile_data),task_quality_validated=False,minimum_host_memory_measured=False,
        required_catalog_family_count=8,required_catalog_runtime_mode_count=sum(map(len,REQUIRED_MODES.values())),
        source_scope='Saved actual original archives and independent audit bytes; private auditors are immutable provenance only, never runtime dependencies.',
        evidence=[{'path':'evidence/'+h+ext,'sha256':h,'bytes':len(raw)} for h,(ext,raw) in sorted(evidence.items())])
    require('torch' not in sys.modules,'Reader imported Torch')
    text=rst_report(result)
    text+='\nAll observed modes\n------------------\n\n'
    text+=csv_table([[profiles[r['family']]['label'],r['cell'],r['mode'],r['precision'],f"{r['p50_ms']:.6f}",json.dumps(r['effective_schedule']['nfe'],sort_keys=True)] for r in sorted(result['cells'],key=lambda v:(FAMILIES.index(v['family']),v['mode'],v['cell']))],['Model','Cell','Mode','Precision','p50 (ms)','Declared NFE'])
    text+='The complete Runtime ratios and action differences are in results.json. Selecting a faster mode does not isolate an FP8 arithmetic benefit.\n'
    output.mkdir(parents=True);(output/'evidence').mkdir()
    for h,(ext,raw) in evidence.items():(output/'evidence'/(h+ext)).write_bytes(raw)
    (output/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');(output/'results.rst').write_text(text)
    (output/'render_results.py').write_bytes(renderer_data);(output/'reproduce_manifest.json').write_bytes(replay_data)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');p.add_argument('--output',required=True);p.add_argument('--require-full',action='store_true');p.add_argument('--require-requested',action='store_true')
    a=p.parse_args()
    try:r=render(a.manifest,a.output)
    except (ValueError,KeyError,OSError,tarfile.TarError) as error:
        print('Evidence admission failed: '+str(error),file=sys.stderr);return 1
    print(json.dumps({'full_qualification':r['full_qualification'],'requested_scope_qualification':r['requested_scope_qualification'],'main_rows':len(r['main_table']),'api_cells':len(r['cells']),'missing':r['missing']}))
    return 2 if ((a.require_full and not r['full_qualification']) or (a.require_requested and not r['requested_scope_qualification'])) else 0

if __name__=='__main__':raise SystemExit(main())

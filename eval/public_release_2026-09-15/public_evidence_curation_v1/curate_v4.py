"""Bounded, create-only public evidence curation; no Git or publication operations."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import zipfile

HERE = Path(__file__).resolve().parent
POLICY_SHA = "a039d80541c3a13c0cad7be9e45e1874970ec2634339c3118d13e9df7dd1bba9"
ASSEMBLER_SHA = "d67aab97892cfd84e404a7bd0cc5b968c6046a0896b900afbbad8d1d2b84fa78"
CONTRACT_SHA = "1086a7e236219f6b8a073bee9e695d870981877cd52dda8646d75dace0146f0d"
STAGE_SHA = "a316bd918ae4426460f0779f90f34985b31a8639f3b6be7f9dd20e86c9cbdbc4"
SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "huggingface_token": re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "URL_credentials": re.compile(rb"https?://[^/\s:@]+:[^/\s@]+@"),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    def unique(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def relative(name):
    require(isinstance(name, str) and name and not any(c in name for c in "\x00\r\n"), "unsafe path")
    path = Path(name)
    require(not path.is_absolute() and all(x not in (".", "..") for x in path.parts)
            and str(path) == name and not name.startswith("-"), "unsafe relative path")
    return path


def safe(root, name, exists=True):
    root = Path(root).absolute()
    require(root.resolve() == root, "root contains a symlink")
    path = root / relative(name)
    require(path.resolve() == path, "path or parent contains a symlink")
    if exists:
        require(path.is_file(), "selected file missing: " + name)
    return path


def checked_ref(value):
    path = Path(value["path"])
    require(path.is_absolute() and path.resolve() == path and path.is_file(), "bound reference missing or symlink")
    require(digest(path.read_bytes()) == value["sha256"], "bound reference hash changed")
    return path


def policy():
    path = HERE / "policy_v4.json"
    require(digest(path.read_bytes()) == POLICY_SHA, "curation policy changed")
    return read(path)


def inspect_file(root, name, rules):
    path = safe(root, name)
    require(path.suffix in rules["allowed_extensions"], "unselected file type: " + name)
    require(path.stat().st_size <= rules["max_file_bytes"], "file exceeds 25 MB limit: " + name)
    data = path.read_bytes()
    require(not any(part in {"rfc", "rfcs", "__pycache__"} for part in path.parts), "retired/cache path")
    if path.suffix == ".npz":
        if path.name == "recorded_inputs_v1.npz":
            require(digest(data) == rules["fixture_sha256"], "recorded fixture changed")
        else:
            require(path.name in {"receipt.npz", "receipt.queue.npz", "actions.npz"}, "unnamed array archive")
            import numpy as np
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                require(sum(x.file_size for x in archive.infolist()) <= rules["max_file_bytes"], "expanded array archive too large")
            with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                require(arrays.files and all(re.fullmatch(r"actions?|action_[0-9]+|queued_actions_[0-9]+", key) for key in arrays.files),
                        "array archive contains non-action data")
                for key in arrays.files:
                    require(arrays[key].dtype.kind in "fi" and np.isfinite(arrays[key]).all(), "invalid numeric action array")
    else:
        for label, expression in SECRET_PATTERNS.items():
            require(expression.search(data) is None, "credential candidate: " + label + " in " + name)
    return {"sha256": digest(data), "bytes": len(data)}


def walk(root, relative_dir, rules):
    directory = safe(root, relative_dir, exists=False)
    require(directory.is_dir(), "selected directory missing: " + relative_dir)
    result = []
    for current, names, files in os.walk(directory, followlinks=False):
        for name in list(names):
            path = Path(current) / name
            require(not path.is_symlink(), "symlink directory in selection")
            if name in rules["ignored_directory_names"]:
                names.remove(name)
        for name in sorted(files):
            path = Path(current) / name
            require(not path.is_symlink(), "symlink file in selection")
            if path.suffix in rules["allowed_extensions"]:
                result.append(str(path.relative_to(root)))
            else:
                raise ValueError("unselected file in bounded evidence tree: " + str(path))
    return sorted(result)


def named_files(repo, rules, strict):
    study = rules["study_relative"]
    names, pending = set(), []
    for name in rules["file_allowlist"]:
        target = study + "/" + name
        if not (repo / target).exists():
            pending.append(target)
        else:
            names.add(target)
    for directory in rules["directory_allowlist"]:
        target = study + "/" + directory
        if not (repo / target).exists():
            pending.append(target)
        else:
            names.update(walk(repo, target, rules))
    if rules["include_existing_historical_comparison_json_at_family_roots"]:
        families = {name.split("/")[1] for name in rules["directory_allowlist"]
                    if name.startswith("qualification/") and not name.startswith("qualification/foreign/")}
        for family_name in sorted(families):
            family = repo / study / "qualification" / family_name
            if family.is_dir():
                for path in family.glob("*.json"):
                    names.add(str(path.relative_to(repo)))
    require(not strict or not pending, "required evidence still pending: " + ", ".join(pending))
    return names, pending


def validate_current(repo, summary, expected):
    summary = summary.absolute()
    require(summary.parent.parent == repo / "eval/public_release_2026-09-15/current_public_pipeline_v1",
            "summary outside current assembly")
    require(digest(summary.read_bytes()) == expected, "summary hash changed")
    value = read(summary)
    require(value["schema"] == "instinctflash.current_public_pipeline_qualification.v1"
            and value["status"] == "current_pipeline_qualified", "current qualification not complete")
    counts = value["counts"]
    require(counts["main_passed"] == 8 and counts["foreign_passed"] == 6
            and counts["historical_families_assessed"] == 8, "incomplete current coverage")
    for key in ("complete_historical_reproduction", "task_quality_certified", "prior_cosmos_SCREEN_transferred"):
        require(value[key] is False, "historical/task claim promotion")
    require(value["assembler"]["sha256"] == ASSEMBLER_SHA
            and value["contract"]["sha256"] == CONTRACT_SHA
            and value["source_stage_manifest"]["sha256"] == STAGE_SHA, "current source binding differs")
    for key in ("assembler", "contract", "source_stage_manifest"):
        checked_ref(value[key])
    selection_path = checked_ref(value["selection"])
    selection = read(selection_path)
    require(selection["template_only"] is False and selection["source_stage_manifest"] == value["source_stage_manifest"],
            "selection differs from summary")
    qualification = repo / "eval/public_release_2026-09-15/qualification"
    for name, entry in selection["snapshot"].items():
        path = safe(qualification, name)
        require(path.stat().st_size == entry["bytes"] and digest(path.read_bytes()) == entry["sha256"],
                "raw evidence changed after current assembly")
    return value, selection_path


def file_ref(path):
    return {"path": str(path), "sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}


def validate_renderers(repo, summary, expected, rules):
    study = repo / rules["study_relative"]
    result = {}
    for kind, spec in rules["renderer_bindings"].items():
        binding = safe(study, spec["binding"])
        value = read(binding)
        source, candidate = safe(study, spec["source"]), safe(study, spec["candidate"])
        require(candidate.stat().st_size <= rules["max_file_bytes"], "renderer candidate exceeds file cap")
        require(value["schema"] == spec["schema"] and value["status"] == "candidate_from_completed_qualification",
                "renderer did not consume a completed qualification")
        require(value["summary"] == {"path": str(summary), "sha256": expected}, "renderer summary differs")
        require(value["renderer"] == {"path": str(source), "sha256": spec["source_sha256"]}
                and digest(source.read_bytes()) == spec["source_sha256"], "renderer source differs")
        require(value["candidate"] == {"path": str(candidate), "sha256": digest(candidate.read_bytes())},
                "renderer candidate changed")
        result[kind] = {"binding": file_ref(binding), "source": file_ref(source), "candidate": file_ref(candidate)}
    entry = safe(study, "results.rst")
    result["reader_entry"] = file_ref(entry)
    result["reader_entry_matches_generated_candidate"] = digest(entry.read_bytes()) == result["results"]["candidate"]["sha256"]
    result["scope"] = "Generated candidates and final reader-entry hashes are bound separately. Root may edit final prose; this does not claim the final README equals its candidate."
    return result


def sde1_candidates(repo, rules):
    spec = rules["optional_SDE1"]
    study = repo / rules["study_relative"]
    found, missing = set(), []
    for name in spec["candidate_files"]:
        path = study / relative(name)
        if path.exists() or path.is_symlink():
            safe(study, name)
            found.add(str(path.relative_to(repo)))
        else:
            missing.append(name)
    return found, missing


def sde1_evidence(repo, rules):
    """Bind existing native validation receipts, without importing or replaying a model."""
    spec = rules["optional_SDE1"]
    study = repo / rules["study_relative"]
    found, missing = sde1_candidates(repo, rules)
    base = study / spec["root"]
    result = {"status": "not_present" if not found else "partial_copied_evidence_not_completed",
              "completed": False, "candidate_files_present": sorted(found), "candidate_files_missing": missing,
              "category": "SCREEN", "task_quality_certified": False, "belongs_to_main_8_plus_6": False,
              "independent_scientific_replay_performed": False}
    if not (base / "completion.json").is_file():
        return result
    done = read(safe(base, "completion.json"))
    result["completion"] = file_ref(base / "completion.json")
    result["native_reported_status"] = done.get("status")
    if done.get("status") != spec["completed_status"]:
        result["status"] = "native_failure_or_partial_preserved_not_completed"
        return result
    required_missing = sorted(set(spec["completion_required_files"]) & set(missing))
    if required_missing:
        result["missing_completion_references"] = required_missing
        return result
    source = safe(study, spec["queue_source"])
    template_path = safe(study, spec["queue_template"])
    require(digest(source.read_bytes()) == spec["queue_source_sha256"]
            and digest(template_path.read_bytes()) == spec["queue_template_sha256"], "SDE1 frozen source/template differs")
    config_path = safe(base, "config.json")
    config, template = read(config_path), read(template_path)
    require(config["schema"] == template["schema"] and config["template_only"] is False,
            "SDE1 final config is still a template")
    for key in ("queue_source_sha256", "installed_sources", "recipes", "python", "ptxas", "run_root", "edge_tmpfs_root",
                "handle_inspector", "protected_service_manifest"):
        require(config[key] == template[key], "SDE1 selected source/config field differs: " + key)
    require(config["queue_source_sha256"] == spec["queue_source_sha256"]
            and config["recipes"] == spec["recipes"], "SDE1 queue source or recipe list differs")
    controls = {}
    for key, control in spec["queue_control_files"].items():
        path = safe(study, control["path"])
        require(digest(path.read_bytes()) == control["sha256"] == config[key]["sha256"]
                and path.stat().st_size == config[key]["bytes"], "SDE1 protected-process control source differs")
        controls[key] = file_ref(path)
    source_guard = read(safe(base, "source_guard.json"))
    matched = set()
    prefix = Path(config["python"]).parent.parent
    for files in config["installed_sources"].values():
        for relative_source, expected_hash in files.items():
            paths = [name for name in source_guard if name.endswith("/" + relative_source)]
            require(len(paths) == 1 and source_guard[paths[0]] == expected_hash
                    and Path(paths[0]).is_relative_to(prefix), "SDE1 installed source guard differs")
            matched.add(paths[0])
    require(matched == set(source_guard), "SDE1 source guard has extra entries")
    # Only the reviewed queue's pure argv constructor is called, never execute(),
    # load_config(), sources(), environment(), subprocesses or native validators.
    module_spec = importlib.util.spec_from_file_location("bound_SDE1_queue_commands", source)
    queue = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(queue)
    operations = queue.commands(config)
    require([row["id"] for row in done["operations"]] == [row["id"] for row in operations],
            "SDE1 completed operation coverage differs")
    for actual, planned in zip(done["operations"], operations):
        process = actual["process"]
        require(type(process["returncode"]) is int and process["returncode"] == 0
                and process.get("failed", False) is False and process["command"] == planned["argv"],
                "SDE1 operation did not close with the selected command")
    require(done["category"] == "SCREEN" and done["task_quality_certified"] is False
            and done["automatic_retry"] is False, "SDE1 completion scope changed")
    require([row["recipe"] for row in done["screens"]] == spec["recipes"], "three SDE1 native validation references required")

    def remote_ref(value, remote, local):
        require(value == {"path": str(remote), "sha256": digest(local.read_bytes()), "bytes": local.stat().st_size},
                "SDE1 original-to-copied evidence reference differs")

    validations = []
    for name, screen in zip(spec["recipes"], done["screens"]):
        archive_root = base / "archives" / name
        validation_path = safe(archive_root, "validation.json")
        value = read(validation_path)
        require(value == screen and value["status"] == "passed" and value["category"] == "SCREEN"
                and value["task_quality_certified"] is False, "SDE1 recipe validation differs from completion")
        run_operation = next(row for row in done["operations"] if row["id"] == name + "-run")
        require(run_operation["screen"] == value, "SDE1 operation/validation reference differs")
        planned_run = next(row for row in operations if row["id"] == name + "-run")
        original_run = Path(config["run_root"]) / "runs" / name
        for key, filename in (("completion", "completion.json"), ("receipt", "receipt.json"), ("actions", "receipt.npz")):
            remote_ref(value[key], original_run / filename, safe(archive_root, "run/" + filename))
        completed, receipt = read(archive_root / "run/completion.json"), read(archive_root / "run/receipt.json")
        require(completed["status"] == "passed" and type(completed["returncode"]) is int and completed["returncode"] == 0
                and receipt["ok"] is True, "SDE1 child receipt failed")
        require(all(row["category"] == "SCREEN" and row["task_quality_certified"] is False for row in (completed, receipt)),
                "SDE1 child quality scope changed")
        p50 = value["p50_ms"]
        require(type(p50) in (int, float) and math.isfinite(p50) and p50 > 0
                and p50 == completed["p50_ms"] == receipt["p50_ms"], "SDE1 linked p50 differs")
        count = 36 if name == "nano-original" else 16
        require(value["requests"] == count and value["action_shape"] == [count, 32, 8], "SDE1 fixed request shape differs")
        require(receipt["actions_sha256"] == digest((archive_root / "run/receipt.npz").read_bytes()), "SDE1 actions hash differs")
        prep = read(safe(archive_root, "preparation/public_preparation.json"))
        require(prep["recipe"] == name and prep["category"] == "SCREEN" and prep["task_quality_certified"] is False,
                "SDE1 preparation recipe/scope differs")
        manifest = "budget_manifest.json" if name == "nano-original" else "instinctcompress_manifest.json"
        require(receipt["checkpoint_manifest_sha256"] == digest((archive_root / "preparation" / manifest).read_bytes()),
                "SDE1 prepared checkpoint manifest differs")
        archive = read(safe(archive_root, "archive.json"))
        require(archive["status"] == "archived_small_evidence" and archive["large_weight_bytes_copied"] == 0,
                "SDE1 archival proof differs")
        copies = {row["copy"]["path"]: row for row in archive["files"]}
        require(len(copies) == len(archive["files"]), "duplicate SDE1 archive reference")
        for selected in found:
            path = repo / selected
            if not path.is_relative_to(archive_root):
                continue
            local = path.relative_to(archive_root)
            if local.parts[0] not in ("run", "preparation"):
                continue
            original_archive = Path(config["run_root"]) / "archives" / name / local
            row = copies[str(original_archive)]
            remote_ref(row["copy"], original_archive, path)
            origin = original_run if local.parts[0] == "run" else Path(planned_run["prepared"])
            remote_ref(row["source"], origin.joinpath(*local.parts[1:]), path)
        validations.append({"recipe": name, "validation": file_ref(validation_path), "p50_ms": p50,
                            "requests": count, "category": "SCREEN", "task_quality_certified": False})
    result.update(status="completed_three_recipe_SCREEN_reference_chain_bound", completed=True,
                  final_config=file_ref(config_path), native_queue_source=file_ref(source),
                  installed_source_guard=file_ref(base / "source_guard.json"), protected_process_controls=controls,
                  validations=validations,
                  scope="Only the frozen native queue's completed validation/reference chain is bound; no new native scientific replay or task-quality certificate.")
    return result


def provenance_index(repo, checkout, names):
    entries = {}
    def visit(value):
        if isinstance(value, dict):
            name, expected = value.get("path"), value.get("sha256")
            if isinstance(name, str) and isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected):
                path = Path(name)
                candidate = path if path.is_absolute() else repo / path
                try:
                    rel = str(candidate.relative_to(repo))
                except ValueError:
                    rel = None
                kind = "local_only_provenance"
                if rel in names and candidate.is_file() and digest(candidate.read_bytes()) == expected:
                    kind = "bundled_exact"
                elif rel is not None:
                    public = checkout / rel
                    if public.is_file() and not public.is_symlink() and public.resolve() == public and digest(public.read_bytes()) == expected:
                        kind = "already_public_exact"
                entries[(name, expected)] = {"original_path": name, "sha256": expected,
                                             "availability": kind, "public_relative_path": rel if kind != "local_only_provenance" else None}
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for name in sorted(names):
        if name.endswith(".json"):
            visit(read(repo / name))
    return {"schema": "instinctflash.public_evidence_provenance.v1", "references": list(entries.values()),
            "scope": "Objects with path+sha256 in selected JSON. Original audit paths stay unchanged. Local-only references are provenance, not requirements for public reproduce/serve_smoke/framework_compare commands."}


def plan(repo, checkout, output, summary=None, summary_sha=None, require_completed_sde1=False):
    repo, checkout, output = repo.absolute(), checkout.absolute(), output.absolute()
    rules = policy()
    require(output.resolve().is_relative_to(HERE) and not output.exists(), "plan output must be a new owned directory")
    final = summary is not None
    names, pending = named_files(repo, rules, strict=final)
    candidates, _ = sde1_candidates(repo, rules)
    names.update(candidates)
    binding = None
    renderer_binding = None
    if final:
        value, selection = validate_current(repo, summary, summary_sha)
        names.update(walk(repo, str(summary.parent.relative_to(repo)), rules))
        names.add(str(selection.relative_to(repo)))
        names.add(rules["study_relative"] + "/results.rst")
        binding = {"summary": {"path": str(summary), "sha256": summary_sha}, "selection": value["selection"]}
    names.update(str((HERE / name).relative_to(repo)) for name in ("curate_v4.py", "policy_v4.json", "test_curate_v4.py"))
    files = {name: inspect_file(repo, name, rules) for name in sorted(names)}
    if final:
        renderer_binding = validate_renderers(repo, summary, summary_sha, rules)
    sde1 = sde1_evidence(repo, rules)
    require(not require_completed_sde1 or final and sde1["completed"], "three completed SDE1 validation chains are not available")
    require(len(files) <= rules["max_files"] and sum(x["bytes"] for x in files.values()) <= rules["max_total_bytes"],
            "evidence selection exceeds total bound")
    output.mkdir(parents=True)
    provenance = output / "provenance_refs.json"
    save(provenance, provenance_index(repo, checkout, names))
    files[str(provenance.relative_to(repo))] = inspect_file(repo, str(provenance.relative_to(repo)), rules)
    require(len(files) <= rules["max_files"] and sum(x["bytes"] for x in files.values()) <= rules["max_total_bytes"],
            "evidence plus provenance index exceeds total bound")
    manifest = {"schema": "instinctflash.public_results_evidence_manifest.v1",
                "status": "ready_for_create_only_copy" if final else "preview_only_waiting_current_qualification",
                "repository": str(repo), "policy_sha256": POLICY_SHA, "curator_sha256": digest(Path(__file__).read_bytes()),
                "current_qualification": binding, "files": files, "pending": pending,
                "report_renderers": renderer_binding, "SDE1": sde1,
                "require_completed_sde1": require_completed_sde1,
                "total_bytes": sum(x["bytes"] for x in files.values()), "unique_blob_bytes": sum({v["sha256"]: v["bytes"] for v in files.values()}.values()),
                "credential_scan": "No private-key/AWS/GitHub/HuggingFace literal candidates; bounded patterns only, no general security certification.",
                "excluded": rules["excluded"], "publication_authorization": False,
                "reproduction_entrypoints": rules["public_reproduction_entrypoints"], "audit_path_portability": rules["frozen_assembly_portability"]}
    path = output / "manifest.json"
    save(path, manifest)
    pathlist = output / "approved_force_add_paths.txt"
    with pathlist.open("x") as stream:
        stream.write("\n".join(sorted([*files, str(path.relative_to(repo))])) + "\n")
    return {"manifest": str(path), "sha256": digest(path.read_bytes()), "paths": str(pathlist),
            "status": manifest["status"], "files": len(files), "bytes": manifest["total_bytes"], "pending": pending}


def copy_plan(manifest_path, expected, checkout, receipt):
    require(digest(manifest_path.read_bytes()) == expected, "manifest hash changed")
    manifest = read(manifest_path)
    require(manifest["status"] == "ready_for_create_only_copy" and manifest["policy_sha256"] == POLICY_SHA
            and manifest["curator_sha256"] == digest(Path(__file__).read_bytes()), "unapproved preview or source drift")
    repo, checkout = Path(manifest["repository"]), checkout.absolute()
    require(repo != checkout and checkout.is_dir() and checkout.resolve() == checkout, "invalid target checkout")
    rules = policy()
    binding = manifest["current_qualification"]
    validate_current(repo, Path(binding["summary"]["path"]), binding["summary"]["sha256"])
    require(validate_renderers(repo, Path(binding["summary"]["path"]), binding["summary"]["sha256"], rules)
            == manifest["report_renderers"], "renderer binding or final reader entry changed")
    require(sde1_evidence(repo, rules) == manifest["SDE1"], "SDE1 evidence changed after planning")
    require(not manifest["require_completed_sde1"] or manifest["SDE1"]["completed"], "completed SDE1 requirement is not satisfied")
    files = dict(manifest["files"])
    files[str(manifest_path.relative_to(repo))] = {"sha256": expected, "bytes": manifest_path.stat().st_size}
    targets = []
    for name, entry in files.items():
        require(inspect_file(repo, name, rules) == entry, "planned source bytes changed")
        target = safe(checkout, name, exists=False)
        if target.exists():
            require(target.is_file() and digest(target.read_bytes()) == entry["sha256"], "existing public evidence conflicts")
        targets.append((name, target, target.exists()))
    require(not receipt.exists(), "copy receipt already exists")
    copied, identical = [], []
    for name, target, exists in targets:
        if exists:
            identical.append(name)
            continue
        safe(checkout, name, exists=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = safe(repo, name).read_bytes()
        require(digest(data) == files[name]["sha256"], "source changed during copy")
        with target.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        require(digest(target.read_bytes()) == files[name]["sha256"], "copied evidence differs")
        copied.append(name)
    save(receipt, {"status": "create_only_evidence_copy_passed", "manifest_sha256": expected,
                   "copied": copied, "already_identical": identical, "Git_index_or_publication_actions": False})
    return {"receipt": str(receipt), "copied": len(copied), "already_identical": len(identical)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("preview", "plan", "copy"))
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--summary-sha256")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--require-completed-sde1", action="store_true", help="Plan only: require all three completed native SCREEN reference chains")
    args = parser.parse_args()
    require(not args.require_completed_sde1 or args.operation == "plan", "completed SDE1 requirement is a final-plan option")
    if args.operation == "copy":
        require(args.manifest is not None and args.manifest_sha256 is not None, "copy requires bound manifest")
        result = copy_plan(args.manifest, args.manifest_sha256, args.checkout, args.output)
    else:
        require((args.summary is not None and args.summary_sha256 is not None) == (args.operation == "plan"),
                "plan requires complete summary; preview cannot admit one")
        result = plan(args.repo, args.checkout, args.output, args.summary, args.summary_sha256, args.require_completed_sde1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

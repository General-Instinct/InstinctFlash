"""Bounded, create-only public evidence curation; no Git or publication operations."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile

HERE = Path(__file__).resolve().parent
POLICY_SHA = "948ed44b4575ad198ace1faa656cccada4399f7096195ae9c786c9bb561ffcfa"
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
    path = HERE / "policy_v3.json"
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


def plan(repo, checkout, output, summary=None, summary_sha=None):
    repo, checkout, output = repo.absolute(), checkout.absolute(), output.absolute()
    rules = policy()
    require(output.resolve().is_relative_to(HERE) and not output.exists(), "plan output must be a new owned directory")
    final = summary is not None
    names, pending = named_files(repo, rules, strict=final)
    binding = None
    if final:
        value, selection = validate_current(repo, summary, summary_sha)
        names.update(walk(repo, str(summary.parent.relative_to(repo)), rules))
        names.add(str(selection.relative_to(repo)))
        names.add(rules["study_relative"] + "/results.rst")
        binding = {"summary": {"path": str(summary), "sha256": summary_sha}, "selection": value["selection"]}
    names.update(str((HERE / name).relative_to(repo)) for name in ("curate_v3.py", "policy_v3.json", "test_curate_v3.py"))
    files = {name: inspect_file(repo, name, rules) for name in sorted(names)}
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
    args = parser.parse_args()
    if args.operation == "copy":
        require(args.manifest is not None and args.manifest_sha256 is not None, "copy requires bound manifest")
        result = copy_plan(args.manifest, args.manifest_sha256, args.checkout, args.output)
    else:
        require((args.summary is not None and args.summary_sha256 is not None) == (args.operation == "plan"),
                "plan requires complete summary; preview cannot admit one")
        result = plan(args.repo, args.checkout, args.output, args.summary, args.summary_sha256)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

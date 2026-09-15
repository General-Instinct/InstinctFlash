"""Portable recorded-input reproduction: plan, prepare, run, and validate reports.

Each model uses its own installed environment. Planning is read-only and imports
no model libraries. Only explicit prepare downloads weights; run captures three
fresh processes (plus an explicit operating point when its schedule changes).
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback


FIXTURE_SHA = "37843e22fa6dd9a2abf2bae390ccb8e5c4319446e3d3fab5d0d477860065f411"
ACTION_SHAPES = {"va": [16, 2, 16], "vla4": [25, 14], "vla2": [50, 14], "pi05": [7],
                 "groot": [40, 17], "edge": [32, 8], "nano": [32, 8], "dreamzero": [24, 8]}
SHAPE_PROVENANCE = {"path": "eval/user_e2e_2026-09-14/matrix_final.json",
                    "sha256": "346154c84a871fd48addd23f563935c29ec0f90a7cbd5b7ef813b7739bf8077d",
                    "scope": "Public action shapes only; historical outcomes and machine paths are not inputs."}
MAIN_ARMS = ("eager_native", "runtime_default", "runtime_selected")
SCHEMA = "instinctflash.public_reproduction.v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_new(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def new_directory(path):
    path = Path(path).absolute()
    require(not path.exists() and not path.is_symlink(), f"output already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def profiles_path():
    packaged = Path(__file__).parent / "fixtures/deployment_profiles.json"
    if packaged.is_file():
        return packaged
    source = Path(__file__).resolve().parents[2] / "release/deployment_profiles.json"
    require(source.is_file(), "deployment profiles missing from installation")
    return source


def fixture_path():
    return Path(__file__).parent / "fixtures/recorded_inputs_v1.npz"


def libraries_from_args(values):
    result = {}
    for value in values:
        key, separator, path = value.partition("=")
        require(separator and key and path and key not in result, "library must be unique NAME=PATH")
        result[key] = path
    return result


def make_plan(model, mode="native", *, catalog_path=None, libraries=None, library_bindings=None, extra_modes=()):
    """Read metadata only. No directories, downloads, imports of Torch or device probes."""
    catalog_path = Path(catalog_path) if catalog_path is not None else profiles_path()
    catalog_bytes = catalog_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    require(catalog.get("schema") == "instinctflash.deployment_profiles.v1", "unknown deployment profile schema")
    matches = [value for value in catalog["models"] if model in (value["id"], value["checkpoint"]["model_id"])]
    require(len(matches) == 1, f"unknown or ambiguous model alias: {model}")
    profile = matches[0]
    family = profile["id"]
    require(family in ACTION_SHAPES, "no recorded-input contract for model")
    require(re.fullmatch(r"[0-9a-f]{40}", profile["checkpoint"]["revision"]) is not None, "checkpoint requires exact revision")
    require(mode in profile["execution_modes"], f"unknown explicit execution mode: {mode}")
    native = profile["execution_modes"]["native"]
    selected = profile["execution_modes"][mode]
    require(isinstance(extra_modes, (list, tuple)) and all(isinstance(v, str) for v in extra_modes),
            "extra modes must be an explicit list")
    require(len(set(extra_modes)) == len(extra_modes) and mode not in extra_modes,
            "duplicate primary or extra execution mode")
    seen_schedules = [native["effective_schedule"], selected["effective_schedule"]]
    for extra in extra_modes:
        require(extra in profile["execution_modes"], f"unknown explicit execution mode: {extra}")
        point = profile["execution_modes"][extra]
        require(point["schedule_changed"] is True and point["effective_schedule"] not in seen_schedules,
                "extra mode must declare a distinct changed schedule; same-policy extras are unsupported")
        seen_schedules.append(point["effective_schedule"])
    require(native["schedule_changed"] is False and native["runtime_kwargs"]["precision"] == "native"
            and not native["environment"], "invalid native baseline profile")
    require(type(selected["schedule_changed"]) is bool, "invalid schedule change declaration")
    require(selected["schedule_changed"] == (selected["effective_schedule"] != native["effective_schedule"]),
            "schedule change flag differs from native policy")
    primary_file_keys = {item["name"] for item in selected["native_requirements"] if item["kind"] == "file_env"}
    file_keys = primary_file_keys | {item["name"] for extra in extra_modes
                                    for item in profile["execution_modes"][extra]["native_requirements"]
                                    if item["kind"] == "file_env"}
    require(not (libraries and library_bindings), "use supplied libraries or archived bindings, not both")
    supplied = libraries or library_bindings or {}
    require(set(supplied) <= file_keys, "library option is not required by selected mode")
    bindings = {}
    environment = dict(selected["environment"])
    for key, value in supplied.items():
        if library_bindings is None:
            path = Path(value).expanduser().resolve(strict=True)
            require(path.is_file(), "selected library must be a regular file")
            bindings[key] = {"path": str(path), "sha256": sha(path)}
        else:
            require(set(value) == {"path", "sha256"} and Path(value["path"]).is_absolute()
                    and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None, "invalid archived library binding")
            bindings[key] = deepcopy(value)
        if key in primary_file_keys:
            environment[key] = bindings[key]["path"]
    require(all(key.startswith("IFL_") and isinstance(value, str) for key, value in environment.items()),
            "profile optimizer environment must contain string IFL_* options only")
    cells = []
    selected_main = native if selected["schedule_changed"] else selected
    for arm in MAIN_ARMS:
        options = {} if arm == "eager_native" else {"device": "cuda:0"} if arm == "runtime_default" else deepcopy(selected_main["runtime_kwargs"])
        cell_environment = environment if arm == "runtime_selected" and not selected["schedule_changed"] else {}
        cells.append({"id": f"{family}-{arm}", "family": family, "arm": arm,
                      "receipt": f"cells/{family}-{arm}/receipt.json", **profile["checkpoint"],
                      "action_shape": ACTION_SHAPES[family], "comparison_group": f"{family}-checkpoint",
                      "experimental": False, "expected_runtime_kwargs": options,
                      "expected_optimizer_environment": cell_environment,
                      "effective_schedule": deepcopy(native["effective_schedule"])})
    if selected["schedule_changed"]:
        identifier = f"{family}-{mode}"
        cells.append({"id": identifier, "family": family, "arm": "operating_point",
                      "receipt": f"cells/{identifier}/receipt.json", **profile["checkpoint"],
                      "action_shape": ACTION_SHAPES[family], "comparison_group": f"{family}-{mode}",
                      "experimental": True, "expected_runtime_kwargs": deepcopy(selected["runtime_kwargs"]),
                      "expected_optimizer_environment": environment,
                      "effective_schedule": deepcopy(selected["effective_schedule"]),
                      "comparison_baseline": True, "cross_policy_baseline": f"{family}-eager_native"})
    extra_requirements = []
    for extra in extra_modes:
        extra_file_keys = {item["name"] for item in profile["execution_modes"][extra]["native_requirements"]
                           if item["kind"] == "file_env"}
        extra_plan = make_plan(model, extra, catalog_path=catalog_path,
                               library_bindings={key: value for key, value in bindings.items() if key in extra_file_keys})
        require(extra_plan["matrix"]["expected_operating_point_cells"] == 1,
                "extra mode must have exactly one operating point")
        cells.append(extra_plan["matrix"]["cells"][-1])
        extra_requirements.extend(extra_plan["native_requirements"])
    require(len({cell["id"] for cell in cells}) == len(cells), "duplicate planned cell")
    matrix = {"schema": 1, "require_observed_schedule": True, "expected_main_cells": 3,
              "expected_families": [family], "expected_operating_point_cells": len(cells) - 3,
              "expected_runtime_update_cells": 0, "quality_certified": False,
              "protocol": {"stateless": {"warmup": 5, "measured": 20},
                           "history": {"warmup_episodes": 1, "measured_episodes": 6, "cycles_per_episode": 3},
                           "pi05_queue_calls": 51},
              "bindings": [{"path": "inputs/recorded_inputs_v1.npz", "sha256": FIXTURE_SHA}], "cells": cells}
    result = {"schema": SCHEMA, "model": family, "execution_mode": mode, "checkpoint": profile["checkpoint"],
            "profiles_sha256": hashlib.sha256(catalog_bytes).hexdigest(), "fixture_sha256": FIXTURE_SHA,
            "shape_provenance": SHAPE_PROVENANCE, "library_bindings": bindings,
            "unresolved_library_options": sorted(file_keys - set(supplied)),
            "native_requirements": deepcopy(selected["native_requirements"]),
            "vendor": deepcopy(profile["upstream"]), "bootstrap": deepcopy(profile["bootstrap"]),
            "matrix": matrix, "task_quality_validated": False, "publication_ready": False,
            "scope": "Recorded cameras and synthetic state; public predict wall time; no network or simulator/task-success measurement.",
            "limitations": ["Each model requires its own compatible installed vendor environment.",
                            "prepare resolves the pinned primary checkpoint; auxiliary tokenizers/base weights must also be prepared in that environment.",
                            "Runtime captures target Jetson Thor SM110 using the unchanged capture contract.",
                            "Changed sampling policies are separate operating points, never architecture-only speedups."]}
    if extra_modes:
        result["extra_execution_modes"] = list(extra_modes)
        for requirement in extra_requirements:
            if requirement not in result["native_requirements"]:
                result["native_requirements"].append(requirement)
    return result


def prepare(model, mode, output, *, cache_dir=None, local_files_only=False, libraries=None, extra_modes=()):
    plan = make_plan(model, mode, libraries=libraries, extra_modes=extra_modes)
    fixture = fixture_path()
    require(sha(fixture) == FIXTURE_SHA, "public fixture hash differs")
    root = new_directory(output)
    profile_bytes = profiles_path().read_bytes()
    require(hashlib.sha256(profile_bytes).hexdigest() == plan["profiles_sha256"], "profiles changed during preparation")
    write_new(root / "plan.json", encoded(plan))
    write_new(root / "matrix.json", encoded(plan["matrix"]))
    write_new(root / "profiles.json", profile_bytes)
    write_new(root / "inputs/recorded_inputs_v1.npz", fixture.read_bytes())
    receipt = {"schema": SCHEMA, "status": "failed", "plan_sha256": sha(root / "plan.json"),
               "matrix_sha256": sha(root / "matrix.json"), "profiles_sha256": plan["profiles_sha256"],
               "fixture_sha256": FIXTURE_SHA, "checkpoint": plan["checkpoint"],
               "cache_dir": str(Path(cache_dir).expanduser().resolve()) if cache_dir else None,
               "local_files_only": bool(local_files_only), "task_quality_validated": False}
    try:
        from huggingface_hub import snapshot_download
        reference = Path(snapshot_download(**{"repo_id": plan["checkpoint"]["model_id"],
                                               "revision": plan["checkpoint"]["revision"]},
                                           cache_dir=receipt["cache_dir"], local_files_only=local_files_only)).absolute()
        require(reference.name == plan["checkpoint"]["revision"] and reference.is_dir(),
                "download did not return the exact pinned snapshot reference")
        snapshot = reference.resolve(strict=True)
        receipt.update(status="prepared_primary_checkpoint", checkpoint_snapshot=str(snapshot),
                       checkpoint_snapshot_reference=str(reference),
                       checkpoint_snapshot_is_linked=reference != snapshot,
                       checkpoint_identity_scope="Requested Hub commit reference only; capture independently binds checkpoint file bytes.")
    except BaseException as error:
        receipt.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_new(root / "preparation.json", encoded(receipt))
    return receipt


def validate_bundle(root, *, check_libraries=True):
    root = Path(root).resolve(strict=True)
    receipt = json.loads((root / "preparation.json").read_text())
    require(receipt["status"] == "prepared_primary_checkpoint", "preparation did not complete successfully")
    for name, key in (("plan.json", "plan_sha256"), ("matrix.json", "matrix_sha256"),
                      ("profiles.json", "profiles_sha256"), ("inputs/recorded_inputs_v1.npz", "fixture_sha256")):
        require(sha(root / name) == receipt[key], f"prepared artifact changed: {name}")
    plan = json.loads((root / "plan.json").read_text())
    require(plan["fixture_sha256"] == FIXTURE_SHA and receipt["fixture_sha256"] == FIXTURE_SHA, "prepared fixture is not the fixed public input")
    regenerated = make_plan(plan["model"], plan["execution_mode"], catalog_path=root / "profiles.json",
                            library_bindings=plan["library_bindings"], extra_modes=plan.get("extra_execution_modes", ()))
    require(plan == regenerated and json.loads((root / "matrix.json").read_text()) == plan["matrix"], "prepared plan no longer matches its profiles/library bytes")
    if check_libraries:
        for binding in plan["library_bindings"].values():
            require(sha(binding["path"]) == binding["sha256"], "selected library bytes changed")
    return plan, receipt


def child_environment(plan, cell, preparation, inherited=None):
    environment = {key: value for key, value in (os.environ if inherited is None else inherited).items()
                   if not key.startswith("IFL_") and key not in {"PYTHONPATH", "PYTHONHOME", "DYNAMIC_CACHE_SCHEDULE", "NUM_DIT_STEPS",
                                                               "LOAD_TRT_ENGINE", "ENABLE_TENSORRT"}}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    environment["ENABLE_TENSORRT"] = "false"
    if preparation.get("cache_dir"):
        environment["HF_HUB_CACHE"] = preparation["cache_dir"]
        environment["HUGGINGFACE_HUB_CACHE"] = preparation["cache_dir"]
    environment.update(cell["expected_optimizer_environment"])
    return environment


def require_installed():
    import instinctflash
    distribution = importlib.metadata.distribution("instinctflash")
    require(Path(instinctflash.__file__).resolve().parent == Path(distribution.locate_file("instinctflash")).resolve(),
            "run requires the current interpreter's installed core wheel")
    require(not json.loads(distribution.read_text("direct_url.json") or "{}").get("dir_info", {}).get("editable"),
            "run requires a noneditable wheel installation")


def report(run_root, output):
    from .user_report import build_report, write_csv
    root = Path(run_root).resolve(strict=True)
    plan, _ = validate_bundle(root, check_libraries=False)
    result = build_report(root / "matrix.json", root)
    for cell in plan["matrix"]["cells"]:
        path = root / cell["receipt"]
        if path.is_file():
            try:
                receipt = json.loads(path.read_text())
                require(receipt.get("matrix_sha256") == sha(root / "matrix.json")
                        and receipt.get("input_archive_sha256") == FIXTURE_SHA,
                        f"{cell['id']}: capture matrix/fixture binding differs")
            except (ValueError, KeyError) as error:
                result["errors"].append(str(error))
                result["status"] = "failed"
    destination = new_directory(output)
    result["reproduction_plan"] = {"sha256": sha(root / "plan.json"), "execution_mode": plan["execution_mode"]}
    if plan.get("extra_execution_modes"):
        result["reproduction_plan"]["extra_execution_modes"] = plan["extra_execution_modes"]
    write_new(destination / "report.json", encoded(result))
    write_csv(result, destination / "report.csv")
    return result


def run(prepared, output):
    prepared = Path(prepared).resolve(strict=True)
    plan, preparation = validate_bundle(prepared)
    require(not plan["unresolved_library_options"], "selected mode requires explicit --library options during prepare")
    require_installed()
    root = new_directory(output)
    for relative in ("plan.json", "matrix.json", "profiles.json", "preparation.json", "inputs/recorded_inputs_v1.npz"):
        write_new(root / relative, (prepared / relative).read_bytes())
    record = {"schema": SCHEMA, "status": "failed", "prepared": str(prepared), "interpreter": sys.executable,
              "plan_sha256": sha(root / "plan.json"), "attempts": [], "task_quality_validated": False}
    try:
        for cell in plan["matrix"]["cells"]:
            command = [sys.executable, "-I", "-B", "-m", "benchmarks.regression.user_e2e", "--matrix", str(root / "matrix.json"),
                       "--cell", cell["id"], "--output-root", str(root), "--fixture", str(root / "inputs/recorded_inputs_v1.npz")]
            attempt = {"cell": cell["id"], "command": command, "optimizer_environment": cell["expected_optimizer_environment"]}
            record["attempts"].append(attempt)
            log_path = root / "logs" / (cell["id"] + ".log")
            log_path.parent.mkdir(exist_ok=True)
            with log_path.open("x") as log:
                completed = subprocess.run(command, cwd=root, env=child_environment(plan, cell, preparation),
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
            attempt.update(exit_code=completed.returncode, log_sha256=sha(log_path))
            if completed.returncode:
                break
        result = report(root, root / "validated_report")
        completed_all = (len(record["attempts"]) == len(plan["matrix"]["cells"])
                         and all(attempt.get("exit_code") == 0 for attempt in record["attempts"]))
        record["status"] = "passed" if result["status"] == "passed" and completed_all else "failed_or_incomplete"
        record["report_status"] = result["status"]
    except BaseException as error:
        record.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_new(root / "run.json", encoded(record))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True, help="profile alias or exact checkpoint Hub ID")
        command.add_argument("--mode", default="native", help="explicit profile execution mode; default native")
        command.add_argument("--extra-mode", action="append", default=[],
                             help="add a distinct schedule-changing operating point sharing the three main arms")
        command.add_argument("--library", action="append", default=[], metavar="NAME=PATH")
        if name == "prepare":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--cache-dir", type=Path)
            command.add_argument("--local-files-only", action="store_true")
    command = commands.add_parser("run")
    command.add_argument("--prepared", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("report")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = make_plan(args.model, args.mode, libraries=libraries_from_args(args.library), extra_modes=args.extra_mode)
    elif args.command == "prepare":
        result = prepare(args.model, args.mode, args.output, cache_dir=args.cache_dir,
                         local_files_only=args.local_files_only, libraries=libraries_from_args(args.library),
                         extra_modes=args.extra_mode)
    elif args.command == "run":
        result = run(args.prepared, args.output)
    else:
        result = report(args.run, args.output)
    print(encoded(result).decode(), end="")
    return 0 if result.get("status", "passed") in {"passed", "prepared_primary_checkpoint"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

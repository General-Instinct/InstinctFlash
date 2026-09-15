"""Explicit prepare/download/run commands for three historical Cosmos SDE1 screens."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
import urllib.request

from cosmos3_sde1.overlay import apply_shard, checked_file, digest, inside, relative, require


DATA = Path(__file__).with_name("data")


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def recipe(name):
    catalog = json.loads((DATA / "recipes.json").read_text())
    require(name in catalog["recipes"], "unknown recipe")
    value = catalog["recipes"][name]
    require(value["category"] == "SCREEN" and value["task_quality_certified"] is False,
            "experimental recipes never certify task quality")
    require(value["execution"]["sampling"]["sigmas"] == [1.0, 0.0]
            and value["execution"]["guidance"]["action"] == {"mode": "cfg", "scale": 1}
            and value["execution"]["nfe"] == {"prefix": 1, "action": 1}, "recipe changed sampler")
    require(re.fullmatch(r"[0-9a-f]{40}", value["base"]["revision"]) is not None,
            "base revision must be immutable")
    return value


def source_file(base, name):
    """Allow ordinary contained files and standard HF content-addressed cache links."""
    base = Path(base).absolute()
    source = base / relative(name)
    resolved = source.resolve()
    if not resolved.is_relative_to(base.resolve()):
        require(base.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", base.name)
                and resolved.parent == (base.parent.parent / "blobs").resolve()
                and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", resolved.name),
                "base symlink leaves checkpoint/cache blobs")
    require(source.is_file(), f"missing base file: {name}")
    return source


def copy_checked(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    h, count = hashlib.sha256(), 0
    with Path(source).open("rb") as original, destination.open("xb") as target:
        for block in iter(lambda: original.read(8 * 1024 * 1024), b""):
            target.write(block)
            h.update(block)
            count += len(block)
        target.flush()
        os.fsync(target.fileno())
    require(count == expected["bytes"] and h.hexdigest() == expected["sha256"],
            f"copied input differs: {destination.name}")


def link_checked_weight(source, destination, expected):
    """Link an immutable, unchanged safetensors shard without editing its inode."""
    require(Path(expected["path"]).suffix == ".safetensors"
            and expected.get("kind") != "overlay_shard" and "data_path" not in expected,
            "hardlinks are only allowed for unchanged weight shards")
    source = Path(source).resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(source.stat().st_dev == destination.parent.stat().st_dev,
            "--link-unmodified requires the same filesystem; no copy fallback")
    require(source.stat().st_size == expected["bytes"] and digest(source) == expected["sha256"],
            "source weight hash differs before linking")
    os.link(source, destination, follow_symlinks=False)
    require(os.path.samefile(source, destination)
            and destination.stat().st_size == expected["bytes"]
            and digest(destination) == expected["sha256"], "linked weight hash differs")


def materialize(name, base, output, overlay=None, *, link_unmodified=False):
    specification = recipe(name)
    require(type(link_unmodified) is bool, "link_unmodified must be explicit boolean")
    base, output = Path(base).absolute(), Path(output).absolute()
    require(base.is_dir(), "base checkpoint directory is missing")
    require(not output.exists() and not output.resolve().is_relative_to(base.resolve())
            and not base.resolve().is_relative_to(output.resolve()), "use fresh output outside base")
    if specification["family"] == "edge":
        require(overlay is not None, "Edge requires the exact released tensor overlay")
        overlay = Path(overlay).absolute()
        require(overlay.stat().st_size == specification["overlay"]["bytes"]
                and digest(overlay) == specification["overlay"]["sha256"], "wrong overlay archive")
        require(not overlay.resolve().is_relative_to(output.resolve()), "output contains overlay input")
    output.mkdir(parents=True, exist_ok=False)
    try:
        linked_weights = []
        for row in specification["files"]:
            destination = output / relative(row["path"])
            if row.get("kind") == "overlay_shard":
                destination.parent.mkdir(parents=True, exist_ok=True)
                apply_shard(source_file(base, row["path"]), overlay, specification["tensors"],
                            destination, row)
            elif "data_path" in row:
                copy_checked(checked_file(DATA, {**row, "path": row["data_path"]}), destination, row)
            elif link_unmodified and Path(row["path"]).suffix == ".safetensors":
                link_checked_weight(source_file(base, row["path"]), destination, row)
                linked_weights.append(row["path"])
            else:
                copy_checked(source_file(base, row["path"]), destination, row)
        if specification["family"] == "nano":
            # Same native config-only change as the historical untrained cost artifact.
            config_path = output / "config.json"
            config = json.loads(config_path.read_text())
            require(config["model"]["config"].get("fixed_step_sampler_config") is None,
                    "base already changed sampler")
            config["model"]["config"]["fixed_step_sampler_config"] = {
                "_type": "fixed_step_sampler_config", "sample_type": "sde", "t_list": [1.0]}
            with config_path.open("w") as stream:
                json.dump(config, stream, indent=2)
                stream.write("\n")
        declaration = {"instinctflash_schema": 1, "execution": specification["execution"],
                       "provenance": {"scope": "Public reproduction of historical SDE1 SCREEN; no task-quality qualification",
                                      "recipe": name, "base_revision": specification["base"]["revision"],
                                      "historical_receipt_sha256": specification["historical_receipt_sha256"]}}
        from instinct_compress.artifacts import write_declaration
        write_declaration(output, execution=declaration["execution"], provenance=declaration["provenance"])
        padding = json.loads((DATA / "sidecars/padding.json").read_text())
        write_json(output / "instinctcompress_action_padding.json", padding)
        binding = {"schema": 1, "recipe": name, "recipe_sha256": hashlib.sha256(
            json.dumps(specification, sort_keys=True).encode()).hexdigest(),
            "category": "SCREEN", "task_quality_certified": False,
            "link_unmodified": link_unmodified, "linked_weight_files": linked_weights,
            "files": [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size,
                       "sha256": digest(p)} for p in sorted(output.rglob("*")) if p.is_file()]}
        write_json(output / "public_preparation.json", binding)
        if specification["family"] == "edge":
            from instinct_compress.artifacts import finalize_artifact, verify_artifact
            # This creates a new public preparation manifest, preserving historical provenance.
            finalize_artifact(output, execution=declaration["execution"], provenance=declaration["provenance"],
                              required_files=("checkpoint.json", "instinctcompress_action_padding.json"))
            verify_artifact(output)
        else:
            rows = [{"relative_path": str(p.relative_to(output)), "bytes": p.stat().st_size,
                     "sha256": digest(p)} for p in sorted(output.rglob("*")) if p.is_file()]
            write_json(output / "budget_manifest.json", {
                "kind": "untrained_original_nano_cost_only", "steps": 1, "files": rows,
                "original_revision": specification["base"]["revision"], "trained_student": False,
                "quality_certified": False,
                "native_config_delta": {"model.config.fixed_step_sampler_config": {
                    "before": None, "after": config["model"]["config"]["fixed_step_sampler_config"]}}})
        return binding
    except BaseException:
        write_json(output / "preparation_failed.json", {"status": "failed", "error": traceback.format_exc()})
        raise


def plan(name):
    value = recipe(name)
    return {"recipe": name, "category": "SCREEN", "task_quality_certified": False,
            "base": value["base"], "overlay": value.get("overlay"), "auxiliary": value["auxiliary"],
            "precision": "native BF16", "tier": "NUMERIC", "execution": value["execution"],
            "warmup_requests": value["warmup_requests"], "measured_requests": value["measured_requests"],
            "historical_p50_ms": value["historical_p50_ms"],
            "current_build_reproduction_not_historical_source_identity": True}


def recipe_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_downloads(name, downloads, output, *, link_unmodified=False):
    """Use the pinned paths returned by fetch without shell-variable extraction."""
    value = recipe(name)
    downloads = Path(downloads).resolve(strict=True)
    record = json.loads(inside(downloads, "downloads.json").read_text())
    require(record.get("schema") == 1 and record.get("recipe") == name
            and record.get("recipe_sha256") == recipe_digest(value), "downloaded recipe binding differs")
    base = Path(record["base"])
    require(base.is_absolute() and base.is_dir() and base.name == value["base"]["revision"],
            "downloaded base is not the pinned snapshot")
    auxiliary = Path(record["auxiliary"])
    require(auxiliary.is_absolute() and auxiliary.is_file()
            and auxiliary.stat().st_size == value["auxiliary"]["bytes"]
            and digest(auxiliary) == value["auxiliary"]["sha256"], "downloaded VAE differs")
    overlay = record.get("overlay")
    if value["family"] == "edge":
        require(isinstance(overlay, str) and Path(overlay).is_absolute()
                and Path(overlay).resolve().is_relative_to(downloads), "downloaded overlay leaves its directory")
    else:
        require(overlay is None, "Nano original weights require no overlay")
    return materialize(name, base, output, overlay, link_unmodified=link_unmodified)


def fetch(name, output, overlay_url=None):
    """Explicit network phase; the run phase is offline."""
    value = recipe(name)
    url = None
    if "overlay" in value:
        url = overlay_url or value["overlay"].get("url")
        require(isinstance(url, str) and url.startswith("https://"),
                "supply the published overlay HTTPS URL; publication URL has not been assigned yet")
    output = Path(output).absolute()
    require(not output.exists(), "use a fresh download directory")
    output.mkdir(parents=True)
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        base = snapshot_download(value["base"]["model_id"], revision=value["base"]["revision"])
        auxiliary = value["auxiliary"]
        vae = hf_hub_download(auxiliary["model_id"], auxiliary["filename"], revision=auxiliary["revision"])
        require(Path(vae).stat().st_size == auxiliary["bytes"] and digest(vae) == auxiliary["sha256"],
                "auxiliary VAE differs")
        result = {"schema": 1, "recipe": name, "recipe_sha256": recipe_digest(value),
                  "base": base, "auxiliary": vae, "overlay": None}
        if "overlay" in value:
            target = output / value["overlay"]["path"]
            count, h = 0, hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response, target.open("xb") as stream:
                require(response.geturl().startswith("https://"), "overlay redirected away from HTTPS")
                for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    count += len(block)
                    require(count <= value["overlay"]["bytes"], "oversized overlay download")
                    h.update(block)
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            require(count == value["overlay"]["bytes"] and h.hexdigest() == value["overlay"]["sha256"],
                    "overlay download hash differs")
            result["overlay"] = str(target)
        write_json(output / "downloads.json", result)
        return result
    except BaseException:
        write_json(output / "download_failed.json", {"status": "failed", "error": traceback.format_exc()})
        raise


def validate_prepared(name, checkpoint):
    value = recipe(name)
    checkpoint = Path(checkpoint)
    binding = json.loads(inside(checkpoint, "public_preparation.json").read_text())
    require(binding["recipe"] == name and binding["recipe_sha256"] == hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()).hexdigest(), "prepared recipe differs")
    recorded = {row["path"] for row in binding["files"]}
    require(len(recorded) == len(binding["files"]), "duplicate preparation entry")
    manifest = "instinctcompress_manifest.json" if value["family"] == "edge" else "budget_manifest.json"
    actual = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file()}
    require(actual == recorded | {manifest, "public_preparation.json"}, "prepared file set differs")
    for row in binding["files"]:
        checked_file(checkpoint, row)
    return value


def run(name, checkpoint, output, fixture=None):
    value = validate_prepared(name, checkpoint)
    output = Path(output).absolute()
    require(not output.exists(), "use a fresh run output")
    if fixture is None:
        from benchmarks.regression.reproduce import fixture_path
        fixture = fixture_path()
    from cosmos3_sde1.fixture import load_frames
    load_frames(fixture)
    output.mkdir(parents=True)
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith("IFL_") and k not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    module = "cosmos3_sde1.benchmark_" + value["family"]
    command = [sys.executable, "-I", "-B", "-m", module, str(Path(checkpoint).absolute()),
               str(output / "receipt.json"), "--attention", "cudnn", "--allow-unqualified",
               "--fixture", str(Path(fixture).absolute())]
    if value["family"] == "nano":
        command.extend(["--cache-mode", "reuse", "--swiglu"])
    write_json(output / "invocation.json", {"command": command, "plan": plan(name),
                                           "source_sha256": digest(Path(__file__))})
    try:
        with (output / "run.log").open("x") as log:
            completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=3600, check=False)
        record = json.loads((output / "receipt.json").read_text()) if (output / "receipt.json").is_file() else {}
        ok = (completed.returncode == 0 and record.get("ok") is True
              and record.get("category") == "SCREEN" and record.get("task_quality_certified") is False)
        result = {"status": "passed" if ok else "failed", "returncode": completed.returncode,
                  "category": "SCREEN", "task_quality_certified": False,
                  "p50_ms": record.get("p50_ms"), "p95_ms": record.get("p95_ms")}
    except BaseException:
        write_json(output / "completion.json", {"status": "failed", "category": "SCREEN",
                   "task_quality_certified": False, "error": traceback.format_exc()})
        raise
    write_json(output / "completion.json", result)
    require(ok, "experimental reproduction failed; logs and partial outputs retained")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="operation", required=True)
    for operation in ["plan", "fetch", "prepare", "run"]:
        sub = subs.add_parser(operation)
        sub.add_argument("recipe", choices=["edge-seed12031", "edge-seed12032", "nano-original"])
        if operation != "plan":
            sub.add_argument("--output", type=Path, required=True)
        if operation == "fetch":
            sub.add_argument("--overlay-url")
        if operation == "prepare":
            inputs = sub.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--base", type=Path)
            inputs.add_argument("--downloads", type=Path)
            sub.add_argument("--overlay", type=Path)
            sub.add_argument("--link-unmodified", action="store_true")
        if operation == "run":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    if args.operation == "plan":
        result = plan(args.recipe)
    elif args.operation == "fetch":
        result = fetch(args.recipe, args.output, args.overlay_url)
    elif args.operation == "prepare":
        if args.downloads:
            if args.overlay:
                parser.error("--downloads cannot be combined with manual --overlay")
            result = prepare_downloads(args.recipe, args.downloads, args.output,
                                       link_unmodified=args.link_unmodified)
        else:
            result = materialize(args.recipe, args.base, args.output, args.overlay,
                                 link_unmodified=args.link_unmodified)
    else:
        result = run(args.recipe, args.checkpoint, args.output, args.fixture)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

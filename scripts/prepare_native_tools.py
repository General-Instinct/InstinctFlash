#!/usr/bin/env python3
"""Prepare Cosmos's unchanged, pinned uvx/HF CLI in an isolated CPU-only cache.

Only ``prepare`` downloads tool packages. It never downloads model weights or
imports a vendor/model module. ``probe`` invokes the native CLI on an already
prepared exact Wan VAE cache with both package and Hub network access disabled.
"""
from __future__ import annotations

import argparse
import datetime as dt
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


CATALOG = Path(__file__).resolve().parents[1] / "release/vendor/cosmos/hf_tool.json"
WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B"
WAN_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
WAN_FILE = "Wan2.2_VAE.pth"
WAN_BYTES = 2818839170


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_catalog(path: Path) -> dict:
    data = json.loads(path.read_text())
    if (data.get("schema") != "instinctflash.cosmos_native_hf_tool.v1" or
            data["uv_version"] != "0.12.5" or data["python_minor"] != "3.13" or
            data["native_argv_prefix"] != ["uvx", "--with", "click", "hf@1.16.4"]):
        raise ValueError("unexpected native tool contract")
    constraint = path.parent / data["constraints"]["path"]
    if constraint.parent != path.parent or constraint.is_symlink() or sha(constraint) != data["constraints"]["sha256"]:
        raise ValueError("tool constraint source mismatch")
    pins = dict(line.split("==") for line in constraint.read_text().splitlines())
    if pins != data["packages"] or pins.get("hf") != "1.16.4" or pins.get("huggingface-hub") != "1.16.4":
        raise ValueError("tool package pins mismatch")
    return data


def runtime_environment(root: Path, python: Path, uvx: Path) -> dict:
    return {"UV_CACHE_DIR": str(root / "cache"), "UV_TOOL_DIR": str(root / "installed_tools"),
            "UV_CONSTRAINT": str(root / "constraints.txt"), "UV_PYTHON": str(python),
            "UV_PYTHON_DOWNLOADS": "never", "UV_NO_CONFIG": "1", "UV_NO_ENV_FILE": "1",
            "UV_NO_BUILD": "1", "UV_ISOLATED": "1", "UV_DEFAULT_INDEX": "https://pypi.org/simple",
            "UV_OFFLINE": "1", "PATH": str(uvx.parent) + os.pathsep + os.environ.get("PATH", os.defpath)}


def clean_environment(values: dict, *, online: bool = False) -> dict:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("UV_", "PIP_")) and key not in
           {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    env.update(values)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", UV_CONCURRENT_DOWNLOADS="1",
               UV_LINK_MODE="copy", PIP_CONFIG_FILE=os.devnull)
    if online:
        env.pop("UV_OFFLINE", None)
    return env


def execute(root: Path, name: str, command: list[str], env: dict, *, timeout: float = 240) -> str:
    with (root / f"{name}.stdout").open("x") as out, (root / f"{name}.stderr").open("x") as err:
        result = subprocess.run(command, env=env, cwd=root, stdout=out, stderr=err,
                                check=False, timeout=timeout, text=True)
    write_json(root / f"{name}.json", {"argv": command, "returncode": result.returncode,
                                      "UV_OFFLINE": env.get("UV_OFFLINE"),
                                      "HF_HUB_OFFLINE": env.get("HF_HUB_OFFLINE")})
    if result.returncode:
        raise RuntimeError(f"native tool stage failed: {name}; preserved original logs")
    return (root / f"{name}.stdout").read_text()


def prepare(root: Path, python: Path, uvx: Path, catalog_path: Path) -> dict:
    catalog = load_catalog(catalog_path)
    root, python = root.expanduser().absolute(), python.expanduser().absolute()
    # Preserve the venv interpreter path; resolving its symlink loses environment identity.
    uvx = uvx.expanduser().absolute()
    if root.exists() or root.is_symlink() or not python.is_file() or not uvx.is_file():
        raise ValueError("tool root must be new and interpreter/uvx must exist")
    root.mkdir(parents=True)
    receipt = {"schema": "instinctflash.native_tool_preparation.v1", "root": str(root),
               "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "helper_sha256": sha(Path(__file__)), "catalog_sha256": sha(catalog_path),
               "uvx": str(uvx), "uvx_sha256": sha(uvx), "python": str(python),
               "GPU_verified": False, "model_constructed": False, "weights_downloaded": False}
    write_json(root / "plan.json", receipt)
    try:
        shutil.copyfile(catalog_path.parent / catalog["constraints"]["path"], root / "constraints.txt")
        values = runtime_environment(root, python, uvx)
        env = clean_environment(values, online=True)
        version = execute(root, "uvx_version", [str(uvx), "--version"], env).strip()
        if not version.startswith(f"uvx {catalog['uv_version']} ("):
            raise ValueError("native uvx version differs from pinned tool")
        minor = execute(root, "python_version", [str(python), "-I", "-B", "-c",
            "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"], env).strip()
        if minor != catalog["python_minor"]:
            raise ValueError("native tool requires the pinned Cosmos Python minor")
        command = [str(uvx), *catalog["native_argv_prefix"][1:], "--help"]
        execute(root, "public_tool_install", command, env)
        execute(root, "offline_tool_reuse", command, clean_environment(values))
        candidates = list((root / "cache/archive-v0").glob("*/lib/python*/site-packages/hf-1.16.4.dist-info/METADATA"))
        if len(candidates) != 1:
            raise ValueError("expected exactly one isolated native HF tool environment")
        site = candidates[0].parent.parent
        packages, inventory = {}, []
        for path in sorted(site.glob("*.dist-info/METADATA")):
            metadata = Parser().parsestr(path.read_text())
            name = normalize(metadata["Name"])
            if name in packages:
                raise ValueError("duplicate native tool distribution")
            packages[name] = metadata["Version"]
            inventory.append({"path": str(path), "sha256": sha(path)})
        if packages != catalog["packages"]:
            raise ValueError("resolved native tool dependency set differs from exact pins")
        tool_python = site.parents[2] / "bin/python"
        uv = uvx.with_name("uv")
        execute(root, "tool_pip_check", [str(uv), "pip", "check", "--python", str(tool_python)],
                clean_environment(values))
        with (root / "run.env").open("x") as out:
            for key, value in values.items():
                if key == "PATH":
                    out.write(f"export PATH={shlex.quote(str(uvx.parent))}:\"$PATH\"\n")
                else:
                    out.write(f"export {key}={shlex.quote(value)}\n")
        receipt.update(status="tool_packages_checked_offline_reuse_passed", environment=values,
                       package_versions=packages, metadata_inventory=inventory,
                       constraint_sha256=sha(root / "constraints.txt"), uvx_version=version,
                       run_env_sha256=sha(root / "run.env"), tool_python=str(tool_python))
        write_json(root / "completion.json", receipt)
        return receipt
    except BaseException as exc:
        receipt.update(status="failed_preserved", error_type=type(exc).__name__, automatic_retry=False)
        write_json(root / "failure.json", receipt)
        raise


def probe(tool_root: Path, cache: Path, output: Path) -> dict:
    tool_root, cache, output = (p.expanduser().absolute() for p in (tool_root, cache, output))
    done = json.loads((tool_root / "completion.json").read_text())
    if done["status"] != "tool_packages_checked_offline_reuse_passed":
        raise ValueError("tool preparation did not pass")
    if (sha(tool_root / "constraints.txt") != done["constraint_sha256"] or
            sha(Path(done["uvx"])) != done["uvx_sha256"]):
        raise ValueError("prepared native tool source changed")
    expected = cache / ("models--" + WAN_REPO.replace("/", "--")) / "snapshots" / WAN_REVISION / WAN_FILE
    if not expected.is_file() or expected.stat().st_size != WAN_BYTES:
        raise ValueError("exact pre-prepared Wan asset is missing; this command never downloads weights")
    output.mkdir(parents=True, exist_ok=False)
    values = runtime_environment(tool_root, Path(done["python"]), Path(done["uvx"]))
    env = clean_environment(values)
    env.update(HF_HUB_OFFLINE="1", HF_HUB_CACHE=str(cache), HUGGINGFACE_HUB_CACHE=str(cache))
    command = ["uvx", "--with", "click", "hf@1.16.4", "download", "--format=json", WAN_REPO,
               "--repo-type", "model", "--revision", WAN_REVISION, WAN_FILE]
    result = json.loads(execute(output, "native_cached_download", command, env))
    actual = Path(result["path"])
    if actual.resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError("native cached resolver returned a different asset")
    receipt = {"schema": "instinctflash.native_tool_cache_probe.v1", "status": "passed",
               "tool_completion_sha256": sha(tool_root / "completion.json"), "helper_sha256": sha(Path(__file__)),
               "native_argv": command, "returned_path": str(actual), "expected_path": str(expected),
               "bytes": expected.stat().st_size, "weights_downloaded": False, "weight_rehashed": False,
               "GPU_initialized": False, "model_constructed": False, "UV_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
               "scope": "Exact native HF subprocess resolves the prepared snapshot and file size offline; prior auxiliary preparation owns content-hash qualification."}
    write_json(output / "completion.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("plan", "prepare", "probe"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--uvx", type=Path, default=Path(shutil.which("uvx") or "uvx"))
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command != "plan" and args.root is None:
        parser.error("prepare/probe require --root")
    if args.command == "probe" and (args.cache_dir is None or args.output is None):
        parser.error("probe requires --cache-dir and a new --output")
    try:
        if args.command == "plan":
            result = load_catalog(args.catalog)
        elif args.command == "prepare":
            result = prepare(args.root, args.python, args.uvx, args.catalog)
        else:
            result = probe(args.root, args.cache_dir, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed_preserved", "error_type": type(exc).__name__,
                          "detail": "See the owned output logs; no automatic retry or model-weight download."}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

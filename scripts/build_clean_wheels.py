#!/usr/bin/env python3
"""Freeze package sources outside the checkout and build core, adapter and serving wheels.

Tracked Markdown is read from HEAD. Untracked Markdown, evaluation receipts,
build directories and native binaries are excluded. Native binaries may only
be included through explicit --native-file SOURCE DESTINATION arguments.
The helper neither imports model stacks nor compiles or executes CUDA code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys


PACKAGES = {
    "instinctflash": ".",
    "pi05_iwm": "examples/pi05_vla",
    "lingbot_vla_iwm": "examples/lingbot_vla",
    "lingbot_vla_v2_iwm": "examples/lingbot_vla_v2",
    "groot_n17_iwm": "examples/groot_n17",
    "dreamzero_iwm": "examples/dreamzero",
    "cosmos3_iwm": "examples/cosmos3_policy",
    "flash_rt": "serving",
}
SOURCE_ROOTS = {"instinctflash", "benchmarks", "examples", "serving", "scripts", "tests"}
ROOT_FILES = {"pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in", "LICENSE", "LICENSE.txt",
              "LICENSE.rst", "NOTICE", "INSTALL.rst", "uv.lock", ".gitignore"}
GENERATED_PARTS = {".git", "__pycache__", "build", "dist", "results", "logs", "runs",
                   "receipts", "matched", "research", "node_modules", "venv", ".venv"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib", ".a", ".o",
                     ".log", ".jsonl", ".npz", ".npy", ".safetensors", ".pt", ".pth",
                     ".bin", ".gz", ".zip", ".whl", ".nsys-rep", ".qdstrm"}


def _git(repository: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repository), *args])


def _source_file(relative: str) -> bool:
    path = PurePosixPath(relative)
    if any(part in GENERATED_PARTS or part.endswith(".egg-info") or part.startswith(".venv-")
           or part.startswith("build-") for part in path.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES or re.search(r"\.so(?:\.\d+)*$", path.name):
        return False
    if path.parts[:3] == ("serving", "third_party", "cutlass"):
        return False
    if len(path.parts) == 1:
        return path.name in ROOT_FILES or (path.name.startswith("requirements") and path.suffix in {".txt", ".in"})
    return path.parts[0] in SOURCE_ROOTS


def _native_destination(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (not path.is_absolute() and "\\" not in relative
            and not any(part in {"", ".", ".."} for part in relative.split("/"))
            and path.parts[:2] in {("serving", "flash_rt"), ("instinctflash", "native")}
            and bool(re.search(r"\.so(?:\.\d+)*$", path.name)))


def create_snapshot(repository: Path, output: Path, *, native_files=()) -> dict:
    repository, output = repository.resolve(), output.resolve()
    if output.is_relative_to(repository):
        raise ValueError("build output must be outside the source checkout")
    if output.exists():
        raise ValueError(f"refusing to replace existing build output: {output}")
    head = _git(repository, "rev-parse", "HEAD").decode().strip()
    tracked_head = set(_git(repository, "ls-tree", "-r", "--name-only", "-z", "HEAD").decode().split("\0"))
    current = set(_git(repository, "ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0"))
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    entries = {}
    # Restoring HEAD Markdown also handles a tracked document deleted in the worktree.
    candidates = current | {name for name in tracked_head if name.lower().endswith(".md")}
    for relative in sorted(candidates - {""}):
        is_markdown = relative.lower().endswith(".md")
        if is_markdown:
            if relative not in tracked_head:
                continue
            # Preserve every tracked Markdown document, except generated build/result trees.
            if any(part in GENERATED_PARTS for part in PurePosixPath(relative).parts):
                continue
            content = _git(repository, "show", f"HEAD:{relative}")
            origin = "HEAD"
            mode = 0o644
        else:
            if not _source_file(relative):
                continue
            original = repository / relative
            if not original.exists():
                continue
            if original.is_symlink():
                resolved = original.resolve()
                if (not resolved.is_relative_to(repository) or not resolved.is_file()
                        or resolved.suffix != original.suffix):
                    raise ValueError(f"source symlink must target an in-repository source file: {relative}")
            if not original.is_file():
                continue
            content = original.read_bytes()
            # Setuptools also materializes package file symlinks in the wheel. Keeping a
            # runtime link into an excluded eval tree would leave a dangling module here.
            origin = "worktree_symlink_materialized" if original.is_symlink() else "worktree"
            mode = original.stat().st_mode & 0o777
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target.chmod(mode)
        entries[relative] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                             "origin": origin}
    injected = []
    for original, relative in native_files:
        if not _native_destination(relative):
            raise ValueError(f"native destination must be a package .so path: {relative!r}")
        original = Path(original).resolve()
        if not original.is_file():
            raise ValueError(f"native input is not a file: {original}")
        if relative in entries:
            raise ValueError(f"duplicate native destination: {relative}")
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        content = target.read_bytes()
        entries[relative] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                             "origin": "explicit_native_file", "input_path": str(original)}
        injected.append(relative)
    manifest = {"schema": 1, "repository": str(repository), "head": head,
                "snapshot": str(source), "files": entries, "native_files": injected,
                "markdown_policy": "tracked HEAD contents only; untracked Markdown excluded",
                "wheel_scope": "package sources only; no model weights or evaluation results",
                "wheels": []}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_wheels(manifest: dict, output: Path, *, python: str) -> None:
    source = Path(manifest["snapshot"])
    wheelhouse = output / "wheels"
    wheelhouse.mkdir()
    logs = output / "logs"
    logs.mkdir()
    # Set timestamps consistently for reproducible archives where the backend supports it.
    epoch = _git(Path(manifest["repository"]), "show", "-s", "--format=%ct", "HEAD").decode().strip()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "SOURCE_DATE_EPOCH": epoch}
    environment.pop("PYTHONPATH", None)
    for name, directory in PACKAGES.items():
        before = set(wheelhouse.glob("*.whl"))
        command = [python, "-m", "pip", "wheel", str(source / directory), "--no-deps",
                   "--wheel-dir", str(wheelhouse), "--disable-pip-version-check"]
        with (logs / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        created = set(wheelhouse.glob("*.whl")) - before
        if len(created) != 1:
            raise RuntimeError(f"{name}: expected exactly one new wheel, found {created}")
        wheel = created.pop()
        has_native = any(path.startswith("serving/flash_rt/" if name == "flash_rt" else "instinctflash/")
                         for path in manifest["native_files"]) if name in {"flash_rt", "instinctflash"} else False
        if has_native and wheel.name.endswith("-any.whl"):
            raise RuntimeError(f"native binaries were packaged into a platform-independent wheel: {wheel.name}")
        manifest["wheels"].append({"package": name, "path": str(wheel), "bytes": wheel.stat().st_size,
                                   "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                                   "native_binaries_included": has_native})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="new directory outside the checkout")
    parser.add_argument("--python", default=sys.executable, help="interpreter used for pip wheel")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--native-file", action="append", nargs=2, metavar=("SOURCE", "DESTINATION"), default=[])
    args = parser.parse_args()
    manifest = create_snapshot(args.source, args.output, native_files=args.native_file)
    if not args.snapshot_only:
        build_wheels(manifest, args.output.resolve(), python=args.python)
    print(json.dumps({"manifest": str(args.output.resolve() / "manifest.json"),
                      "files": len(manifest["files"]), "wheels": len(manifest["wheels"]),
                      "native_files": manifest["native_files"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

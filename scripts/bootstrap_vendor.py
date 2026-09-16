#!/usr/bin/env python3
"""Create a separate public-source inference environment; never load model weights.

Targets have separate, pinned vendor profiles; the default is Linux/aarch64
Thor. Source checkout and environment destinations must not already exist.
Public dependency resolution, both package checks, and the offline doctor are
recorded separately. A successful package install is not GPU qualification.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import unquote, urldefrag, urlsplit

import prepare_native_tools
import repair_vendor_wheel

CHECKOUT = Path(__file__).resolve().parents[1]
FAMILIES = ("pi05", "vla4", "vla2", "groot", "va", "edge", "nano", "dreamzero")
TARGETS = {
    "jetson_thor": {"machine": "aarch64", "ptxas_target": "sm_110a", "ptx_version": "9.0"},
    "rtx4090": {"machine": "x86_64", "ptxas_target": "sm_89", "ptx_version": "7.8"},
    "rtx5090": {"machine": "x86_64", "ptxas_target": "sm_120", "ptx_version": "9.0"},
}
PURE_SDISTS = {
    "antlr4-python3-runtime": ("4.9.3", "f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b"),
    "iopath": ("0.1.10", "3311c16a4d9137223e20f141655759933e1eda24f8bff166af834af3c645ef01"),
}


def sha(path: Path) -> str:
    return repair_vendor_wheel.sha256(path)


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def checked_path(base: Path, relative: str, expected: str | None = None) -> Path:
    p = base / relative
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"invalid source input: {relative}")
    if expected is not None and sha(p) != expected:
        raise ValueError(f"source input hash mismatch: {relative}")
    return p


def load_profile(family: str, checkout: Path = CHECKOUT, *, target: str = "jetson_thor") -> dict:
    if target not in TARGETS:
        raise ValueError(f"unsupported deployment target: {target}")
    base = checkout / "release/vendor"
    prefix = "" if target == "jetson_thor" else f"{target}/"
    p = checked_path(base, f"{prefix}{family}/bootstrap.json")
    profile = json.loads(p.read_text())
    if profile["family"] != family or profile["schema"] != "instinctflash.vendor_bootstrap.v1":
        raise ValueError("invalid bootstrap profile")
    if (profile["target"]["system"] != "Linux"
            or profile["target"]["machine"] != TARGETS[target]["machine"]
            or profile.get("deployment_target", "jetson_thor") != target):
        raise ValueError("bootstrap profile does not match requested deployment target")
    for key in ("constraints", "requirements"):
        checked_path(base, profile[key]["path"], profile[key]["sha256"])
    source = profile["source_manifest"]
    if source:
        manifest = json.loads(checked_path(base, source).read_text())
        info = manifest["source"]
        if not info["repository"].startswith("https://github.com/") or "@" in info["repository"]:
            raise ValueError("vendor source must be a public GitHub HTTPS URL")
        if len(info["revision"]) != 40 or any(c not in "0123456789abcdef" for c in info["revision"]):
            raise ValueError("vendor revision must be an exact commit")
        if manifest["patch"]:
            patch = manifest["patch"]
            checked_path(base / Path(source).parent, patch["path"], patch["sha256"])
        profile["source"] = manifest
    patch = profile["packaging_patch"]
    if patch:
        checked_path(base, patch["path"], patch["sha256"])
        if set(patch["before"]) - {"pyproject.toml", "setup.py", "requirements.txt"}:
            raise ValueError("inference packaging patch must not change runtime source")
    if profile.get("native_tool"):
        tool = profile["native_tool"]
        path = checked_path(base, tool["path"], tool["sha256"])
        if family not in {"edge", "nano"}:
            raise ValueError("unexpected native subprocess tool for this family")
        prepare_native_tools.load_catalog(path)
    return profile


def clean_environment(cache_dir: Path | None) -> dict:
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith(("PIP_", "UV_", "CONDA_", "IFL_")) or key in {
            "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "DYNAMIC_CACHE_SCHEDULE",
            "NUM_DIT_STEPS", "LOAD_TRT_ENGINE", "ENABLE_TENSORRT",
            "LINGBOT_ROOT", "LINGBOT_VLA_ROOT", "LINGBOT_VLA_V2_ROOT", "DREAMZERO_ROOT", "GR00T_ROOT",
        }:
            env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES="", GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0",
               PYTHONDONTWRITEBYTECODE="1", UV_NO_CONFIG="1", UV_CONCURRENT_DOWNLOADS="1",
               UV_CONCURRENT_BUILDS="1", UV_LINK_MODE="copy", PIP_CONFIG_FILE=os.devnull)
    if cache_dir:
        env["UV_CACHE_DIR"] = str(cache_dir)
    return env


def activation_environment(profile: dict, vendor: Path) -> dict:
    values = {}
    if profile["root_env"]:
        values[profile["root_env"]] = str(vendor)
    if profile["family"] == "dreamzero":
        # Documented package-update opt-out; no model or preprocessing change.
        values["NO_ALBUMENTATIONS_UPDATE"] = "1"
    return values


def vendor_source_transport(profile: dict, source: Path | None) -> dict | None:
    """Describe an explicit local transport without replacing the pinned source."""
    if not profile["source_manifest"]:
        if source is not None:
            raise ValueError("--vendor-source requires a family with a Git source profile")
        return None
    if source is None:
        return {"kind": "public_https", "url": profile["source"]["source"]["repository"]}
    path = source.expanduser().resolve(strict=True)
    if path.is_dir():
        return {"kind": "local_git_repository", "path": str(path)}
    if path.is_file():
        with path.open("rb") as f:
            header = f.readline(32)
        if header in (b"# v2 git bundle\n", b"# v3 git bundle\n"):
            return {"kind": "local_git_bundle", "path": str(path),
                    "bytes": path.stat().st_size, "sha256": sha(path)}
    raise ValueError("--vendor-source must be a local Git repository or Git bundle")


def admit_dependency_wheelhouse(directory: Path, profile: dict, *, checkout: Path = CHECKOUT) -> dict:
    """Verify a transported cache before any wheel is installed or imported.

    Requirements/constraints remain authoritative. Explicit public URL pins
    additionally require their original SHA256, even if a manifest was edited.
    A local wheelhouse is an explicit caller-supplied dependency source, whose
    complete manifest and hashes are recorded separately from public resolution.
    """
    directory = directory.expanduser().resolve(strict=True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    target = profile.get("deployment_target", "jetson_thor")
    expected = {"requirements_sha256": profile["requirements"]["sha256"],
                "constraints_sha256": profile["constraints"]["sha256"],
                "public_wheel_overrides": profile["public_wheel_overrides"]}
    schema = manifest.get("schema")
    if schema == "instinctflash.dependency_wheelhouse.v2":
        expected.update(python_metadata_overlays=profile.get("python_metadata_overlays", []),
                        audited_pure_python_sdists=profile.get("audited_pure_python_sdists", {}))
    elif profile.get("python_metadata_overlays") or profile.get("audited_pure_python_sdists"):
        raise ValueError("this recipe requires wheelhouse v2 original dependency inputs")
    elif manifest.get("overlay_inputs") or manifest.get("source_archives"):
        raise ValueError("original dependency inputs require wheelhouse v2")
    if (schema not in {"instinctflash.dependency_wheelhouse.v1", "instinctflash.dependency_wheelhouse.v2"}
            or manifest.get("family") != profile["family"] or manifest.get("target") != target
            or manifest.get("inputs") != expected):
        raise ValueError("dependency wheelhouse does not match the selected pinned recipe")
    relative = manifest.get("wheel_directory", ".")
    if not isinstance(relative, str) or relative not in (".", "wheels"):
        raise ValueError("dependency wheelhouse uses an unsupported archive directory")
    wheel_directory = directory / relative
    if wheel_directory.is_symlink() or not wheel_directory.is_dir():
        raise ValueError("dependency wheelhouse archive directory is missing or linked")
    entries = manifest.get("wheels")
    if not isinstance(entries, list) or not entries:
        raise ValueError("dependency wheelhouse has no verified wheels")
    seen, by_name = set(), {}
    public_urls = {urldefrag(url)[0] for url in profile["public_wheel_overrides"].values()}
    for entry in entries:
        filename = entry["filename"]
        if (not isinstance(filename, str) or Path(filename).name != filename
                or not filename.endswith(".whl") or filename in seen):
            raise ValueError("dependency wheelhouse contains an invalid or repeated wheel path")
        seen.add(filename)
        url = urlsplit(entry["url"])
        if (url.scheme != "https" or url.username or url.password
                or (url.hostname not in {"files.pythonhosted.org", "download.pytorch.org",
                                         "download-r2.pytorch.org"}
                    and urldefrag(entry["url"])[0] not in public_urls)
                or unquote(Path(url.path).name) != filename):
            raise ValueError("dependency wheelhouse wheel has an invalid public source binding")
        digest = entry["sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("dependency wheelhouse requires full SHA256 digests")
        path = wheel_directory / filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"dependency wheelhouse file missing or wrong size: {filename}")
        if sha(path) != digest:
            raise ValueError(f"dependency wheelhouse hash mismatch: {filename}")
        with zipfile.ZipFile(path) as archive:
            metadata = [info for info in archive.infolist()
                        if info.filename.endswith(".dist-info/METADATA") and info.filename.count("/") == 1]
            if len(metadata) != 1 or metadata[0].file_size > 1 << 20:
                raise ValueError(f"dependency wheel metadata is invalid: {filename}")
            from email.parser import BytesParser
            message = BytesParser().parsebytes(archive.read(metadata[0]))
        normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
        if (normalize(message.get("Name", "")) != normalize(entry["name"])
                or message.get("Version") != entry["version"]):
            raise ValueError(f"dependency wheel metadata differs from manifest: {filename}")
        by_name.setdefault(normalize(entry["name"]), []).append(entry)
    actual = {path.name for path in wheel_directory.glob("*.whl")}
    if actual != seen:
        raise ValueError("dependency wheelhouse contains wheels outside the verified manifest")
    overrides = {}
    for name, original_url in profile["public_wheel_overrides"].items():
        url, fragment = urldefrag(original_url)
        if not fragment.startswith("sha256="):
            raise ValueError("public wheel override requires its original SHA256")
        matches = [entry for entry in by_name.get(re.sub(r"[-_.]+", "-", name).lower(), [])
                   if entry["sha256"] == fragment.removeprefix("sha256=")
                   and urldefrag(entry["url"])[0] == url]
        if len(matches) != 1:
            raise ValueError(f"dependency wheelhouse lacks the exact public wheel: {name}")
        overrides[name] = str(wheel_directory / matches[0]["filename"])
    originals = (_admit_original_dependency_inputs(directory, manifest, profile, checkout, set(by_name))
                 if schema == "instinctflash.dependency_wheelhouse.v2" else {})
    return {"directory": str(directory), "wheel_directory": str(wheel_directory),
            "manifest_sha256": sha(manifest_path),
            "target": target, "family": profile["family"], "verified_wheels": len(seen),
            "explicit_override_paths": overrides,
            "scope": "caller-supplied wheelhouse; recipe, file hashes and metadata verified", **originals}


def _admit_original_dependency_inputs(directory: Path, manifest: dict, profile: dict,
                                     checkout: Path, wheel_names: set[str]) -> dict:
    """Bind original overlay wheels and audited sources before offline preparation."""
    normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
    overlay_rules = {}
    for reference in profile.get("python_metadata_overlays", []):
        if reference["path"] in overlay_rules:
            raise ValueError("repeated metadata overlay recipe")
        rule_path = checked_path(checkout / "release/vendor", reference["path"], reference["sha256"])
        overlay_rules[reference["path"]] = (reference, json.loads(rule_path.read_text()))
    sources = profile.get("audited_pure_python_sdists", {})
    paths = {"overlay_input_paths": {}, "source_archive_paths": {}}
    selected_names = set(wheel_names)
    for group, expected_count in (("overlay_inputs", len(overlay_rules)), ("source_archives", len(sources))):
        entries = manifest.get(group, [])
        if not isinstance(entries, list) or len(entries) != expected_count:
            raise ValueError(f"dependency wheelhouse lacks exact {group}")
        folder = directory / group
        if (entries or folder.exists() or folder.is_symlink()) and (folder.is_symlink() or not folder.is_dir()):
            raise ValueError(f"dependency wheelhouse {group} directory is missing or linked")
        seen, bindings = set(), set()
        for entry in entries:
            filename = entry["filename"]
            if not isinstance(filename, str) or Path(filename).name != filename or filename in seen:
                raise ValueError(f"invalid original dependency path in {group}")
            seen.add(filename)
            name = normalize(entry["name"])
            if name in selected_names:
                raise ValueError("original dependency input collides with another selected package")
            selected_names.add(name)
            if group == "overlay_inputs":
                key = entry["rule_path"]
                if key not in overlay_rules or key in bindings:
                    raise ValueError("unrecognized or repeated metadata overlay input")
                reference, rule = overlay_rules[key]
                if (entry.get("rule_sha256") != reference["sha256"]
                        or any(entry[field] != rule[field] for field in ("filename", "url", "bytes", "sha256"))):
                    raise ValueError("original metadata overlay input differs from pinned rule")
                if not filename.endswith(".whl"):
                    raise ValueError("metadata overlay input must be the original wheel")
            else:
                key = name
                if key not in sources or key in bindings or key not in PURE_SDISTS:
                    raise ValueError("unrecognized or repeated audited source archive")
                source = sources[key]
                url, fragment = urldefrag(source["url"])
                parsed = urlsplit(url)
                expected_version, expected_sha = PURE_SDISTS[key]
                if (source["version"] != expected_version or fragment != "sha256=" + expected_sha
                        or parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org"
                        or parsed.username or parsed.password or parsed.query
                        or entry["version"] != expected_version or entry["sha256"] != expected_sha
                        or entry["bytes"] != source["bytes"] or entry["url"] != url
                        or filename != unquote(Path(urlsplit(url).path).name)):
                    raise ValueError("original source archive differs from audited profile")
            bindings.add(key)
            path = folder / filename
            if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]
                    or sha(path) != entry["sha256"]):
                raise ValueError(f"original dependency artifact differs from full size or SHA256: {filename}")
            if group == "overlay_inputs":
                with zipfile.ZipFile(path) as archive:
                    from email.parser import BytesParser
                    metadata = BytesParser().parsebytes(archive.read(rule["metadata_member"]))
                if normalize(metadata.get("Name", "")) != name or metadata.get("Version") != entry["version"]:
                    raise ValueError("original overlay wheel metadata differs from manifest")
                paths["overlay_input_paths"][key] = str(path)
            else:
                paths["source_archive_paths"][key] = str(path)
        if folder.exists() and (folder.is_symlink() or {p.name for p in folder.iterdir()} != seen):
            raise ValueError(f"dependency wheelhouse contains unlisted {group}")
    return {**paths, "verified_overlay_inputs": len(paths["overlay_input_paths"]),
            "verified_source_archives": len(paths["source_archive_paths"])}


def share_dependency_artifacts(admission: dict, cache: Path) -> dict:
    """Give identical wheels a stable file URL across separate family installs.

    uv keys local archives by their source location. A shared extraction cache
    alone does not deduplicate wheels arriving in different family directories.
    Exact file-URL constraints also keep the selected manifest authoritative.
    """
    manifest_path = Path(admission["directory"]) / "manifest.json"
    if sha(manifest_path) != admission["manifest_sha256"]:
        raise ValueError("dependency manifest changed after admission")
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["wheels"]
    names = [re.sub(r"[-_.]+", "-", entry["name"]).lower() for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("shared artifacts require one exact wheel per package")
    cache = cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    def publish(entry: dict, source: Path) -> str:
        directory = cache / entry["sha256"]
        directory.mkdir(exist_ok=True)
        if directory.is_symlink():
            raise ValueError("shared dependency artifact directory cannot be linked")
        target = directory / entry["filename"]
        if not target.exists() and not target.is_symlink():
            # uv keys local archives by ctime on Unix. Keep a dedicated inode:
            # adding/removing links in a caller's wheelhouse must not invalidate it.
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
                staging = Path(temporary.name)
            try:
                shutil.copyfile(source, staging)
                if sha(staging) != entry["sha256"]:
                    raise ValueError("dependency artifact changed while copying")
                try:
                    os.link(staging, target)
                except FileExistsError:
                    pass  # Another installer may have published the same bytes.
            finally:
                staging.unlink()
        if (target.is_symlink() or not target.is_file()
                or target.stat().st_size != entry["bytes"] or sha(target) != entry["sha256"]):
            raise ValueError(f"shared dependency artifact differs from manifest: {entry['name']}")
        return str(target)

    paths = {name: publish(entry, Path(admission["wheel_directory"]) / entry["filename"])
             for name, entry in zip(names, entries)}
    overlay_paths, source_paths = {}, {}
    for entry in manifest.get("overlay_inputs", []):
        key = entry["rule_path"]
        overlay_paths[key] = publish(entry, Path(admission["overlay_input_paths"][key]))
    for entry in manifest.get("source_archives", []):
        name = re.sub(r"[-_.]+", "-", entry["name"]).lower()
        if name in paths:
            raise ValueError("source archive collides with an installable wheel")
        source_paths[name] = publish(entry, Path(admission["source_archive_paths"][name]))
        paths[name] = source_paths[name]
    constraints = "".join(f"{name} @ {Path(path).as_uri()}\n" for name, path in sorted(paths.items()))
    overrides = {name: paths[re.sub(r"[-_.]+", "-", name).lower()]
                 for name in admission["explicit_override_paths"]}
    return {"cache": str(cache), "paths": paths, "constraints": constraints,
            "explicit_override_paths": overrides,
            "overlay_input_paths": overlay_paths, "source_archive_paths": source_paths,
            "scope": "same verified wheels at stable file URLs; dependency versions unchanged"}


def prepare_ptxas(path: Path, output: Path, *, target: str = "jetson_thor") -> dict:
    """Check the selected target's assembler without importing Torch or using a GPU."""
    if target not in TARGETS:
        raise ValueError(f"unsupported deployment target: {target}")
    architecture = TARGETS[target]["ptxas_target"]
    path = path.expanduser().resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("--ptxas must name an executable CUDA assembler")
    output.mkdir(exist_ok=False)
    source = output / "probe.ptx"
    binary = output / "probe.cubin"
    source.write_text(f".version {TARGETS[target]['ptx_version']}\n.target {architecture}\n.address_size 64\n"
                      ".visible .entry instinctflash_ptxas_probe() { ret; }\n")
    command = [str(path), f"--gpu-name={architecture}", str(source), "-o", str(binary)]
    done = subprocess.run(command, capture_output=True, timeout=30, check=False)
    (output / "stdout.log").write_bytes(done.stdout)
    (output / "stderr.log").write_bytes(done.stderr)
    values = {"TRITON_PTXAS_PATH": str(path)}
    if target in {"jetson_thor", "rtx5090"}:
        values["TRITON_PTXAS_BLACKWELL_PATH"] = str(path)
    passed = done.returncode == 0 and binary.is_file() and binary.stat().st_size > 0
    receipt = {"status": "passed" if passed else "failed", "command": command,
               "returncode": done.returncode, "ptxas": str(path), "sha256": sha(path),
               "environment": values, "target": architecture, "deployment_target": target, "GPU_used": False,
               "probe_sha256": sha(source), "cubin_sha256": sha(binary) if passed else None}
    write_json(output / "receipt.json", receipt)
    if not passed:
        raise RuntimeError(f"ptxas cannot assemble {architecture}; see {output / 'stderr.log'}")
    (output / "run.env").write_text("".join(
        f"export {name}={shlex.quote(value)}\n" for name, value in values.items()))
    return receipt


def prepare_inference_metadata(original: Path, output: Path, rule: dict) -> dict:
    """Scope a pinned pure-Python wheel to the documented VLA inference stack.

    This changes dependency declarations, not implementation. Broad LeRobot
    training/device extras are explicitly outside this wheel's qualification.
    """
    if sha(original) != rule["sha256"] or original.stat().st_size != rule["bytes"]:
        raise ValueError("public Python dependency artifact mismatch")
    if output.exists() or output.is_symlink():
        raise ValueError("metadata overlay output must be new")
    with zipfile.ZipFile(original) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate Python wheel member")
        data = {name: z.read(name) for name in names}
        meta_name, record_name = rule["metadata_member"], rule["record_member"]
        rows = [r for r in csv.reader(io.StringIO(data[record_name].decode())) if r]
        if (any(len(r) != 3 for r in rows) or len(rows) != len(names) or
                len({r[0] for r in rows}) != len(names) or {r[0] for r in rows} != set(names)):
            raise ValueError("invalid original Python wheel RECORD")
        for name, digest, size in rows:
            if name == record_name:
                if digest or size:
                    raise ValueError("hashed RECORD self-entry")
            elif digest != repair_vendor_wheel.digest(data[name]) or size != str(len(data[name])):
                raise ValueError("original Python wheel RECORD mismatch")
        for name, expected in rule["historical_imported_source_sha256"].items():
            if hashlib.sha256(data[name]).hexdigest() != expected:
                raise ValueError("public Python source differs from recorded imported source")
        old = data[meta_name]
        if hashlib.sha256(old).hexdigest() != rule["metadata_sha256"]:
            raise ValueError("unexpected original dependency metadata")
        header, body = old.split(b"\n\n", 1)
        lines = [line for line in header.splitlines() if not line.startswith((b"Requires-Dist:", b"Provides-Extra:"))]
        deps = rule["inference_dependencies"]
        if not deps or any("\n" in value or "\r" in value or " @ " in value for value in deps):
            raise ValueError("invalid inference dependency declaration")
        lines += [f"Requires-Dist: {value}".encode() for value in deps]
        new = b"\n".join(lines) + b"\n\n" + body
        if new == old:
            raise ValueError("empty metadata overlay")
        new_rows = [[name, repair_vendor_wheel.digest(new), str(len(new))] if name == meta_name else row
                    for row in rows for name in [row[0]]]
        record = io.StringIO(newline="")
        csv.writer(record, lineterminator="\n").writerows(new_rows)
        changed = {meta_name: new, record_name: record.getvalue().encode()}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as raw, zipfile.ZipFile(raw, "w") as out:
            for info in z.infolist():
                out.writestr(copy.copy(info), changed.get(info.filename, data[info.filename]))
    with zipfile.ZipFile(output) as z:
        if set(z.namelist()) != set(data):
            raise ValueError("wheel overlay changed member inventory")
        for name in data:
            if z.read(name) != changed.get(name, data[name]):
                raise ValueError("wheel overlay altered undeclared payload")
    return {"status": "metadata_overlay_prepared", "original_sha256": sha(original), "repacked_sha256": sha(output),
            "changed_members": sorted(changed), "all_implementation_and_notice_bytes_preserved": True,
            "historical_imported_source_files_verified": len(rule["historical_imported_source_sha256"]),
            "inference_dependencies": deps, "scope": rule["scope"], "GPU_verified": False,
            "source_sha256": sha(Path(__file__))}


class Bootstrap:
    def __init__(self, args, profile: dict):
        self.args, self.profile = args, profile
        self.root = args.root.expanduser().absolute()
        self.env_dir = (args.env_dir or self.root / "env").expanduser().absolute()
        self.vendor = (args.vendor_dir or self.root / "vendor").expanduser().absolute()
        self.environment = clean_environment(args.cache_dir)
        self.environment["UV_LINK_MODE"] = getattr(args, "link_mode", "copy")
        self.source_transport = vendor_source_transport(profile, getattr(args, "vendor_source", None))
        if self.source_transport and self.source_transport["kind"].startswith("local_git_"):
            # Fetch objects from the explicit local input only, including when a
            # partial clone would otherwise lazily contact its configured remote.
            for key in tuple(self.environment):
                if key.startswith("GIT_"):
                    self.environment.pop(key)
            self.environment.update(GIT_ALLOW_PROTOCOL="file", GIT_NO_LAZY_FETCH="1",
                                    GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                                    GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
        self.receipts = self.root / "receipts"
        self.python = self.env_dir / "bin/python"
        self.commands = []

    def command(self, label: str, argv: list[str], *, cwd: Path | None = None,
                acceptable: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
        logfile = self.receipts / f"{len(self.commands):02d}_{label}.log"
        start = dt.datetime.now(dt.timezone.utc).isoformat()
        with logfile.open("xb") as out:
            result = subprocess.run([str(x) for x in argv], cwd=cwd or self.root,
                                    env=self.environment, stdout=out, stderr=subprocess.STDOUT,
                                    timeout=self.args.timeout, check=False)
        entry = {"label": label, "argv": [str(x) for x in argv], "cwd": str(cwd or self.root),
                 "started_at": start, "returncode": result.returncode, "log": str(logfile),
                 "log_sha256": sha(logfile)}
        self.commands.append(entry)
        write_json(self.receipts / f"{len(self.commands):02d}_{label}.json", entry)
        if result.returncode not in acceptable:
            raise RuntimeError(f"{label} failed (exit {result.returncode}); see {logfile}")
        return result

    def prepare_roots(self) -> None:
        targets = [self.root, self.env_dir]
        if self.profile["source_manifest"]:
            targets.append(self.vendor)
        if self.source_transport and self.source_transport["kind"] == "local_git_repository":
            source = Path(self.source_transport["path"])
            if any(target.resolve().is_relative_to(source) for target in targets):
                raise ValueError("bootstrap destinations must be outside the local vendor source")
        for target in targets:
            if target.exists() or target.is_symlink():
                raise ValueError(f"refusing existing destination: {target}")
        if self.env_dir == self.vendor or self.root == self.env_dir or self.root == self.vendor:
            raise ValueError("bootstrap, environment and source destinations must be distinct")
        if self.env_dir.is_relative_to(self.vendor) or self.vendor.is_relative_to(self.env_dir):
            raise ValueError("environment and vendor destinations must not contain one another")
        self.root.mkdir(parents=True, exist_ok=False)
        self.receipts.mkdir()
        write_json(self.receipts / "plan.json", {"profile": self.profile, "source_sha256": sha(Path(__file__)),
                    "vendor_source_transport": self.source_transport,
                    "model_constructed": False, "GPU_verified": False, "weights_downloaded": False})

    def prepare_source(self) -> None:
        if not self.profile["source_manifest"]:
            return
        manifest = self.profile["source"]
        transport = self.source_transport
        local = transport["kind"].startswith("local_git_")
        git = ["git", "-c", "core.hooksPath=/dev/null"]
        if local:
            git += ["-c", "core.fsmonitor=false", "-c", "fetch.fsckObjects=true"]
        self.vendor.mkdir(parents=True, exist_ok=False)
        self.command("git_init", [*git, "init", str(self.vendor)])
        if transport["kind"] == "local_git_bundle":
            self.command("git_bundle_verify", [*git, "bundle", "verify", transport["path"]], cwd=self.vendor)
        fetch = [*git, "fetch"]
        if local:
            fetch += ["--no-tags"]
        # Git imports the bundle's pack; --depth is unsupported for bundles.
        if transport["kind"] != "local_git_bundle":
            fetch += ["--depth=1"]
        fetch += [transport["path"] if local else transport["url"], manifest["source"]["revision"]]
        self.command("git_fetch", fetch, cwd=self.vendor)
        if transport["kind"] == "local_git_bundle" and (
                Path(transport["path"]).stat().st_size != transport["bytes"]
                or sha(Path(transport["path"])) != transport["sha256"]):
            raise ValueError("local vendor bundle changed during fetch")
        self.command("git_checkout", [*git, "checkout", "--detach", "FETCH_HEAD"], cwd=self.vendor)
        if local:
            self.command("git_connectivity", [*git, "fsck", "--connectivity-only", "HEAD"], cwd=self.vendor)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.vendor, env=self.environment, text=True).strip()
        if head != manifest["source"]["revision"]:
            raise ValueError("fetched vendor commit mismatch")
        if manifest["patch"]:
            folder = self.args.checkout / "release/vendor" / Path(self.profile["source_manifest"]).parent
            patch = checked_path(folder, manifest["patch"]["path"], manifest["patch"]["sha256"])
            self.command("recorded_patch_check", ["git", "apply", "--check", str(patch)], cwd=self.vendor)
            self.command("recorded_patch", ["git", "apply", str(patch)], cwd=self.vendor)
        for name, expected in manifest["expected_patched_files"].items():
            checked_path(self.vendor, name, expected["sha256"])
        packaging = self.profile["packaging_patch"]
        for name, expected in packaging["before"].items():
            checked_path(self.vendor, name, expected)
        patch = checked_path(self.args.checkout / "release/vendor", packaging["path"], packaging["sha256"])
        touched = subprocess.check_output(["git", "apply", "--numstat", str(patch)],
                                          cwd=self.vendor, env=self.environment, text=True)
        names = {line.split("\t", 2)[2] for line in touched.splitlines()}
        if names != set(packaging["before"]) or names != set(packaging["after"]):
            raise ValueError("packaging patch touches undeclared files")
        self.command("inference_metadata_check", ["git", "apply", "--check", str(patch)], cwd=self.vendor)
        self.command("inference_metadata_patch", ["git", "apply", str(patch)], cwd=self.vendor)
        for name, expected in packaging["after"].items():
            checked_path(self.vendor, name, expected)
        write_json(self.receipts / "vendor_source.json", {"repository": manifest["source"]["repository"],
                   "revision": head, "transport": transport, "recorded_patch": manifest["patch"],
                   "inference_metadata_patch": packaging, "runtime_files_changed_by_metadata_patch": False})

    def repaired_wheel(self) -> Path:
        catalog = self.args.checkout / "release/vendor/wheel_repairs.json"
        rule = json.loads(catalog.read_text())["repairs"][self.profile["wheel_metadata_repair"]]
        folder = self.root / "wheel_repair"
        folder.mkdir()
        original = folder / "original" / rule["filename"]
        original.parent.mkdir()
        if self.args.repaired_wheel:
            # A caller-supplied repair is accepted only with its preservation receipt.
            receipt = json.loads(self.args.repair_receipt.read_text())
            candidate = self.args.repaired_wheel
            if (receipt["public_source"] != rule or receipt["status"] != "passed" or
                    not receipt["all_other_payload_bytes_preserved"] or
                    receipt["repair_source_sha256"] != sha(Path(repair_vendor_wheel.__file__)) or
                    receipt["repaired"]["sha256"] != "de240f3d9b6285807eae0bc77a22de8a63f71b2fadeca4d54e321a60b78f95b6" or
                    sha(candidate) != receipt["repaired"]["sha256"]):
                raise ValueError("provided repaired wheel/receipt binding mismatch")
            write_json(folder / "repair_receipt.json", receipt)
            write_json(folder / "verified_external_artifact.json", {
                "path": str(candidate.resolve()), "sha256": sha(candidate), "copied": False,
                "scope": "Read-only verified input; avoids copying the 220 MB shared wheel per environment."})
            return candidate.resolve()
        with urllib.request.urlopen(rule["url"], timeout=120) as stream, original.open("xb") as f:
            for block in iter(lambda: stream.read(4 << 20), b""):
                f.write(block)
        target = folder / "repaired" / rule["filename"]
        receipt = repair_vendor_wheel.repair(original, target, rule)
        write_json(folder / "repair_receipt.json", receipt)
        return target

    def auxiliary_python_wheels(self) -> list[Path]:
        outputs = []
        for reference in self.profile.get("python_metadata_overlays", []):
            path = checked_path(self.args.checkout / "release/vendor", reference["path"], reference["sha256"])
            rule = json.loads(path.read_text())
            if rule["sha256"] != "e53800ead8216861540ad3aebaf12e3cf87a399b3c1f234eeead33716c9c24fd":
                raise ValueError("unaudited Python dependency metadata overlay")
            root = self.root / "python_metadata_overlay"
            root.mkdir()
            wheelhouse = getattr(self, "dependency_wheelhouse", None)
            if wheelhouse is not None:
                original = Path(wheelhouse["overlay_input_paths"][reference["path"]])
            else:
                original = root / "original" / rule["filename"]
                original.parent.mkdir()
                with urllib.request.urlopen(rule["url"], timeout=120) as src, original.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            output = root / "inference_only" / rule["filename"]
            receipt = prepare_inference_metadata(original, output, rule)
            write_json(root / "receipt.json", receipt)
            outputs.append(output)
        return outputs

    def install(self) -> dict:
        self.prepare_roots()
        try:
            deployment_target = self.profile.get("deployment_target", "jetson_thor")
            native_wheelhouse_dir = getattr(self.args, "native_tool_wheelhouse", None)
            native_wheelhouse = None
            if native_wheelhouse_dir:
                tool = self.profile.get("native_tool")
                if tool is None:
                    raise ValueError("--native-tool-wheelhouse requires this family's native tool catalog")
                catalog = checked_path(self.args.checkout / "release/vendor", tool["path"], tool["sha256"])
                native_wheelhouse = prepare_native_tools.admit_wheelhouse(
                    native_wheelhouse_dir, catalog, target=deployment_target)
                write_json(self.root / "native_tool_wheelhouse_admission.json", native_wheelhouse)
            wheelhouse_dir = getattr(self.args, "dependency_wheelhouse", None)
            wheelhouse = (admit_dependency_wheelhouse(wheelhouse_dir, self.profile, checkout=self.args.checkout)
                          if wheelhouse_dir else None)
            if wheelhouse:
                write_json(self.root / "dependency_wheelhouse_admission.json", wheelhouse)
            indexes = (["--offline", "--no-index", "--find-links", wheelhouse["wheel_directory"]]
                       if wheelhouse else ["--index-url", "https://pypi.org/simple"])
            shared_artifacts = None
            artifact_cache = getattr(self.args, "dependency_artifact_cache", None)
            if artifact_cache:
                if wheelhouse is None:
                    raise ValueError("--dependency-artifact-cache requires --dependency-wheelhouse")
                shared_artifacts = share_dependency_artifacts(wheelhouse, artifact_cache)
                write_json(self.root / "shared_dependency_artifacts.json", shared_artifacts)
                artifact_constraints = self.root / "dependency_artifacts.constraints.txt"
                artifact_constraints.write_text(shared_artifacts["constraints"])
                indexes += ["-c", str(artifact_constraints)]
                wheelhouse = {**wheelhouse, "explicit_override_paths": shared_artifacts["explicit_override_paths"],
                              "overlay_input_paths": shared_artifacts["overlay_input_paths"],
                              "source_archive_paths": shared_artifacts["source_archive_paths"]}
            self.dependency_wheelhouse = wheelhouse
            compiler = (prepare_ptxas(self.args.ptxas, self.root / "compiler", target=deployment_target)
                        if getattr(self.args, "ptxas", None) else None)
            if compiler:
                self.environment.update(compiler["environment"])
            probe = subprocess.check_output([self.args.python, "-I", "-B", "-c",
                 "import sys,platform,json;print(json.dumps([platform.system(),platform.machine(),f'{sys.version_info.major}.{sys.version_info.minor}']))"],
                 env=self.environment, text=True)
            target = self.profile["target"]
            if json.loads(probe) != [target["system"], target["machine"], target["python_minor"]]:
                raise ValueError("interpreter does not match the selected target's system, machine and Python version")
            self.prepare_source()
            self.command("venv", [self.args.uv, "venv", "--python", self.args.python, str(self.env_dir)])
            repaired = self.repaired_wheel() if self.profile.get("wheel_metadata_repair") else None
            if repaired is None and self.args.repaired_wheel:
                raise ValueError("this target does not use a repaired dependency wheel")
            auxiliary_wheels = self.auxiliary_python_wheels()
            base = self.args.checkout / "release/vendor"
            requirements = checked_path(base, self.profile["requirements"]["path"], self.profile["requirements"]["sha256"])
            constraints = checked_path(base, self.profile["constraints"]["path"], self.profile["constraints"]["sha256"])
            deps = [self.args.uv, "pip", "install", "--python", str(self.python), *indexes,
                    "--only-binary", ":all:", "-c", str(constraints), "-r", str(requirements),
                    *([str(repaired)] if repaired else []),
                    *[str(p) for p in auxiliary_wheels]]
            deps += ([f"{name} @ {Path(path).as_uri()}" for name, path in wheelhouse["explicit_override_paths"].items()]
                     if wheelhouse else [f"{name} @ {url}" for name, url in self.profile["public_wheel_overrides"].items()])
            # The local source constraint also applies to later source-wheel
            # installs, even when that dependency is already installed.
            offline_source_build_options = []
            for name, source in self.profile.get("audited_pure_python_sdists", {}).items():
                if (name not in PURE_SDISTS or source["version"] != PURE_SDISTS[name][0] or
                        not source["url"].endswith("#sha256=" + PURE_SDISTS[name][1])):
                    raise ValueError("unaudited source-build exception")
                source_url = (Path(wheelhouse["source_archive_paths"][name]).as_uri()
                              if wheelhouse else source["url"])
                deps += ["--no-binary", name, f"{name} @ {source_url}"]
                if wheelhouse:
                    offline_source_build_options += ["--no-binary", name]
            if wheelhouse and self.profile.get("audited_pure_python_sdists"):
                # Runtime constraints do not constrain PEP 517's isolated build.
                # Bind that resolver to the same admitted original wheel bytes.
                manifest = json.loads((Path(wheelhouse["directory"]) / "manifest.json").read_text())
                build_constraints = self.root / "dependency_build.constraints.txt"
                build_lines = []
                for entry in manifest["wheels"]:
                    name = re.sub(r"[-_.]+", "-", entry["name"]).lower()
                    path = (Path(shared_artifacts["paths"][name]) if shared_artifacts
                            else Path(wheelhouse["wheel_directory"]) / entry["filename"])
                    build_lines.append(f"{name} @ {path.as_uri()}\n")
                build_constraints.write_text("".join(build_lines))
                deps += ["--build-constraints", str(build_constraints)]
                offline_source_build_options += ["--build-constraints", str(build_constraints)]
            self.command("dependency_resolution", deps)
            # Only our public source wheels and Python-only inference metadata are built.
            # No upstream training/native build extras are selected.
            self.command("build_tools", [self.args.uv, "pip", "install", "--python", str(self.python),
                         *indexes, "--only-binary", ":all:",
                         *offline_source_build_options,
                         "-c", str(constraints), "build", "wheel", "setuptools>=77"])
            wheel_dir = self.root / "wheels"
            wheel_dir.mkdir()
            projects = []
            if self.profile["source_manifest"]:
                projects.append(self.vendor)
            if self.args.package_wheel_dir:
                catalog_root = self.args.checkout / "release"
                if deployment_target != "jetson_thor":
                    catalog_root /= deployment_target
                catalog = json.loads((catalog_root / "deployment_profiles.json").read_text())
                if catalog.get("target") != deployment_target:
                    raise ValueError("supplied-wheel catalog does not match the deployment target")
                row = next(m for m in catalog["models"] if m["id"] == self.profile["family"])
                distributions = {"instinctflash", "flash-rt", row["adapter"]["distribution"]}
                supplied = []
                for distribution in sorted(distributions):
                    candidates = list(self.args.package_wheel_dir.glob(distribution.replace("-", "_") + "-*.whl"))
                    if len(candidates) != 1 or candidates[0].is_symlink():
                        raise ValueError(f"require one exact supplied wheel for {distribution}")
                    supplied += candidates
                for p in supplied:
                    shutil.copyfile(p, wheel_dir / p.name)
            else:
                projects += [self.args.checkout, self.args.checkout / "serving"]
                adapter = self.profile["adapter_directory"]
                if adapter:
                    projects.append(self.args.checkout / adapter)
            for index, project in enumerate(projects):
                self.command(f"build_{index}", [str(self.python), "-m", "build", "--wheel", "--no-isolation",
                                              "--outdir", str(wheel_dir), str(project)])
            wheels = sorted(wheel_dir.glob("*.whl"))
            self.command("install_source_wheels", [self.args.uv, "pip", "install", "--python", str(self.python),
                         *indexes, "--only-binary", ":all:",
                         *offline_source_build_options,
                         "-c", str(constraints), *[str(p) for p in wheels]])
            self.command("pip_check", [str(self.python), "-m", "pip", "check"])
            self.command("uv_pip_check", [self.args.uv, "pip", "check", "--python", str(self.python)])
            values = activation_environment(self.profile, self.vendor)
            if compiler:
                values.update(compiler["environment"])
            native_tool = None
            if self.profile.get("native_tool"):
                tool = self.profile["native_tool"]
                catalog = checked_path(base, tool["path"], tool["sha256"])
                native_tool = prepare_native_tools.prepare(
                    self.root / "native_tools", self.python, self.python.with_name("uvx"), catalog,
                    wheelhouse=native_wheelhouse_dir, target=deployment_target)
                values.update(native_tool["environment"])
            env_text = ""
            for key, value in values.items():
                if key == "PATH":
                    env_text += f"export PATH={shlex.quote(str(self.python.parent))}:\"$PATH\"\n"
                else:
                    env_text += f"export {key}={shlex.quote(value)}\n"
            (self.root / "run.env").write_text(env_text)
            (self.root / "activate.sh").write_text(
                f". {shlex.quote(str(self.env_dir / 'bin/activate'))}\n. {shlex.quote(str(self.root / 'run.env'))}\n")
            self.environment.update(values)
            doctor = None
            if not self.args.defer_doctor:
                doctor = self.command("cpu_doctor", [str(self.python), str(self.args.checkout / "scripts/public_deploy.py"),
                                       "doctor", self.profile["family"], "--target", deployment_target], acceptable=(0, 1))
            result = {"status": "packages_checked", "family": self.profile["family"], "python": str(self.python),
                      "deployment_target": deployment_target,
                      "vendor": str(self.vendor) if self.profile["source_manifest"] else None,
                      "CPU_doctor_passed": None if doctor is None else doctor.returncode == 0,
                      "CPU_doctor_deferred": doctor is None, "GPU_verified": False, "model_constructed": False,
                      "task_quality_certified": False, "weights_downloaded": False, "commands": self.commands,
                      "native_tool_preparation": native_tool,
                      "native_tool_wheelhouse": native_wheelhouse,
                      "compiler_preparation": compiler,
                      "dependency_wheelhouse": wheelhouse,
                      "shared_dependency_artifacts": shared_artifacts,
                      "wheels": [{"path": str(p), "sha256": sha(p)} for p in wheels]}
            write_json(self.root / "completion.json", result)
            return result
        except Exception as error:
            write_json(self.root / "failure.json", {"status": "failed_preserved", "error": str(error),
                       "error_type": type(error).__name__, "commands": self.commands,
                       "GPU_verified": False, "automatic_retry": False})
            raise


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("command", choices=("plan", "install"))
    p.add_argument("model", choices=FAMILIES)
    p.add_argument("--target", choices=tuple(TARGETS), default="jetson_thor")
    p.add_argument("--checkout", type=Path, default=CHECKOUT)
    p.add_argument("--root", type=Path)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--uv", default="uv")
    p.add_argument("--env-dir", type=Path)
    p.add_argument("--vendor-dir", type=Path)
    p.add_argument("--vendor-source", type=Path,
                   help="Fetch the profile's exact commit from a local Git repository or self-contained bundle.")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--link-mode", choices=("copy", "hardlink"), default="copy",
                   help="Use hardlink only with an executable shared cache on the environment filesystem.")
    p.add_argument("--package-wheel-dir", type=Path)
    p.add_argument("--dependency-wheelhouse", type=Path,
                   help="Use a manifest-bound, hash-verified dependency wheel cache; keep the selected recipe's pins.")
    p.add_argument("--dependency-artifact-cache", type=Path,
                   help="Give verified wheelhouse files stable shared paths for reuse across family environments.")
    p.add_argument("--native-tool-wheelhouse", type=Path,
                   help="Prepare Cosmos's pinned native HF subprocess offline from its separate verified wheel cache.")
    p.add_argument("--repaired-wheel", type=Path)
    p.add_argument("--repair-receipt", type=Path)
    p.add_argument("--ptxas", type=Path,
                   help="CUDA assembler for the selected target; verify its architecture and persist Triton overrides.")
    p.add_argument("--defer-doctor", action="store_true", help="Install/check packages only; explicitly record the offline doctor as pending.")
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--json", action="store_true", help="Output is always structured JSON.")
    a = p.parse_args(argv)
    a.checkout = a.checkout.expanduser().resolve()
    if bool(a.repaired_wheel) != bool(a.repair_receipt):
        p.error("--repaired-wheel requires --repair-receipt, and conversely")
    if not 0 < a.timeout <= 7200:
        p.error("--timeout must be between zero and 7200 seconds")
    if a.command == "install" and a.root is None:
        p.error("install requires --root at a new destination")
    try:
        profile = load_profile(a.model, a.checkout, target=a.target)
        if a.command == "plan":
            result = profile
            if a.vendor_source is not None:
                result = {**profile, "vendor_source_transport": vendor_source_transport(profile, a.vendor_source)}
            if a.native_tool_wheelhouse is not None:
                tool = profile.get("native_tool")
                if tool is None:
                    raise ValueError("--native-tool-wheelhouse requires this family's native tool catalog")
                catalog = checked_path(a.checkout / "release/vendor", tool["path"], tool["sha256"])
                result = {**result, "native_tool_wheelhouse_admission": prepare_native_tools.admit_wheelhouse(
                    a.native_tool_wheelhouse, catalog, target=a.target)}
        else:
            result = Bootstrap(a, profile).install()
        print(json.dumps(result, indent=2, sort_keys=True))
        # Keep the complete package/doctor receipt available to callers, while
        # making a failed required check visible to shell automation. An
        # explicitly deferred doctor records None and remains a successful install.
        return 1 if a.command == "install" and result["CPU_doctor_passed"] is False else 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

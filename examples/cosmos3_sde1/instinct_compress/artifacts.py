"""Native checkpoint packaging, independent of any training method or model family.

The model adapter writes its native weights, architecture and processors. This
module records execution facts for InstinctFlash and content hashes for humans
and reproducibility tools. Training provenance is never an execution option.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DECLARATION_FILE = "instinctflash.json"
MANIFEST_FILE = "instinctcompress_manifest.json"
WEIGHTS_FILES = (
    "model.safetensors", "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.safetensors.index.json",
)
# Kept in sync with instinctflash.descriptors.checkpoint.FORBIDDEN_IN_EXECUTION.
# No InstinctFlash import is required in a training environment.
FORBIDDEN_IN_EXECUTION = frozenset((
    "recipe", "training_method", "teacher", "student", "solver", "dataset", "optimizer",
    "coverage_gate_pass", "min_updates_per_head", "head_updates_min", "endpoint_rmse",
    "trainable", "paper", "training_diagnostics", "certification", "distillation",
))


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_execution(execution: Mapping[str, Any]) -> None:
    """Reject missing semantics and training metadata at the publication boundary."""
    leaked = sorted(FORBIDDEN_IN_EXECUTION.intersection(execution))
    if leaked:
        raise ValueError(f"Training fields belong in provenance, not execution: {leaked}")
    for field in ("model_id", "backbone"):
        value = execution.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"execution.{field} must be a non-empty string")
    if not isinstance(execution.get("servable"), bool):
        raise ValueError("execution.servable must be an explicit boolean")
    nfe = execution.get("nfe", {})
    if not isinstance(nfe, Mapping):
        raise ValueError("execution.nfe must map phase names to positive integers")
    for phase, steps in nfe.items():
        if not isinstance(phase, str) or not phase or isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("execution.nfe must map phase names to positive integers")
    guidance = execution.get("guidance", {})
    if not isinstance(guidance, Mapping):
        raise ValueError("execution.guidance must be a mapping")
    for phase, value in guidance.items():
        if not isinstance(phase, str) or not phase:
            raise ValueError("execution.guidance phase names must be non-empty strings")
        if isinstance(value, Mapping):
            if set(value) - {"mode", "scale"}:
                raise ValueError(f"Unknown guidance fields for {phase}")
            mode, scale = value.get("mode"), value.get("scale")
            if mode not in {"none", "cfg", "positive_only"}:
                raise ValueError(f"Invalid guidance mode for {phase}: {mode!r}")
            if scale is not None:
                _validate_scale(scale)
                if mode != "cfg" and float(scale) != 1.0:
                    raise ValueError(f"Non-CFG guidance for {phase} must have scale 1")
        elif isinstance(value, str):
            if value not in {"none", "cfg", "positive_only"}:
                raise ValueError(f"Invalid guidance mode for {phase}: {value!r}")
        else:
            _validate_scale(value)
    # Validate serializability and finite floats before touching the output.
    json.dumps(dict(execution), allow_nan=False)


def _validate_scale(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError("Guidance scale must be a finite nonnegative number")


def write_declaration(destination: str | Path, *, execution: Mapping[str, Any],
                      provenance: Mapping[str, Any] | None = None) -> Path:
    """Write Flash schema 1. ``servable`` is explicitly supplied by the adapter.

    This describes loadability, not downstream task qualification. Model-specific
    schedule/conditioning semantics must already have been checked by the adapter.
    """
    validate_execution(execution)
    path = Path(destination) / DECLARATION_FILE
    _write_json(path, {"instinctflash_schema": 1, "execution": dict(execution),
                       "provenance": dict(provenance or {})})
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Artifact path must stay inside its package: {name!r}")
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"Artifact symlink escapes its package: {name!r}") from None
    if not path.is_file():
        raise ValueError(f"Missing artifact file: {name}")
    return path


def validate_native_files(destination: str | Path, *, required_files: Sequence[str] = ()) -> tuple[str, ...]:
    """Check complete native weights, including all referenced shards.

    Full local weights are required here. A pointer to an unmodified base cannot
    substitute for the compressed student's payload. Native modeling libraries
    remain responsible for matching tensor keys and shapes against architecture.
    """
    root = Path(destination)
    for name in ("config.json", *required_files):
        _local_file(root, name)
    if not any((root / name).is_file() for name in WEIGHTS_FILES):
        raise ValueError(f"No native weights in {root}: expected one of {WEIGHTS_FILES}")
    weight_names: set[str] = set()
    for name in WEIGHTS_FILES:
        if not (root / name).exists():
            continue
        path = _local_file(root, name)
        weight_names.add(name)
        if not name.endswith(".index.json"):
            continue
        document = json.loads(path.read_text())
        weight_map = document.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{name} must contain a non-empty weight_map")
        for key, shard in weight_map.items():
            if not isinstance(key, str) or not key or not isinstance(shard, str) or not shard.endswith(".safetensors"):
                raise ValueError(f"{name} contains an invalid tensor/shard mapping")
            _local_file(root, shard)
            weight_names.add(shard)
    return tuple(sorted(weight_names))


def finalize_artifact(destination: str | Path, *, execution: Mapping[str, Any],
                      provenance: Mapping[str, Any] | None = None,
                      required_files: Sequence[str] = ()) -> dict[str, Any]:
    """Stamp a native export and inventory its exact local content.

    A returned manifest proves bytes/layout only; it does not claim inference or
    task quality. Call the native and Flash runtimes on held-out inputs separately.
    """
    root = Path(destination)
    weights = validate_native_files(root, required_files=required_files)
    write_declaration(root, execution=execution, provenance=provenance)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_FILE:
            continue
        name = path.relative_to(root).as_posix()
        _local_file(root, name)
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    manifest = {
        "instinctcompress_schema": 1,
        "model_id": execution["model_id"],
        "backbone": execution["backbone"],
        "weight_files": list(weights),
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files),
        "validation_scope": "Native file presence and content integrity; inference and task quality are separate evaluations.",
    }
    _write_json(root / MANIFEST_FILE, manifest)
    return manifest


def verify_artifact(destination: str | Path) -> dict[str, Any]:
    """Check the package has exactly the files and hashes recorded at export."""
    root = Path(destination)
    manifest = json.loads((root / MANIFEST_FILE).read_text())
    if manifest.get("instinctcompress_schema") != 1:
        raise ValueError("Unsupported InstinctCompress artifact schema")
    declaration = json.loads((root / DECLARATION_FILE).read_text())
    if declaration.get("instinctflash_schema") != 1:
        raise ValueError("Unsupported InstinctFlash declaration schema")
    validate_execution(declaration.get("execution", {}))
    validate_native_files(root)
    recorded: set[str] = set()
    for row in manifest["files"]:
        name = row["path"]
        if name in recorded:
            raise ValueError(f"Duplicate manifest file: {name}")
        recorded.add(name)
        path = _local_file(root, name)
        if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Artifact content changed after export: {name}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name != MANIFEST_FILE}
    if recorded != actual:
        raise ValueError(f"Artifact inventory mismatch: missing={sorted(recorded - actual)}, extra={sorted(actual - recorded)}")
    return manifest

"""Small, dependency-free utilities shared by the benchmark pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


class ConfigurationError(ValueError):
    """A benchmark declaration is invalid or internally inconsistent."""


ENV_ITEM = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
SECRET_NAME = re.compile(r"(?:TOKEN|PASSWORD|SECRET|API_KEY|PRIVATE_KEY)", re.IGNORECASE)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError as error:
        raise ConfigurationError(f"missing JSON file: {path}") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"invalid JSON in {path}: {error}") from error


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def require_keys(value: dict[str, Any], keys: Iterable[str], where: str) -> None:
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ConfigurationError(f"{where}: missing required field(s): {', '.join(missing)}")


def require_revision(value: str, where: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", value or ""):
        raise ConfigurationError(f"{where}: revision must be a full 40-character lowercase SHA")


def expand_command(argv: list[str], environment: dict[str, str] | None = None) -> list[str]:
    """Expand exact ``${NAME}`` argv entries without invoking a shell.

    Exact-item expansion is intentionally less flexible than shell interpolation: a driver path
    may come from the environment, while metacharacters and word splitting remain inert.
    """
    source = os.environ if environment is None else environment
    expanded = []
    for item in argv:
        match = ENV_ITEM.fullmatch(item)
        if match:
            name = match.group(1)
            if name not in source or not source[name]:
                raise ConfigurationError(f"driver command requires non-empty environment variable {name}")
            expanded.append(source[name])
        else:
            expanded.append(item)
    return expanded


def validate_public_environment(values: dict[str, str], where: str) -> None:
    for name, value in values.items():
        if SECRET_NAME.search(name):
            raise ConfigurationError(
                f"{where}: {name} looks secret; pass credentials through the process environment, "
                "never a committed benchmark manifest"
            )
        if not isinstance(value, str):
            raise ConfigurationError(f"{where}: environment value for {name} must be a string")


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty list")
    ordered = sorted(float(item) for item in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def command_output(argv: list[str], cwd: Path, timeout: int = 10) -> str | None:
    try:
        completed = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None

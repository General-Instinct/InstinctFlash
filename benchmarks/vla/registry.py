"""Load and validate the immutable model, dataset, suite, and profile registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import ConfigurationError, load_json, require_keys, require_revision, sha256_json


HERE = Path(__file__).resolve().parent
DEFAULT_REGISTRY = HERE / "config" / "registry.json"
SUITE_KINDS = {"contract", "latency", "open_loop", "closed_loop"}
SEED_STRATEGIES = {"fixed", "increment_until_stable"}


@dataclass(frozen=True)
class Registry:
    raw: dict[str, Any]
    digest: str
    path: Path

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        return {entry["id"]: entry for entry in self.raw["models"]}

    @property
    def datasets(self) -> dict[str, dict[str, Any]]:
        return {entry["id"]: entry for entry in self.raw["datasets"]}

    @property
    def suites(self) -> dict[str, dict[str, Any]]:
        return {entry["id"]: entry for entry in self.raw["suites"]}

    @property
    def profiles(self) -> dict[str, dict[str, Any]]:
        return self.raw["profiles"]

    @property
    def builtin_model_ids(self) -> set[str]:
        return {model["id"] for model in self.raw["models"] if model.get("builtin", False)}


def _unique(entries: list[dict[str, Any]], label: str) -> None:
    identifiers = [entry.get("id") for entry in entries]
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if None in identifiers or "" in identifiers:
        raise ConfigurationError(f"registry {label}: every entry needs a non-empty id")
    if duplicates:
        raise ConfigurationError(f"registry {label}: duplicate id(s): {', '.join(duplicates)}")


def _validate(raw: dict[str, Any]) -> None:
    require_keys(raw, ("schema_version", "models", "datasets", "suites", "profiles"), "registry")
    if raw["schema_version"] != 1:
        raise ConfigurationError(f"registry: unsupported schema_version {raw['schema_version']!r}")
    for key in ("models", "datasets", "suites"):
        if not isinstance(raw[key], list) or not raw[key]:
            raise ConfigurationError(f"registry: {key} must be a non-empty list")
        _unique(raw[key], key)
    if not isinstance(raw["profiles"], dict) or not raw["profiles"]:
        raise ConfigurationError("registry: profiles must be a non-empty object")

    backbones = set()
    model_ids = set()
    for index, model in enumerate(raw["models"]):
        where = f"registry models[{index}]"
        require_keys(model, ("id", "revision", "backbone", "builtin", "source_env", "determinism"), where)
        require_revision(model["revision"], where)
        if not isinstance(model["builtin"], bool):
            raise ConfigurationError(f"{where}: builtin must be boolean")
        if model["determinism"] not in {"bitexact", "numeric_envelope"}:
            raise ConfigurationError(f"{where}: unsupported determinism {model['determinism']!r}")
        if model["determinism"] == "numeric_envelope" and model.get("repeatability_max_abs") is None:
            raise ConfigurationError(f"{where}: numeric_envelope needs repeatability_max_abs")
        if not isinstance(model["source_env"], list):
            raise ConfigurationError(f"{where}: source_env must be a list")
        backbones.add(model["backbone"])
        model_ids.add(model["id"])

    dataset_ids = set()
    for index, dataset in enumerate(raw["datasets"]):
        where = f"registry datasets[{index}]"
        require_keys(dataset, ("id", "kind", "source", "revision"), where)
        require_revision(dataset["revision"], where)
        dataset_ids.add(dataset["id"])

    suite_ids = set()
    for index, suite in enumerate(raw["suites"]):
        where = f"registry suites[{index}]"
        require_keys(
            suite,
            ("id", "kind", "dataset", "tasks", "compatible_backbones", "seed_strategy", "seed_base"),
            where,
        )
        if suite["kind"] not in SUITE_KINDS:
            raise ConfigurationError(f"{where}: unsupported kind {suite['kind']!r}")
        if suite["dataset"] not in dataset_ids:
            raise ConfigurationError(f"{where}: unknown dataset {suite['dataset']!r}")
        if not suite["tasks"] or len(set(suite["tasks"])) != len(suite["tasks"]):
            raise ConfigurationError(f"{where}: tasks must be unique and non-empty")
        unknown_backbones = sorted(set(suite["compatible_backbones"]) - backbones)
        if unknown_backbones:
            raise ConfigurationError(f"{where}: unknown backbone(s): {', '.join(unknown_backbones)}")
        unknown_models = sorted(set(suite.get("model_ids", ())) - model_ids)
        if unknown_models:
            raise ConfigurationError(f"{where}: unknown model id(s): {', '.join(unknown_models)}")
        if suite["seed_strategy"] not in SEED_STRATEGIES:
            raise ConfigurationError(f"{where}: unsupported seed strategy {suite['seed_strategy']!r}")
        suite_ids.add(suite["id"])

    required_kinds = SUITE_KINDS
    present_kinds = {suite["kind"] for suite in raw["suites"]}
    if required_kinds - present_kinds:
        raise ConfigurationError(
            f"registry: missing suite kind(s): {', '.join(sorted(required_kinds - present_kinds))}"
        )
    for name, profile in raw["profiles"].items():
        where = f"registry profiles.{name}"
        require_keys(profile, ("suites", "limits", "latency", "arm_repeats"), where)
        unknown_suites = sorted(set(profile["suites"]) - suite_ids)
        if unknown_suites:
            raise ConfigurationError(f"{where}: unknown suite(s): {', '.join(unknown_suites)}")
        require_keys(profile["limits"], ("tasks", "seeds_per_task"), f"{where}.limits")
        require_keys(profile["latency"], ("warmup", "iterations"), f"{where}.latency")


def load_registry(path: str | Path | None = None) -> Registry:
    resolved = Path(path).resolve() if path else DEFAULT_REGISTRY
    raw = load_json(resolved)
    if not isinstance(raw, dict):
        raise ConfigurationError(f"registry root must be an object: {resolved}")
    _validate(raw)
    return Registry(raw=raw, digest=sha256_json(raw), path=resolved)

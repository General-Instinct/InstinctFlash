"""Strict YAML loading for optimization pipelines."""

from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from instinctwm.config.schema import (
    ConfigurationError,
    OptimizationLayer,
    PassMode,
    PassSelection,
    PipelineConfig,
    PipelinePolicy,
)


MAX_CONFIG_BYTES = 1024 * 1024
PRESETS = frozenset({"stock", "bitexact", "shipped"})


class _UniqueSafeLoader(yaml.SafeLoader):
    yaml_implicit_resolvers = deepcopy(yaml.SafeLoader.yaml_implicit_resolvers)


# PyYAML implements YAML 1.1 and therefore treats ``on``/``off`` as booleans. They are pass
# modes in this schema, so use the YAML 1.2 boolean spelling instead: only true/false are bools.
for _first, _resolvers in list(_UniqueSafeLoader.yaml_implicit_resolvers.items()):
    _UniqueSafeLoader.yaml_implicit_resolvers[_first] = [
        (tag, regexp) for tag, regexp in _resolvers
        if tag != "tag:yaml.org,2002:bool"
    ]
_UniqueSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$", re.IGNORECASE), list("tTfF"))


def _construct_unique_mapping(loader, node, deep=False):
    out = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in out
        except TypeError as e:
            raise ConfigurationError("YAML mapping keys must be scalar and hashable") from e
        if duplicate:
            raise ConfigurationError(f"duplicate YAML key {key!r}")
        out[key] = loader.construct_object(value_node, deep=deep)
    return out


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def preset_path(name: str) -> Path:
    if name not in PRESETS:
        raise ConfigurationError(f"unknown optimization preset {name!r}; available: {sorted(PRESETS)}")
    return Path(str(files("instinctwm.config").joinpath("presets", f"{name}.yaml")))


def _strict_keys(where: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    if not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{where} field names must be strings")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(f"{where} contains unknown field(s) {unknown}; allowed: {sorted(allowed)}")


def _mapping(where: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _parse(raw: Mapping[str, Any], source: str) -> PipelineConfig:
    _strict_keys("optimization config", raw,
                 {"schema_version", "kind", "name", "policy", "layers"})
    if raw.get("schema_version") != 1:
        raise ConfigurationError(
            f"{source}: schema_version must be 1, got {raw.get('schema_version')!r}")
    if raw.get("kind") != "OptimizationPipeline":
        raise ConfigurationError(
            f"{source}: kind must be 'OptimizationPipeline', got {raw.get('kind')!r}")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError(f"{source}: name must be a non-empty string")

    p = _mapping("policy", raw.get("policy", {}))
    _strict_keys("policy", p, {"tier_ceiling", "allow_experimental", "unlisted"})
    tier = p.get("tier_ceiling", "bitexact")
    if tier not in {"bitexact", "numeric", "behavioral"}:
        raise ConfigurationError(
            "policy.tier_ceiling must be bitexact, numeric, or behavioral")
    allow_experimental = p.get("allow_experimental", False)
    if not isinstance(allow_experimental, bool):
        raise ConfigurationError("policy.allow_experimental must be a boolean")
    try:
        unlisted = PassMode(p.get("unlisted", "off"))
    except ValueError as e:
        raise ConfigurationError("policy.unlisted must be auto, on, off, or required") from e
    policy = PipelinePolicy(tier, allow_experimental, unlisted)

    layers = _mapping("layers", raw.get("layers", {}))
    if not all(isinstance(key, str) for key in layers):
        raise ConfigurationError("layers field names must be strings")
    valid_layers = {x.value for x in OptimizationLayer}
    unknown_layers = sorted(set(layers) - valid_layers)
    if unknown_layers:
        extra = "; Layer 1 model recipes are deliberately outside YAML v1" \
            if "model" in unknown_layers else ""
        raise ConfigurationError(
            f"layers contains unsupported layer(s) {unknown_layers}; allowed: {sorted(valid_layers)}{extra}")

    selections = []
    seen = set()
    for layer in OptimizationLayer:
        rows = layers.get(layer.value, [])
        if not isinstance(rows, list):
            raise ConfigurationError(f"layers.{layer.value} must be a list")
        for index, row in enumerate(rows):
            where = f"layers.{layer.value}[{index}]"
            row = _mapping(where, row)
            _strict_keys(where, row, {"id", "mode", "params"})
            pass_id = row.get("id")
            if not isinstance(pass_id, str) or not pass_id.strip():
                raise ConfigurationError(f"{where}.id must be a non-empty string")
            if pass_id in seen:
                raise ConfigurationError(f"pass {pass_id!r} appears more than once in the YAML")
            seen.add(pass_id)
            try:
                mode = PassMode(row.get("mode", "auto"))
            except ValueError as e:
                raise ConfigurationError(f"{where}.mode must be auto, on, off, or required") from e
            params = _mapping(f"{where}.params", row.get("params", {}))
            selections.append(PassSelection(pass_id, layer, mode, dict(params)))
    return PipelineConfig(name.strip(), policy, tuple(selections), source=source)


def load_config(source: str | Path | Mapping[str, Any] | PipelineConfig) -> PipelineConfig:
    if isinstance(source, PipelineConfig):
        return source
    if isinstance(source, Mapping):
        return _parse(source, "<memory>")
    if not isinstance(source, (str, Path)):
        raise TypeError("optimization config must be a preset name, path, mapping, or PipelineConfig")

    if isinstance(source, str) and source in PRESETS:
        path = preset_path(source)
        source_name = f"preset:{source}"
    else:
        path = Path(source)
        source_name = str(path)
    if not path.is_file():
        raise ConfigurationError(f"optimization config does not exist: {path}")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ConfigurationError(
            f"optimization config is larger than {MAX_CONFIG_BYTES} bytes: {path}")
    try:
        raw = yaml.load(path.read_text(), Loader=_UniqueSafeLoader)
    except ConfigurationError:
        raise
    except yaml.YAMLError as e:
        raise ConfigurationError(f"invalid YAML in {path}: {e}") from e
    if raw is None:
        raw = {}
    return _parse(_mapping("optimization config", raw), source_name)

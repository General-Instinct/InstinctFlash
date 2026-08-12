"""Typed schema for the Layer 2-6 optimization pipeline configuration.

The YAML is policy, never executable Python.  It names registered passes and supplies only
parameters that the pass definition explicitly exposes.  This module deliberately imports no
torch so a configuration can be inspected on the same laptop-only installation as a checkpoint.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


class ConfigurationError(ValueError):
    """The optimization configuration is invalid or cannot be resolved safely."""


class OptimizationLayer(str, enum.Enum):
    GRAPH = "graph"
    CACHE = "cache"
    ATTENTION = "attention"
    KERNEL = "kernel"
    HARDWARE = "hardware"


class PassMode(str, enum.Enum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"
    REQUIRED = "required"


class PassMaturity(str, enum.Enum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    UNAVAILABLE = "unavailable"
    DEPRECATED = "deprecated"


class InstallPhase(str, enum.Enum):
    PRE_BUILD = "pre_build"
    POST_BUILD = "post_build"
    POST_RESET = "post_reset"


_MISSING = object()


@dataclass(frozen=True)
class ParameterSpec:
    """One YAML-visible constructor parameter."""

    type: type | tuple[type, ...]
    default: Any = _MISSING
    choices: tuple[Any, ...] = ()
    minimum: int | float | None = None
    maximum: int | float | None = None
    description: str = ""

    def validate(self, pass_id: str, name: str, value: Any) -> Any:
        expected = self.type if isinstance(self.type, tuple) else (self.type,)
        # bool is an int in Python; accepting it for an integer tuning knob is almost never intended.
        if isinstance(value, bool) and bool not in expected:
            ok = False
        else:
            ok = isinstance(value, expected)
        if not ok:
            want = " | ".join(t.__name__ for t in expected)
            raise ConfigurationError(
                f"pass {pass_id!r} parameter {name!r} must be {want}, got {type(value).__name__}")
        if self.choices and value not in self.choices:
            raise ConfigurationError(
                f"pass {pass_id!r} parameter {name!r} must be one of {list(self.choices)!r}")
        if self.minimum is not None and value < self.minimum:
            raise ConfigurationError(
                f"pass {pass_id!r} parameter {name!r} must be >= {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ConfigurationError(
                f"pass {pass_id!r} parameter {name!r} must be <= {self.maximum}")
        return value


Installer = Callable[[object, object | None, object | None, Mapping[str, Any]], list[str]]
PassFactory = Callable[[Mapping[str, Any]], object]


@dataclass(frozen=True)
class PassDefinition:
    """A registered, YAML-addressable optimization module."""

    id: str
    version: str
    layer: OptimizationLayer
    factory: PassFactory
    installer: Installer | None = None
    install_phase: InstallPhase = InstallPhase.PRE_BUILD
    params: Mapping[str, ParameterSpec] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    before: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    requires_capabilities: frozenset[str] = frozenset()
    maturity: PassMaturity = PassMaturity.STABLE
    auto_eligible: bool = False
    legacy_flags: tuple[str, ...] = ()
    no_runtime_action: str = ""
    description: str = ""

    def validate_params(self, supplied: Mapping[str, Any]) -> dict[str, Any]:
        if not all(isinstance(name, str) for name in supplied):
            raise ConfigurationError(f"pass {self.id!r} parameter names must be strings")
        unknown = sorted(set(supplied) - set(self.params))
        if unknown:
            raise ConfigurationError(
                f"pass {self.id!r} has no configurable parameter(s) {unknown}; "
                f"allowed: {sorted(self.params)}")
        out: dict[str, Any] = {}
        for name, spec in self.params.items():
            if name in supplied:
                out[name] = spec.validate(self.id, name, supplied[name])
            elif spec.default is not _MISSING:
                out[name] = spec.default
        try:
            json.dumps(out, sort_keys=True)
        except (TypeError, ValueError) as e:
            raise ConfigurationError(
                f"pass {self.id!r} parameters must be JSON-serializable: {e}") from e
        return out


@dataclass(frozen=True)
class PassSelection:
    id: str
    layer: OptimizationLayer
    mode: PassMode = PassMode.AUTO
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelinePolicy:
    tier_ceiling: str = "bitexact"
    allow_experimental: bool = False
    unlisted: PassMode = PassMode.OFF


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    policy: PipelinePolicy
    selections: tuple[PassSelection, ...]
    schema_version: int = 1
    kind: str = "OptimizationPipeline"
    source: str = "<memory>"


@dataclass(frozen=True)
class ResolvedSelection:
    definition: PassDefinition
    mode: PassMode
    params: Mapping[str, Any]
    enabled_by: str = "user"
    reason: str = "selected"

    @property
    def id(self) -> str:
        return self.definition.id


@dataclass(frozen=True)
class SkippedSelection:
    id: str
    layer: OptimizationLayer
    mode: PassMode
    reason: str


@dataclass(frozen=True)
class ResolvedPipeline:
    name: str
    policy: PipelinePolicy
    ordered: tuple[ResolvedSelection, ...]
    skipped: tuple[SkippedSelection, ...]
    source: str
    fingerprint: str

    def explain(self) -> str:
        out = [f"optimization pipeline {self.name!r} [{self.fingerprint[:12]}]",
               f"  source: {self.source}",
               f"  tier ceiling: {self.policy.tier_ceiling}"]
        for i, item in enumerate(self.ordered, 1):
            suffix = f"; enabled by {item.enabled_by}" if item.enabled_by != "user" else ""
            out.append(
                f"  {i:02d} {item.definition.layer.value:<9} {item.id:<28} "
                f"mode={item.mode.value}{suffix}")
        for item in self.skipped:
            out.append(
                f"  -- {item.layer.value:<9} {item.id:<28} mode={item.mode.value}: {item.reason}")
        return "\n".join(out)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy": {
                "tier_ceiling": self.policy.tier_ceiling,
                "allow_experimental": self.policy.allow_experimental,
                "unlisted": self.policy.unlisted.value,
            },
            "passes": [
                {"id": p.id, "version": p.definition.version,
                 "layer": p.definition.layer.value, "mode": p.mode.value,
                 "params": dict(p.params), "enabled_by": p.enabled_by}
                for p in self.ordered
            ],
            "fingerprint": self.fingerprint,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.canonical_dict(), indent=2, sort_keys=True) + "\n")
        return path


def pipeline_fingerprint(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()

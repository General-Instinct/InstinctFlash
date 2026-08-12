"""Pass definitions and third-party discovery."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from instinctwm.config.schema import (
    ConfigurationError,
    InstallPhase,
    OptimizationLayer,
    ParameterSpec,
    PassDefinition,
    PassMaturity,
)


ENTRY_POINT_GROUP = "instinctwm.passes"


class PassRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, PassDefinition] = {}
        self._plugins_discovered = False
        self.plugin_problems: list[str] = []

    def register(self, definition: PassDefinition) -> None:
        if not isinstance(definition, PassDefinition):
            raise TypeError(f"expected PassDefinition, got {type(definition).__name__}")
        if not isinstance(definition.id, str) or not definition.id:
            raise ConfigurationError("optimization pass id must be a non-empty string")
        if definition.id in self._definitions:
            raise ConfigurationError(f"optimization pass {definition.id!r} is already registered")
        if not definition.id.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ConfigurationError(
                f"optimization pass id {definition.id!r} may contain only letters, numbers, '.', '-', '_'")
        if not isinstance(definition.version, str) or not definition.version.strip():
            raise ConfigurationError(f"optimization pass {definition.id!r} needs a version")
        if not isinstance(definition.layer, OptimizationLayer):
            raise ConfigurationError(f"optimization pass {definition.id!r} has an invalid layer")
        if not isinstance(definition.install_phase, InstallPhase):
            raise ConfigurationError(
                f"optimization pass {definition.id!r} has an invalid install phase")
        if not isinstance(definition.maturity, PassMaturity):
            raise ConfigurationError(f"optimization pass {definition.id!r} has invalid maturity")
        if not callable(definition.factory):
            raise ConfigurationError(f"optimization pass {definition.id!r} factory is not callable")
        if definition.installer is not None and not callable(definition.installer):
            raise ConfigurationError(f"optimization pass {definition.id!r} installer is not callable")
        relation_groups = (definition.requires, definition.before, definition.after,
                           definition.conflicts)
        if not all(isinstance(group, tuple) for group in relation_groups):
            raise ConfigurationError(
                f"optimization pass {definition.id!r} relationships must be tuples")
        relations = tuple(other for group in relation_groups for other in group)
        if not all(isinstance(other, str) and other for other in relations):
            raise ConfigurationError(
                f"optimization pass {definition.id!r} relationships must be non-empty strings")
        if definition.id in relations:
            raise ConfigurationError(f"optimization pass {definition.id!r} cannot relate to itself")
        if not isinstance(definition.params, Mapping) or not all(
                   isinstance(name, str) and isinstance(spec, ParameterSpec)
                   for name, spec in definition.params.items()):
            raise ConfigurationError(
                f"optimization pass {definition.id!r} has an invalid parameter schema")
        if not isinstance(definition.requires_capabilities, frozenset) or not all(
                   isinstance(capability, str) and capability
                   for capability in definition.requires_capabilities):
            raise ConfigurationError(
                f"optimization pass {definition.id!r} capabilities must be non-empty strings")
        if not isinstance(definition.auto_eligible, bool):
            raise ConfigurationError(
                f"optimization pass {definition.id!r} auto_eligible must be boolean")
        if not isinstance(definition.legacy_flags, tuple) or not all(
                isinstance(flag, str) and flag.startswith("--") for flag in definition.legacy_flags):
            raise ConfigurationError(
                f"optimization pass {definition.id!r} legacy flags must be '--' strings")
        self._definitions[definition.id] = definition

    def get(self, pass_id: str) -> PassDefinition:
        try:
            return self._definitions[pass_id]
        except KeyError:
            problems = (f"; plugin discovery problems: {self.plugin_problems}"
                        if self.plugin_problems else "")
            raise ConfigurationError(
                f"unknown optimization pass {pass_id!r}; registered: {list(self.names())}"
                f"{problems}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def definitions(self) -> tuple[PassDefinition, ...]:
        return tuple(self._definitions[n] for n in self.names())

    def discover_plugins(self) -> tuple[str, ...]:
        if self._plugins_discovered:
            return tuple(self.plugin_problems)
        self._plugins_discovered = True
        try:
            from importlib.metadata import entry_points
            try:
                eps = entry_points(group=ENTRY_POINT_GROUP)
            except TypeError:  # pragma: no cover - Python 3.10 compatibility shape
                eps = (entry_points() or {}).get(ENTRY_POINT_GROUP, ())
        except Exception:  # pragma: no cover - a broken metadata environment
            return ()
        for ep in eps:
            try:
                value = ep.load()
                value = value() if callable(value) and not isinstance(value, PassDefinition) else value
                definitions: Iterable[PassDefinition]
                if isinstance(value, PassDefinition):
                    definitions = (value,)
                elif isinstance(value, Iterable):
                    definitions = tuple(value)
                else:
                    raise TypeError("entry point must return PassDefinition or an iterable of them")
                staged = PassRegistry()
                for definition in definitions:
                    # External ids must be namespaced; a plugin cannot impersonate a built-in pass.
                    if "." not in definition.id:
                        raise ConfigurationError(
                            f"third-party pass id {definition.id!r} must contain a package namespace")
                    if definition.id in self._definitions:
                        raise ConfigurationError(
                            f"optimization pass {definition.id!r} is already registered")
                    staged.register(definition)
                for definition in staged.definitions():
                    self.register(definition)
            except Exception as e:  # one bad plugin must not break unrelated configs
                self.plugin_problems.append(
                    f"{ep.name} ({ep.value}): {type(e).__name__}: {e}")
        return tuple(self.plugin_problems)


_DEFAULT: PassRegistry | None = None


def default_registry() -> PassRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        from instinctwm.config.builtins import register_builtins
        _DEFAULT = PassRegistry()
        register_builtins(_DEFAULT)
    _DEFAULT.discover_plugins()
    return _DEFAULT

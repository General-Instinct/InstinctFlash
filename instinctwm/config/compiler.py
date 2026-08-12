"""Resolve a YAML policy into a deterministic, dependency-safe pass order."""

from __future__ import annotations

from typing import Mapping

from instinctwm.config.loader import load_config
from instinctwm.config.registry import PassRegistry, default_registry
from instinctwm.config.schema import (
    ConfigurationError,
    InstallPhase,
    PassMaturity,
    PassMode,
    PipelineConfig,
    ResolvedPipeline,
    ResolvedSelection,
    SkippedSelection,
    pipeline_fingerprint,
)


def resolve_pipeline(source, registry: PassRegistry | None = None) -> ResolvedPipeline:
    config: PipelineConfig = load_config(source)
    registry = registry or default_registry()
    configured = {s.id: s for s in config.selections}
    active: dict[str, ResolvedSelection] = {}
    skipped: list[SkippedSelection] = []

    def activate(pass_id: str, *, enabled_by: str = "user", inherited_mode: PassMode | None = None):
        is_dependency = enabled_by not in {"user", "policy.unlisted"}
        definition = registry.get(pass_id)
        selection = configured.get(pass_id)
        if selection is not None and selection.layer is not definition.layer:
            raise ConfigurationError(
                f"pass {pass_id!r} belongs to layer {definition.layer.value!r}, "
                f"not {selection.layer.value!r}")
        mode = selection.mode if selection is not None else (inherited_mode or config.policy.unlisted)
        raw_params: Mapping = selection.params if selection is not None else {}
        params = definition.validate_params(raw_params)
        if mode is PassMode.OFF:
            if is_dependency:
                raise ConfigurationError(
                    f"pass {enabled_by!r} requires {pass_id!r}, but {pass_id!r} is explicitly off")
            skipped.append(SkippedSelection(pass_id, definition.layer, mode, "explicitly disabled"))
            return False
        if mode is PassMode.AUTO and not definition.auto_eligible:
            if is_dependency:
                raise ConfigurationError(
                    f"pass {enabled_by!r} requires {pass_id!r}, but {pass_id!r} is not eligible "
                    f"for automatic selection")
            skipped.append(SkippedSelection(
                pass_id, definition.layer, mode, "not eligible for automatic selection"))
            return False
        if definition.maturity is PassMaturity.UNAVAILABLE:
            reason = "registered for explanation but has no executable implementation"
            if mode is PassMode.REQUIRED or is_dependency:
                raise ConfigurationError(f"required pass {pass_id!r} is unavailable: {reason}")
            skipped.append(SkippedSelection(pass_id, definition.layer, mode, reason))
            return False
        if definition.maturity is PassMaturity.EXPERIMENTAL and not config.policy.allow_experimental:
            reason = "experimental passes are disabled by policy.allow_experimental=false"
            if mode is PassMode.REQUIRED or is_dependency:
                raise ConfigurationError(f"required pass {pass_id!r} is experimental: {reason}")
            skipped.append(SkippedSelection(pass_id, definition.layer, mode, reason))
            return False
        if definition.maturity is PassMaturity.DEPRECATED and mode is PassMode.AUTO:
            reason = "deprecated passes require an explicit mode: on or required"
            if is_dependency:
                raise ConfigurationError(f"required pass {pass_id!r} is deprecated: {reason}")
            skipped.append(SkippedSelection(pass_id, definition.layer, mode, reason))
            return False
        active[pass_id] = ResolvedSelection(definition, mode, params, enabled_by=enabled_by)
        return True

    for selection in config.selections:
        activate(selection.id)
    if config.policy.unlisted is not PassMode.OFF:
        for pass_id in registry.names():
            if pass_id not in configured:
                activate(pass_id, enabled_by="policy.unlisted",
                         inherited_mode=config.policy.unlisted)

    # Dependencies are transitive and may cross layer numbers. Layer is organization, not order.
    pending = list(active)
    while pending:
        pass_id = pending.pop(0)
        item = active[pass_id]
        for dependency in item.definition.requires:
            if dependency in active:
                continue
            mode = PassMode.REQUIRED if item.mode is PassMode.REQUIRED else PassMode.ON
            if activate(dependency, enabled_by=pass_id, inherited_mode=mode):
                pending.append(dependency)

    for pass_id, item in active.items():
        for other in item.definition.conflicts:
            if other in active:
                raise ConfigurationError(
                    f"optimization passes {pass_id!r} and {other!r} conflict; disable one explicitly")

    # dependency -> dependent edges. before/after are definition-owned, never YAML-owned.
    edges = {pass_id: set() for pass_id in active}
    indegree = {pass_id: 0 for pass_id in active}

    def add_edge(first: str, second: str) -> None:
        if first not in active or second not in active or second in edges[first]:
            return
        edges[first].add(second)
        indegree[second] += 1

    for pass_id, item in active.items():
        for dep in item.definition.requires:
            add_edge(dep, pass_id)
        for earlier in item.definition.after:
            add_edge(earlier, pass_id)
        for later in item.definition.before:
            add_edge(pass_id, later)

    # Lifecycle is part of execution order: all pre-build mutations happen before a model exists,
    # post-build transforms see that model, and post-reset transforms see initialized episode state.
    # Adding these edges also turns an impossible dependency (POST_RESET required by PRE_BUILD) into
    # a closed cycle instead of silently executing the declared DAG in the wrong order.
    phase_rank = {InstallPhase.PRE_BUILD: 0, InstallPhase.POST_BUILD: 1,
                  InstallPhase.POST_RESET: 2}
    for first, first_item in active.items():
        for second, second_item in active.items():
            if phase_rank[first_item.definition.install_phase] < \
                    phase_rank[second_item.definition.install_phase]:
                add_edge(first, second)

    def sort_key(pass_id: str):
        d = active[pass_id].definition
        return phase_rank[d.install_phase], d.layer.value, pass_id

    ready = sorted((p for p, n in indegree.items() if n == 0), key=sort_key)
    ordered_ids = []
    while ready:
        current = ready.pop(0)
        ordered_ids.append(current)
        for nxt in sorted(edges[current], key=sort_key):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort(key=sort_key)
    if len(ordered_ids) != len(active):
        cycle = sorted(p for p, n in indegree.items() if n > 0)
        raise ConfigurationError(f"optimization pass dependency cycle: {cycle}")

    ordered = tuple(active[p] for p in ordered_ids)
    payload = {
        "schema_version": config.schema_version,
        "name": config.name,
        "policy": {
            "tier_ceiling": config.policy.tier_ceiling,
            "allow_experimental": config.policy.allow_experimental,
            "unlisted": config.policy.unlisted.value,
        },
        "passes": [
            {"id": x.id, "version": x.definition.version, "layer": x.definition.layer.value,
             "mode": x.mode.value, "params": dict(x.params), "enabled_by": x.enabled_by}
            for x in ordered
        ],
    }
    return ResolvedPipeline(config.name, config.policy, ordered, tuple(skipped), config.source,
                            pipeline_fingerprint(payload))

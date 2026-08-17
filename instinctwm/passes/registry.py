"""Which passes exist. A registry, so the planner does not have to know any model's name.

    from instinctwm.passes.registry import default_passes
    default_passes()            # everything installed, in evaluation order

THE LEAK THIS CLOSES. `Optimizer.__init__` did `from instinctwm.passes.lingbot import default_passes`,
so the default pass set for every model in the ecosystem was one model's list. It was the only
model-name reference left in executable code anywhere in the generic layers -- measured, not guessed --
and it is exactly the shape the architecture is supposed to forbid: a new family could not add a pass
without editing the planner, and every family inherited passes written for a world model.

A registry naming its own builtins is fine; a planner naming a model is not. `runtime/loader.py` makes
the same distinction for adapters, and this is the symmetric half for passes.

EXTERNAL PASSES. A third-party package advertises passes the same way it advertises adapters:

    [project.entry-points."instinctwm.passes"]
    my_family = "my_package.passes:default_passes"

The entry point names a zero-argument callable returning a list of pass instances. Discovery failures
are reported, never raised: one broken third-party pass package must not stop a runtime from planning a
model that has nothing to do with it.
"""

from __future__ import annotations

from typing import Callable, Sequence

#: The entry-point group. Stable, and part of the public extension surface.
ENTRY_POINT_GROUP = "instinctwm.passes"

#: Providers registered in this process, in registration order. Order matters: some passes are
#: preconditions for others, and the planner evaluates in the order it receives.
_PROVIDERS: list[tuple[str, Callable[[], Sequence]]] = []
_DISCOVERED = False


def register_passes(name: str, provider: Callable[[], Sequence]) -> None:
    """Add a provider of pass instances. Re-registering a name replaces it."""
    global _PROVIDERS
    _PROVIDERS = [(n, p) for n, p in _PROVIDERS if n != name] + [(name, provider)]


def _register_builtins() -> None:
    """The passes that ship in this repository.

    Imported here rather than in the planner because a registry is allowed to know what it ships.
    Lazy inside the function: the pass modules import the planner's contract, so a module-scope
    import would be circular.
    """
    def lingbot():
        from instinctwm.passes.lingbot import default_passes as _d
        return _d()

    register_passes("lingbot", lingbot)


def discover(strict: bool = False) -> list[str]:
    """Register builtins plus any installed third-party providers. Returns failures, never raises."""
    global _DISCOVERED
    if _DISCOVERED:
        return []
    _DISCOVERED = True
    problems: list[str] = []
    try:
        _register_builtins()
    except Exception as e:                                        # noqa: BLE001
        problems.append(f"builtin passes: {type(e).__name__}: {e}")
    try:
        from importlib.metadata import entry_points
        try:
            eps = entry_points(group=ENTRY_POINT_GROUP)
        except TypeError:                                        # pragma: no cover  py<3.10 shape
            eps = (entry_points() or {}).get(ENTRY_POINT_GROUP, ())   # type: ignore[assignment]
        for ep in eps:
            try:
                register_passes(ep.name, ep.load())
            except Exception as e:                               # noqa: BLE001
                problems.append(f"{ep.name} ({ep.value}): {type(e).__name__}: {e}")
    except Exception as e:                                       # noqa: BLE001
        problems.append(f"entry-point discovery: {type(e).__name__}: {e}")
    if strict and problems:
        raise RuntimeError("; ".join(problems))
    return problems


def providers() -> list[str]:
    """Names of every registered provider, in evaluation order."""
    discover()
    return [n for n, _ in _PROVIDERS]


def default_passes() -> list:
    """Fresh pass instances from every registered provider, in evaluation order.

    Fresh instances rather than a shared list: passes are stateless today, but a shared mutable
    default is the kind of thing that stops being true quietly.
    """
    discover()
    out: list = []
    for _, provider in _PROVIDERS:
        try:
            out.extend(provider())
        except Exception:                                        # noqa: BLE001  a bad provider
            continue                                             # is reported by discover()
    return out

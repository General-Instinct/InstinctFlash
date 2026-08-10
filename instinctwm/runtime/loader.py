"""`load()` — the entry point, and the registry behind it.

`load("some-model-id")` returns a `BackendAdapter`: the object that states facts about a model
and knows how to apply a plan to that model's server. It deliberately does NOT touch the
filesystem, import torch, or read a checkpoint — reading a model's declarations has to work on
a laptop with no GPU, or the optimizer stops being something you can reason with offline.

Registration is by factory rather than by instance so that adapters stay cheap to enumerate:
`available_models()` should not construct six model backends to print six strings.
"""

from __future__ import annotations

from typing import Callable, Dict

from instinctwm.adapters.base import BackendAdapter

_REGISTRY: Dict[str, Callable[[], BackendAdapter]] = {}


def register(model_id: str, factory: Callable[[], BackendAdapter]) -> None:
    """Add a backend to the registry.

    Re-registering the same id is an error rather than an overwrite: two adapters claiming one
    model id means one of them is silently never used, and which one depends on import order.
    """
    if model_id in _REGISTRY:
        raise KeyError(f"{model_id!r} is already registered")
    _REGISTRY[model_id] = factory


def available_models() -> list[str]:
    """Every registered model id, sorted. Discovers installed plugins first."""
    discover_plugins()
    return sorted(_REGISTRY)


#: Packages already scanned, so discovery is idempotent and cheap to call on every lookup.
_DISCOVERED = False

#: The entry-point group an external model family advertises itself under.
ENTRY_POINT_GROUP = "instinctwm.adapters"


def discover_plugins() -> list[str]:
    """Register adapters advertised by INSTALLED packages, without importing them by name.

    WHY THIS EXISTS. Before it, `register()` was the only way in, so an external model family was
    reachable only if the user already knew to `import their_package` before calling
    `Runtime.from_pretrained`. That is precisely the hidden knowledge a Hub repo id is supposed to
    replace: the checkpoint declares `backbone: gridworld_ar` and the runtime could not turn that
    into "you need the gridworld-wm package", so a correct, published, servable checkpoint failed
    for a reason the user had no way to see.

    An external author now declares, in their own pyproject.toml:

        [project.entry-points."instinctwm.adapters"]
        gridworld_ar = "gridworld_wm.adapter:GridworldAdapter"

    and `pip install gridworld-wm` is the whole integration. No PR to InstinctWM, no import order to
    get right, and the backbone name lives next to the code that implements it.

    Failures here are deliberately non-fatal: one broken third-party plugin must not stop a runtime
    from loading a model that has nothing to do with it. The names that failed are returned so a
    caller can report them.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return []
    _DISCOVERED = True
    problems: list[str] = []
    try:
        from importlib.metadata import entry_points
    except Exception:                                            # pragma: no cover
        return problems
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:                                            # pragma: no cover  py<3.10 shape
        eps = (entry_points() or {}).get(ENTRY_POINT_GROUP, ())   # type: ignore[assignment]
    for ep in eps:
        if ep.name in _REGISTRY:
            continue
        try:
            _REGISTRY[ep.name] = ep.load()
        except Exception as e:                                   # noqa: BLE001
            problems.append(f"{ep.name} ({ep.value}): {type(e).__name__}: {e}")
    return problems


def load(model_id: str, **kwargs) -> BackendAdapter:
    """Return the backend adapter for `model_id`.

    Extra keyword arguments go to the backend's constructor — `lingbot_root=` for LingBot-VA,
    for instance. Nothing here reads a checkpoint; that happens in `serve()`.
    """
    if model_id not in _REGISTRY:
        discover_plugins()
    try:
        factory = _REGISTRY[model_id]
    except KeyError:
        raise KeyError(
            f"unknown model {model_id!r}. Registered: {available_models()}. "
            f"Add one with instinctwm.register(model_id, factory)."
        ) from None
    return factory(**kwargs) if kwargs else factory()


def _register_builtins() -> None:
    """Register the backends that ship with InstinctWM.

    Imported lazily inside the function so a broken third-party backend cannot make
    `import instinctwm` fail.
    """
    from instinctwm.adapters.lingbot_va import LingBotVA

    # The BACKBONE id is what a checkpoint declares; many checkpoints share one backbone. The old
    # registration used a CHECKPOINT id as a backbone id, which is the conflation the platform claim
    # forbids -- it is kept as an alias so nothing that names it breaks.
    register(LingBotVA.BACKBONE, LingBotVA)
    register(LingBotVA.model_id, LingBotVA)          # deprecated alias


_register_builtins()

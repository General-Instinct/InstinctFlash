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
    """Every registered model id, sorted."""
    return sorted(_REGISTRY)


def load(model_id: str, **kwargs) -> BackendAdapter:
    """Return the backend adapter for `model_id`.

    Extra keyword arguments go to the backend's constructor — `lingbot_root=` for LingBot-VA,
    for instance. Nothing here reads a checkpoint; that happens in `serve()`.
    """
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

    register(LingBotVA.model_id, LingBotVA)


_register_builtins()

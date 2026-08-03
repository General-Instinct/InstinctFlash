"""Recipe registry.

One entry, deliberately. An earlier version of this file also shipped declaration-only `scm`, `rcm`
and `dmd2` classes to demonstrate that the interface could carry them. That was abstraction built
for papers we have not committed to, and it has been removed: the Layer 2 pass framework got its
interfaces right by implementing real optimizations and letting the seams emerge, and Layer 1 gets
the same treatment. Once PDD works end to end on LingBot-VA we ask whether the interface extends,
with a working recipe to argue from.

`Capabilities` stays -- it predates this and already earns its place by turning "this box has no
flash-attn" into a startup error rather than a discovery hours into a job.
"""

from __future__ import annotations

from typing import Callable

from instinctwm.train.recipes.pdd import ParallelDecoding

#: name -> factory taking the per-phase NFE mapping.
REGISTRY: dict[str, Callable[..., object]] = {
    "pdd": ParallelDecoding,
}


def register(name: str, factory: Callable[..., object]) -> None:
    if name in REGISTRY:
        raise KeyError(f"recipe {name!r} already registered")
    REGISTRY[name] = factory


def build(name: str, *args, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown recipe {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](*args, **kwargs)


def available() -> list[str]:
    return sorted(REGISTRY)


__all__ = ["ParallelDecoding", "REGISTRY", "register", "build", "available"]

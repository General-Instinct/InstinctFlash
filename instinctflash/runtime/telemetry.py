"""Read statistics from an already loaded local backend without initiating work."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import inspect


def backend_stats_snapshot(backend) -> dict:
    """Return a detached statistics envelope; loading and worker RPC are never requested.

    A loaded implementation may expose a property or a legacy zero-argument
    method. Errors from either are propagated: broken statistics must not be
    reported as missing statistics.
    """
    from instinctflash.runtime.engine_backend import EngineBackend
    from instinctflash.runtime.execution import InProcessBackend, WorkerBackend

    name = type(backend).__name__
    if isinstance(backend, WorkerBackend):
        return {"status": "unsupported", "backend": name, "stats": None,
                "reason": "Worker placement does not provide a live statistics RPC."}
    if isinstance(backend, InProcessBackend):
        implementation = vars(backend).get("_impl")
    elif isinstance(backend, EngineBackend):
        implementation = vars(backend).get("_loop")
    else:
        return {"status": "unsupported", "backend": name, "stats": None,
                "reason": "This execution backend does not provide a local statistics snapshot."}
    if implementation is None:
        return {"status": "not_loaded", "backend": name, "stats": None,
                "reason": "The local model has not been loaded or has been closed."}

    missing = object()
    # Distinguish an absent attribute from AttributeError raised inside a broken
    # property. getattr(..., default) alone would silently swallow the latter.
    if inspect.getattr_static(implementation, "backend_stats", missing) is missing:
        return {"status": "unsupported", "backend": name, "stats": None,
                "reason": "The loaded model does not expose backend_stats."}
    statistics = implementation.backend_stats
    if callable(statistics):
        statistics = statistics()
    if not isinstance(statistics, Mapping):
        raise TypeError("Loaded backend_stats must return a mapping, "
                        f"got {type(statistics).__name__}")
    return {"status": "available", "backend": name,
            "stats": deepcopy(dict(statistics))}

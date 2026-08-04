"""InstinctWM — load, optimize, and deploy world-action models.

    from instinctwm import load, Optimizer, Tier

    model = load("lingbot-va-posttrain-robotwin")          # the adapter states facts
    plan  = Optimizer(tier_ceiling=Tier.BITEXACT).compile(model.spec())
    print(plan.explain())                                  # what fired, and why
    server = plan.serve(model, port=29056)                 # deploy

Everything above the `serve()` line is pure analysis: no torch, no checkpoint, no GPU. That is
intentional — deciding which optimizations are legal for a model is something you should be
able to do on a laptop, and a framework that needs the weights loaded before it can tell you
what it would do is a runtime wearing a framework's clothes.
"""

from __future__ import annotations

import os as _os
import sys as _sys


def _bootstrap_submodules() -> None:
    """Make the vendored `instinct-pdd` submodule importable without an install step.

    Layer 1's algorithm lives in its own repository (`instinct-pdd`), included here as a git
    submodule so there is exactly one copy of it. Its package sits at `instinct-pdd/src/`, which is
    not importable by default.

    A FALLBACK, NOT A HIJACK: if `instinct_pdd` already imports -- because it was pip-installed, or
    because a caller put it on the path deliberately -- this does nothing. That ordering matters,
    since silently shadowing an installed release with whatever revision the submodule happens to be
    pinned at is the kind of thing that makes a bug report unreproducible.

    The alternative was `pip install -e ./instinct-pdd`, which is cleaner in principle but mutates a
    deliberately pinned environment; see eval/lingbot_va_robotwin/env.sh on why that environment is
    left alone. The alternative to THAT was a sys.path insert in every entry point, which is worse.
    """
    try:
        import instinct_pdd  # noqa: F401
        return
    except ImportError:
        pass
    src = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                        "instinct-pdd", "src")
    if _os.path.isdir(src) and src not in _sys.path:
        _sys.path.insert(0, src)


_bootstrap_submodules()

from instinctwm.adapter.base import (
    AdapterSpec,
    BackendAdapter,
    CommitMode,
    GuidanceMode,
    GuidanceRule,
    KVLifetime,
    KVStreamSpec,
    PhaseSpec,
    PurityKey,
)
from instinctwm.deployment import DeploymentSpec
from instinctwm.loader import available_models, load, register
from instinctwm.optimizer.base import (
    OptimizationPass,
    Optimizer,
    PassResult,
    Plan,
    Tier,
)
from instinctwm.optimizer.passes import default_passes

__version__ = "0.1.0"

__all__ = [
    # entry points
    "load",
    "register",
    "available_models",
    "Optimizer",
    # what an adapter declares
    "AdapterSpec",
    "BackendAdapter",
    "CommitMode",
    "GuidanceMode",
    "GuidanceRule",
    "KVLifetime",
    "KVStreamSpec",
    "PhaseSpec",
    "PurityKey",
    # how it is deployed
    "DeploymentSpec",
    # what the optimizer produces
    "OptimizationPass",
    "PassResult",
    "Plan",
    "Tier",
    "default_passes",
    "__version__",
]

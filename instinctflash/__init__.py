"""InstinctFlash — one runtime for robot world-action models.

    from instinctflash import load, Optimizer, Tier

    model = load("lingbot-va-posttrain-robotwin")          # the adapter states facts
    plan  = Optimizer(tier_ceiling=Tier.BITEXACT).compile(model.spec())
    print(plan.explain())                                  # what fired, and why
    server = plan.serve(model, port=29056)                 # deploy

ONE RUNTIME. MANY CHECKPOINTS. SHARED INFRASTRUCTURE.

There is no "fast runtime" and no "quality runtime". There is one runtime, and it serves every
compatible checkpoint by reading what that checkpoint declares. Training recipes -- PDD, DMD2, LCM,
DreamZero -- produce different CHECKPOINTS, never different runtimes. Nothing below this line knows
which recipe produced the weights it is serving, and nothing is allowed to ask: see
`descriptors/` for the facts a checkpoint publishes and CHECKPOINTS.md for why the method name is
deliberately absent from them.

    descriptors/   what a checkpoint declares          (capabilities, not recipes)
    adapters/      WHERE things are, per backbone      (publish sites)
    passes/        WHAT to do there                    (consume sites, return rewrites)
    planners/      which passes are legal + profitable (declarations only, no GPU)
    executors/     apply a plan to a live server       (the only layer that touches the model)
    backends/      kernels                             (triton / torch, chosen by measurement)
    runtime/       load, install, serve
    train/         Layer 1 -- recipes that MAKE checkpoints, not part of serving

Everything above the `serve()` line is pure analysis: no torch, no checkpoint, no GPU. That is
intentional -- deciding which optimizations are legal for a model is something you should be
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

from instinctflash.adapters.base import (
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
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.runtime.loader import available_models, load, register
from instinctflash.planners.planner import (
    OptimizationPass,
    Optimizer,
    PassResult,
    Plan,
    Tier,
)
from instinctflash.passes.lingbot import default_passes
from instinctflash.runtime.facade import Episode, Runtime, UnknownBackboneError, describe
from instinctflash.descriptors.package import from_pretrained


__version__ = "0.1.0"

__all__ = [
    # --- the public API. Everything else is one import deeper and rarely needed.
    "Runtime",
    "Episode",
    "from_pretrained",
    "describe",
    "UnknownBackboneError",
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

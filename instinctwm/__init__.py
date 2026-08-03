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

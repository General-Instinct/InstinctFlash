"""Planners — decide WHAT to do, from declared facts alone.

A planner reads an `AdapterSpec` (what the model is) plus `DeploymentSpec` (how it is being served)
and returns a `Plan`. It never imports a model module, never touches a checkpoint, and never asks
which training recipe produced the weights. Every decision is derived from declarations, which is
what lets `Optimizer(...).compile(spec)` run on a laptop with no GPU.
"""
from instinctflash.planners.planner import OptimizationPass, Optimizer, PassResult, Plan, Tier

__all__ = ["Optimizer", "Plan", "Tier", "OptimizationPass", "PassResult"]

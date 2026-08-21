"""The pass registry.

Registration order is evaluation order, and it is load-bearing where one pass is a
precondition for another. The current order removes substrate overhead first — those passes
are unconditional and bit-exact — and then runs the passes derived from model declarations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from instinctflash.passes.lingbot.cfg_elision import CFGBranchElision
from instinctflash.passes.lingbot.conditioning_prefill import ConditioningPrefill
from instinctflash.passes.lingbot.substrate import (
    AllocatorChurnElision,
    DebugDumpElision,
    FSDPElision,
    ObsDecodeElision,
)

if TYPE_CHECKING:
    from instinctflash.planners.planner import OptimizationPass


def default_passes() -> "list[OptimizationPass]":
    """Every pass InstinctFlash ships, in evaluation order.

    Returns fresh instances rather than a shared module-level list: passes are stateless
    today, but a shared mutable default is the kind of thing that stops being true quietly.
    """
    return [
        FSDPElision(),
        AllocatorChurnElision(),
        DebugDumpElision(),
        ObsDecodeElision(),
        ConditioningPrefill(),
        CFGBranchElision(),
    ]


__all__ = [
    "AllocatorChurnElision",
    "CFGBranchElision",
    "ConditioningPrefill",
    "DebugDumpElision",
    "FSDPElision",
    "ObsDecodeElision",
    "default_passes",
]

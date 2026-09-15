"""The pass registry.

Registration order is evaluation order, and it is load-bearing where one pass is a
precondition for another. The current order removes substrate overhead first — those passes
are unconditional and bit-exact — and then runs the passes derived from model declarations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from instinctflash.passes.lingbot.action_terminal_elision import ActionTerminalForwardElision
from instinctflash.passes.lingbot.cfg_elision import CFGBranchElision
from instinctflash.passes.lingbot.conditioning_prefill import ConditioningPrefill
from instinctflash.passes.lingbot.conv_layout_ndhwc import ConvLayoutAutotune
from instinctflash.passes.lingbot.ring_kv import RingKVAddressing
from instinctflash.passes.lingbot.sm120_gated_residual import SM120GatedResidual
from instinctflash.passes.lingbot.sm120_wan_stage2 import SM120WanStage2
from instinctflash.passes.lingbot.sm120_wan_stage3 import SM120WanStage3
from instinctflash.passes.lingbot.sm120_wan_qk_rope import SM120WanQKRoPE
from instinctflash.passes.lingbot.sm120_wan_gemm import SM120WanGEMM
from instinctflash.passes.lingbot.substrate import (
    AllocatorChurnElision,
    DebugDumpElision,
    FSDPElision,
    ObsDecodeElision,
    PromptEncoderStaging,
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
        # after ConditioningPrefill on purpose: the staging mechanism lives in its reset wrapper,
        # and install_plan enforces the pairing.
        PromptEncoderStaging(),
        RingKVAddressing(),
        # after RingKVAddressing by convention only: the elision detects the allocator per cache at its
        # first reservation, so it composes with or without P003 in either order.
        ActionTerminalForwardElision(),
        SM120GatedResidual(),
        SM120WanStage2(),
        SM120WanStage3(),
        SM120WanQKRoPE(),
        SM120WanGEMM(),
        ConvLayoutAutotune(),
        CFGBranchElision(),
    ]


__all__ = [
    "ActionTerminalForwardElision",
    "AllocatorChurnElision",
    "CFGBranchElision",
    "ConditioningPrefill",
    "ConvLayoutAutotune",
    "DebugDumpElision",
    "FSDPElision",
    "ObsDecodeElision",
    "PromptEncoderStaging",
    "RingKVAddressing",
    "SM120GatedResidual",
    "SM120WanStage2",
    "SM120WanStage3",
    "SM120WanQKRoPE",
    "SM120WanGEMM",
    "default_passes",
]

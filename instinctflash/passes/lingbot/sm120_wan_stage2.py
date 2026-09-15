"""P009-A2: BITEXACT SM120 fusion of Wan's middle block regions.

The pass depends on P009-A1 and replaces two eager residual/LayerNorm chains per block with
Welford kernels reproducing PyTorch 2.9's exact D=3072 reduction tree. Attention, norm1, FFN,
and the A1 final residual remain unchanged.
"""

from __future__ import annotations

from instinctflash.adapters.base import AdapterSpec
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import HardwareReq
from instinctflash.planners.planner import PassResult, Tier


class SM120WanStage2:
    name = "sm120_wan_stage2"
    requires_capabilities = frozenset({"backbone:wan_va"})
    hardware = HardwareReq(
        min_capability=(12, 0),
        requires=frozenset({"cuda", "sm120_kernels", "sm120_stage2_kernels"}),
    )

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        device = getattr(deployment, "device", None)
        if device is None:
            # Same doctrine as P009-A1 (see sm120_gated_residual.py): an unprobed target must
            # decline, not stay applies=True under the planner's UNCHECKED annotation and then
            # fail in the kernel constructor at install time.
            return PassResult(
                self.name,
                False,
                Tier.BITEXACT,
                "no probed device: the kernel is compiled and certified for SM120 exactly, so an "
                "unprobed target declines instead of deferring to install time",
            )
        if device.capability != (12, 0):
            return PassResult(
                self.name,
                False,
                Tier.BITEXACT,
                f"kernel is compiled and certified for SM120 exactly; device is "
                f"sm_{device.capability[0]}{device.capability[1]}",
            )
        return PassResult(
            self.name,
            True,
            Tier.BITEXACT,
            "P009-A1 and the independent Wan stage2 ABI are present; the two Welford kernels "
            "preserve eager BF16 boundaries and PyTorch 2.9's exact D=3072 reduction order",
            params={
                "depends_on": "sm120_gated_residual",
                "regions_per_block": 2,
                "blocks": 30,
                "certified_rows": [64, 480],
                "hidden_dim": 3072,
                "library_feature": "sm120_stage2_kernels",
                "buffer_plan": "three BF16 buffers per block/shape",
            },
            expected_win=(
                "RTX 5090 P003+P007+P009-A1, integrated 42-cycle locked ABBA: mean "
                "349.86 -> 337.62 ms (1.0362x); saturated 396.01 -> 384.72 ms "
                "(1.0294x), 168/168 actions bitwise equal"
            ),
        )


__all__ = ["SM120WanStage2"]

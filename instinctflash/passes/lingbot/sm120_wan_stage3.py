"""P009-A3: BITEXACT SM120 fusion of Wan norm1 + Ada modulation."""

from __future__ import annotations

from instinctflash.adapters.base import AdapterSpec
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import HardwareReq
from instinctflash.planners.planner import PassResult, Tier


class SM120WanStage3:
    name = "sm120_wan_stage3"
    requires_capabilities = frozenset({"backbone:wan_va"})
    hardware = HardwareReq(
        min_capability=(12, 0),
        requires=frozenset({
            "cuda", "sm120_kernels", "sm120_stage2_kernels", "sm120_stage3_kernels",
        }),
    )

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        device = getattr(deployment, "device", None)
        if device is None:
            return PassResult(
                self.name, False, Tier.BITEXACT,
                "unprobed target: the kernel is compiled and certified for SM120 exactly",
            )
        if device.capability != (12, 0):
            return PassResult(
                self.name, False, Tier.BITEXACT,
                f"kernel is compiled and certified for SM120 exactly; device is "
                f"sm_{device.capability[0]}{device.capability[1]}",
            )
        return PassResult(
            self.name,
            True,
            Tier.BITEXACT,
            "P009-A1/A2 and the independent A3 ABI are present; norm1 FP32 LayerNorm and "
            "Ada modulation use PyTorch 2.9's exact D=3072 Welford order",
            params={
                "depends_on": "sm120_wan_stage2",
                "regions_per_block": 1,
                "blocks": 30,
                "certified_rows": [64, 480],
                "hidden_dim": 3072,
                "library_feature": "sm120_stage3_kernels",
            },
            expected_win=(
                "RTX 5090 A1+A2 baseline, 42-cycle A-B-B-A: mean "
                "409.50 -> 404.34 ms (1.0127x), growing 406.53 -> 401.23 ms, "
                "saturated 454.95 -> 449.44 ms; 168/168 actions bitwise equal"
            ),
        )


__all__ = ["SM120WanStage3"]

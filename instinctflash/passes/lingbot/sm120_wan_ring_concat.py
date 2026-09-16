"""P009-A6: bitexact SM120 dual K/V ring-wrap concatenation."""

from __future__ import annotations

from instinctflash.adapters.base import AdapterSpec
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import HardwareReq
from instinctflash.planners.planner import PassResult, Tier


class SM120WanRingConcat:
    name = "sm120_wan_ring_concat"
    requires_capabilities = frozenset({"backbone:wan_va"})
    hardware = HardwareReq(
        min_capability=(12, 0),
        requires=frozenset(
            {
                "cuda",
                "sm120_kernels",
                "sm120_stage2_kernels",
                "sm120_stage3_kernels",
                "sm120_qk_rope_kernels",
                "sm120_gemm_kernels",
                "sm120_ring_concat_kernels",
            }
        ),
    )

    def evaluate(
        self,
        spec: AdapterSpec,
        deployment: DeploymentSpec,
    ) -> PassResult:
        device = getattr(deployment, "device", None)
        if device is None:
            return PassResult(
                self.name,
                False,
                Tier.BITEXACT,
                "unprobed target: A6 is certified for SM120 exactly",
            )
        if device.capability != (12, 0):
            return PassResult(
                self.name,
                False,
                Tier.BITEXACT,
                f"A6 is certified for SM120 exactly; device is "
                f"sm_{device.capability[0]}{device.capability[1]}",
            )
        return PassResult(
            self.name,
            True,
            Tier.BITEXACT,
            "A1-A5 and the independent dual-K/V ring-concat ABI are present; "
            "wrapped ring intervals are copied into one shared scratch arena without "
            "changing BF16 words or attention order",
            params={
                "depends_on": "sm120_wan_gemm",
                "pool_shape": [2, 9792, 24, 128],
                "module_sites": 30,
                "library_feature": "sm120_ring_concat_kernels",
            },
            expected_win=(
                "RTX 5090 A5 baseline, 42-cycle A-B-B-A: 389.80 -> 387.73 ms; "
                "late cycles 426.60 -> 422.26 ms; 168/168 actions bitwise equal"
            ),
        )


__all__ = ["SM120WanRingConcat"]

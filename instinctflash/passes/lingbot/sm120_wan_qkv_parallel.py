"""P009-A7: bitexact parallel Q/K/V projections on SM120."""

from __future__ import annotations

from instinctflash.adapters.base import AdapterSpec
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import HardwareReq
from instinctflash.planners.planner import PassResult, Tier


class SM120WanParallelQKV:
    name = "sm120_wan_qkv_parallel"
    requires_capabilities = frozenset({"backbone:wan_va"})
    hardware = HardwareReq(
        min_capability=(12, 0),
        requires=frozenset(
            {
                "cuda",
                "cublas",
                "sm120_kernels",
                "sm120_stage2_kernels",
                "sm120_stage3_kernels",
                "sm120_qk_rope_kernels",
                "sm120_gemm_kernels",
                "sm120_ring_concat_kernels",
                "sm120_qkv_parallel_kernels",
            }
        ),
    )

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        device = getattr(deployment, "device", None)
        if device is None:
            return PassResult(
                self.name,
                False,
                Tier.BITEXACT,
                "unprobed target: A7 is certified for SM120 exactly",
            )
        if device.capability != (12, 0):
            return PassResult(
                self.name,
                False,
                Tier.BITEXACT,
                f"A7 is certified for SM120 exactly; device is "
                f"sm_{device.capability[0]}{device.capability[1]}",
            )
        return PassResult(
            self.name,
            True,
            Tier.BITEXACT,
            "A1-A6 and the independent parallel-QKV ABI are present; Q/K/V keep "
            "their original no-split-K tactics and BF16 bias epilogues on three "
            "private streams joined to the caller stream",
            params={
                "depends_on": "sm120_wan_ring_concat",
                "certified_rows": [64, 480],
                "hidden_dim": 3072,
                "module_sites": 30,
                "split_k": 1,
                "library_feature": "sm120_qkv_parallel_kernels",
            },
            expected_win=(
                "RTX 5090 neutral-prewarmed 42-cycle A-B-B-A: 389.33 -> 383.45 ms "
                "(1.0153x); late cycles 423.91 -> 416.66 ms (1.0174x); "
                "168/168 actions bitwise equal"
            ),
        )


__all__ = ["SM120WanParallelQKV"]

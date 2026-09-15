"""P009-A1: SM120 Wan gated-residual fusion.

Two regions in every Wan block have the same eager chain::

    (hidden.float() + update * gate_fp32).type_as(hidden)

The SM120 kernel replaces its separate multiply, add, and cast launches while retaining separate
RN FP32 multiply/add instructions and the final BF16 RNE conversion.  It is therefore BITEXACT,
not merely close; the release gate compared 50,160 kernel calls across two 42-cycle candidate arms
and every action word matched both baseline arms.
"""

from __future__ import annotations

from instinctflash.adapters.base import AdapterSpec
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import HardwareReq
from instinctflash.planners.planner import PassResult, Tier


class SM120GatedResidual:
    name = "sm120_gated_residual"
    requires_capabilities = frozenset({"backbone:wan_va"})
    hardware = HardwareReq(
        min_capability=(12, 0), requires=frozenset({"cuda", "sm120_kernels"})
    )

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        device = getattr(deployment, "device", None)
        if device is None:
            # contract.py doctrine: a feature the target cannot be shown to have "must decline
            # rather than be left undecided". Staying applies=True here rode the planner's
            # APPLICABILITY UNCHECKED annotation into install_plan, which then crashed in the
            # kernel constructor on every non-5090 machine.
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
            "Wan declares two FP32 gated-residual chains per block and the SM120 native library "
            "is present; separate FMUL/FADD plus final BF16 RNE reproduce eager bits",
            params={
                "regions_per_block": 2,
                "blocks": 30,
                "certified_rows": [64, 480],
                "hidden_dim": 3072,
                "library_feature": "sm120_kernels",
            },
            expected_win=(
                "RTX 5090 P003+P007 Fast operating point, 42-cycle ABBA: mean "
                "354.81 -> 345.05 ms (1.0283x); saturated mean 399.90 -> 390.81 ms "
                "(1.0232x). Both baseline and candidate arm spreads <0.18%."
            ),
        )


__all__ = ["SM120GatedResidual"]

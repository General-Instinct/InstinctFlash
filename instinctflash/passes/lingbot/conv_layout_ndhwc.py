"""P007 planner integration for per-device Wan VAE Conv3D layout autotuning.

The implementation and benchmark site live in ``backends.conv.apply`` and import torch.  This
module is the declaration-only half: it stays torch-free so checkpoint planning works on a laptop,
then the LingBot runtime imports the implementation lazily only when a NUMERIC plan applies it.
"""

from __future__ import annotations

from instinctflash.adapters.base import AdapterSpec
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import HardwareReq
from instinctflash.planners.planner import PassResult, Tier


class ConvLayoutAutotune:
    """Select NCDHW or NDHWC by measurement on the serving device at first model load."""

    name = "conv_layout_ndhwc"
    requires_capabilities = frozenset({"backbone:wan_va"})
    hardware = HardwareReq(requires=("cuda", "cudnn"))

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        if deployment.want_pixels:
            return PassResult(
                self.name, False, Tier.NUMERIC,
                "caller requested predicted pixels; P007's paired certificate covers the "
                "observation encoder/action path, not a converted VAE decoder",
            )

        device = getattr(deployment, "device", None)
        measured = "this device" if device is not None else "the serving device at first load"
        return PassResult(
            self.name, True, Tier.NUMERIC,
            f"two Wan VAE Conv3D subgraphs can choose NCDHW or NDHWC by measurement on {measured}; "
            "NDHWC changes convolution accumulation order, so the swap is NUMERIC",
            params={
                "autotune_site": "va_conv_layout",
                "candidates": ["stock", "ndhwc"],
                "certified_on": "sm90 H100 / 555 paired episodes",
                "target_device_certificate": "required before default-on",
            },
            expected_win=(
                "Measured ON H100: 1.405x complete cycle. RTX 5090 operator screen before "
                "integration: NCDHW 2.780 ms vs NDHWC 1.206 ms (2.30x); end-to-end must be "
                "measured and SM120 quality evidence is not inherited from H100."
            ),
        )


__all__ = ["ConvLayoutAutotune"]

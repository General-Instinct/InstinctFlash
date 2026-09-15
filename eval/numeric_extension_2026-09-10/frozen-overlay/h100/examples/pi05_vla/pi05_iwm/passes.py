"""Planner pass for the explicitly declared pi0.5 TF32 operating points on H100 and Thor."""
from __future__ import annotations

from instinctflash.adapters.base import AdapterSpec
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import HardwareReq
from instinctflash.planners.planner import PassResult, Tier

CAPABILITY = "declares:pi05_tf32_numeric"
PASS_NAME = "pi05_tf32_numeric"

_EVIDENCE = {
    "lerobot/pi05_base": {
        "max_abs_action_delta": 0.002482,
        "evidence": "examples/pi05_vla/tf32_static_h100_results.json",
        "reason": (
            "four real-weight A/B/A prompt cases were deterministic, static replay equaled "
            "TF32 eager exactly, and max |delta action vs FP32| was 2.482e-3"
        ),
        "expected_win": (
            "real H100/lerobot-pi05_base: full 50-action chunk 261.37 -> 66.48 ms "
            "(3.93x); TF32 prefill 36.29 ms"
        ),
    },
    "lerobot/pi05_libero_finetuned_v044": {
        "max_abs_action_delta": 0.0008987784385681152,
        "evidence": "examples/pi05_vla/tf32_v044_static_h100_results.json",
        "reason": (
            "paired independent-process v044 inference on the same schema-valid LIBERO-like "
            "observation/noise was deterministic in each arm; max |delta postprocessed action "
            "vs FP32| was 8.988e-4, and a second image/state/prompt replayed at the same static "
            "three-view/200-token extent"
        ),
        "expected_win": (
            "real H100/lerobot-pi05_libero_finetuned_v044: FP32-highest static-KV -> "
            "TF32 static-KV full 50-action chunk 185.960 -> 72.407 ms (2.568x)"
        ),
    },
}


class Pi05TF32Numeric:
    """The measured TF32 + static-KV execution contract, never a hidden speed switch."""

    name = PASS_NAME
    requires_capabilities = frozenset({"backbone:pi05", CAPABILITY})
    hardware = HardwareReq(
        min_capability=(9, 0), requires=frozenset({"cuda", "cuda_graphs", "cublas"})
    )

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        # Evidence is keyed by the UPSTREAM WEIGHTS the pointer package selects, carried in
        # spec.notes['base_weights'] by Pi05Adapter.spec_for_checkpoint. It must not key on
        # spec.model_id: the facade's plan-header override rewrites model_id to the declared
        # package id (e.g. 'instinctflash/pi05-base-tf32-h100'), which correctly labels the plan
        # but is not the identity the numerics were measured on -- keying on it made both pointer
        # packages plan without the pass and fail at build time.
        weights = spec.notes.get("base_weights") or spec.model_id
        target = spec.notes.get("tf32_hardware", "sm90")
        evidence = _EVIDENCE.get(weights)
        if evidence is None:
            return PassResult(
                self.name, False, Tier.NUMERIC,
                f"measured only for {sorted(_EVIDENCE)}, not {weights!r}",
            )
        device = deployment.device
        capability = getattr(device, "capability", None) if device is not None else None
        required_capability = {"sm90": (9, 0), "sm110": (11, 0)}.get(target)
        if required_capability is None or (capability is not None and capability != required_capability):
            return PassResult(
                self.name, False, Tier.NUMERIC,
                f"TF32 declaration targets {target}; probed capability is {capability}",
            )
        if target == "sm110":
            # Execution support is separate from numerical/task qualification. Never inherit
            # H100's FP32-reference margin onto Thor or onto a native-BF16 reference.
            evidence = {
                "max_abs_action_delta": None,
                "evidence": "eval/numeric_extension_2026-09-10/README.md",
                "reason": "Thor execution support; target-device action/task quality unqualified; exact graph replay self-check remains required",
                "expected_win": "Measure the complete configuration against native upstream on Thor; no H100 speed or margin is transferred",
            }
        return PassResult(
            name=self.name,
            applies=True,
            tier=Tier.NUMERIC,
            reason=(
                f"checkpoint explicitly declares the {target} TF32 operating point: FP32 GEMMs use "
                "TF32 tensor cores and the action loop uses replay-safe static-KV CUDA Graphs; "
                f"measured weights are {weights}; {evidence['reason']}"
            ),
            params={
                # The optimizer only honors this when the exact token is present in the checkpoint
                # capabilities. This is why an external pass cannot spend a tier by assertion.
                "required_by_checkpoint": CAPABILITY,
                "precision": "tf32",
                "static_kv_graph": True,
                "max_abs_action_delta": evidence["max_abs_action_delta"],
                "hardware": target,
                **({"qualification": "unqualified"} if target == "sm110" else {}),
                "evidence": evidence["evidence"],
            },
            expected_win=evidence["expected_win"],
        )


def default_passes() -> list:
    return [Pi05TF32Numeric()]


__all__ = ["CAPABILITY", "PASS_NAME", "Pi05TF32Numeric", "default_passes"]

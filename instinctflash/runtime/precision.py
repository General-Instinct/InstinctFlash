"""Explicit arithmetic policy, independent of placement and evidence tiers."""
from dataclasses import replace


def resolve_precision(precision: str, tier_ceiling: str | None,
                      placement: str = "auto") -> str | None:
    if placement not in {"auto", "in_process", "worker", "engine"}:
        raise ValueError(f"Unknown placement {placement!r}")
    if precision not in {"native", "fp8"}:
        raise ValueError(f"precision must be 'native' or 'fp8'; got {precision!r}")
    if precision == "native":
        if placement == "engine":
            raise ValueError("The current engine uses FP8; select precision='fp8' explicitly.")
        return tier_ceiling or "bitexact"
    if placement == "worker":
        raise ValueError("precision='fp8' currently requires in-process execution; use placement='auto'.")
    if tier_ceiling == "bitexact":
        raise ValueError("precision='fp8' changes arithmetic and conflicts with tier_ceiling='bitexact'.")
    return tier_ceiling or "numeric"


def constrain_precision(plan, precision: str) -> None:
    """Keep a native plan truthful without turning a policy into a caller exclusion."""
    if precision != "native":
        return
    for i, result in enumerate(getattr(plan, "results", ())):
        if (result.name == "engine_offload" and not getattr(result, "excluded", False)
                and not result.reason.startswith("precision='native':")):
            plan.results[i] = replace(
                result, applies=False,
                reason="precision='native': FP8 engine is not authorized; " + result.reason)


def require_fp8_plan(plan, backbone: str):
    """Declaration/device gate shared by preflight and execution; no kernel probe."""
    from instinctflash.runtime.engine_backend import ENGINE_BACKBONES
    if backbone == "dreamzero":
        require_dreamzero_fp8_environment()
        require_dreamzero_fp8_schedule(getattr(plan, "resolved_step_cache", None))
    result = next((r for r in getattr(plan, "results", ()) if r.name == "engine_offload"), None)
    if result is not None and getattr(result, "excluded", False):
        raise RuntimeError("precision='fp8' conflicts with excluded engine_offload.")
    if backbone not in ENGINE_BACKBONES:
        raise RuntimeError(f"No FP8 engine frontend for backbone {backbone!r}.")
    if result is None or not result.applies or result.params.get("backend") != "engine":
        detail = getattr(result, "reason", "no eligible engine pass")
        raise RuntimeError(f"precision='fp8' refused: planner declined engine_offload: {detail}")
    return result


def require_dreamzero_fp8_environment():
    """The native loader honors LOAD_TRT_ENGINE independently of its enable flag."""
    import os
    if (os.environ.get("ENABLE_TENSORRT", "false").lower() == "true"
            or os.environ.get("LOAD_TRT_ENGINE") is not None):
        raise ValueError("DreamZero FP8 requires the native PyTorch path; unset "
                         "LOAD_TRT_ENGINE and disable ENABLE_TENSORRT")


def require_dreamzero_fp8_schedule(step_cache):
    """Keep metadata preflight and the native builder on the same supported mask."""
    if step_cache is not None and step_cache.fixed_steps != 8:
        raise ValueError("DreamZero FP8 requires the native fixed 8-of-16 mask or its dynamic profile")


def install_requested_fp8(model, plan, family):
    """Native adapters never import or initialize the optional quantization implementation."""
    if any(r.name == "engine_offload" and r.applies and
           r.params.get("executor") == "sm89_torch_fp8"
           for r in getattr(plan, "results", ())):
        from .sm89_fp8 import maybe_install_sm89_fp8
        return maybe_install_sm89_fp8(model, plan, family)
    if not any(r.name == "engine_offload" and r.applies and
               r.params.get("executor") == "h100_torch_fp8"
               for r in getattr(plan, "results", ())):
        return
    from .h100_fp8 import maybe_install_h100_fp8
    return maybe_install_h100_fp8(model, plan, family)


def require_transform_permission(plan, minimum, name):
    """Environment overrides cannot bypass the caller's arithmetic permission."""
    from instinctflash.planners.planner import Tier
    allowed = getattr(plan, "tier_ceiling", Tier.BITEXACT)
    if allowed < minimum:
        raise ValueError(f"{name} changes computation and requires explicit "
                         f"tier_ceiling='{minimum.name.lower()}'; selected {allowed.name}")

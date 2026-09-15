"""Resolve approximate step reuse before device placement or lazy model loading."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os


@dataclass(frozen=True)
class ResolvedStepCache:
    """One immutable selection; actual dynamic compute counts require execution telemetry."""

    dynamic: bool
    fixed_steps: int
    profile: str | None
    source: str

    def __post_init__(self):
        if type(self.dynamic) is not bool:
            raise ValueError("Resolved step-cache dynamic must be a bool")
        if type(self.fixed_steps) is not int or self.fixed_steps not in (5, 6, 7, 8):
            raise ValueError("Resolved DreamZero fixed_steps must be 5, 6, 7, or 8")
        expected = "dreamzero_velocity_v1" if self.dynamic else None
        if self.profile != expected:
            raise ValueError(f"Resolved step-cache profile must be {expected!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def _flag(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def resolve_step_cache(checkpoint, *, step_cache: str | None = None,
                       tier_ceiling: str | None = None, placement: str = "auto",
                       family: str | None = None) -> ResolvedStepCache | None:
    """Sample legacy DreamZero environment options once, without modifying them.

    Explicit selections override the environment. ``checkpoint`` restores the
    declaration, which may itself require BEHAVIORAL permission. ``None`` retains
    legacy environment compatibility, including an explicitly false cache flag.
    """
    if step_cache not in (None, "dynamic", "checkpoint"):
        raise ValueError("step_cache must be None, 'dynamic', or 'checkpoint'")
    backbone = family or checkpoint.execution.backbone
    if backbone != "dreamzero":
        if step_cache == "dynamic":
            raise ValueError(f"step_cache='dynamic' is not supported for backbone {backbone!r}")
        return None

    extra = dict(getattr(checkpoint.execution, "extra", None) or {})
    dynamic = _flag(extra.get("dynamic_cache_schedule", False),
                    "execution.dynamic_cache_schedule")
    fixed_steps = 8
    source = "checkpoint"
    if step_cache == "dynamic":
        dynamic, source = True, "runtime.step_cache=dynamic"
    elif step_cache == "checkpoint":
        source = "runtime.step_cache=checkpoint"
    else:
        selected_env = []
        if "DYNAMIC_CACHE_SCHEDULE" in os.environ:
            dynamic = _flag(os.environ["DYNAMIC_CACHE_SCHEDULE"], "DYNAMIC_CACHE_SCHEDULE")
            selected_env.append("DYNAMIC_CACHE_SCHEDULE")
        if "NUM_DIT_STEPS" in os.environ:
            try:
                fixed_steps = int(os.environ["NUM_DIT_STEPS"])
            except ValueError as error:
                raise ValueError("DreamZero NUM_DIT_STEPS must be 5, 6, 7, or 8") from error
            selected_env.append("NUM_DIT_STEPS")
        if selected_env:
            source = "environment:" + ",".join(selected_env)
    if fixed_steps not in (5, 6, 7, 8):
        raise ValueError("DreamZero NUM_DIT_STEPS must be 5, 6, 7, or 8")
    changed = dynamic or fixed_steps != 8
    if changed and tier_ceiling != "behavioral":
        raise ValueError("DreamZero altered step schedule requires explicit "
                         "tier_ceiling='behavioral'; FP8 permission alone does not authorize it")
    if changed and placement == "worker":
        raise ValueError("DreamZero altered step schedule is not supported by worker placement; "
                         "use placement='in_process'")
    return ResolvedStepCache(dynamic, fixed_steps,
                             "dreamzero_velocity_v1" if dynamic else None, source)


def annotate_step_cache_plan(plan, resolved: ResolvedStepCache | None) -> None:
    """Record selected schedule semantics before loading; never invent dynamic call counts."""
    if resolved is None:
        return
    if not isinstance(resolved, ResolvedStepCache):
        raise TypeError("step_cache must be a ResolvedStepCache")
    from instinctflash.planners.planner import PassResult, Tier
    from instinctflash.runtime.precision import require_transform_permission

    changed = resolved.dynamic or resolved.fixed_steps != 8
    if changed:
        require_transform_permission(plan, Tier.BEHAVIORAL, "DreamZero altered step schedule")
    previous = next((result for result in plan.results
                     if result.name == "dreamzero_schedule"), None)
    if previous is not None and previous.excluded:
        if changed:
            raise ValueError("Selected DreamZero step schedule conflicts with excluded dreamzero_schedule")
        plan.resolved_step_cache = resolved
        return
    schedule = {"grid_steps": 16,
                "computed_steps": None if resolved.dynamic else resolved.fixed_steps,
                "fixed_mask_steps": resolved.fixed_steps,
                "dynamic_cache_schedule": resolved.dynamic,
                "profile": resolved.profile, "source": resolved.source}
    if resolved.dynamic:
        from instinctflash.runtime.step_cache import StepCacheConfig
        schedule["config"] = StepCacheConfig(profile=resolved.profile).to_dict()
    result = PassResult(
        "dreamzero_schedule", True, Tier.BEHAVIORAL if changed else Tier.BITEXACT,
        (f"Dynamic {resolved.profile} video/action prediction reuse; "
         "realized compute count unknown; SCREEN evidence"
         if resolved.dynamic else f"Selected fixed {resolved.fixed_steps}-of-16 DreamZero mask"),
        params={"execution_schedule": schedule, "step_cache": resolved.to_dict()})
    if previous is None:
        plan.results.append(result)
    else:
        plan.results[plan.results.index(previous)] = result
    plan.resolved_step_cache = resolved

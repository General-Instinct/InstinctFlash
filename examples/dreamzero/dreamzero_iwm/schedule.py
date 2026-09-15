"""Instance-local legacy schedule reads during native head construction.

No process environment or native class is mutated. The original constructor's
code, closure and all other globals are retained; only its two legacy getenv
lookups see the already resolved Runtime selection.
"""
from __future__ import annotations

import hashlib
import inspect
import textwrap
from types import FunctionType


_INIT_HASH = "a8c984241e93c51361e181abb69f2f8f922f265250d83b0cf66ffc5000a2d2e3"


class _ScheduleEnvironment:
    def __init__(self, original, dynamic, fixed_steps):
        self._original = original
        self._values = {"DYNAMIC_CACHE_SCHEDULE": "true" if dynamic else "false",
                        "NUM_DIT_STEPS": str(fixed_steps)}

    def getenv(self, name, default=None):
        if name in self._values:
            return self._values[name]
        return self._original.getenv(name, default)

    def __getattr__(self, name):
        return getattr(self._original, name)


def _construct(cls, config, *, dynamic, fixed_steps):
    initializer = cls.__init__
    if cls.__new__ is not object.__new__ or not isinstance(initializer, FunctionType):
        raise ValueError("DreamZero head construction changed; requalify instance-local schedule")
    globals_for_instance = dict(initializer.__globals__)
    globals_for_instance["os"] = _ScheduleEnvironment(
        globals_for_instance["os"], dynamic, fixed_steps)
    local_init = FunctionType(initializer.__code__, globals_for_instance,
                             initializer.__name__, initializer.__defaults__, initializer.__closure__)
    local_init.__kwdefaults__ = initializer.__kwdefaults__
    instance = cls.__new__(cls)
    local_init(instance, config)
    return instance


def build_head(config, ifl_dynamic_cache_schedule, ifl_fixed_dit_steps):
    """Hydra factory returning the original native head class and state layout."""
    from instinctflash.runtime.step_cache_policy import ResolvedStepCache
    from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import WANPolicyHead

    selection = ResolvedStepCache(
        ifl_dynamic_cache_schedule, ifl_fixed_dit_steps,
        "dreamzero_velocity_v1" if ifl_dynamic_cache_schedule else None, "owned native constructor")
    source = textwrap.dedent(inspect.getsource(WANPolicyHead.__init__)).strip()
    if hashlib.sha256(source.encode()).hexdigest() != _INIT_HASH:
        raise ValueError("DreamZero constructor changed; re-audit instance-local schedule reads")
    return _construct(WANPolicyHead, config, dynamic=selection.dynamic,
                      fixed_steps=selection.fixed_steps)

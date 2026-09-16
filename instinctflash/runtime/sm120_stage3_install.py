"""Plan-scoped runtime installation for P009-A3 Wan norm1 fusion."""

from __future__ import annotations

import threading


def install_sm120_wan_stage3(server_module, va_server_cls) -> list[str]:
    """Arm P009-A3 for the next server built by this thread, after A1/A2."""
    from instinctflash.backends.sm120_wan_stage3 import (
        SM120WanStage3Kernels,
        install_wan_stage3,
    )

    kernels = SM120WanStage3Kernels()
    state = getattr(va_server_cls, "_ifl_sm120_wan_stage3_state", None)
    if state is None:
        state = threading.local()
        original_init = va_server_cls.__init__

        def _init_with_plan_scoped_stage3(self, *args, **kwargs):
            pending = getattr(state, "kernels", None)
            if pending is not None:
                del state.kernels
            original_init(self, *args, **kwargs)
            if pending is not None:
                install_wan_stage3(self.transformer, pending)

        va_server_cls.__init__ = _init_with_plan_scoped_stage3
        va_server_cls._ifl_sm120_wan_stage3_state = state
        va_server_cls._ifl_sm120_wan_stage3_installed = True

    state.kernels = kernels
    va_server_cls._ifl_sm120_wan_stage3_kernels = kernels
    return ["sm120_wan_stage3"]


__all__ = ["install_sm120_wan_stage3"]

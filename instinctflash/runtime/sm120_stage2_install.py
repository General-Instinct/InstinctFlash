"""Plan-scoped runtime installation for P009-A2 Wan block fusion."""

from __future__ import annotations

import threading


def install_sm120_wan_stage2(server_module, va_server_cls) -> list[str]:
    """Arm P009-A2 for the next server built by this thread, after P009-A1."""
    from instinctflash.backends.sm120_wan_stage2 import (
        SM120WanStage2Kernels,
        install_wan_stage2,
    )

    kernels = SM120WanStage2Kernels()
    state = getattr(va_server_cls, "_ifl_sm120_wan_stage2_state", None)
    if state is None:
        state = threading.local()
        original_init = va_server_cls.__init__

        def _init_with_plan_scoped_stage2(self, *args, **kwargs):
            pending = getattr(state, "kernels", None)
            if pending is not None:
                # Consume before construction so an upstream/A1 exception cannot leak this
                # P009-A2 activation into a later excluded Runtime.
                del state.kernels
            original_init(self, *args, **kwargs)
            if pending is not None:
                install_wan_stage2(self.transformer, pending)

        va_server_cls.__init__ = _init_with_plan_scoped_stage2
        va_server_cls._ifl_sm120_wan_stage2_state = state
        va_server_cls._ifl_sm120_wan_stage2_installed = True

    state.kernels = kernels
    va_server_cls._ifl_sm120_wan_stage2_kernels = kernels
    return ["sm120_wan_stage2"]


__all__ = ["install_sm120_wan_stage2"]

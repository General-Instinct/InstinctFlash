"""Runtime installation for the optional SM120 Wan kernel pack."""

from __future__ import annotations

import threading


def install_sm120_gated_residual(server_module, va_server_cls) -> list[str]:
    """Arm P009-A1 for exactly the next server built by this thread's plan."""
    # Resolve and load before mutating the class. A missing/incompatible library therefore leaves
    # the upstream server untouched and fails the plan honestly.
    from instinctflash.backends.sm120_residual import (
        SM120GatedResidualKernel,
        install_wan_blocks,
    )

    kernel = SM120GatedResidualKernel()
    state = getattr(va_server_cls, "_ifl_sm120_gated_residual_state", None)
    if state is None:
        state = threading.local()
        original_init = va_server_cls.__init__

        def _init_with_plan_scoped_sm120_residual(self, *args, **kwargs):
            pending = getattr(state, "kernel", None)
            if pending is not None:
                # Consume before construction: an upstream exception cannot leak this plan's
                # kernel into the next, possibly excluded, Runtime in the same interpreter.
                del state.kernel
            original_init(self, *args, **kwargs)
            if pending is not None:
                install_wan_blocks(self.transformer, pending)

        va_server_cls.__init__ = _init_with_plan_scoped_sm120_residual
        va_server_cls._ifl_sm120_gated_residual_state = state
        va_server_cls._ifl_sm120_gated_residual_installed = True

    # The class wrapper persists because plan installation precedes model construction, but its
    # activation is one-shot and thread-local. An excluded later plan leaves no pending kernel.
    state.kernel = kernel
    va_server_cls._ifl_sm120_gated_residual_kernel = kernel
    return ["sm120_gated_residual"]


__all__ = ["install_sm120_gated_residual"]

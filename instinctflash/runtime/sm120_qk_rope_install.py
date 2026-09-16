"""Plan-scoped runtime installation for P009-A4 Q/K RMSNorm+RoPE fusion."""
from __future__ import annotations
import threading
def install_sm120_wan_qk_rope(server_module,va_server_cls)->list[str]:
    from instinctflash.backends.sm120_wan_qk_rope import SM120WanQKRoPEKernels,install_wan_qk_rope
    kernels=SM120WanQKRoPEKernels(); state=getattr(va_server_cls,"_ifl_sm120_wan_qk_rope_state",None)
    if state is None:
        state=threading.local(); original_init=va_server_cls.__init__
        def _init(self,*args,**kwargs):
            pending=getattr(state,"kernels",None)
            if pending is not None: del state.kernels
            original_init(self,*args,**kwargs)
            if pending is not None: install_wan_qk_rope(self.transformer,pending)
        va_server_cls.__init__=_init; va_server_cls._ifl_sm120_wan_qk_rope_state=state; va_server_cls._ifl_sm120_wan_qk_rope_installed=True
    state.kernels=kernels; va_server_cls._ifl_sm120_wan_qk_rope_kernels=kernels
    return ["sm120_wan_qk_rope"]
__all__=["install_sm120_wan_qk_rope"]

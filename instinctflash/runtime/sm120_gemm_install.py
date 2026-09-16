"""Plan-scoped runtime installation for P009-A5 pinned Wan GEMM tactics."""
from __future__ import annotations
import threading
def install_sm120_wan_gemm(server_module,va_server_cls)->list[str]:
    from instinctflash.backends.sm120_wan_gemm import SM120WanGEMMKernels,install_wan_gemm
    kernels=SM120WanGEMMKernels();state=getattr(va_server_cls,"_ifl_sm120_wan_gemm_state",None)
    if state is None:
        state=threading.local();original_init=va_server_cls.__init__
        def _init(self,*args,**kwargs):
            pending=getattr(state,"kernels",None)
            if pending is not None:del state.kernels
            original_init(self,*args,**kwargs)
            if pending is not None:install_wan_gemm(self.transformer,pending)
        va_server_cls.__init__=_init;va_server_cls._ifl_sm120_wan_gemm_state=state;va_server_cls._ifl_sm120_wan_gemm_installed=True
    state.kernels=kernels;va_server_cls._ifl_sm120_wan_gemm_kernels=kernels
    return ["sm120_wan_gemm"]
__all__=["install_sm120_wan_gemm"]

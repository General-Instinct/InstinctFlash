"""Experiment-only reuse of the unchanged Qwen3.6 BF16 CUDA kernel."""
import ctypes
from pathlib import Path
import torch


def install(service, library):
    lib = ctypes.CDLL(str(Path(library).resolve()))
    lib.probe_swiglu.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_void_p]
    lib.probe_swiglu.restype = None

    @torch.library.custom_op('ifl_cosmos_probe::swiglu', mutates_args=())
    def fused(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if not (gate.is_cuda and up.device == gate.device and gate.dtype == up.dtype == torch.bfloat16
                and gate.shape == up.shape and gate.is_contiguous() and up.is_contiguous()
                and 0 < gate.numel() < 2**31):
            raise ValueError('Unsupported engine SwiGLU inputs')
        out = torch.empty_like(gate)
        with torch.cuda.device(gate.device):
            lib.probe_swiglu(gate.data_ptr(), up.data_ptr(), out.data_ptr(), gate.numel(),
                            torch.cuda.current_stream(gate.device).cuda_stream)
        return out

    @fused.register_fake
    def fake(gate, up):
        return torch.empty_like(gate)

    names = []
    for name, module in service.model.net.named_modules():
        if type(module).__name__ != 'Qwen3VLTextMLP':
            continue
        if module.config.hidden_act != 'silu':
            raise ValueError('Unexpected activation')
        def forward(x, mod=module):
            return mod.down_proj(fused(mod.gate_proj(x), mod.up_proj(x)))
        module.forward = forward
        names.append(name)
    if not names:
        raise RuntimeError('No compatible Cosmos MLPs found')
    return {'kernel': 'silu_mul_qwen36_bf16', 'modules': names, 'library': str(library)}

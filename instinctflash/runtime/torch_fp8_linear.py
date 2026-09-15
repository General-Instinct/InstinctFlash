"""Explicit inference-only FP8 projection for audited Thor model call sites.

This is not an automatic model converter. Callers must preserve native processors,
attention, scheduling and any projections whose weights are read directly.
"""
import torch
from torch import nn


class ThorFP8Linear(nn.Module):
    recipe = "e4m3fn_weights_dynamic_per_tensor_activations_fp32_accum_bf16_output"

    def __init__(self, linear: nn.Linear):
        super().__init__()
        if type(linear) is not nn.Linear:
            raise TypeError("FP8 packing requires a plain nn.Linear with no custom forward")
        w = linear.weight.detach()
        if w.device.type != "cuda" or torch.cuda.get_device_capability(w.device) not in ((9, 0), (11, 0)):
            raise ValueError("FP8 projections require an SM90 or SM110 CUDA device")
        if w.dtype != torch.bfloat16:
            raise ValueError("This FP8 recipe requires native BF16 projection weights")
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError("FP8 projection input/output dimensions must be multiples of 16")
        if not torch.isfinite(w).all():
            raise ValueError("Cannot quantize non-finite projection weights")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        scale = w.float().abs().amax().clamp_min(1e-12).reshape(1) / 448.0
        packed = (w.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        self.register_buffer("weight_fp8", packed.contiguous())
        self.register_buffer("weight_scale", scale)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        if bias is not None and (bias.dtype != torch.bfloat16 or not torch.isfinite(bias).all()):
            raise ValueError("This FP8 recipe requires finite BF16 bias")
        self.register_buffer("bias", bias)

    @torch.no_grad()
    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("FP8 projection input has incompatible feature dimension")
        if x.dtype != torch.bfloat16 or x.device != self.weight_fp8.device:
            raise ValueError("FP8 projection input must be BF16 on the packed weight device")
        if self.weight_fp8.dtype != torch.float8_e4m3fn or self.weight_scale.dtype != torch.float32:
            raise RuntimeError("FP8 packed buffers were cast after construction; pack after model.to()")
        shape = (*x.shape[:-1], self.out_features)
        # Cosmos may call both towers while one token partition is empty.
        if x.numel() == 0:
            return x.new_empty(shape)
        from .fp8_pack import dynamic_pack_bf16_e4m3

        rows = x.reshape(-1, self.in_features).contiguous()
        # abs/max select BF16 values exactly; widen only the scalar before
        # FP32 clamping/division, avoiding a full FP32 activation allocation.
        # Preserve amax's NaN propagation and the reference scale arithmetic.
        packed, scale = dynamic_pack_bf16_e4m3(rows)
        out = torch._scaled_mm(
            packed, self.weight_fp8.t(), scale_a=scale, scale_b=self.weight_scale,
            bias=self.bias, out_dtype=torch.bfloat16, use_fast_accum=False,
        )
        return out.reshape(shape)

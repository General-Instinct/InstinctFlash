"""Explicit inference-only FP8 projections for audited model call sites.

This is not an automatic model converter. Callers must preserve native processors,
attention, scheduling and any projections whose weights are read directly.
"""
import torch
from torch import nn


class ThorFP8Linear(nn.Module):
    recipe = "e4m3fn_weights_dynamic_per_tensor_activations_fp32_accum_bf16_output"
    supported_capabilities = ((9, 0), (11, 0))
    allow_cpu_source = False

    def __init__(self, linear: nn.Linear, *, device=None, storage_device=None):
        super().__init__()
        if type(linear) is not nn.Linear:
            raise TypeError("FP8 packing requires a plain nn.Linear with no custom forward")
        w = linear.weight.detach()
        target = torch.device(device) if device is not None else w.device
        if (target.type != "cuda"
                or torch.cuda.get_device_capability(target) not in self.supported_capabilities):
            raise ValueError(f"FP8 projections require CUDA capability in {self.supported_capabilities}")
        storage = torch.device(storage_device) if storage_device is not None else target
        if storage != target and not (self.allow_cpu_source and storage.type == "cpu"):
            raise ValueError("Separate CPU packed storage is supported only by a desktop projection recipe")
        if w.device != target and not (self.allow_cpu_source and w.device.type == "cpu"):
            # CUDA without an index and cuda:current identify the same target.
            target_index = target.index if target.index is not None else torch.cuda.current_device()
            if w.device != torch.device("cuda", target_index):
                raise ValueError("Pack FP8 weights on their target device; CPU packing requires a desktop projection recipe")
        if w.dtype != torch.bfloat16:
            raise ValueError("This FP8 recipe requires native BF16 projection weights")
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError("FP8 projection input/output dimensions must be multiples of 16")
        if not torch.isfinite(w).all():
            raise ValueError("Cannot quantize non-finite projection weights")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        maximum = w.float().abs().amax().clamp_min(1e-12).reshape(1)
        if self.allow_cpu_source and w.device.type == "cpu":
            # CUDA division by a host scalar multiplies by its FP32 reciprocal.
            # CPU true division can differ by one ULP. Keep CPU preplacement
            # on the same numerical recipe as the original CUDA weight packing.
            reciprocal = torch.tensor(1.0 / 448.0, dtype=torch.float32, device=w.device)
            scale = maximum * reciprocal
        else:
            scale = maximum / 448.0
        packed = (w.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        self.register_buffer("weight_fp8", packed.to(storage).contiguous())
        self.register_buffer("weight_scale", scale.to(storage))
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        if bias is not None and (bias.dtype != torch.bfloat16 or not torch.isfinite(bias).all()):
            raise ValueError("This FP8 recipe requires finite BF16 bias")
        self.register_buffer("bias", bias.to(storage) if bias is not None else None)

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


class SM89FP8Linear(ThorFP8Linear):
    """SM89 E4M3 projection; CPU packing avoids a full BF16 CUDA loading peak.

    Call after the native BF16 cast. A later model-wide dtype cast would corrupt
    the packed buffers and is detected by forward(), just as for the Thor path.
    CPU packing preserves this FP8 recipe, not the original BF16 arithmetic.
    """

    supported_capabilities = ((8, 9),)
    allow_cpu_source = True

    def __init__(self, linear: nn.Linear, *, device=None, storage_device=None):
        if device is None and linear.weight.device.type == "cpu":
            device = "cuda"
        super().__init__(linear, device=device, storage_device=storage_device)

    @property
    def weight(self):
        # Audited attention code may inspect dtype/device, but a direct-weight
        # GEMM must fail instead of silently bypassing the quantized projection.
        return torch.empty(0, device=self.weight_fp8.device, dtype=torch.bfloat16)

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(self, x):
        return super().forward(x)


class SM120FP8Linear(SM89FP8Linear):
    """Explicit SM120 E4M3 arithmetic; device qualification remains separate."""

    supported_capabilities = ((12, 0),)

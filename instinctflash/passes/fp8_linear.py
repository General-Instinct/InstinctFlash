"""FP8 W+A linear serving, torch-level, from the serving engine's verified recipe.

The recipe (anchors in serving/flash_rt/, all source-verified):
  * weights: per-tensor symmetric FP8 E4M3, scale = max(amax/448, 1e-12) computed in FP32
    (frontends/torch/pi05_rtx.py:337-343);
  * gate+up projections concatenated BEFORE quantization so they share one GEMM and one scale
    (:590-603);
  * RMSNorm (1+scale) folded into the adjacent linear IN FP32, with a bit-truncate
    (fp32 -> bf16 -> fp32) applied to the weight before scale computation — bf16 rounds values
    near -1.0 to exactly -1.0, collapsing (1+scale) to 0 and zeroing whole channels; upstream
    measured that mistake at -10% LIBERO (:182-224);
  * activations: STATIC per-tensor scales from percentile-99.9 calibration over a small
    stratified sample set (core/calibration.py:38-64, 174-276), baked before serving so the
    hot path carries no amax reductions;
  * only the large GEMMs quantize (attention qkv/o and FFN in vision/encoder/decoder);
    everything else stays bf16.

Hot path: `torch._scaled_mm(x_fp8, w_fp8.t(), scale_a, scale_b, bias, out_dtype=bf16)` —
tensorwise scales, verified available on this torch (2.11/cu130). All bookkeeping lives in
buffers so the module is CUDA-graph-capturable: quantizing an activation is clamp+cast at a
fixed address, no host math.
"""

from __future__ import annotations

import fnmatch
import re

import torch
from torch import nn

FP8_MAX = 448.0
E4M3 = torch.float8_e4m3fn


def _call_amax(x: torch.Tensor) -> float:
    """True amax of one calibration call. The recipe's percentile-99.9 is a reduction ACROSS
    calls (a robust-max over the sample set, calibration.py:38-64) — NOT an element-level
    clip, which was measured here to cost 4.1% max error on pure gaussians before this fix."""
    return float(x.detach().abs().amax())


class FP8Linear(nn.Module):
    """An nn.Linear served as FP8 W+A through torch._scaled_mm.

    Two phases: constructed in CALIBRATION mode it runs the original bf16 linear while
    recording the input's p99.9 amax; `freeze()` bakes the static activation scale and swaps
    the forward to the fp8 path. The weight is quantized once at construction, in fp32.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach().to(torch.float32)
        scale_w = torch.clamp(w.abs().amax() / FP8_MAX, min=1e-12)
        self.register_buffer("weight_fp8", (w / scale_w).clamp(-FP8_MAX, FP8_MAX).to(E4M3))
        self.register_buffer("scale_w", scale_w.to(torch.float32))
        self.bias = linear.bias  # kept in bf16, fed to _scaled_mm's epilogue
        self.out_features, self.in_features = linear.weight.shape
        # calibration state
        self._orig = linear
        self.dynamic_rows = True
        self._calibrating = True
        self._amax_seen = 0.0
        self.register_buffer("scale_x", torch.tensor(1.0, dtype=torch.float32, device=linear.weight.device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._calibrating:
            self._amax_seen = max(self._amax_seen, _call_amax(x))
            return self._orig(x)
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        bias = self.bias.to(torch.bfloat16) if self.bias is not None else None
        if self.dynamic_rows:
            sa = (x2.abs().amax(dim=1, keepdim=True).float() / FP8_MAX).clamp(min=1e-12)
            xq = (x2 / sa).clamp(-FP8_MAX, FP8_MAX).to(E4M3)
            sb = self.scale_w.expand(1, self.out_features).contiguous() if self.scale_w.dim() == 0 \
                else self.scale_w
            y = torch._scaled_mm(xq, self.weight_fp8.t(), scale_a=sa, scale_b=sb,
                                 bias=bias, out_dtype=torch.bfloat16)
        else:
            xq = (x2.float() / self.scale_x).clamp(-FP8_MAX, FP8_MAX).to(E4M3)
            y = torch._scaled_mm(xq, self.weight_fp8.t(), scale_a=self.scale_x,
                                 scale_b=self.scale_w, bias=bias, out_dtype=torch.bfloat16)
        return y.reshape(*shape[:-1], self.out_features)


    @property
    def weight(self):
        """Upstream dtype-sniffs (e.g. `q_proj.weight.dtype`); zero-size so a real matmul
        against it fails loudly instead of silently bypassing the fp8 path."""
        return torch.empty(0, dtype=torch.bfloat16, device=self.weight_fp8.device)

    def freeze(self) -> None:
        if not self._calibrating:
            return
        amax = self._amax_seen if self._amax_seen > 0 else 1.0
        self.scale_x.fill_(max(amax / FP8_MAX, 1e-12))
        self._calibrating = False
        self._orig = None  # drop the bf16 copy


class FP8GateUp(nn.Module):
    """Gemma-style gate+up served as ONE fp8 GEMM (the recipe's concat-before-quantize)."""

    def __init__(self, gate: nn.Linear, up: nn.Linear):
        super().__init__()
        w = torch.cat([gate.weight, up.weight], dim=0).detach().to(torch.float32)
        scale_w = torch.clamp(w.abs().amax() / FP8_MAX, min=1e-12)
        self.register_buffer("weight_fp8", (w / scale_w).clamp(-FP8_MAX, FP8_MAX).to(E4M3))
        self.register_buffer("scale_w", scale_w.to(torch.float32))
        assert gate.bias is None and up.bias is None, "Gemma MLP is bias-free"
        self.split = gate.weight.shape[0]
        self._gate, self._up = gate, up
        self.dynamic_rows = True
        self._calibrating = True
        self._amax_seen = 0.0
        self.register_buffer("scale_x", torch.tensor(1.0, dtype=torch.float32, device=gate.weight.device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if self._calibrating:
            self._amax_seen = max(self._amax_seen, _call_amax(x))
            return self._gate(x), self._up(x)
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if self.dynamic_rows:
            sa = (x2.abs().amax(dim=1, keepdim=True).float() / FP8_MAX).clamp(min=1e-12)
            xq = (x2 / sa).clamp(-FP8_MAX, FP8_MAX).to(E4M3)
            sb = self.scale_w.expand(1, self.weight_fp8.shape[0]).contiguous() if self.scale_w.dim() == 0 \
                else self.scale_w
            y = torch._scaled_mm(xq, self.weight_fp8.t(), scale_a=sa, scale_b=sb,
                                 out_dtype=torch.bfloat16)
        else:
            xq = (x2.float() / self.scale_x).clamp(-FP8_MAX, FP8_MAX).to(E4M3)
            y = torch._scaled_mm(xq, self.weight_fp8.t(), scale_a=self.scale_x,
                                 scale_b=self.scale_w, out_dtype=torch.bfloat16)
        y = y.reshape(*shape[:-1], -1)
        return y[..., :self.split], y[..., self.split:]


    @property
    def weight(self):
        """Upstream dtype-sniffs (e.g. `q_proj.weight.dtype`); zero-size so a real matmul
        against it fails loudly instead of silently bypassing the fp8 path."""
        return torch.empty(0, dtype=torch.bfloat16, device=self.weight_fp8.device)

    def freeze(self) -> None:
        if not self._calibrating:
            return
        self.scale_x.fill_(max((self._amax_seen or 1.0) / FP8_MAX, 1e-12))
        self._calibrating = False
        self._gate = self._up = None


class _MergedGemmaMLP(nn.Module):
    """Drop-in for Gemma MLP: one fused gate+up fp8 GEMM, activation, fp8 down."""

    def __init__(self, mlp, fused: FP8GateUp, down: FP8Linear):
        super().__init__()
        self.fused, self.down = fused, down
        self.act_fn = mlp.act_fn

    def forward(self, x):
        gate, up = self.fused(x)
        return self.down(self.act_fn(gate) * up)


def fold_rmsnorm_fp32(norm: nn.Module, linears: "list[nn.Linear]") -> None:
    """Fold a Gemma RMSNorm's (1 + weight) into following linears, in FP32.

    The bit-truncate first: the norm weight is stored bf16-representable; truncating
    fp32->bf16->fp32 BEFORE computing (1+w) reproduces exactly what the bf16 forward saw, so
    the fold changes column scaling by the same factor the runtime used — computed in fp32 so
    (1 + w) near zero keeps its low bits instead of collapsing (the -10% LIBERO failure).
    After folding, the norm weight is zeroed: (1 + 0) = identity.
    """
    w = norm.weight.detach()
    w32 = w.to(torch.bfloat16).to(torch.float32)          # bit-truncate, then fp32 math
    gain = (1.0 + w32)                                     # per-channel column gain
    for lin in linears:
        lin.weight.data = (lin.weight.detach().to(torch.float32) * gain.unsqueeze(0)) \
            .to(lin.weight.dtype)
    norm.weight.data = torch.zeros_like(norm.weight)


def apply_fp8(model: nn.Module, include: "list[str]",
              merge_gate_up: bool = True,
              min_dim: int = 256) -> "list[FP8Linear | FP8GateUp]":
    """Swap matching nn.Linear modules for FP8 wrappers (calibration mode).

    `include` is a list of fnmatch patterns against module names. Gemma-style MLPs matching
    both gate_proj and up_proj are fused when merge_gate_up. Linears smaller than min_dim on
    either side stay bf16 (fp8 GEMM needs dims %16 and tiny GEMMs do not pay).
    Returns the wrapped modules; call freeze_fp8(model) after calibration passes.
    """
    wrapped = []
    by_name = dict(model.named_modules())
    handled = set()

    if merge_gate_up:
        mlp_names = sorted({n.rsplit(".", 1)[0] for n in by_name
                            if n.endswith("mlp.gate_proj")
                            and any(fnmatch.fnmatch(n, p) for p in include)})
        for mn in mlp_names:
            mlp = by_name[mn]
            gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
            if min(gate.weight.shape) < min_dim or gate.weight.shape[1] % 16:
                continue
            fused = FP8GateUp(gate, up)
            downw = FP8Linear(down)
            parent = by_name[mn.rsplit(".", 1)[0]]
            setattr(parent, mn.rsplit(".", 1)[1], _MergedGemmaMLP(mlp, fused, downw))
            wrapped += [fused, downw]
            handled.update({f"{mn}.gate_proj", f"{mn}.up_proj", f"{mn}.down_proj"})

    for name, mod in list(by_name.items()):
        if not isinstance(mod, nn.Linear) or name in handled:
            continue
        if not any(fnmatch.fnmatch(name, p) for p in include):
            continue
        if min(mod.weight.shape) < min_dim or mod.weight.shape[0] % 16 or mod.weight.shape[1] % 16:
            continue
        parent_name, _, leaf = name.rpartition(".")
        parent = by_name[parent_name] if parent_name else model
        fp8 = FP8Linear(mod)
        setattr(parent, leaf, fp8)
        wrapped.append(fp8)
    return wrapped


def freeze_fp8(model: nn.Module) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, (FP8Linear, FP8GateUp)) and m._calibrating:
            m.freeze()
            n += 1
    return n


def fold_gemma_norms(model: nn.Module, layer_pattern: str) -> int:
    """Fold input_layernorm into qkv and post_attention_layernorm into gate/up for every
    matching Gemma layer. Run BEFORE apply_fp8 (folding must precede weight quantization)."""
    n = 0
    rx = re.compile(re.escape(layer_pattern).replace(r"\*", r"\d+") + "$")
    for name, mod in model.named_modules():
        if not rx.fullmatch(name):
            continue
        sa, mlp = mod.self_attn, mod.mlp
        fold_rmsnorm_fp32(mod.input_layernorm, [sa.q_proj, sa.k_proj, sa.v_proj])
        fold_rmsnorm_fp32(mod.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj])
        n += 1
    return n

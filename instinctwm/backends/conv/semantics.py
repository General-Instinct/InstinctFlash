"""The convolution vocabulary: what is computed, over what layout.

WHY LAYOUT IS IN THE VOCABULARY AND NOT IN AN IMPLEMENTATION

Every other axis of a convolution — kernel, stride, groups — is obviously part of what the operator IS.
Memory layout looks like a detail of how a tensor happens to be stored, and that framing is what cost
us 158.7 ms per encode: the VAE's 3x3x3 convolutions were falling back to `slow_conv_dilated3d` purely
because they were called in NCDHW, and reach cuDNN at 4.35-7.24x in NDHWC. Nothing about the model, the
weights or the arithmetic needed to change. See PROFILE.md.

A backend does not "prefer" a layout as a matter of taste. It ACCEPTS a set of layouts and declines the
rest, exactly as an attention backend accepts a set of mask kinds. So `MemoryLayout` is declared, a
`ConvCapabilities` names which layouts it serves, and selection ranges over the product of (backend,
layout) rather than over backends alone. That product is the thing that was invisible before: there was
no place in the system to express "this operator would be legal for a fast backend if it arrived
transposed".

AND THE TRANSPOSE IS NOT FREE, in two different currencies:

    time      converting layout costs a copy. It only pays when the layout PROPAGATES through a
              subgraph, so the conversion is amortised over many operators rather than paid per call.
              `ConvCapabilities.layout_conversion_amortises_over` records that.
    numerics  NDHWC changes the accumulation order inside the convolution. Measured on the VAE
              encoder's output: max|delta| 1.25e-01, relative 6.67e-03, about 1.7x bf16 resolution.
              That is a NUMERIC-tier change and cannot ship under a max|delta| = 0 gate.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ConvSemantics(enum.Enum):
    """WHAT the convolution computes. A checkpoint-scoped fact; the runtime never changes it.

    As with attention, backends may only be exchanged within one member. A causal convolution is not a
    standard one with different padding — the causality is part of the function, and substituting
    across that boundary would serve a different model.
    """

    #: standard cross-correlation, zero-padded.
    STANDARD = "standard"
    #: causal in the leading spatial (time) axis: output t depends only on inputs <= t. Wan's
    #: `WanCausalConv3d` pads 2*pad on the left and 0 on the right, then convolves with padding 0.
    CAUSAL_TIME = "causal_time"
    #: grouped/depthwise, where groups == in_channels.
    DEPTHWISE = "depthwise"
    #: transposed / fractionally-strided, as decoders use.
    TRANSPOSED = "transposed"


class MemoryLayout(enum.Enum):
    """Physical layout of an activation tensor. A CAPABILITY axis, not a storage detail.

    The `channels_last_3d` member is the one that matters here: it is PyTorch's name for NDHWC, and it
    is the difference between `cudnn_convolution` and `slow_conv_dilated3d` for every 3x3x3 bf16
    convolution in the VAE on H100 / torch 2.9 / cuDNN 9.10.
    """

    NCHW = "nchw"                       # 2D contiguous
    NHWC = "nhwc"                       # 2D channels_last
    NCDHW = "ncdhw"                     # 3D contiguous  <- what the VAE runs today
    NDHWC = "ndhwc"                     # 3D channels_last_3d  <- what cuDNN wants

    def rank(self) -> int:
        return 4 if self in (MemoryLayout.NCHW, MemoryLayout.NHWC) else 5

    def is_channels_last(self) -> bool:
        return self in (MemoryLayout.NHWC, MemoryLayout.NDHWC)

    def torch_memory_format(self):
        """The `torch.memory_format` this corresponds to, or None for contiguous."""
        import torch
        return {
            MemoryLayout.NCHW: torch.contiguous_format,
            MemoryLayout.NHWC: torch.channels_last,
            MemoryLayout.NCDHW: torch.contiguous_format,
            MemoryLayout.NDHWC: torch.channels_last_3d,
        }[self]

    @staticmethod
    def of(t) -> "MemoryLayout":
        """Classify a live tensor. Used by probes and by legality checks, never to pick a backend."""
        import torch
        if t.dim() == 5:
            return (MemoryLayout.NDHWC
                    if t.is_contiguous(memory_format=torch.channels_last_3d)
                    else MemoryLayout.NCDHW)
        if t.dim() == 4:
            return (MemoryLayout.NHWC
                    if t.is_contiguous(memory_format=torch.channels_last)
                    else MemoryLayout.NCHW)
        raise ValueError(f"conv layout is defined for rank 4 or 5, got rank {t.dim()}")


@dataclass(frozen=True)
class ConvShape:
    """The signature a backend is asked to serve.

    Deliberately the full signature rather than a summary: the VAE investigation turned on the fact
    that 1x1x1 convolutions already reached cuDNN while 3x3x3 ones did not, and a summary that dropped
    the kernel size would have hidden that 16 of 62 convolutions were never on the slow path.
    """

    in_channels: int
    out_channels: int
    kernel: tuple[int, ...]
    stride: tuple[int, ...] = (1, 1, 1)
    padding: tuple[int, ...] = (0, 0, 0)
    dilation: tuple[int, ...] = (1, 1, 1)
    groups: int = 1
    dtype: str = "bfloat16"
    #: spatial extent, for memory legality and for reporting. (T, H, W) in 3D.
    spatial: tuple[int, ...] = ()
    batch: int = 1

    def spatial_rank(self) -> int:
        return len(self.kernel)

    def is_pointwise(self) -> bool:
        """1x1(x1): already served by cuDNN in either layout, so never a conversion candidate."""
        return all(k == 1 for k in self.kernel)

    def is_dilated(self) -> bool:
        return any(d != 1 for d in self.dilation)

    def elements(self) -> int:
        n = self.batch * self.in_channels
        for s in self.spatial:
            n *= s
        return n

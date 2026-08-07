"""Declared envelopes for the conv backends we can dispatch to today. NO KERNELS WRITTEN.

Every backend here already exists — cuDNN and PyTorch's fallbacks ship with torch. That is the point of
this layer: the 10.34x measured on the VAE encoder came from *dispatching differently*, not from new
code, and until this module existed there was nowhere to express which layouts each backend serves.

The envelopes are declarations of MEASURED behaviour, not of documentation:

    cudnn_conv3d     serves 1x1x1 in either layout, and 3x3x3 ONLY channels-last. Verified per
                     signature in probe_vae_conv_backend.py: 4 of 5 executed signatures fell back in
                     NCDHW, all reached cuDNN in NDHWC at 4.35-7.24x.
    torch_fallback   the reference path. `slow_conv_dilated3d` despite nothing being dilated: it is
                     simply where PyTorch lands when its 3D backends decline. Always legal, always
                     bit-exact against itself, and the incumbent every candidate must beat.
"""

from __future__ import annotations

from instinctwm.backends.conv.capabilities import ConvCapabilities
from instinctwm.backends.conv.semantics import ConvSemantics as C
from instinctwm.backends.conv.semantics import MemoryLayout as L
from instinctwm.passes.contract import HardwareReq

_UNBUILT = ("declaration only: this layer selects among backends that already exist. bind() is what a "
            "pass would install and is deliberately not wired here.")


class _Declared:
    name = "unnamed"
    version = "0.0.0"

    def capabilities(self) -> ConvCapabilities:      # pragma: no cover - overridden
        raise NotImplementedError

    def measure(self, shape, layout, device):
        raise NotImplementedError(f"{self.name}: {_UNBUILT}")

    def bind(self, shape, layout):
        raise NotImplementedError(f"{self.name}: {_UNBUILT}")


class TorchFallbackConv(_Declared):
    """PyTorch's own dispatch, whatever it picks. The incumbent and the numerical reference.

    On the VAE's 3x3x3 bf16 convolutions in NCDHW this resolves to `slow_conv_dilated3d`, which lowers
    via `vol2col` and materialises column buffers — 16.69 ms of convolution inside a 175.72 ms encode,
    the remainder being the surrounding work that materialisation generates.
    """

    name = "torch_fallback"
    version = "1.0.0"

    def capabilities(self) -> ConvCapabilities:
        return ConvCapabilities(
            semantics=frozenset(C),
            layouts=frozenset(L),
            spatial_ranks=frozenset({2, 3}),
            is_reference_path=True,
            layout_changes_reduction_order=False,
            amortises_over=1,
        )


class CuDNNConv3d(_Declared):
    """cuDNN, as reached through PyTorch's dispatcher. Fast, and layout-conditional.

    `pointwise_only_off_preferred_layout=True` is the declaration that encodes the measured fallback
    cause: a 3x3x3 kernel in NCDHW is declined outright, while the same kernel in NDHWC is served at
    4.35-7.24x. `amortises_over=62` is the VAE encoder's convolution count — this backend is only
    viable when the whole subgraph is converted, not one operator at a time.
    """

    name = "cudnn_conv3d"
    version = "1.0.0"

    def capabilities(self) -> ConvCapabilities:
        return ConvCapabilities(
            semantics=frozenset({C.STANDARD, C.CAUSAL_TIME, C.DEPTHWISE}),
            layouts=frozenset({L.NCDHW, L.NDHWC, L.NCHW, L.NHWC}),
            dtypes=frozenset({"bfloat16", "float16", "float32"}),
            spatial_ranks=frozenset({2, 3}),
            pointwise_only_off_preferred_layout=True,
            hardware=HardwareReq(requires=frozenset({"cudnn"})),
            is_reference_path=False,
            layout_changes_reduction_order=True,
            amortises_over=62,
            conversion_ms=0.0,      # measured by the probe, not guessed here
        )


DECLARED_BACKENDS = (TorchFallbackConv(), CuDNNConv3d())


def register_declared(registry) -> tuple[str, ...]:
    for b in DECLARED_BACKENDS:
        if b.name not in registry.names():
            registry.register(b)
    return registry.names()

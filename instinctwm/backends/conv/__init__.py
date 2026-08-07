"""Conv backend dispatch: layout is a capability, not an implementation detail.

    from instinctwm.backends.conv import REGISTRY, register_declared
    register_declared(REGISTRY)
    print(REGISTRY.explain(semantics=..., shape=..., have_layout=...))

The measured result this layer exists to express: the VAE encoder's 3x3x3 bf16 convolutions are declined
by cuDNN in NCDHW and served at 4.35-7.24x in NDHWC, and converting the whole encoder once takes the
encode from 175.72 ms to 17.00 ms. No kernel was written. See PROFILE.md.
"""

from instinctwm.backends.conv.capabilities import ConvCapabilities, legality
from instinctwm.backends.conv.reference import DECLARED_BACKENDS, register_declared
from instinctwm.backends.conv.registry import (
    REGISTRY,
    Candidate,
    ConvBackendRegistry,
    ConvPlan,
)
from instinctwm.backends.conv.semantics import ConvSemantics, ConvShape, MemoryLayout

__all__ = ["ConvSemantics", "MemoryLayout", "ConvShape", "ConvCapabilities", "legality",
           "ConvBackendRegistry", "REGISTRY", "Candidate", "ConvPlan",
           "DECLARED_BACKENDS", "register_declared"]

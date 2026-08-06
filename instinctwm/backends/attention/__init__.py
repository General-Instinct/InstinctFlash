"""Attention backends: one runtime, many attention implementations, chosen by measurement.

ARCHITECTURE ONLY. Selection raises `NotImplementedError` and no backend is installed anywhere. See
ATTENTION.md for the design and for what is deliberately not built.

    from instinctwm.backends.attention import REGISTRY, register_declared, read_site
    register_declared(REGISTRY)
    print(REGISTRY.explain(**facts))       # which backends are legal for a site, and why not

The one rule that makes this layer safe: a backend may be substituted only within the SEMANTICS the
checkpoint declares. Softmax attention computed two ways is a legality-and-speed question. A sliding
window where the checkpoint declared full attention is a different model.
"""

from instinctwm.backends.attention.backend import (
    AttentionBackend,
    AttentionBinding,
    AttentionMeasurement,
    plan_penalty_ms,
)
from instinctwm.backends.attention.capabilities import AttentionCapabilities, legality
from instinctwm.backends.attention.reference import DECLARED_BACKENDS, register_declared
from instinctwm.backends.attention.registry import (
    REGISTRY,
    AttentionBackendRegistry,
    Candidate,
)
from instinctwm.backends.attention.semantics import (
    AttentionSemantics,
    AttentionShape,
    Distribution,
    MaskKind,
    MaskSpec,
    QKVLayout,
)
from instinctwm.backends.attention.site import (
    AttentionSiteFacts,
    attention_site,
    read_site,
)

__all__ = [
    # what a checkpoint declares
    "AttentionSemantics", "MaskKind", "MaskSpec", "QKVLayout", "AttentionShape", "Distribution",
    # what a backend declares
    "AttentionCapabilities", "AttentionBackend", "AttentionBinding", "AttentionMeasurement",
    # the seam
    "attention_site", "read_site", "AttentionSiteFacts",
    # selection (legality only; ranking not implemented)
    "AttentionBackendRegistry", "REGISTRY", "Candidate", "legality", "plan_penalty_ms",
    "DECLARED_BACKENDS", "register_declared",
]

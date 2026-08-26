"""GR00T N1.7 adapter package: the Runtime adapter plus the bitexact fastpaths.

`fast_decode` and `backbone_fastpath` are pure-torch/numpy rewrites gated bitexact against
stock under the 6-case H100 protocol (see ../verify_fastpaths.py and its committed results).
Nothing imports torch at package import time; each module is imported by the caller that
installs it.
"""

__all__ = ["GR00TN17Adapter"]


def __getattr__(name):
    # Importing ``groot_n17_iwm.static_capture`` from FlashRT must not pull the Runtime/core
    # package into a serving-only environment. The adapter entry point imports its module
    # directly, so a lazy convenience export keeps both packages independent.
    if name == "GR00TN17Adapter":
        from .adapter import GR00TN17Adapter

        return GR00TN17Adapter
    raise AttributeError(name)

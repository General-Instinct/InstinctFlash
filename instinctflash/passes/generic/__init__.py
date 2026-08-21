"""Passes that reason about any model, from its declaration alone.

Everything under `passes/lingbot/` is one family's; everything here fires on whatever declares the
structure it needs. The split is the point: a provider named after a model is allowed to know that
model, and this one is not allowed to know any.
"""

from __future__ import annotations

from instinctflash.passes.generic.graph_capture import GraphCaptureApplicable


def default_passes() -> list:
    return [GraphCaptureApplicable()]


__all__ = ["GraphCaptureApplicable", "default_passes"]

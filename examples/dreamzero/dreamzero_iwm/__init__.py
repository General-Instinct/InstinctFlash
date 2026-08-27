"""`dreamzero` for InstinctFlash — registered from outside the core.

    pip install ./examples/dreamzero

The binding is the entry point in pyproject.toml, not an import here.
"""
from dreamzero_iwm.adapter import BACKBONE, DreamZeroAdapter

__all__ = ["DreamZeroAdapter", "BACKBONE"]

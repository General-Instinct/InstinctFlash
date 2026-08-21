"""`pi05` for InstinctFlash — a VLA family, registered from outside the core.

    pip install ./examples/pi05_vla

The binding is the entry point in pyproject.toml, not an import here.
"""
from pi05_iwm.adapter import BACKBONE, Pi05Adapter

__all__ = ["Pi05Adapter", "BACKBONE"]

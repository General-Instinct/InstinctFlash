"""`cosmos3_policy` for InstinctFlash — Edge and Nano share this one adapter.

    pip install ./examples/cosmos3_policy

The binding is the entry point in pyproject.toml, not an import here.
"""
from cosmos3_iwm.adapter import BACKBONE, Cosmos3PolicyAdapter

__all__ = ["Cosmos3PolicyAdapter", "BACKBONE"]

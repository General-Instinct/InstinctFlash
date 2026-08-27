"""`lingbot_vla` (the 4B family) for InstinctFlash — registered from outside the core.

    pip install ./examples/lingbot_vla

The binding is the entry point in pyproject.toml, not an import here.
"""
from lingbot_vla_iwm.adapter import BACKBONE, LingBotVLA4BAdapter

__all__ = ["LingBotVLA4BAdapter", "BACKBONE"]

"""`gridworld-ar` for InstinctFlash — an external plugin. No InstinctFlash source was modified.

    pip install gridworld-wm
    # then, with no import of this package anywhere:
    from instinctflash import Runtime
    runtime = Runtime.from_pretrained("some-org/my-world-model")

The binding is the entry point in pyproject.toml, not an import here:

    [project.entry-points."instinctflash.adapters"]
    gridworld_ar = "gridworld_wm.adapter:GridworldAdapter"

This module imports nothing from InstinctFlash at package-import time on purpose. `register()` at
import time only works if something imports this package first, which is the hidden knowledge the
entry point removes.
"""

from gridworld_wm.adapter import ADAPTER, GridworldAdapter

__all__ = ["GridworldAdapter", "ADAPTER"]

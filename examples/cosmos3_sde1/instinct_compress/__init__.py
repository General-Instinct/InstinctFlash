"""Shared compression infrastructure with model-owned native checkpoint adapters.

The existing :mod:`pi05_compress` package remains the authoritative pi05 recipe.
Heavy model dependencies are imported only by the selected adapter.
"""

__version__ = "0.4.0"


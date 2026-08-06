"""Generic nested data binding for capture units.

WHY

`CaptureUnit` assumed a unit takes named TENSORS and returns ONE tensor. That is a property of
LingBot-VA, not of world-action models. Cosmos3-Edge's decoder layer takes a `SequencePack` -- a
dict of tensors plus host metadata -- and returns a 3-tuple. `GraphExecutor.run` bound inputs with
`buf.copy_(inputs[name])`, which cannot bind a dict, so the generic executor could not carry the
model at all.

THE SPLIT THAT MATTERS FOR CAPTURE

A nested value has two parts, and they behave completely differently under graph capture:

  LEAVES   the tensors. These get bound to stable addresses and copied into per replay.
  SPEC     everything else -- container types, dict keys, ordering, and any non-tensor value
           (ints, strings, None, SplitInfo objects). None of it is re-evaluated on replay, so it
           is FROZEN into the captured graph and therefore belongs in the graph cache key.

That is the whole protocol. `flatten` separates them, `unflatten` puts them back, and the spec is
hashable precisely so the executor can key graphs on it without knowing what a SequencePack is.

Adapters may supply their own `Binder` when the default tree walk is wrong (a custom container, a
tensor that must not be bound, metadata that should be ignored for keying). Most will not need to:
`TreeBinder` handles tensors, tuples, lists, dicts and dataclass-free objects by identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class Leaf:
    """Placeholder marking where a tensor sat in the original structure."""
    index: int


def _hashable(x: Any) -> Any:
    """Best-effort hashable view of a non-tensor value, for graph keying."""
    if isinstance(x, (int, float, bool, str, bytes, type(None))):
        return x
    if isinstance(x, (list, tuple)):
        return tuple(_hashable(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((str(k), _hashable(v)) for k, v in x.items()))
    if isinstance(x, torch.dtype) or isinstance(x, torch.device):
        return str(x)
    # An opaque object (e.g. Cosmos's SplitInfo). Identity is the honest answer: if the adapter
    # swaps it for a different instance, the capture may no longer be valid and the key SHOULD
    # change. An adapter that knows better can supply its own Binder.
    return f"{type(x).__name__}@{id(x)}"


class Binder(Protocol):
    def flatten(self, value: Any) -> tuple[list[torch.Tensor], Any]: ...
    def unflatten(self, leaves: list[torch.Tensor], spec: Any) -> Any: ...


class TreeBinder:
    """Default binder: walk tensors / tuples / lists / dicts; everything else is spec."""

    def flatten(self, value: Any) -> tuple[list[torch.Tensor], Any]:
        leaves: list[torch.Tensor] = []

        def go(v):
            if isinstance(v, torch.Tensor):
                leaves.append(v)
                return Leaf(len(leaves) - 1)
            if isinstance(v, tuple):
                return ("tuple", tuple(go(x) for x in v))
            if isinstance(v, list):
                return ("list", tuple(go(x) for x in v))
            if isinstance(v, dict):
                # key order is part of the structure; sort so two equal dicts key identically
                return ("dict", tuple((k, go(v[k])) for k in sorted(v, key=str)))
            return ("leafless", v)

        spec = go(value)
        return leaves, spec

    def unflatten(self, leaves: list[torch.Tensor], spec: Any) -> Any:
        def go(s):
            if isinstance(s, Leaf):
                return leaves[s.index]
            kind, payload = s
            if kind == "tuple":
                return tuple(go(x) for x in payload)
            if kind == "list":
                return [go(x) for x in payload]
            if kind == "dict":
                return {k: go(v) for k, v in payload}
            return payload

        return go(spec)


def spec_key(spec: Any) -> Any:
    """A hashable projection of a spec, for the graph cache key.

    Tensor positions collapse to their slot index; everything else is the frozen host metadata the
    capture baked in.
    """
    def go(s):
        if isinstance(s, Leaf):
            return ("leaf", s.index)
        kind, payload = s
        if kind in ("tuple", "list"):
            return (kind, tuple(go(x) for x in payload))
        if kind == "dict":
            return ("dict", tuple((k, go(v)) for k, v in payload))
        return ("leafless", _hashable(payload))

    return go(spec)


def leaf_shapes(leaves: list[torch.Tensor]) -> tuple:
    """Shapes and dtypes of the bound leaves -- also part of the key, since capture freezes them."""
    return tuple((tuple(t.shape), str(t.dtype)) for t in leaves)

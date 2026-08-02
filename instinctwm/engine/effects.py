"""Detect host-side effects in a region before capturing it.

WHY THIS EXISTS

The first end-to-end graph capture was 2.17x and produced wrong actions (max|delta| 1.398 against
a chunk-to-chunk movement of 1.031). The cause was not the graph: the captured region mutated
**host** state -- P003's ring bookkeeping advanced with plain Python interleaved among the GPU ops.
Capture records the GPU work and freezes it; on replay the Python never runs again, so the ring
stopped advancing while the graph kept writing the slots baked in at capture time.

Nothing about that is visible in a shape, a dtype, or a tier. It is only visible by running the
region twice and noticing that the world changed. That is what this module does, and it is why the
planner refuses to capture a unit whose effects are not declared.

The check is deliberately dumb and empirical: snapshot, run, snapshot, diff. It cannot prove purity
-- an effect that happens only on the third call, or only at a wrap boundary, will slip through --
so it is a necessary condition for capture, not a sufficient one. The bit-exact gate remains the
backstop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch


def snapshot_host_state(roots: Iterable[Any], max_depth: int = 3) -> dict[str, Any]:
    """Collect non-tensor, non-callable attributes reachable from `roots`.

    Tensors are excluded on purpose: a GPU tensor write IS part of the captured graph and replays
    correctly. It is the Python-side scalars, dicts and counters that do not.
    """
    out: dict[str, Any] = {}
    seen: set[int] = set()

    def walk(obj, path, depth):
        if depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (int, float, bool, str, type(None))):
                    out[f"{path}[{k!r}]"] = v
                elif isinstance(v, dict):
                    walk(v, f"{path}[{k!r}]", depth + 1)
            return
        d = getattr(obj, "__dict__", None)
        if not d:
            return
        for k, v in d.items():
            if k.startswith("__"):
                continue
            if isinstance(v, (int, float, bool, str)):
                out[f"{path}.{k}"] = v
            elif isinstance(v, dict):
                walk(v, f"{path}.{k}", depth + 1)

    for i, r in enumerate(roots):
        walk(r, f"root{i}", 0)
    return out


@dataclass
class EffectReport:
    pure: bool
    changed: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    undeclared: tuple[str, ...] = ()

    def __str__(self) -> str:
        if self.pure:
            return "no host-side effects detected"
        lines = [f"{len(self.changed)} host-state key(s) mutated by one call:"]
        for k, (before, after) in list(self.changed.items())[:8]:
            lines.append(f"    {k}: {before!r} -> {after!r}")
        if len(self.changed) > 8:
            lines.append(f"    ... and {len(self.changed) - 8} more")
        return "\n".join(lines)


def detect_host_effects(fn: Callable[[], Any], roots: Iterable[Any],
                        declared: Iterable[str] = ()) -> EffectReport:
    """Run `fn` once and report host state it changed.

    `declared` names keys the caller knows about and has arranged to run outside the graph. A
    substring match is used so a caller can declare `"_ring"` and cover every layer's ring dict.
    """
    roots = list(roots)
    before = snapshot_host_state(roots)
    with torch.no_grad():
        fn()
    after = snapshot_host_state(roots)

    changed = {k: (before.get(k), after[k]) for k in after
               if k in before and before[k] != after[k]}
    changed.update({k: (None, after[k]) for k in after if k not in before})
    undeclared = tuple(k for k in changed if not any(d in k for d in declared))
    return EffectReport(pure=not undeclared, changed=changed, undeclared=undeclared)


class UndeclaredHostEffects(RuntimeError):
    """A unit mutates host state it did not declare. Capturing it would silently freeze that."""


def require_capturable(unit_name: str, fn: Callable[[], Any], roots: Iterable[Any],
                       declared: Iterable[str] = ()) -> EffectReport:
    rep = detect_host_effects(fn, roots, declared)
    if not rep.pure:
        raise UndeclaredHostEffects(
            f"refusing to capture {unit_name!r}: it mutates host state that graph replay will not "
            f"re-execute.\n  {rep}\n"
            f"  Move this bookkeeping outside the captured region, or declare it and arrange for "
            f"the runtime to advance it per replay.")
    return rep

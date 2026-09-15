"""Pure operating-point ranking shared by public evaluation and training."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def within_one_episode(success: float | None, best: float | None, n_pairs: int | None) -> bool:
    """Is `success` tied with `best` at the resolution of the pairing?

    Two configurations whose paired success differs by less than ONE episode of the pairing are
    not separated by the screen — the campaign's tie rule (h1_report §4b: 1V/4A@w1 vs @w3 is
    −0.4 pp [−0.054, +0.044], "quality tie; cost rule -> batch-1"). With no pairing size known
    the tolerance is zero.
    """
    if success is None or best is None:
        return False
    tolerance = (1.0 / n_pairs) if n_pairs else 0.0
    return success >= best - tolerance - 1e-12


def rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Best first: highest paired success; ties (within one episode) go to fewer batch-2
    forwards, then the lower total guidance scale — the cost rule."""
    measured = [dict(c) for c in candidates if c.get("success") is not None]
    if not measured:
        return []
    top = max(c["success"] for c in measured)
    n_pairs = min((c["n_pairs"] for c in measured if c.get("n_pairs")), default=None)

    def key(c):
        tied = within_one_episode(c["success"], top, n_pairs)
        b2 = c.get("forwards_batch2")
        scale = sum((g.get("scale") or 0.0) for g in (c.get("guidance") or {}).values()
                    if isinstance(g, Mapping))
        return (0 if tied else 1, -(c["success"] if not tied else top),
                b2 if b2 is not None else 10**6, scale)

    return sorted(measured, key=key)

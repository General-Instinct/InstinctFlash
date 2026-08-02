"""HoistInvariant -- evaluate a value at its declared scope instead of a tighter one.

The second true engine pass. Compare with `optimizer/passes/hoist_invariant_casts.py`, which knows
`FP32LayerNorm`, `scale_shift_table`, `_iwm_w32`, and rewrites `WanTransformerBlock.forward`. None
of that is about hoisting.

WHAT HOISTING IS, GENERICALLY

A value has a true scope -- the coarsest binding time at which it stops changing -- and an
evaluation scope, where it is actually computed. When the first is coarser than the second, the
value is recomputed for no reason. Hoisting is memoizing its producer at the true scope.

That is the entire pass. It works on anything an adapter can present as
"a producer, plus the scope at which its result is stable".

WHY EVERY DECLINE IS EXPLAINED

Hoisting is the most dangerous thing in the optimizer: a wrong `scope` produces silently wrong
outputs, not a crash. A value declared EPISODE-scoped that actually varies per cycle will be
computed once and reused forever, and nothing will fail. So this pass refuses by default and
records why, and scope claims stay adapter-DECLARED -- never inferred by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from instinctwm.passes.interface import Rewrite, RewriteKind, Scope, Site, SiteKind


@dataclass
class Decline:
    site_id: str
    reason: str

    def __str__(self) -> str:
        return f"{self.site_id}: {self.reason}"


class HoistInvariant:
    name = "hoist_invariant"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.declines: list[Decline] = []
        self.caches: list = []          # every memo installed, so a scope change can invalidate

    def sites_required(self):
        return (SiteKind.INVARIANT_CONDITIONING,)

    # ---- decide -------------------------------------------------------------------------------
    def plan_rewrites(self, sites, device) -> list[Rewrite]:
        self.declines.clear()
        out: list[Rewrite] = []
        for site in sites.get(SiteKind.INVARIANT_CONDITIONING, []):
            why = self._why_not(site)
            if why:
                self.declines.append(Decline(site.id, why))
                if self.verbose:
                    print(f"[hoist_invariant] DECLINE {site.id}: {why}", flush=True)
                continue
            out.append(Rewrite(
                site_id=site.id, kind=RewriteKind.WRAP, payload=self._memoize(site),
                note=f"evaluate at {site.scope().name} instead of {site.evaluated_at().name}"))
        return out

    def _why_not(self, site: Site) -> str | None:
        scope, ev = site.scope(), site.evaluated_at()
        if scope is None or ev is None:
            return ("no declared scope (a hoist on an undeclared scope is a guess, and a wrong "
                    "guess is silent)")
        if not (scope < ev):
            return (f"declared scope {scope.name} is not coarser than its evaluation scope "
                    f"{ev.name} -- nothing to hoist")
        if site.attrs.get("pure") is False:
            return "adapter declares the producer impure; its result cannot be reused"
        if site.attrs.get("produce") is None:
            return "site exposes no producer to memoize"
        return None

    # ---- what to install ----------------------------------------------------------------------
    def _memoize(self, site: Site):
        scope: Scope = site.scope()

        def wrap(produce):
            box: dict = {}

            def cached():
                if "v" not in box:
                    box["v"] = produce()
                return box["v"]

            cached.iwm_scope = scope
            cached.iwm_invalidate = box.clear
            self.caches.append(cached)
            return cached

        return wrap

    # ---- lifecycle ----------------------------------------------------------------------------
    def invalidate(self, scope: Scope) -> int:
        """Drop every cache whose scope is at least as tight as `scope`.

        The runtime calls this when a scope boundary is crossed -- e.g. `invalidate(Scope.EPISODE)`
        on reset. A MODEL-scoped cache survives; an EPISODE-scoped one does not.
        """
        n = 0
        for c in self.caches:
            if c.iwm_scope.value >= scope.value:
                c.iwm_invalidate()
                n += 1
        return n

    def stats(self) -> str:
        return f"hoisted={len(self.caches)} declined={len(self.declines)}"

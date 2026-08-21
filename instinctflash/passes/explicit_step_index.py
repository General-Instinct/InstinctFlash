"""ExplicitStepIndex -- construct the loop index instead of searching for it.

Fifth engine pass. Found by re-profiling the migrated default: 223 host synchronizations per
control cycle, every one of them from three lines in the flow-matching solver:

    timestep    = timestep.cpu()                                   # D2H
    timestep_id = argmin((self.timesteps - timestep).abs())        # search the table
    if to_final or timestep_id + 1 >= len(self.timesteps):         # forces .item()

The solver is handed a timestep VALUE and recovers the step INDEX by searching the schedule for
it -- while the caller is sitting in `for i, t in enumerate(timesteps)` and already has `i`. The
sigma/timestep tables are PLAN-scoped and device-resident; only the index was ever in question.

This is the same shape as P003: a quantity known by construction, rediscovered from data at the
tightest scope, at the cost of a host round trip. P003 removed it from KV addressing; this removes
it from the solver. Naming the pattern is the point -- it will appear again.

Every sync also blocks whole-cycle capture, because a host round trip inside the loop body makes
the loop uncapturable. So the value of this pass is mostly what it unlocks.
"""

from __future__ import annotations

from dataclasses import dataclass

from instinctflash.passes.interface import Rewrite, RewriteKind, Site, SiteKind


@dataclass
class Decline:
    site_id: str
    reason: str

    def __str__(self) -> str:
        return f"{self.site_id}: {self.reason}"


class ExplicitStepIndex:
    name = "explicit_step_index"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.declines: list[Decline] = []
        self.rewritten: list[str] = []
        self.syncs_removed_per_call = 0

    def sites_required(self):
        return (SiteKind.STATE_ADDRESSING,)

    def plan_rewrites(self, sites, device) -> list[Rewrite]:
        self.declines.clear()
        out: list[Rewrite] = []
        for site in sites.get(SiteKind.STATE_ADDRESSING, []):
            why = self._why_not(site)
            if why:
                self.declines.append(Decline(site.id, why))
                if self.verbose:
                    print(f"[explicit_step_index] DECLINE {site.id}: {why}", flush=True)
                continue
            self.rewritten.append(site.id)
            self.syncs_removed_per_call += site.attrs.get("syncs_per_call", 0)
            out.append(Rewrite(site_id=site.id, kind=RewriteKind.WRAP,
                               payload=self._counter(site),
                               note="index by construction; no host round trip"))
        return out

    def _why_not(self, site: Site) -> str | None:
        a = site.attrs
        if a.get("addressing") != "index_by_search":
            return (f"addressing is {a.get('addressing')!r}, not a search for an index that the "
                    f"caller already knows")
        if not a.get("monotonic_calls"):
            return ("adapter does not declare that the site is called once per step in order; "
                    "a counter would drift from the true index")
        if a.get("reset_hook") is None:
            return "no reset hook, so the counter could not be rewound between loops"
        return None

    def _counter(self, site: Site):
        """Replace `resolve(value) -> index` with a counter advanced per call."""
        engine = self

        def wrap(_search):
            box = {"i": 0}

            def resolve(_value=None):
                i = box["i"]
                box["i"] = i + 1
                return i

            resolve.iwm_reset = lambda: box.update(i=0)
            site.attrs["reset_hook"](resolve.iwm_reset)
            return resolve

        return wrap

    def stats(self) -> str:
        return (f"rewritten={len(self.rewritten)} declined={len(self.declines)} "
                f"syncs_removed_per_call={self.syncs_removed_per_call}")

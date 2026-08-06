"""StablePools -- allocate storage once, clear it logically in place.

Third true engine pass. Compare with `passes/lingbot/stable_pools.py`, which knows
`WanAttention.init_kv_cache`, `clear_cache`, the keys `k`/`v`/`mask`/`id`/`is_pred`, the `_ring`
dict, and P002's `populate_cross_cache`. None of that is about storage lifetime.

THE DISTINCTION THE PASS IS BUILT ON

A buffer has a PHYSICAL lifetime -- how long its storage must stay at one address -- and a LOGICAL
reset scope -- how often its contents must be forgotten. Models routinely conflate them: LingBot
reallocates its whole KV pool at every episode reset because the contents are episode-scoped, even
though the storage is model-scoped. Anything holding a pointer into that storage (a captured CUDA
graph) is silently invalidated.

So the pass asks one question per site: is the physical lifetime coarser than the scope at which
allocation currently happens? If yes, allocate once and let the adapter's declared `clear` handle
the logical reset.

WHY EXTENT MUST BE STATIC

Reusing storage is only sound if the next request fits it exactly. A site that cannot state its
extent might ask for a different shape next episode, and silently returning the old buffer would
be a correctness bug, not a slow path. Undeclared extent is therefore a decline, not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from instinctwm.passes.interface import Rewrite, RewriteKind, Scope, Site, SiteKind


@dataclass
class Decline:
    site_id: str
    reason: str

    def __str__(self) -> str:
        return f"{self.site_id}: {self.reason}"


class StablePools:
    name = "stable_pools"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.declines: list[Decline] = []
        self.n_allocs = 0
        self.n_reuses = 0
        self.pointers: dict[str, tuple] = {}
        #: sites allocated since the last `set_baseline`. Non-empty means some storage moved and
        #: nothing has re-certified it yet, so `pointers_stable` must refuse.
        self.pending: set[str] = set()

    def sites_required(self):
        return (SiteKind.ALLOCATION,)

    # ---- decide -------------------------------------------------------------------------------
    def plan_rewrites(self, sites, device) -> list[Rewrite]:
        self.declines.clear()
        out: list[Rewrite] = []
        for site in sites.get(SiteKind.ALLOCATION, []):
            why = self._why_not(site)
            if why:
                self.declines.append(Decline(site.id, why))
                if self.verbose:
                    print(f"[stable_pools] DECLINE {site.id}: {why}", flush=True)
                continue
            out.append(Rewrite(
                site_id=site.id, kind=RewriteKind.WRAP, payload=self._stabilize(site),
                note=(f"allocate once at {site.attrs['physical_lifetime'].name}, clear at "
                      f"{site.attrs['logical_reset'].name}")))
        return out

    def _why_not(self, site: Site) -> str | None:
        a = site.attrs
        phys, ev = a.get("physical_lifetime"), a.get("evaluated_at")
        if phys is None or ev is None:
            return "no declared physical_lifetime / evaluated_at"
        if not (phys < ev):
            return (f"physical lifetime {phys.name} is not coarser than the scope allocation "
                    f"happens at ({ev.name}) -- nothing to stabilize")
        if a.get("extent") is None:
            return ("extent is dynamic or undeclared; reusing storage across a shape change "
                    "would return a buffer that does not fit the request")
        if a.get("allocate") is None:
            return "site exposes no allocator to wrap"
        if a.get("clear") is None and a.get("copy_into") is None:
            return ("no declared reset semantics; storage could be reused but the previous "
                    "episode's contents would leak. Declare `clear` (reset to empty) or "
                    "`copy_into` (recompute fresh contents into the same storage)")
        return None

    # ---- what to install ----------------------------------------------------------------------
    def _stabilize(self, site: Site):
        """Two reset semantics, because storage reuse has two shapes.

        CLEAR      the contents are logically empty after a reset, so wipe in place.
                   (LingBot's KV pool: mask/id/is_pred reset, k/v left alone as unreachable.)
        COPY_INTO  the contents are recomputed every episode but must land in the SAME storage.
                   (P002's cross-attention K/V: new text, same buffers.)

        The second only turned up when migrating the shipped server. A site model with only
        `clear` would have forced cross-attention K/V to stay backend-specific, which is how
        parallel implementations start.
        """
        extent = site.attrs["extent"]
        clear = site.attrs.get("clear")
        copy_into = site.attrs.get("copy_into")
        engine = self
        sid = site.id

        def wrap(allocate):
            box: dict = {}

            def stable(*a, **kw):
                want = kw.get("extent", extent)
                if "v" in box and box["extent"] == want:
                    if copy_into is not None:
                        copy_into(box["v"], allocate(*a, **kw))   # fresh values, same storage
                    else:
                        clear(box["v"])                           # logical reset, same storage
                    engine.n_reuses += 1
                    return box["v"]
                # first call, or the extent genuinely changed: allocate and re-record
                v = allocate(*a, **kw)
                box["v"], box["extent"] = v, want
                engine.n_allocs += 1
                # NOTE the absence of `engine.pointers[sid] = ...` here.
                #
                # Recording the baseline at allocation time makes the check FAIL OPEN: a reset
                # that reallocates refreshes the baseline before anyone compares against it, so
                # `pointers_stable` returns True while every captured graph points at freed
                # memory. That is exactly what happened -- max|delta action| = 1.527 with graph
                # preservation on, while the same passes without preservation were bit-exact.
                # It is also the SECOND time this class of bug appeared: the backend-specific
                # P006 had it in `_record`, and the generic rewrite reintroduced it.
                #
                # The baseline moves only when someone explicitly calls `set_baseline`.
                engine.pending.add(sid)
                return v

            stable.iwm_site = sid
            return stable

        return wrap

    # ---- verification the pass owns, because it made the claim ---------------------------------
    @staticmethod
    def _ptrs(value) -> tuple:
        import torch

        out = []

        def go(v):
            if isinstance(v, torch.Tensor):
                out.append(v.untyped_storage().data_ptr())
            elif isinstance(v, dict):
                for k in sorted(v, key=str):
                    go(v[k])
            elif isinstance(v, (list, tuple)):
                for x in v:
                    go(x)

        go(value)
        return tuple(out)

    def set_baseline(self, current: dict[str, object]) -> None:
        """Declare the CURRENT addresses to be the reference. The only way the baseline moves.

        Callers do this once stabilization is genuinely in effect -- i.e. after the first wrapped
        allocation of every site, not before.
        """
        self.pointers = {sid: self._ptrs(v) for sid, v in current.items()}
        self.pending.clear()

    def pointers_stable(self, current: dict[str, object]) -> tuple[bool, str]:
        """Have the storages this pass stabilized stayed put?

        The pass made the claim, so the pass carries the check. A stale pointer does not raise --
        it returns plausible garbage -- so anything relying on stability must be able to ask.
        """
        if self.pending:
            return False, (f"{len(self.pending)} site(s) allocated since the last baseline "
                           f"(e.g. {sorted(self.pending)[:2]}); storage moved and has not been "
                           f"re-certified")
        if not self.pointers:
            return False, "no baseline recorded; refusing to certify stability"
        for sid, want in self.pointers.items():
            if sid not in current:
                return False, f"{sid} is no longer present"
            got = self._ptrs(current[sid])
            if got != want:
                return False, f"{sid} moved: {want[:2]} -> {got[:2]}"
        return True, f"all {len(self.pointers)} stabilized site(s) at their original addresses"

    def stats(self) -> str:
        return (f"allocs={self.n_allocs} reuses={self.n_reuses} "
                f"stabilized={len(self.pointers)} declined={len(self.declines)}")

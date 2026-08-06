"""GraphCapture -- a generic pass. Captures whatever the adapter calls a capture unit.

Compare with `passes/lingbot/graph_capture.py`, the LingBot version: that one imports
`modules.model`, reaches for `WanTransformer3DModel`, rewrites the source of its `forward` to find
`for block in self.blocks:`, and knows the names `update_cache` and `cache_name`. None of that is
about graph capture. It is about LingBot.

This one knows three things, all of them true of any model:

  * a capture unit is a callable plus the arguments it is called with
  * replay is only valid while the structural signature holds, so key on it
  * host-state mutation inside the region makes capture unsound, so refuse

Everything else -- which callable, which arguments, when the signature changes -- is the adapter's
answer to "where".
"""

from __future__ import annotations

import torch

from instinctwm.executors.binding import TreeBinder, leaf_shapes, spec_key
from instinctwm.planners.effects import detect_host_effects
from instinctwm.passes.interface import Rewrite, RewriteKind, Site, SiteKind


class GraphCapture:
    name = "graph_capture"

    def __init__(self, max_graphs: int = 64, verbose: bool = False):
        self.max_graphs = max_graphs
        self.verbose = verbose
        self.graphs: dict = {}
        self.stable: dict = {}
        self.outputs: dict = {}
        self.n_captures = 0
        self.n_replays = 0
        self.refused: list[str] = []

    def sites_required(self):
        return (SiteKind.CAPTURE_UNIT,)

    # ---- decide -------------------------------------------------------------------------------
    def plan_rewrites(self, sites, device) -> list[Rewrite]:
        out: list[Rewrite] = []
        for site in sites.get(SiteKind.CAPTURE_UNIT, []):
            if site.attrs.get("capturable") is False:
                self.refused.append(f"{site.id}: adapter declares capturable=False")
                continue
            out.append(Rewrite(site_id=site.id, kind=RewriteKind.WRAP,
                               payload=self._make_wrapper(site),
                               note="replay a captured graph keyed on the structural signature"))
        return out

    # ---- what to install ----------------------------------------------------------------------
    def _make_wrapper(self, site: Site):
        binder = site.attrs.get("binder") or TreeBinder()
        # The adapter may name a host-state root set so the effect check has something to watch.
        roots = site.attrs.get("effect_roots", ())
        engine = self

        def wrapper(orig):
            def run(*args):
                flat, key = [], []
                for v in args:
                    leaves, spec = binder.flatten(v)
                    flat.append((leaves, spec))
                    key.append((spec_key(spec), leaf_shapes(leaves)))
                slot = (site.id, tuple(key), site.attrs.get("extent_fn", lambda: 0)())

                if slot not in engine.graphs:
                    ok = engine._capture(site, orig, binder, flat, slot, roots)
                    if not ok:
                        return orig(*args)          # refused: fall back, never guess
                for (leaves, _s), (owned, _o) in zip(flat, engine.stable[slot]):
                    for dst, src in zip(owned, leaves):
                        dst.copy_(src)
                engine.graphs[slot].replay()
                engine.n_replays += 1
                return engine.outputs[slot]

            return run

        return wrapper

    def _capture(self, site, orig, binder, flat, slot, roots) -> bool:
        stable = [([t.detach().clone() for t in leaves], spec) for leaves, spec in flat]
        args = [binder.unflatten(o, sp) for o, sp in stable]

        # GATE: a region that mutates host state cannot be captured -- replay does not re-run
        # Python. This is model-independent, so it lives in the pass, not the adapter.
        if roots:
            rep = detect_host_effects(lambda: orig(*args), roots)
            if not rep.pure:
                msg = f"{site.id}: mutates host state {list(rep.undeclared)[:3]}"
                if msg not in self.refused:
                    self.refused.append(msg)
                    if self.verbose:
                        print(f"[graph_capture] REFUSED {msg}", flush=True)
                return False

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                orig(*args)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g), torch.no_grad():
                out = orig(*args)
        except Exception as ex:
            msg = f"{site.id}: capture failed ({type(ex).__name__})"
            if msg not in self.refused:
                self.refused.append(msg)
            return False

        while len(self.graphs) >= self.max_graphs:
            oldest = next(iter(self.graphs))
            del self.graphs[oldest], self.stable[oldest], self.outputs[oldest]
        self.graphs[slot], self.stable[slot], self.outputs[slot] = g, stable, out
        self.n_captures += 1
        return True

    def stats(self) -> str:
        return (f"captures={self.n_captures} replays={self.n_replays} "
                f"held={len(self.graphs)} refused={len(self.refused)}")

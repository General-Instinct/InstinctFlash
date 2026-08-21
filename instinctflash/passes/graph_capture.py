"""GraphCapture -- a generic pass. Captures whatever the adapter calls a capture unit.

Compare with `passes/lingbot/graph_capture.py`, the LingBot version: that one imports
`modules.model`, reaches for `WanTransformer3DModel`, rewrites the source of its `forward` to find
`for block in self.blocks:`, and knows the names `update_cache` and `cache_name`. None of that is
about graph capture. It is about LingBot.

This one knows four things, all of them true of any model:

  * a capture unit is a callable plus the arguments it is called with
  * replay is only valid while the structural signature holds, so key on it
  * host-state mutation inside the region makes capture unsound, so refuse
  * capturing is not evidence of replaying: check the graph against an input it was NOT captured
    from, once, and discard it if the answers differ

Everything else -- which callable, which arguments, when the signature changes -- is the adapter's
answer to "where".
"""

from __future__ import annotations

import torch

from instinctflash.executors.binding import TreeBinder, leaf_shapes, spec_key
from instinctflash.planners.effects import detect_host_effects
from instinctflash.passes.interface import Rewrite, RewriteKind, Site, SiteKind


class GraphCapture:
    name = "graph_capture"

    def __init__(self, max_graphs: int = 64, verbose: bool = False):
        self.max_graphs = max_graphs
        self.verbose = verbose
        self.graphs: dict = {}
        self.stable: dict = {}
        self.outputs: dict = {}
        #: slots captured but not yet checked against an input they were not captured from
        self.pending: dict = {}
        #: slots whose replay was measured to disagree with eager. Permanent: a region that replays
        #: wrong once will replay wrong again, and retrying it per call would pay the validation cost
        #: forever while still producing the eager answer.
        self.rejected: set = set()
        self.n_captures = 0
        self.n_replays = 0
        self.n_validated = 0
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

                if slot in engine.rejected:
                    return orig(*args)              # proven unsound here; never try again
                if slot not in engine.graphs:
                    ok = engine._capture(site, orig, binder, flat, slot, roots)
                    if not ok:
                        return orig(*args)          # refused: fall back, never guess
                    engine.pending[slot] = True     # unvalidated until a DIFFERENT input arrives
                for (leaves, _s), (owned, _o) in zip(flat, engine.stable[slot]):
                    for dst, src in zip(owned, leaves):
                        dst.copy_(src)

                # VALIDATE ON THE SECOND, DIFFERENT INPUT -- once, then never again.
                #
                # Capturing successfully proves nothing about replaying. A graph replayed with the
                # very inputs it was captured from is exact by construction, so the only check that
                # means anything uses an input the capture never saw. Without this the pass shipped a
                # measured 1.55x on pi05 whose actions were WRONG by up to 48% of the signal, and it
                # looked correct three separate ways: capture succeeded, the host-effect gate passed,
                # and a per-step comparison at the captured operating point read 0.000e+00.
                #
                # The gate could not have caught it. `detect_host_effects` watches state reachable
                # from the declared roots, and pi05's denoise step mutates a `DynamicCache` that it
                # CREATES inside the region -- 50 entries appended per call, invisible to any root
                # set because the object does not exist when the snapshot is taken. Structural
                # analysis has a floor; comparing outputs does not.
                if engine.pending.get(slot):
                    ref = orig(*[binder.unflatten(le, sp) for le, sp in flat])
                    engine.graphs[slot].replay()
                    got = engine.outputs[slot]
                    bad = engine._mismatch(ref, got)
                    engine.pending[slot] = False
                    if bad is not None:
                        engine.rejected.add(slot)
                        del engine.graphs[slot], engine.stable[slot], engine.outputs[slot]
                        msg = (f"{site.id}: replay disagrees with eager by {bad:.3e} on an input it "
                               f"was not captured from -- discarded, falling back to eager")
                        if msg not in self.refused:
                            self.refused.append(msg)
                        print(f"[graph_capture] REJECTED {msg}", flush=True)
                        return ref
                    engine.n_validated += 1
                    return got

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

        # CAPTURING MUST NOT MOVE THE RNG. Capture runs the region three extra times (two warm-ups and
        # the capture itself) and `torch.cuda.graph` registers the default generator with the graph
        # pool, so without this the global stream sits at a different offset afterwards than it would
        # have. Anything the MODEL samples outside the captured region then diverges.
        #
        # This was not theoretical. pi05 draws its initial flow-matching noise in `sample_actions`,
        # outside the region. A paired end-to-end check reported the first chunk bit-exact across 50
        # actions and every later chunk different by ~2.1 against an action scale of 0.5: replay was
        # exact and the NEXT chunk started from different noise. A pass that silently reseeds the model
        # it is optimizing cannot claim BITEXACT, and the tier is the product here.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
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
        finally:
            # In `finally` on purpose: a refused or failed capture has already run the region, so it
            # has already moved the stream. Restoring only on success would leave the fallback path
            # -- the one that is supposed to be indistinguishable from not having the pass -- as the
            # one that perturbs the model.
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        while len(self.graphs) >= self.max_graphs:
            oldest = next(iter(self.graphs))
            del self.graphs[oldest], self.stable[oldest], self.outputs[oldest]
        self.graphs[slot], self.stable[slot], self.outputs[slot] = g, stable, out
        self.n_captures += 1
        return True

    @staticmethod
    def _mismatch(ref, got, atol: float = 0.0):
        """`None` if replay reproduced eager, else the largest absolute disagreement.

        atol=0 by design. A captured graph replays the SAME kernels in the same order on the same
        addresses, so bit-exact is the correct expectation and any drift is evidence that something is
        not being re-read. A tolerance here would convert exactly the bug this catches into a pass.
        """
        import torch as _t

        def walk(a, b):
            if _t.is_tensor(a) and _t.is_tensor(b):
                if a.shape != b.shape:
                    return float("inf")
                d = (a.detach().float() - b.detach().float()).abs().max().item()
                return d if d > atol else None
            if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                if len(a) != len(b):
                    return float("inf")
                worst = [w for w in (walk(x, y) for x, y in zip(a, b)) if w is not None]
                return max(worst) if worst else None
            if isinstance(a, dict) and isinstance(b, dict):
                if set(a) != set(b):
                    return float("inf")
                worst = [w for w in (walk(a[k], b[k]) for k in a) if w is not None]
                return max(worst) if worst else None
            return None if a == b else float("inf")

        return walk(ref, got)

    def stats(self) -> str:
        return (f"captures={self.n_captures} replays={self.n_replays} "
                f"validated={self.n_validated} held={len(self.graphs)} "
                f"rejected={len(self.rejected)} refused={len(self.refused)}")

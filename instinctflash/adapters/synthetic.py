"""A synthetic Backend Adapter. Not a transformer, not a WAM, barely a model.

Its only purpose is to prove the pass interface has no hidden coupling to either real model. If a
pass fires here, it fired on a structure nobody designed it around.

It publishes two INVARIANT_CONDITIONING sites that differ in exactly one property:

    "episode_projection"  scope=EPISODE, evaluated_at=STEP   -> hoistable
    "per_step_noise"      scope=STEP,    evaluated_at=STEP   -> NOT hoistable

The second exists so a correct no-op can be observed, rather than inferred from a pass that
happened to find nothing.
"""

from __future__ import annotations

import torch


class SyntheticSurface:
    model_id = "synthetic"

    def __init__(self, device, dim: int = 256, dtype=torch.float32):
        g = torch.Generator(device="cpu").manual_seed(0)
        self.W = torch.randn(dim, dim, generator=g).to(device, dtype)
        self.x = torch.randn(dim, dim, generator=g).to(device, dtype)
        self.device, self.dim, self.dtype = device, dim, dtype
        self.calls = {"episode_projection": 0, "per_step_noise": 0}
        self._prod = {}
        self._installed = {}

    # -- the "model" --------------------------------------------------------------------------
    def _episode_projection(self):
        """Expensive, and constant for the whole episode -- but recomputed every step."""
        self.calls["episode_projection"] += 1
        return self.W @ self.W.T

    def _per_step_noise(self):
        """Genuinely varies per step. Hoisting this would be a correctness bug."""
        self.calls["per_step_noise"] += 1
        return torch.full((self.dim, self.dim), float(self.calls["per_step_noise"]),
                          device=self.device, dtype=self.dtype)

    def step(self):
        proj = self._installed.get("episode_projection", self._episode_projection)()
        noise = self._installed.get("per_step_noise", self._per_step_noise)()
        return (self.x @ proj) + noise

    # -- WHERE --------------------------------------------------------------------------------
    def sites(self, kind):
        from instinctflash.passes.interface import Scope, Site, SiteKind

        if kind is SiteKind.ALLOCATION:
            yield from self.alloc_sites()
            return
        if kind is not SiteKind.INVARIANT_CONDITIONING:
            return
        self._prod["episode_projection"] = self._episode_projection
        self._prod["per_step_noise"] = self._per_step_noise
        yield Site(kind=kind, id="synthetic.episode_projection",
                   attrs={"scope": Scope.EPISODE, "evaluated_at": Scope.STEP, "pure": True,
                          "produce": self._episode_projection})
        yield Site(kind=kind, id="synthetic.per_step_noise",
                   attrs={"scope": Scope.STEP, "evaluated_at": Scope.STEP, "pure": True,
                          "produce": self._per_step_noise})

    def alloc_sites(self):
        """One ALLOCATION site whose extent is genuinely dynamic -- it must be declined."""
        from instinctflash.passes.interface import Scope, Site, SiteKind

        yield Site(kind=SiteKind.ALLOCATION, id="synthetic.growing_buffer",
                   attrs={"physical_lifetime": Scope.MODEL, "logical_reset": Scope.EPISODE,
                          "evaluated_at": Scope.EPISODE,
                          "extent": None,          # grows with the episode; cannot be reused
                          "ownership": "synthetic",
                          "allocate": lambda **kw: torch.zeros(8, device=self.device),
                          "clear": lambda t: t.zero_()})

    def apply(self, rewrite):
        from instinctflash.passes.interface import RewriteKind

        key = rewrite.site_id.split(".", 1)[1]
        if rewrite.kind is not RewriteKind.WRAP or key not in self._prod:
            raise NotImplementedError(f"synthetic surface cannot apply {rewrite}")
        self._installed[key] = rewrite.payload(self._prod[key])

"""Backend Adapter helpers for LingBot-VA.

Only the parts the generic engine needs from the adapter side. The optimization passes are still
model-specific `install()` functions elsewhere; that is a known gap, not this file's job.
"""

from __future__ import annotations


def state_roots(model) -> dict:
    """Named roots the engine walks to name device buffers.

    The engine used to hard-code this schema. Moving it here is what lets the same tracer name
    Cosmos3's state without knowing what a `SequencePack` is.
    """
    # Specific roots FIRST: `build_name_map` keeps the first name it sees, and a generic walk from
    # the model reaches the same tensors by an unhelpful path
    # (`model.blocks[0]._modules.attn1.attn_caches.pos.k`). Order is the whole mechanism.
    roots: dict = {}
    for i, blk in enumerate(getattr(model, "blocks", []) or []):
        for attr in ("attn1", "attn2"):
            a = getattr(blk, attr, None)
            if a is None:
                continue
            if getattr(a, "attn_caches", None):
                roots[f"kv[{i}].{attr}"] = a.attn_caches
            kv = getattr(a, "_iwm_cross_kv", None)
            if kv is not None:
                roots[f"cross_kv[{i}].{attr}"] = kv
        # P004's hoisted fp32 parameter views, which live on the modules themselves
        for name, mod in blk.named_modules():
            for cache in ("_iwm_w32", "_iwm_b32", "_iwm_sst32"):
                t = getattr(mod, cache, None)
                if t is not None:
                    roots[f"hoisted[{i}].{name or 'block'}.{cache}"] = t
    roots["model"] = model        # catch-all, deliberately last
    return roots


# =================================================================================================
# AdapterSurface: LingBot-VA published as sites.
# =================================================================================================

class LingBotSurface:
    """Site publisher for LingBot-VA.

    Everything model-specific about capture lives HERE now: which callable is the unit, what its
    arguments are, what makes its structural signature change (the ring interval), and which host
    state must be advanced outside the captured region. The pass reads none of it by name.
    """

    model_id = "lingbot-va"

    def __init__(self, model, cache_name="pos"):
        self.model = model
        self.cache_name = cache_name
        self._wrapped = {}
        self._cast_sites: dict = {}

    # -- WHERE ------------------------------------------------------------------------------
    def sites(self, kind):
        from instinctwm.passes.interface import Scope, Site, SiteKind

        blocks = getattr(self.model, "blocks", []) or []
        if kind is SiteKind.CAPTURE_UNIT:
            a0 = blocks[0].attn1 if blocks else None
            deferred = bool(getattr(type(a0), "_iwm_defer_commit", False)) if a0 else False
            yield Site(
                kind=kind, id="lingbot.block_stack",
                attrs={
                    # Capture is only sound once the ring bookkeeping has been deferred out of the
                    # region. The ADAPTER knows that; the pass just reads the flag.
                    "capturable": deferred,
                    "effect_roots": tuple(b.attn1.attn_caches for b in blocks
                                          if getattr(b.attn1, "attn_caches", None)),
                    "arity": 4,
                    # the ring interval is baked into every captured graph
                    "extent_fn": self._extent,
                    "note": "argument 0 is hidden states; 1 encoder; 2 timestep_proj; 3 rope",
                })
        elif kind is SiteKind.EXECUTION_REGION:
            for i, _b in enumerate(blocks):
                yield Site(kind=kind, id=f"lingbot.block[{i}]", attrs={"index": i})
        elif kind is SiteKind.STATE_ADDRESSING:
            for i, b in enumerate(blocks):
                r = (getattr(b.attn1, "attn_caches", None) or {}).get(self.cache_name, {})
                if isinstance(r, dict) and "_ring" in r:
                    yield Site(kind=kind, id=f"lingbot.kv_ring[{i}]",
                               attrs={"addressing": "ring_interval",
                                      "scope": Scope.CYCLE, "evaluated_at": Scope.LAYER})
        elif kind is SiteKind.INVARIANT_CONDITIONING:
            LN = self._install_producer_shim()
            for i, b in enumerate(blocks):
                for name, mod in b.named_modules():
                    # ONLY modules whose consumer actually reads the producer. Publishing a site
                    # the adapter cannot route means the pass installs a cache nothing calls and
                    # then reports a rewrite that did nothing: an inflated count, and worse, a
                    # correctness claim about a code path that never ran. An earlier version
                    # published every weight/bias in the block and claimed 81 rewrites where only
                    # the layer-norm ones were live.
                    if not isinstance(mod, LN):
                        continue
                    for src in ("weight", "bias"):
                        t = getattr(mod, src, None)
                        if t is None or not hasattr(t, "float"):
                            continue
                        # A PARAMETER cast to fp32. The parameter does not change once the model is
                        # loaded, so the cast is MODEL-scoped; the model performs it inside every
                        # layer of every forward, i.e. at LAYER scope. 4,740 casts of a constant
                        # per control cycle.
                        yield Site(
                            kind=kind, id=f"lingbot.cast[{i}].{name or 'block'}.{src}",
                            attrs={"scope": Scope.MODEL, "evaluated_at": Scope.LAYER,
                                   "pure": True, "dtype": "fp32",
                                   "produce": self._remember(
                                       f"lingbot.cast[{i}].{name or 'block'}.{src}", mod, src)})

    # -- invariant-conditioning plumbing ----------------------------------------------------
    # The adapter owns the mechanism; the pass owns the policy. `produce` computes the value,
    # `install` puts whatever the pass hands back where the consumer will call it.

    def _remember(self, site_id, mod, src):
        """Record producer+installer for this site so `apply` can find them by id alone."""
        produce, install = self._producer(mod, src), self._installer(mod, src)
        self._cast_sites[site_id] = (produce, install)
        return produce

    @staticmethod
    def _producer(mod, src):
        def produce():
            return getattr(mod, src).float()
        return produce

    @staticmethod
    def _installer(mod, src):
        def install(cached):
            setattr(mod, f"_iwm_cast_{src}", cached)
        return install

    def _install_producer_shim(self):
        """Make FP32LayerNorm read its fp32 params through a producer, so a pass can wrap it.

        Without this the cast is buried inside `F.layer_norm(x.float(), ..., self.weight.float())`
        and there is nothing for a generic pass to hold on to. Exposing it is the adapter's job:
        the model must present a rewritable surface before anything can rewrite it.
        """
        import modules.model as M
        LN = M.FP32LayerNorm
        if getattr(LN, "_iwm_producer_shim", False):
            return LN
        import torch.nn.functional as F
        _orig = LN.forward

        def forward(self, inputs):
            wc = getattr(self, "_iwm_cast_weight", None)
            bc = getattr(self, "_iwm_cast_bias", None)
            if wc is None and bc is None:
                return _orig(self, inputs)
            w = wc() if wc is not None else (None if self.weight is None else self.weight.float())
            b = bc() if bc is not None else (None if self.bias is None else self.bias.float())
            return F.layer_norm(inputs.float(), self.normalized_shape, w, b,
                                self.eps).to(inputs.dtype)

        LN.forward = forward
        LN._iwm_producer_shim = True
        return LN

    def _extent(self) -> int:
        blocks = getattr(self.model, "blocks", []) or []
        if not blocks:
            return 0
        sig = getattr(blocks[0].attn1, "_iwm_ring_signature", None)
        s = sig(self.cache_name) if sig else None
        return 0 if not s else s[0] * 100003 + s[1]     # fold (start, count) into one int

    # -- APPLY ------------------------------------------------------------------------------
    def apply(self, rewrite):
        from instinctwm.passes.interface import RewriteKind

        if rewrite.site_id == "lingbot.block_stack" and rewrite.kind is RewriteKind.WRAP:
            self._wrapped["block_stack"] = rewrite.payload(self._raw_stack)
            return
        if rewrite.site_id.startswith("lingbot.cast[") and rewrite.kind is RewriteKind.WRAP:
            spec = self._cast_sites.get(rewrite.site_id)
            if spec is None:
                raise KeyError(f"unknown cast site {rewrite.site_id}")
            produce, install = spec
            install(rewrite.payload(produce))
            return
        raise NotImplementedError(f"lingbot surface cannot apply {rewrite}")

    def _raw_stack(self, hidden, encoder, tproj, rot):
        x = hidden
        for b in self.model.blocks:
            x = b(x, encoder, tproj, rot, update_cache=0, cache_name=self.cache_name)
        return x

    def stack(self, hidden, encoder, tproj, rot):
        fn = self._wrapped.get("block_stack", self._raw_stack)
        return fn(hidden, encoder, tproj, rot)

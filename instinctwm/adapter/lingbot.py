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

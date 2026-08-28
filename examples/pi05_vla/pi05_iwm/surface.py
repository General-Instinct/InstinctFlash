"""The Layer-5 surface for pi05: where the generic passes are allowed to act on this model.

A checkpoint adapter says WHAT pi05 is (`adapter.py`). This says WHERE a pass may cut into it. The
split matters because `instinctflash/passes/graph_capture.py` is already generic -- it keys graphs on a
structural signature, binds inputs to stable addresses, refuses regions that mutate host state, and
falls back to eager when it refuses. None of that needed writing again. What it cannot know is which
callable of pi05's is the capture unit, and how to flatten a `DynamicCache`. That is what is here.

WHY pi05 IS CAPTURABLE AND LingBot-VA IS NOT

Not a better runtime -- a different model, and `AdapterSpec.shapes_static_across_cycles()` derives it
from the declaration rather than taking anyone's word:

    LingBot-VA   EPISODE-lifetime streams, ring KV grows 152 slots/cycle, saturates near cycle 64.
                 Real RoboTwin episodes are a median of 15.6 cycles, so no episode ever reaches a
                 shape-stable state and capture measured 1.43x SLOWER from recapture alone.
    pi05         one CHUNK-lifetime prefix, rebuilt every chunk. The prompt is padded to 200 tokens
                 whatever the text (measured across a 4-char and a 130-char prompt), the suffix is
                 always chunk_size=50. Shapes repeat forever.

WHAT ACTUALLY BLOCKED IT, AND IT WAS NOT THE ARCHITECTURE

Capture failed with "Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the
CPU tensor is pinned". `embed_suffix` builds `torch.tensor([1] + [0] * 49, device=cuda)` from a host
list on every denoise step -- a constant, rebuilt ten times per chunk, and building a CUDA tensor
from a Python list is an unpinned H2D copy. One redundant constant made a 2187-launch,
99.9%-submit-bound loop uncapturable.

`denoise_step` also re-assigns `_attn_implementation = "eager"` every call. The purity gate compares
host state by VALUE, so pre-establishing it once means the same assignment is a no-op diff and the
region tests pure -- which is the honest way to pass that gate rather than declining to run it.

MEASURED ON ONE H100, pi05_base, 50-step chunk, AND THE ANSWER IS THAT CAPTURE DOES NOT SHIP:

    upstream                     16.19 ms/denoise step,  302 ms/chunk
    hoisting the constants       15.84 ms/step   BITEXACT, no win alone -- it is only the enabler
    + captured replay             5.01 ms/step   1.55x on the chunk, and WRONG

The region is not replay-safe. `denoise_step` calls `clone_past_key_values` and the forward then
appends the 50 suffix entries to that clone, so the region CREATES and mutates a `DynamicCache` on
every call. Replay does not re-run Python, and the result is correct only for the exact input the
capture was taken from:

    replayed with the captured input                  max |d| 0.000e+00
    replayed with a new x_t, same cache object        max |d| 2.116e-01
    replayed with a new cache of identical values     max |d| 2.036e+00     (v_t scale 4.27)

Three separate things said this was fine before that was measured. Capture succeeded. The host-effect
gate passed -- and could not have failed, because it snapshots state reachable from the declared roots
and this cache does not exist yet when the snapshot is taken. And a per-step check read 0.000e+00,
because it compared replay against eager at the operating point the graph was captured from, which is
exact by construction and therefore evidence of nothing.

End to end it looked better still: the first chunk was bit-exact across all 50 actions. That was not
capture working. It was capture NOT yet installed -- the first chunk ran eager while the graph was
being taken, and every chunk after it replayed and was wrong by ~2.1 against an action scale of 0.5.

The engine pass now compares replay against eager on the second, DIFFERENT input and discards the
graph, so pi05 runs eager and returns upstream's actions exactly. The site stays published: the
rejection is a measurement about this region in this version of lerobot, and a preallocated cache of
prefix+chunk extent -- the serving engine's approach (serving/), where a max-sized buffer is what makes capture valid -- is
the thing that would make it replay-safe.

RESOLVED, AND THE FIX IS THE ONE THE REJECTION NAMED. `static_capture.py` replaces the per-step
DynamicCache clone+append with one K/V buffer per layer of extent prefix+suffix: prefix slots are
rewritten once per chunk OUTSIDE the graph, suffix slots are overwritten INSIDE it at fixed
addresses, and attention always runs over the constant full extent under the 4D mask pi05 already
builds for exactly that width. Nothing in the region allocates and no Python container exists to
go stale. Measured on the same H100, same checkpoint (`verify_static_capture.py`):

    replay vs eager, captured input            max |d| 0.000e+00
    replay vs eager, 3 new (x_t, t)            max |d| 0.000e+00   (was 2.116e-01)
    replay vs eager, NEW PROMPT, refilled      max |d| 0.000e+00   (was 2.036e+00)
    denoise step                               16.25 -> 4.57 ms    3.55x
    chunk (prefill + 10 steps)                 298.7 -> 181.3 ms   1.65x

BITEXACT on inputs the capture never saw, which is the tier the DynamicCache version could not
reach at any speed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from instinctflash.executors.binding import TreeBinder
from instinctflash.passes.interface import RewriteKind, Site, SiteKind

CAPTURE_UNIT_ID = "pi05.denoise_step"


class Pi05CacheBinder:
    """`TreeBinder`, plus the one container it cannot walk: transformers' `DynamicCache`.

    The default walk treats an unknown object as opaque and keys it by identity, which is the right
    conservative default and exactly wrong here: pi05 allocates a FRESH cache every chunk, so an
    identity key would miss on every chunk and recapture 20 times a second. The cache is really just
    2N tensors plus a sliding-window flag per layer, so it flattens like any other container -- and
    once it does, the pass's own stable-buffer machinery makes the per-chunk prefill land in fixed
    addresses without pi05 having to preallocate anything itself.
    """

    TAG = "pi05.dyncache"

    def __init__(self):
        self._tree = TreeBinder()

    @staticmethod
    def _is_cache(value) -> bool:
        return type(value).__name__ in ("DynamicCache", "Cache") and hasattr(value, "__iter__")

    def flatten(self, value):
        if not self._is_cache(value):
            return self._tree.flatten(value)
        leaves, windows = [], []
        for keys, values, sliding_window in value:
            leaves.extend((keys, values))
            windows.append(sliding_window)
        return leaves, (self.TAG, tuple(windows))

    def unflatten(self, leaves, spec):
        if not (isinstance(spec, tuple) and len(spec) == 2 and spec[0] == self.TAG):
            return self._tree.unflatten(leaves, spec)
        from transformers.cache_utils import DynamicCache
        windows = spec[1]
        return DynamicCache(tuple((leaves[2 * i], leaves[2 * i + 1], windows[i])
                                 for i in range(len(windows))))


class Pi05Surface:
    """pi05 as a set of sites a pass may act on. Two methods, per `AdapterSurface`."""

    model_id = "pi05"

    def __init__(self, flow_model):
        self._m = flow_model
        # THE ORIGINAL, bound now. `install()` replaces `type(model).denoise_step`, so a `_raw_denoise`
        # that went through the attribute would call the wrapper that calls it -- and it did: capture
        # recursed 164 frames into a RecursionError inside the purity gate, because the gate runs the
        # raw callable to check it, and the raw callable was no longer raw.
        self._orig_denoise = type(flow_model).denoise_step
        self._wrapped = None
        self.hoisted: list[str] = []
        self._const: dict = {}

    # -- prerequisites ---------------------------------------------------------------------------
    def hoist_loop_constants(self) -> list[str]:
        """Make the denoise step capturable, bit-exactly. Must run before the pass.

        Both of these establish a CONSTANT on every step. Neither is an optimization on its own --
        measured at 15.84 ms against 16.19 ms, which is noise -- and together they are the only
        reason capture is legal at all.
        """
        m = self._m
        for cfg in (m.paligemma_with_expert.paligemma.model.language_model.config,
                    m.paligemma_with_expert.gemma_expert.model.config):
            cfg._attn_implementation = "eager"        # pre-establish: makes the host diff empty
        self.hoisted.append("attn_implementation pre-established (was re-assigned per step)")

        const = self._const

        def embed_suffix(self_m, noisy_actions, timestep):
            from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
            time_emb = create_sinusoidal_pos_embedding(
                timestep, self_m.action_in_proj.out_features,
                min_period=self_m.config.min_period, max_period=self_m.config.max_period,
                device=timestep.device)
            time_emb = time_emb.type(dtype=timestep.dtype)
            action_emb = self_m._apply_checkpoint(lambda a: self_m.action_in_proj(a), noisy_actions)

            def time_mlp(t):
                return F.silu(self_m.time_mlp_out(F.silu(self_m.time_mlp_in(t))))

            time_emb = self_m._apply_checkpoint(time_mlp, time_emb)
            bsize, action_time_dim = action_emb.shape[:2]

            key = (action_emb.dtype, str(action_emb.device), bsize, action_time_dim)
            if key not in const:
                pad = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
                att = torch.tensor([1] + [0] * (self_m.config.chunk_size - 1),
                                   dtype=action_emb.dtype, device=action_emb.device)
                const[key] = (pad, att[None, :].expand(bsize, att.shape[0]))
            pad_masks, att_masks = const[key]
            return action_emb, pad_masks, att_masks, time_emb

        type(self._m).embed_suffix = embed_suffix
        self.hoisted.append("pad_masks + att_masks cached (att_masks was an unpinned H2D copy from "
                            "a Python list -- the capture blocker)")
        return list(self.hoisted)

    # -- AdapterSurface --------------------------------------------------------------------------
    #: Publish the denoise step as capturable for the GENERIC engine pass. OFF, on evidence --
    #: see the module docstring. IFL_PI05_CAPTURE=1 still flips the site attr so the negative
    #: result stays reproducible through `run_pass` (the measurement is the useful artifact),
    #: but `Pi05Adapter.install` no longer routes through this site at all: the DynamicCache
    #: region is retired from serving, and the flag is a no-op-with-notice there.
    #:
    #: The DynamicCache decision, stated plainly: replay is 1.53x on the chunk and the actions are
    #: wrong (1.139e-02 end-to-end delta after the engine pass's replay check; not bit-exact, no
    #: non-inferiority evidence, no tier under which it can ship). The SHIPPABLE capture is the
    #: static-KV path in `static_capture.py`: bitexact on unseen inputs and prompts, 3.55x on the
    #: denoise step, 1.65x on the chunk -- see its module docstring for the gate numbers. It is
    #: the DEFAULT for pi05-class checkpoints on capture-capable devices, gated per process by
    #: the post-capture bit-exact self-check; IFL_PI05_NO_CAPTURE=1 is the kill-switch and
    #: IFL_PI05_STATIC_CAPTURE=1 (the old opt-in) is a no-op-with-notice.
    CAPTURE_OPT_IN = "IFL_PI05_CAPTURE"
    STATIC_CAPTURE_OPT_IN = "IFL_PI05_STATIC_CAPTURE"

    def sites(self, kind):
        import os

        if kind is SiteKind.CAPTURE_UNIT:
            yield Site(
                kind=kind, id=CAPTURE_UNIT_ID,
                attrs={"capturable": os.environ.get(self.CAPTURE_OPT_IN) == "1",
                       "binder": Pi05CacheBinder(),
                       # the whole flow-matching module: let the gate watch everything pi05 could
                       # mutate, rather than a subset chosen to make the gate pass
                       "effect_roots": (self._m,),
                       "arity": 4,
                       "note": "(prefix_pad_masks, past_key_values, x_t, timestep) -> v_t; "
                               "shapes static because the prompt is padded to a fixed budget"})
        # EXECUTION_REGION / STATE_ADDRESSING / ALLOCATION: pi05 publishes none. It keeps no
        # resident KV pool -- the prefix cache is passed in and discarded per chunk -- so a pass
        # asking for those gets an honest empty answer instead of a symbol that does not exist.

    def apply(self, rewrite):
        if rewrite.site_id != CAPTURE_UNIT_ID or rewrite.kind is not RewriteKind.WRAP:
            raise NotImplementedError(f"the pi05 surface cannot apply {rewrite}")
        self._wrapped = rewrite.payload(self._raw_denoise)

    # -- the unit itself -------------------------------------------------------------------------
    def _raw_denoise(self, prefix_pad_masks, past_key_values, x_t, timestep):
        return self._orig_denoise(self._m, prefix_pad_masks=prefix_pad_masks,
                                  past_key_values=past_key_values, x_t=x_t, timestep=timestep)

    def install(self) -> bool:
        """Route pi05's own `denoise_step` through whatever the passes left behind."""
        if self._wrapped is None:
            return False
        surface = self

        def denoise_step(self_m, prefix_pad_masks, past_key_values, x_t, timestep):
            return surface._wrapped(prefix_pad_masks, past_key_values, x_t, timestep)

        type(self._m).denoise_step = denoise_step
        return True

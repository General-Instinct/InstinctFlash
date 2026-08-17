"""ConditioningPrefill — hoist episode-constant conditioning out of the denoise loop.

This is loop-invariant code motion applied to a denoise loop, and on LingBot-VA it is the
largest single arithmetic win available.

The measurement
---------------
`model.py:331` gives the KV cache to self-attention and withholds it from cross-attention::

    self.attn_caches = {} if cross_attention_dim_head is None else None

So on every forward, in all 30 blocks, `WanAttention.forward` recomputes the text K/V from
scratch (`model.py:426`, `to_k`/`to_v` over the 512-token text embedding). The operands trace
back only to `input_dict["text_emb"]`, which is `self.prompt_embeds` — assigned at exactly two
lines in the whole tree, both inside `_reset` (`wan_va_server.py:424`, `:426`). It is
episode-constant, and it is recomputed 77 times per control step.

The cost is lopsided because the *query* is short and the *text* is not (D=3072, 30 layers,
batch 2 under CFG):

| forward | query rows | cross K/V GF/layer | other GF/layer | cross K/V share |
|---|---|---|---|---|
| video (240 tok) | 480 | 38.65 | 138.92 | 21.8% |
| action (32 tok) | 64 | 38.65 | 18.52 | **67.6%** |

Per control cycle: 26 video x 5.33 TFLOP + 51 action x 1.72 TFLOP = 226 TFLOP total, of which
**89 TFLOP (39%) is recomputing a constant.**

Cost to cache: 30 layers x (K + V) x [2, 512, 24, 128] bf16 = **360 MiB** resident, ~5% on top
of the 6.72 GiB self-attention pool.

Why it is BITEXACT
------------------
The cached tensors are the identical values the model would recompute from an unchanged input,
produced by the same kernels in the same order. The only removed ops are `to_k`/`to_v`/`norm_k`
on a constant. Two structural preconditions hold and are asserted at install time, because if
either were false the cache would be wrong rather than slow:

  * `rotary_emb is None` for cross-attention (`model.py:552` passes `None`), so no
    position-dependent transform is applied to the cached K/V.
  * `update_cache == 0` for cross-attention (`model.py:553`), so the paged self-attention cache
    path is not taken and there is nothing to roll back.

Do NOT key the cache on `text_emb.data_ptr()`: the server `.clone()`s a fresh tensor every
forward (`wan_va_server.py:257`, `:290`, `:314`), so the pointer changes while the value never
does. Key on the episode instead.

Invalidation
------------
`_reset` is the only writer of `prompt_embeds`, and it also recomputes `use_cfg` (the only thing
that can change the batch dimension) — so `_reset` releases and repopulates. `_compute_kv_cache`
calls `clear_pred_cache`, which clears *self*-attention provisional slots only; the text cache
must be RETAINED there. That distinction is explicit here rather than incidental, mirroring
vLLM-Omni's `retain_cross_attention` vs `release_cross_attention` split
(`ar_diffusion/kv_cache/manager.py`), which is the productized form of this and worth adopting.
"""

from __future__ import annotations

from instinctwm.adapters.base import AdapterSpec, KVLifetime
from instinctwm.descriptors.deployment import DeploymentSpec
from instinctwm.planners.planner import PassResult, Tier


class ConditioningPrefill:
    """Fires whenever an adapter declares any episode- or chunk-scoped conditioning purity."""

    name = "conditioning_prefill"

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        hoistable = [
            p for p in spec.purity
            if p.scope in (KVLifetime.EPISODE, KVLifetime.CHUNK)
        ]
        if not hoistable:
            return PassResult(self.name, False, Tier.BITEXACT,
                              "model declares no episode- or chunk-constant conditioning")

        n_fwd = spec.total_forwards()
        return PassResult(
            name=self.name,
            applies=True,
            tier=Tier.BITEXACT,
            reason=(
                f"{[p.artifact for p in hoistable]} declared pure in "
                f"{sorted({p.scope.value for p in hoistable})} scope, but recomputed on all "
                f"{n_fwd} forwards per control step ({spec.forwards_breakdown()})"
            ),
            params={
                "artifacts": [p.artifact for p in hoistable],
                "retain_across": ["kv_commit"],   # clear_pred_cache must NOT drop it
                "release_on": ["reset"],
            },
            # ATTRIBUTED, because this pass is genuinely generic and now fires on model families
            # these numbers were never measured on. Planning a pi05 VLA produced a plan quoting
            # "89 of 226 TFLOP" and "360 MiB resident" as its expectation -- LingBot-VA's figures,
            # presented as a prediction for a different architecture with a different layer count and
            # a different conditioning artifact. A cross-model expectation has to name the model it
            # came from or it reads as a forecast.
            expected_win=(
                f"hoists {len(hoistable)} artifact(s) out of the denoise loop. Measured ON "
                f"LingBot-VA: 89 of 226 TFLOP per control cycle (39%), 67.6% of an action forward's "
                f"layer FLOPs, 360 MiB resident. Those figures are that model's; for any other "
                f"backbone the FLOP share and the residency cost follow its own layer count and "
                f"artifact size. Launch-bound share means the wall-clock win is less than the FLOP "
                f"win either way — measure it"
            ),
        )

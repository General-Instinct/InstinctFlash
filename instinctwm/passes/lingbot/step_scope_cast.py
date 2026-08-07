"""P008 candidate: hoist the timestep-modulation cast from LAYER scope to STEP scope.

THE SCOPE ERROR, measured rather than inferred (LAYER5_CAST_FAMILY.md):

    # model.py:524, inside WanTransformerBlock.forward -- runs once per block
    temb_scale_shift_table = self.scale_shift_table[None] + temb.float()

`temb` is `timestep_proj`, computed ONCE per forward at model.py:858-859 and passed unchanged into all
30 blocks. So `temb.float()` produces 30 identical fp32 tensors per forward and discards 29 of them:
300 casts per cycle where 10 would do. Measured 4.69 ms/cycle at that callsite, 290 of 300 calls
removable, ~1.4% of a 330 ms cycle.

DELIBERATELY NARROW. This is NOT a generic `StepInvariantCastHoisting` pass. The family analysis
checked all 22 `_to_copy` callsites and found exactly one hoistable: 21 of 22 cast a genuinely
different value every call, because a block's activations vary by definition. An abstraction over one
instance would be speculation. The generic legality rule is written down in LAYER5_CAST_FAMILY.md so a
second backbone can be tested against it cheaply; generalize with two data points, not one.

WHY IT IS BITEXACT BY CONSTRUCTION. `.float()` on an fp32 tensor returns `self` -- no copy, no
arithmetic. So pre-casting `timestep_proj` once and handing the fp32 tensor to every block makes each
block's `temb.float()` a no-op returning the identical object. Every block then reads exactly the bits
it read before. There is no rounding to reproduce because nothing new is computed.

THE FOUR LEGALITY CONDITIONS, checked rather than assumed:

  1. STEP strictly encloses LAYER.                                    structural, holds by definition
  2. The value is invariant across the outer scope.                    MEASURED: 1.0 distinct value per
     forward against 30 calls (probe_cast_lifetime.py). Note this was established by VALUE identity,
     not storage identity -- a storage digest overstates redundancy 15x because the caching allocator
     recycles addresses.
  3. Purity: no effectful op between the hoisted evaluation and its consumers.
     `temb` appears EXACTLY ONCE in the block body, at line 524. Verified by inspection of the whole
     forward: there is no other reference, so nothing can intervene.
  4. No consumer mutates the result. The single consumer is `scale_shift_table[None] + temb.float()`,
     an out-of-place add. One cast shared by 30 blocks is therefore safe; this is the condition a
     naive implementation gets wrong, and it is why this is a gated pass and not a one-line edit.

WHY A COUNTER AND NOT A VALUE CACHE. The obvious implementation memoises `temb.float()` keyed on
`(data_ptr, version, shape)`. That is exactly the trap the family analysis exposed: the caching
allocator hands the same address to later tensors, so a pointer-keyed cache can serve a stale cast. The
counter is bumped by the transformer's own forward, so it changes exactly once per forward and cannot
alias.
"""

from __future__ import annotations

from dataclasses import dataclass

from instinctwm.passes.contract import (
    Applicability,
    BenchResult,
    CostTerm,
    Discovery,
    HardwareReq,
    Tier,
    VerifyResult,
)


@dataclass
class _State:
    """Which forward we are in, and the fp32 cast made for it."""

    token: int = 0
    cast_for: int = -1
    value: object = None
    hoisted: int = 0        # casts avoided
    passthrough: int = 0    # calls that already had the right dtype


class StepScopeCastHoist:
    """Cast `timestep_proj` once per forward instead of once per block.

    Installs two wrappers, and needs both:
      * `WanTransformer3DModel.forward` bumps a per-forward token
      * `WanTransformerBlock.forward` replaces its `temb` argument with the fp32 cast for that token
    """

    name = "step_scope_cast_hoist"
    hardware = HardwareReq()
    cost_term = CostTerm.PER_STEP

    def __init__(self) -> None:
        self.state = _State()

    # ---- 1 + 2. detection and applicability --------------------------------------------------
    def applicability(self, spec, device) -> Applicability:
        """DECLARED, not auto-detected. The invariance was established by measurement on this
        backbone; asserting it for an arbitrary adapter would be exactly the over-generalisation the
        family analysis argued against."""
        return Applicability(
            True,
            "timestep conditioning is computed per forward and consumed per block; the cast is "
            "hoistable to STEP scope (measured: 1.0 distinct value per forward over 30 calls)",
            discovery=Discovery.DECLARED, cost_term=CostTerm.PER_STEP, claimed_tier=Tier.BITEXACT)

    def expected_delta_ms(self, spec, device) -> float:
        """29 of 30 casts per forward, at the measured per-call cost."""
        per_call_ms = 4.69 / 300.0
        blocks, forwards = 30, 10
        return per_call_ms * (blocks - 1) * forwards

    # ---- install -----------------------------------------------------------------------------
    def install(self, server_module, server_cls) -> list[str]:
        import importlib

        M = importlib.import_module("modules.model")
        Block = M.WanTransformerBlock
        Model = M.WanTransformer3DModel
        st = self.state

        if getattr(Block, "_iwm_step_scope_cast", False):
            return []

        orig_model_forward = Model.forward

        def model_forward(self, *a, **k):
            # One token per forward. The blocks key off this rather than off any tensor identity, so
            # allocator address reuse cannot produce a stale cast.
            st.token += 1
            return orig_model_forward(self, *a, **k)

        orig_block_forward = Block.forward

        def block_forward(self, hidden_states, encoder_hidden_states, temb, *a, **k):
            if temb is None:
                return orig_block_forward(self, hidden_states, encoder_hidden_states, temb, *a, **k)
            if temb.dtype is not __import__("torch").float32:
                if st.cast_for != st.token:
                    st.value = temb.float()          # the ONE cast for this forward
                    st.cast_for = st.token
                else:
                    st.hoisted += 1                  # a cast that did not happen
                temb = st.value
            else:
                st.passthrough += 1
            # `temb.float()` inside the block now returns `self`: no copy, identical bits.
            return orig_block_forward(self, hidden_states, encoder_hidden_states, temb, *a, **k)

        Model.forward = model_forward
        Block.forward = block_forward
        Block._iwm_step_scope_cast = True
        return [self.name]

    def stats(self) -> dict:
        return {"casts_avoided": self.state.hoisted,
                "already_fp32": self.state.passthrough,
                "forwards": self.state.token}

    # ---- 3 + 4. gates ------------------------------------------------------------------------
    def verify(self, harness) -> VerifyResult:
        """BITEXACT or nothing. A no-op cast cannot perturb a single bit, so a nonzero delta here
        means a legality condition is violated -- most likely condition 4, a consumer mutating the
        shared tensor -- and the pass must be withdrawn rather than re-tiered."""
        d = harness.max_abs_action_delta()
        return VerifyResult(
            passed=(d == 0.0),
            tier_achieved=Tier.BITEXACT if d == 0.0 else Tier.NUMERIC,
            max_abs_delta=d,
            detail="hoisting a cast to a scope where the value is invariant cannot change any bit; "
                   "a nonzero delta means the invariance or the read-only assumption is false")

    def benchmark(self, harness) -> BenchResult:
        return harness.latency_ab(self.name)

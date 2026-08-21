"""Post-saturation Plan Buffer: one captured graph surviving ring advancement. **NOT SHIPPED.**

VERDICT FIRST. Every correctness gate passes. The latency gate says do not ship.

    byte-identical writes   index_copy_ vs slice assignment: 0 differing words across every pool
                            shape, key size and offset tested
    action-level exactness  max|delta action| = 0 over 45 seeded cycles SPANNING saturation --
                            every cycle including the transition at 36
    reset isolation         two identical runs with a mid-run reset agree bit-for-bit; the ring
                            restarts unsaturated so the plan path disengages
    capture count           270 -> 238 over 45 cycles, 12% fewer
    EPISODE-MODE ABBA       post-saturation cycle, three configurations:
                                capture OFF (shipped)          351.4 ms
                                capture ON, no plan buffer      936.1 ms
                                capture ON + plan buffer        503.5 ms
                            The fix recovers 432 of the 585 ms capture penalty and is still 1.43x
                            SLOWER than not capturing at all. Graph capture remains unprofitable at
                            the Fast operating point even with ring progression removed from the key.

WHAT THIS MEANS FOR THE HOST-DISPATCH HYPOTHESIS, which is the more important outcome. The
critical-path analysis predicted 1.72x from graph replay removing 94.3% of the cycle's dispatcher
operations. Replay does not deliver it. Two explanations remain and they are distinguishable:

  (a) the surviving recaptures swamp the benefit. 238 captures over 45 cycles is still 5.3/cycle at
      ~111 ms each, and only 32 of the ~48 theoretically removable ones went away. The video ring
      saturates; the action stream is declared `kv_lifetime: cycle` and may never saturate, so its
      keys keep changing and its graphs keep re-capturing.
  (b) replay does not actually remove the host cost attributed to it -- the launch path has its own
      per-replay overhead, or the 94.3% figure counts ops that replay does not subsume.

Distinguishing them needs one measurement: dispatcher-op count per cycle with capture ON and the plan
buffer installed. If it falls ~94% and the cycle still regresses, (a) is the answer and the remaining
recaptures are the target. If it does not fall, (b) is, and the critical-path model needs revision.
That measurement has not been made, so the 1.72x prediction stands neither confirmed nor refuted.

WHAT IS WORTH KEEPING. The mechanism is correct and gated, and the bug it exposed is the instructive
part: buffers are keyed (cache_name, key_size) because their LENGTH differs -- 240 for a video write,
32 for an action write -- but `head` depends only on ring state. Refreshing only the buffer belonging
to the commit that just ran left the other holding a stale head, and the next write of the other size
landed on live slots. The gate localised it exactly: cycles 0-35 exact, 36-44 wrong, max|delta| 0.453.
One buffer refreshed is not "the buffer refreshed".

WHY THIS IS A PASS AND NOT AN EDIT. `ring_kv` (P003) and `graph_block_stack` (P005) are both frozen. The
capability they need -- writing KV through a device-resident index -- is added to ring_kv as an OPT-IN
that defaults off and is byte-identical to v1.0.0 when off. This pass is the thing that opts in, so the
change is inert until something asks for it and the freeze protocol is satisfied by a version bump plus a
gate re-run rather than by an unannounced behavioural change.

WHAT IT DOES, and the scope is deliberately one line of state:

    once `count == total`, the write offset stops being baked into the graph.

Nothing else changes. The read path is untouched -- post-saturation it was already `key_all = kp`, the
whole pool. Pre-saturation is untouched entirely, because there `count` is the read EXTENT and no graph
can absorb a changing shape. There is no count bucketing and no padded attention: those would be NUMERIC
and are out of scope.

Measured basis (LAYER5_GRAPH_PERSISTENCE.md): the ring saturates at cycle 36 of a ~53-cycle episode;
94.3% of the cycle's 38,442 dispatcher operations occur inside the region a replay covers; a fully
replayed cycle should reach the device-bound floor of 196 ms, 1.72x.

STATUS: NEGATIVE RESULT
Every correctness gate passes and the latency gate refuses it: 503.5 ms against
351.4 ms with capture off -- 1.43x SLOWER. The plan buffer recovers 432 of the 585 ms capture penalty and
it does not matter, because 5.3 surviving captures at ~111 ms each exceed the whole cycle.
See HISTORICAL.md.
"""

from __future__ import annotations

from instinctflash.passes.contract import (
    Applicability,
    BenchResult,
    CostTerm,
    Discovery,
    HardwareReq,
    Tier,
    VerifyResult,
)


class PersistentRingGraph:
    """Turn on the Plan Buffer write path, which lets the graph key drop `start` when saturated."""

    name = "persistent_ring_graph"
    hardware = HardwareReq(requires=frozenset({"cuda_graphs"}))
    cost_term = CostTerm.FIXED      # it removes host dispatch, not per-forward device work

    def applicability(self, spec, device) -> Applicability:
        return Applicability(
            True,
            "post-saturation the KV read is the whole pool and the write offset is the only ring "
            "state a captured graph bakes; routing the write through a device-resident index makes "
            "the graph independent of ring position",
            discovery=Discovery.DECLARED, cost_term=CostTerm.FIXED, claimed_tier=Tier.BITEXACT)

    def expected_delta_ms(self, spec, device) -> float:
        # 94.3% of 38,442 dispatcher ops at ~8.8 us, on post-saturation cycles only.
        return 0.943 * 38442 * 8.8 / 1000.0

    def install(self, server_module, server) -> list[str]:
        blocks = list(server.transformer.blocks)
        if not blocks:
            return []
        a0 = blocks[0].attn1
        if not hasattr(a0, "_iwm_plan_buffer"):
            print("InstinctFlash persistent-graph: ring_kv does not expose _iwm_plan_buffer; "
                  "NOT INSTALLED (needs P003 v1.1.0 or later)", flush=True)
            return []
        n = 0
        for blk in blocks:
            blk.attn1._iwm_use_plan_buffer = True
            n += 1
        print(f"InstinctFlash persistent-graph: Plan Buffer write enabled on {n} layers; the graph key "
              f"drops `start` once count == total", flush=True)
        return [self.name]

    def verify(self, harness) -> VerifyResult:
        d = harness.max_abs_action_delta()
        return VerifyResult(
            passed=(d == 0.0),
            tier_achieved=Tier.BITEXACT if d == 0.0 else Tier.NUMERIC,
            max_abs_delta=d,
            detail="index_copy_ with a contiguous ascending index writes the same bytes to the same "
                   "slots as the slice assignment, so a nonzero delta means the Plan Buffer contents "
                   "were stale at replay -- a refresh-ordering bug, not a numerical one")

    def benchmark(self, harness) -> BenchResult:
        return harness.latency_ab(self.name)

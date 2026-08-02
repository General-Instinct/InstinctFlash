"""Released passes. Frozen: change only for a correctness bug.

A frozen pass is one whose behaviour other measurements are calibrated against. The restated
baseline in `RESULTS.md` section 8 was produced by exactly this set at exactly these settings, so
editing one silently invalidates every later comparison. Performance tuning is not a reason to
touch them; a correctness bug is.

To change a frozen pass: bump its version, re-run its gates, and re-run the full restated table
(`probe_latency.py --repeats 3` across all cumulative configs). If the table moves, every number
downstream of it moves too.
"""

from __future__ import annotations

from dataclasses import dataclass

from instinctwm.optimizer.contract import Tier


@dataclass(frozen=True)
class Released:
    pid: str
    name: str
    version: str
    tier: Tier
    step_speedup: float
    gates: str
    frozen: bool = True


RELEASED = (
    Released(
        pid="P001", name="substrate_elision", version="1.0.0", tier=Tier.BITEXACT,
        step_speedup=2.11,
        gates="max|delta action| = 0 over 6 paired seeded cycles; "
              "removes FSDP-at-world-size-1, per-chunk empty_cache, blocking debug dumps"),
    Released(
        pid="P002", name="conditioning_prefill", version="1.0.0", tier=Tier.BITEXACT,
        step_speedup=1.12,
        gates="max|delta action| = 0 over 6 paired seeded cycles; caches episode-constant "
              "cross-attention K/V for all 30 layers (+360 MiB), removes 89 of 226 TFLOP/cycle"),
    Released(
        pid="P004", name="hoist_invariant_casts", version="1.0.0", tier=Tier.BITEXACT,
        step_speedup=1.02,
        gates="max|delta action| = 0 over 8 paired seeded cycles; casts FP32LayerNorm weight/bias "
              "and the block scale_shift_table once per episode instead of once per forward, "
              "removing 7,110 casts of a constant per control cycle. Cost model predicted 47.4 ms, "
              "measured 49.7 ms (6% error)"),
    Released(
        pid="P005", name="graph_block_stack", version="1.0.0", tier=Tier.BITEXACT,
        step_speedup=1.38,
        gates="max|delta action| = 0 over 6 paired seeded cycles, verified with the gate run AFTER "
              "an episode reset (the ordering that exposed a nan); 2539.9 -> 1842.0 ms under "
              "probe_latency --repeats 3, spread 0.5%. Runs the 30-block stack from a captured "
              "CUDA graph: per-op dispatch 6.2 us (83.6% of it cudaLaunchKernel) becomes ~1.17 us "
              "replay. Requires P003, whose slice addressing is what makes the stack capturable "
              "at all -- a stock block raises cudaErrorStreamCaptureInvalidated"),
    Released(
        pid="P003", name="ring_kv_addressing", version="1.0.0", tier=Tier.BITEXACT,
        step_speedup=1.40,
        gates="max|delta action| = 0 over 40 cycles past the wrap at ~36; 800/800 allocator "
              "parity checks across 5.6 full wraps; 3/3 bitwise-identical action streams on "
              "put_bottles_dustbin (1700 steps, ~53 cycles/episode)"),
)

#: The measured chain these produce together, under `probe_latency.py --repeats 3`.
BASELINE = {
    "stock": 8431.5,
    "P001": 3994.0,
    "P001+P002": 3567.5,
    "P001+P002+P003": 2553.9,
    "P001+P002+P003+P004": 2539.9,
    "P001+P002+P003+P004+P005": 1842.0,
    "cumulative_speedup": 4.58,
    #: P005 recaptures every episode because `_reset` reallocates the KV pool, so a captured graph
    #: would point at freed memory (measured: nan on episode 2). With graphs surviving resets the
    #: same build measured 1208.2 ms -- that is the size of the prize for E1 (a stable arena), and
    #: it is NOT claimable today because it is only correct with a stable pool.
    "P005_with_persistent_pool_projected": 1208.2,
    "protocol": "probe_latency.py --cycles 10 --repeats 3; first run discarded; "
                "all spreads <= 0.7% (P005 arm: 0.5%)",
}


def summary() -> str:
    out = ["Released passes (frozen)"]
    for r in RELEASED:
        out.append(f"  {r.pid} {r.name:22s} v{r.version}  {r.tier.name:9s} "
                   f"{r.step_speedup:.2f}x step")
    out.append(f"  chain: {BASELINE['stock']:.0f} -> {BASELINE['P001+P002+P003+P004+P005']:.0f} ms "
               f"= {BASELINE['cumulative_speedup']:.2f}x")
    return "\n".join(out)

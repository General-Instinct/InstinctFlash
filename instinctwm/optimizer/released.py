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
        pid="P006", name="stable_state_pools", version="1.0.0", tier=Tier.BITEXACT,
        step_speedup=1.52,
        gates="max|delta action| = 0 over 6 paired seeded cycles run AFTER 5 episode resets; "
              "probe_reset_isolation = 0 (episode 2 bitwise identical to a fresh episode); "
              "1842.0 -> 1211.3 ms, spread 0.0%. Reset clears logical KV state in place instead "
              "of reallocating, so P005's graphs survive -- gated by a runtime pointer "
              "certificate that fails closed",
        ),
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
    "P001+P002+P003+P004+P005+P006": 1211.3,
    "cumulative_speedup": 6.96,
    #: P006 delivered the recapture gap P005 left open: graphs now survive resets.
    "protocol": "probe_latency.py --cycles 10 --repeats 3; first run discarded; "
                "all spreads <= 0.7% (P005 arm: 0.5%)",
    #: EPISODE MODE (probe_episode.py, 45 cycles, ONE reset). probe_latency resets between
    #: repeats, which rewinds the ring to (0,0) so every repeat replays the keys the discarded
    #: first run captured. That hides any per-cycle cost that depends on ring position -- and
    #: graph capture has one, because the graph key contains (start, count).
    #:
    #: Restated under episode mode, current default:
    #:                          pre-saturation   post-saturation   whole episode
    #:   with graph capture        2925.4 ms        2302.8 ms        2800.8 ms
    #:   without graph capture     3572.5 ms        2710.5 ms        3400.1 ms
    #:
    #: So graph capture is still a NET WIN (1.18x post-saturation, 1.21x whole episode) but far
    #: from the 1.5-2x probe_latency implied. Captures never stop: 6.0/cycle even after the pool
    #: saturates, 92.5% cache hit rate.
    #:
    #: OUTSTANDING: the stock and intermediate rows above are all probe_latency-protocol and have
    #: NOT been restated in episode mode, so `cumulative_speedup` is not an episode-mode number.
    "episode_mode": {
        "protocol": "probe_episode.py --cycles 45 (one reset, ring never rewound)",
        "default_whole_episode_ms": 2800.8,
        "default_post_saturation_ms": 2302.8,
        "no_graph_whole_episode_ms": 3400.1,
        "no_graph_post_saturation_ms": 2710.5,
        "captures_per_cycle_after_saturation": 6.0,
        "graph_cache_hit_rate": 0.92457,
        #: The FULL chain, 45 cycles, one reset, all six rungs. THIS is the long-horizon number.
        #: `cumulative_speedup` above is probe_latency-protocol and overstates by 2.13x.
        "chain_whole_episode_ms": {
            "stock": 9585.1,
            "P001": 5260.6,
            "P001+P002": 5107.8,
            "P001+P002+P003": 3330.8,
            "+generic_passes": 3588.1,       # REGRESSION vs P003 without capture; see below
            "+graph_capture(default)": 2832.1,
        },
        "chain_post_saturation_ms": {
            "stock": 9486.3, "P001": 5195.1, "P001+P002": 5059.1,
            "P001+P002+P003": 2635.7, "+generic_passes": 2966.7,
            "+graph_capture(default)": 2298.7,
        },
        "cumulative_speedup_episode": 3.38,
        #: The generic passes cost ~330 ms post-saturation WITHOUT capture (2966.7 vs 2635.7).
        #: The adapter's rewritable-site shims add ~2,370 Python calls per cycle, which capture
        #: hides and eager does not. Confirm sequentially before acting on it.
        "generic_pass_eager_regression_ms": 331.0,
        "evictions_per_episode": 204,
    },
}


def summary() -> str:
    out = ["Released passes (frozen)"]
    for r in RELEASED:
        out.append(f"  {r.pid} {r.name:22s} v{r.version}  {r.tier.name:9s} "
                   f"{r.step_speedup:.2f}x step")
    # Episode mode leads, because it is the protocol that describes a real episode. probe_latency
    # resets between repeats, which rewinds the ring and hides per-cycle recapture; it overstated
    # this chain by 2.13x.
    e = BASELINE["episode_mode"]
    ch, cp = e["chain_whole_episode_ms"], e["chain_post_saturation_ms"]
    out.append("  EPISODE MODE (45 cycles, one reset) -- the reporting standard:")
    out.append(f"    whole episode  : {ch['stock']:.0f} -> {ch['+graph_capture(default)']:.0f} ms "
               f"= {e['cumulative_speedup_episode']:.2f}x")
    out.append(f"    post-saturation: {cp['stock']:.0f} -> "
               f"{cp['+graph_capture(default)']:.0f} ms")
    out.append(f"    captures {e['captures_per_cycle_after_saturation']:.1f}/cycle even AFTER "
               f"saturation, {e['evictions_per_episode']} evictions: the cache does NOT converge")
    out.append(f"  short-horizon (probe_latency, resets between repeats): "
               f"{BASELINE['stock']:.0f} -> "
               f"{BASELINE['P001+P002+P003+P004+P005+P006']:.0f} ms "
               f"= {BASELINE['cumulative_speedup']:.2f}x  [OVERSTATES by 2.13x]")
    return "\n".join(out)

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

from instinctwm.passes.contract import Tier


@dataclass(frozen=True)
class Released:
    pid: str
    name: str
    version: str
    tier: Tier
    step_speedup: float
    gates: str
    frozen: bool = True
    #: Set when a version has SHIPPED CODE but its gates have not been re-run yet. A frozen pass may
    #: only change for a correctness bug, and the fix lands before the gate can run if the fleet is
    #: busy -- so the honest state is "fixed, not yet re-verified", not "verified". Never leave this
    #: set once the gates pass; never clear it without running them.
    gates_owed: str = ""

    #: REQUIRED for any pass whose tier is not BITEXACT. A NUMERIC or BEHAVIORAL pass changes outputs,
    #: so `max|delta action| = 0` is unavailable and the only defensible evidence is a paired
    #: non-inferiority certificate. `is_verified()` refuses such a pass without one, because the
    #: failure mode is a lossy pass inheriting the credibility of six bit-exact ones.
    certificate: str = ""

    def is_verified(self) -> bool:
        if self.gates_owed:
            return False
        if self.tier is not Tier.BITEXACT and not self.certificate:
            return False
        return True

    def evidence_kind(self) -> str:
        return "bit-exactness" if self.tier is Tier.BITEXACT else "paired non-inferiority"


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
        pid="P005", name="graph_block_stack", version="1.0.1", tier=Tier.BITEXACT,
        step_speedup=1.38,
        gates="max|delta action| = 0 over 6 paired seeded cycles, verified with the gate run AFTER "
              "an episode reset (the ordering that exposed a nan); 2539.9 -> 1842.0 ms under "
              "probe_latency --repeats 3, spread 0.5%. Runs the 30-block stack from a captured "
              "CUDA graph: per-op dispatch 6.2 us (83.6% of it cudaLaunchKernel) becomes ~1.17 us "
              "replay. Requires P003, whose slice addressing is what makes the stack capturable "
              "at all -- a stock block raises cudaErrorStreamCaptureInvalidated. "
              "v1.0.1: the eager FALLBACK did not advance the ring. `install` sets "
              "_iwm_defer_commit permanently, so only _commit_all advances it, and both fallback "
              "returns skipped that -- from the first capture failure onwards the ring froze, every "
              "later forward rewrote the same slots, and attention read a stale window with no error "
              "raised. A correctness bug, which is the only reason a frozen pass may change. No "
              "reported number is affected: no log contains CAPTURE FAILED and no eval server ran "
              "--graph-blocks",
        gates_owed="",   # CLEARED 2026-08-07, all three on an idle fleet:
        #   1. eager-fallback ring advance: 8160 slots/cycle on BOTH the captured and the forced
        #      fallback path, identical rate (probe_graph_fallback.py). This gate reported 0 -> 0
        #      twice before, because the probe read _iwm_count / _iwm_ring_count / kv_count, none of
        #      which exist -- the accessor is _iwm_ring_signature(cache_name). A gate that cannot
        #      read its own observable now raises RingUnreadable and reports NOT EVALUATED rather
        #      than a number. Same bug class this file's own docstring records about the original
        #      graph-capture integration ("keyed on an attribute that did not exist").
        #   2. max|delta action| = 0 with --graph-blocks: exit 0.
        #   3. latency under ABBA (base, treat, treat, base): 3417.7 -> 2836.1 ms = 1.205x, with
        #      0.3% drift on the repeated base arm.
        ),
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
    Released(
        pid="P007", name="conv_layout_ndhwc", version="1.0.0", tier=Tier.NUMERIC,
        step_speedup=1.405,
        gates="THE FIRST NON-BITEXACT RELEASE, and the first Layer 5 one. Every 3x3x3 convolution in "
              "both observation VAEs was declining cuDNN in NCDHW and landing on "
              "slow_conv_dilated3d; serving them in NDHWC reaches cudnn_convolution at 4.35-7.24x "
              "per signature. NO KERNEL WAS WRITTEN -- this is backend/layout dispatch, chosen by "
              "instinctwm/backends/conv/ and applied by backends/conv/apply.py to both "
              "streaming_vae and streaming_vae_half (62 + 62 Conv3d weights; converting only the "
              "first leaves the two wrist cameras on the fallback path). "
              "cudnn.benchmark=True changes nothing (1.00x on all four signatures), so this is not "
              "heuristic search: there is no NCDHW bf16 3D kernel for these shapes on H100 / "
              "torch 2.9 / cuDNN 9.10. "
              "LATENCY: episode mode, post-saturation steady state, ABBA-ordered (base, treat, "
              "treat, base) -- baseline 519.2/522.7 -> mean 521.0 ms, conv-layout 358.8/382.8 -> "
              "mean 370.8 ms = 1.405x, +150.2 ms/cycle. Drift on the repeated base arm 0.7%. "
              "NOTE the asymmetry: the two treatment arms differ by 6.4% while the base arms differ "
              "by 0.7%, so the conv-layout path is the noisier of the two -- plausibly cuDNN "
              "re-selecting a kernel between runs. The ABBA mean is the number; 1.45x was an earlier "
              "figure from the FIRST arm pair before the ordering completed, and quoting it would "
              "have been picking the better of two treatment arms. "
              "Corroborated in-process at 490.4 -> 330.2 ms = 1.49x by an independent harness; the "
              "in-process figure runs slightly higher because it excludes websocket transport. "
              "SIDE EFFECT that explains an older mystery: aten::copy_ falls 34,710 -> 6,385 calls "
              "and fill_ 29,681 -> 1,361, because 82% of the copy population was vol2col lowering "
              "inside the fallback. copy_ was the largest line in the profile and a copy kernel "
              "would have been wasted work. "
              "ATTRIBUTION, stated explicitly because it was got wrong once: this speedup is "
              "DEVICE-SIDE. 62 convolutions x ~2.1 ms saved each is ~130 ms against the measured "
              "+150.2 ms/cycle, so the kernel change accounts for the gain on its own. The ~56,600 "
              "dispatcher calls that vanished with vol2col are a CONSEQUENCE of the layout decision, "
              "not its mechanism, and LAYER5_CRITICAL_PATH.md section 4 briefly credited them as the "
              "main effect -- that claim is retracted there. P007 is a backend/layout win and is not "
              "evidence for any host-dispatch model.",
        certificate="paired non-inferiority, margin -0.05 declared BEFORE the run, both arms 2V/4A "
                    "on identical pinned seeds so only the layout differs. 555 paired episodes: "
                    "baseline 506/555 = 0.9117, conv-layout 504/555 = 0.9081, delta -0.0036. "
                    "Discordant 60 (31 baseline-only / 29 layout-only); exact McNemar two-sided "
                    "p = 0.897 (no detectable difference); one-sided non-inferiority p = 0.00031. "
                    "NON-INFERIOR. Required because NDHWC changes the convolution's accumulation "
                    "order: max|delta| 1.25e-01 on the encoder output, relative 6.67e-03, ~1.7x bf16 "
                    "resolution -- and the latents feed the KV cache, so it propagates to actions. "
                    "max|delta action| = 0 is unavailable by construction, which is why the conv "
                    "backend layer derives NUMERIC for this pair and refuses to select it without an "
                    "explicit prefer_bitexact=False."),
)

#: MEASUREMENT PROTOCOL, and a caveat that applies to every number below.
#:
#: These were measured with SEQUENTIAL A/B ordering: baseline arm first, treatment arm second. PR #2
#: showed that this box drifts UPWARD within a session -- 3214 -> 3730 -> 3964 ms across three rounds
#: of the same configuration -- so whichever arm runs second is systematically penalised. Every number
#: here is therefore PRE-ORDER-CONTROL: usable, and not equivalent to an ABBA-ordered measurement.
#:
#: The direction of the bias is knowable even if the size is not. Where the treatment ran second, its
#: speedup is UNDERSTATED; where a regression was reported for a second-running arm, part of it may be
#: drift. ABBA (base, treat, treat, base) is the default protocol from now on.
#:
#: THESE ARE ALSO QUALITY-OPERATING-POINT NUMBERS -- 25 video / 50 action steps, ~79 forwards per cycle. The
#: shipped Fast operating point runs 6 forwards, so the per-step term is ~20x smaller while the fixed term is
#: unchanged. Do not quote any of this for the Fast operating point; it is being re-measured.
#: 'Operating point', not 'profile': a profile sounds like a mode of the engine, and there is no
#: such mode. It is a declared step schedule -- a descriptor delta -- and the planner re-derives
#: the pass set from it. See AUDIT.md F6.
MEASUREMENT_PROTOCOL = ("sequential A/B, pre-order-control; Quality operating point (25 video / 50 action)")

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
    #: Graph capture is a NET WIN in episode mode -- 1.21x whole episode -- but far from the
    #: 1.5-2x probe_latency implied. Captures never stop: 6.0/cycle, 92.5% hit rate.
    #:
    #: WHY THE KEY NEVER CONVERGES, measured directly: the ring advances 152 slots/cycle and
    #: `start` stays 0 for the whole episode (no wraparound), so `count` -- the attention KV
    #: length -- grows every single cycle. The graph key follows the attention shape. Making the
    #: WRITE offset device-resident would not help, because it is the READ EXTENT that moves.
    #: Padding it to a fixed extent is ruled out: masked SDPA is not bit-exact. So the key cannot
    #: converge within an episode without changing numerics -- a property of the model, not an
    #: engineering gap.
    "episode_mode": {
        "protocol": "probe_episode.py --cycles 45 (one reset, ring never rewound)",
        "default_whole_episode_ms": 2800.8,
        "default_late_episode_ms": 2302.8,
        "no_graph_whole_episode_ms": 3400.1,
        "no_graph_late_episode_ms": 2710.5,
        "captures_per_cycle": 6.0,
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
        #: MISLABELLED WHEN WRITTEN. The ring advances 152 slots/cycle (measured), not 272, so
        #: saturation is at cycle ~64 and a 45-cycle run never reaches it. These are LATE-EPISODE
        #: numbers with a warm graph cache, not steady state. They are still the right rows to
        #: compare against each other; they are not "post-saturation".
        "chain_late_episode_ms": {
            "stock": 9486.3, "P001": 5195.1, "P001+P002": 5059.1,
            "P001+P002+P003": 2635.7, "+generic_passes": 2966.7,
            "+graph_capture(default)": 2298.7,
        },
        "cumulative_speedup_episode": 3.38,
        #: RETRACTED. The +331 ms "generic pass regression" came from six servers measured
        #: CONCURRENTLY and does not reproduce. Sequential A/B (one server, one GPU, 45 cycles
        #: each) gives, late-episode:
        #:     p003_base 2728.9 | shims_only 2771.6 | +pools 2758.9
        #:     +hoist 2892.3    | +promote 2857.2   | +stepidx 2702.4
        #: The full generic stack is 26.5 ms FASTER than P003 alone. What survives: the adapter
        #: shims cost ~43 ms (1.6%), and HoistInvariant costs +133 ms in eager mode -- real, but
        #: more than repaid by ExplicitStepIndex at -155 ms.
        "generic_stack_vs_p003_ms": -26.5,
        "shim_cost_ms": 42.7,
        "hoist_eager_cost_ms": 133.4,
        "stepidx_gain_ms": -154.8,
        "evictions_per_episode": 204,
    },
}


def summary() -> str:
    out = ["Released passes (frozen)"]
    for r in RELEASED:
        flag = "" if r.is_verified() else "   [GATES OWED]"
        out.append(f"  {r.pid} {r.name:22s} v{r.version}  {r.tier.name:9s} "
                   f"{r.step_speedup:.2f}x step   [{r.evidence_kind()}]{flag}")
    owed = [r.pid for r in RELEASED if not r.is_verified()]
    if owed:
        out.append(f"  NOT FULLY VERIFIED: {', '.join(owed)}. Either gates are owed, or a "
                   f"non-BITEXACT pass is missing its certificate. See Released.gates_owed / "
                   f".certificate.")
    lossy = [r.pid for r in RELEASED if r.tier is not Tier.BITEXACT]
    if lossy:
        out.append(f"  TIER: the chain is NOT bit-exact end to end -- {', '.join(lossy)} "
                   f"{'is' if len(lossy) == 1 else 'are'} NUMERIC. A plan containing one of these "
                   f"cannot claim max|delta action| = 0, however many BITEXACT passes sit beside it.")
    # Episode mode leads, because it is the protocol that describes a real episode. probe_latency
    # resets between repeats, which rewinds the ring and hides per-cycle recapture; it overstated
    # this chain by 2.13x.
    e = BASELINE["episode_mode"]
    ch, cp = e["chain_whole_episode_ms"], e["chain_late_episode_ms"]
    out.append("  EPISODE MODE (45 cycles, one reset) -- the reporting standard:")
    out.append(f"    whole episode  : {ch['stock']:.0f} -> {ch['+graph_capture(default)']:.0f} ms "
               f"= {e['cumulative_speedup_episode']:.2f}x")
    out.append(f"    late episode   : {cp['stock']:.0f} -> "
               f"{cp['+graph_capture(default)']:.0f} ms")
    out.append(f"    captures {e['captures_per_cycle']:.1f}/cycle throughout, "
               f"{e['evictions_per_episode']} evictions: the cache does NOT converge")
    out.append(f"  short-horizon (probe_latency, resets between repeats): "
               f"{BASELINE['stock']:.0f} -> "
               f"{BASELINE['P001+P002+P003+P004+P005+P006']:.0f} ms "
               f"= {BASELINE['cumulative_speedup']:.2f}x  [OVERSTATES by 2.13x]")
    return "\n".join(out)

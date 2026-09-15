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

from instinctflash.passes.contract import Tier


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

    #: Where step_speedup was measured, when it is NOT this project's H100 protocol box. Rendered
    #: inline in summary() rows: a 5090-measured number printed beside H100 rows with the label
    #: only in a Disposition note reads as fleet-verified, and it is not.
    measured_on: str = ""

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
        pid="P003", name="ring_kv_addressing", version="1.0.1", tier=Tier.BITEXACT,
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
              "instinctflash/backends/conv/ and applied by backends/conv/apply.py to both "
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
              "ATTRIBUTION, and it took three attempts to get right -- see LAYER6_REGIMES.md. It is "
              "NOT purely device-side and NOT purely host-side. A positive control toggling the "
              "layout in-process measured, with both off-arms agreeing to 0.04%: device busy "
              "246.3 -> 190.1 ms (-56.2 ms) and device events 46,992 -> 18,603 (-28,387 launches). "
              "Against the published in-process delta of 160.2 ms (490.4 -> 330.2) and a measured "
              "VAE device slope of ~1.0, that is ~56-61 ms of device time (~37%) and ~99-104 ms from "
              "the removed launches at ~3.5 us each (~63%). The earlier ~130 ms device estimate "
              "extrapolated the LARGEST conv signature (2.659 -> 0.581 ms) across all 62 and was too "
              "high. WHAT P007 REALLY DID: it moved the VAE encode ACROSS a regime boundary. With "
              "vol2col it issued 46,992 tiny kernels and was host-bound; with cuDNN it issues large "
              "convolutions and is device-bound at slope ~1.09. It won on both terms because it "
              "changed which term binds. That is why it is the only pass here that has ever paid.",
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
    Released(
        pid="P009", name="sm120_gated_residual", version="1.0.1", tier=Tier.BITEXACT,
        step_speedup=1.0283,
        gates="SM120 native FMUL+FADD+BF16-RNE kernel; PTX contains separate mul.rn.f32 and "
              "add.rn.f32 with no fma.rn.f32. 0 differing words over 17,694,720 adversarial "
              "elements across 12 exponent scales and both production shapes. Two 42-cycle "
              "candidate arms (50,160 live kernel calls total) were action-bit-exact against both "
              "baseline arms through ring saturation; same-runtime reset produced identical "
              "actions and stable buffer pointers. ABBA mean 354.81 -> 345.05 ms = 1.0283x; "
              "saturated mean 399.90 -> 390.81 ms = 1.0232x; every arm spread <0.18%. "
              "v1.0.1 correctness hardening: plan-scoped thread-local constructor token prevents "
              "enabled->excluded Runtime leakage; target device reaches DeviceProfile.probe; native "
              "capability verifies loadable ABI v1; full-body rewrite pins upstream source hash; "
              "rank/stride/storage-overlap and single-stream paths fail closed. Reverified 42 cycles.",
        measured_on="RTX 5090, author-measured",
    ),
    Released(
        pid="P009-A2", name="sm120_wan_stage2", version="1.0.0", tier=Tier.BITEXACT,
        step_speedup=1.0362,
        gates="Independent SM120 ABI v1 reproduces PyTorch 2.9's float4, threads=(32,4), "
              "two-level Welford tree for D=3072 while preserving both eager BF16 residual "
              "boundaries. Production launcher: 0 differing words over 66,868,480 adversarial "
              "fields across seven input patterns, 12 exponent scales, and both production "
              "shapes; dtype/shape/alignment/alias guards all refused invalid inputs. Integrated "
              "42-cycle A-B-B-A under one physical-GPU0 lock: 168/168 actions bitwise equal, "
              "349.857 -> 337.622 ms = 1.0362x; saturated 396.012 -> 384.716 ms = 1.0294x; "
              "candidate-arm spread 0.00094%, no foreign GPU0 PID. Same-Runtime reset: 3/3 "
              "actions bitwise equal, all 30 blocks retained both buffer shapes and every pointer, "
              "with exact 1680 A2 and 840 A1 calls per episode. Adds about 0.187 GiB peak memory.",
        measured_on="RTX 5090, author-measured",
    ),
    Released(
        pid="P009-A3", name="sm120_wan_stage3", version="1.0.0", tier=Tier.BITEXACT,
        step_speedup=1.0127,
        gates="Independent SM120 ABI v1 fuses norm1 FP32 LayerNorm with Ada scale/shift while "
              "reproducing PyTorch 2.9's D=3072 Welford order and final BF16 RNE. Operator gate: "
              "0 differing words for output, mean, and rstd across random/constant/alternating "
              "patterns, three exponent scales, and rows 64/480; dtype/shape/alias guards refused "
              "invalid inputs; local region speedup 2.22x/2.64x. Real 5B model 42-cycle A-B-B-A: "
              "168/168 actions bitwise equal, 409.50 -> 404.34 ms = 1.0127x; growing "
              "406.53 -> 401.23 ms; saturated 454.95 -> 449.44 ms; candidate spread 0.073%. "
              "Adds about 0.094 GiB peak memory and executes exactly 12,540 A3 calls per arm.",
        measured_on="RTX 5090, author-measured",
    ),
    Released(
        pid="P009-A4", name="sm120_wan_qk_rope", version="1.0.1", tier=Tier.BITEXACT,
        step_speedup=1.0209,
        gates="Independent SM120 ABI v1 fuses Torch 2.9's vec4/threads(32,4) BF16 RMSNorm, "
              "its BF16 materialization boundary, and the following FP64-complex RoPE. Operator "
              "gate: zero differing output/rstd words across random/constant/alternating patterns, "
              "three exponent scales, rows 64/480; dtype/shape/alias guards refused invalid inputs; "
              "local region 1.72x/1.75x. Real 5B model 42-cycle A-B-B-A: 168/168 actions bitwise "
              "equal, 404.71 -> 396.42 ms = 1.0209x; growing 401.27 -> 393.12 ms; saturated "
              "452.30 -> 442.12 ms; candidate spread 0.086%. Adds about 0.188 GiB peak memory "
              "and executes exactly 25,080 fused Q/K calls per arm.",
        measured_on="RTX 5090, author-measured",
    ),
    Released(
        pid="P009-A5", name="sm120_wan_gemm", version="1.0.0", tier=Tier.BITEXACT,
        step_speedup=1.0175,
        gates="Independent SM120 ABI v1 pins two cuBLASLt BF16+bias tactics with split-K fixed "
              "to one; all K=14336 and weak/noisy K=3072 candidates retain upstream PyTorch. "
              "Exhaustive screen: 0 differences over 85,229,568 outputs across four candidate "
              "shapes; the selected production gate rechecked 21,528,576 outputs over random, "
              "constant, and alternating inputs at three exponent scales. The two selected shapes "
              "measured 1.20x and 1.69x locally. Real 5B model 42-cycle A-B-B-A: 168/168 actions "
              "bitwise equal, 395.03 -> 388.24 ms = 1.0175x; growing saves 5.18 ms and saturated "
              "saves 5.09 ms; candidate-arm spread 0.167%. Static outputs add about 0.561 GiB peak "
              "memory. Shape, module-count, dtype, contiguity, alignment, weight-version, pointer, "
              "device, ABI, Torch/CUDA/cuBLASLt-version, no-workspace, and one-stream conditions fail closed.",
        measured_on="RTX 5090, author-measured",
    ),
    Released(
        pid="P009-A6", name="sm120_wan_ring_concat", version="1.0.0", tier=Tier.BITEXACT,
        step_speedup=1.0053,
        gates="Independent SM120 ABI v1 replaces only wrapped P003 K/V torch.cat pairs with one "
              "uint4-vectorized CUDA copy into shared persistent scratch; physical-slot order and "
              "every BF16 word are unchanged. Operator gate: 0 differing K/V words at counts "
              "1000/4000/7000/9000; local speedups 4.64x/1.93x/1.72x/1.80x. Real 5B model "
              "42-cycle A-B-B-A: 168/168 actions bitwise equal, 389.80 -> 387.73 ms = 1.0053x; "
              "late cycles 426.60 -> 422.26 ms = 1.0103x. Candidate arms executed exactly 300 "
              "A6 calls, both baseline arms executed zero; reset replay was bitwise equal with "
              "stable scratch pointers. The scratch capacity is 0.224 GiB; observed peak allocation "
              "was unchanged within 0.003 GiB. Shape, module-count, wrap-state, dtype, contiguity, "
              "alignment, alias, device, ABI, Torch/CUDA-version, and one-stream conditions fail closed.",
        measured_on="RTX 5090, author-measured",
    ),
    Released(
        pid="P010", name="action_terminal_forward_elision", version="1.0.0", tier=Tier.BITEXACT,
        step_speedup=1.098,
        gates="Skips the action loop's padded terminal forward (wan_va_server.py:542-546, action_mode=True "
              "and update_cache=1): its output is discarded at :548 and its provisional K/V is dropped by "
              "clear_pred_cache (:574) before any forward reads it. Only the forward's SLOT ALLOCATION is "
              "replayed -- stock: update_cache minus its K/V lines; --ring-kv: the forward's metadata "
              "writes + _commit (count/pred/start advance) -- which is what the naive skip lacked when it "
              "was refuted on 2026-08-09 (0 through cycle ~37, then 0.0297..0.406 past the ring wrap). "
              "GATE: probe_action_terminal_elision.py, --deterministic-seed serving, 48 seeded cycles per "
              "episode, ABBA ON/EL/EL/ON, wrap crossed at cycle 36 (ring start 0 -> 272) with 12 post-wrap "
              "cycles: max|delta action| = 0.000e+00 on EVERY cycle and arm pair, same-arm repeats "
              "0.000e+00, on both allocators (stock mask, --ring-kv) x both operating points (2V/4A@w5 "
              "batch-2, 2V/2A@w1 batch-1), replicated in two independent runs (8/8); no materialization, "
              "no self-disable. ALLOCATOR LEVEL (tests/test_ring_allocator.py, real stock WanAttention "
              "allocator + real RingKVAddressing ring, payload-stamped, 80 cycles = 2.2 wraps): bookkeeping "
              "exact at 2V/4A, 2V/1A, 1V/2A on both allocators; ring-naive diverges at cycle 36 at every "
              "point; stock-naive diverges only at 1A (with any transient action forward the first "
              "transient already performs the eviction). LATENCY (H100, in-process, ABBA, all same-arm "
              "drifts <= 1.2%): shipped chain 2V/4A@w5 257.0 -> 234.0 ms/cycle (-23.0, 1.098x; infer "
              "186.3 -> 163.4; saturated 276.2 -> 253.6), 2V/2A@w1 217.6 -> 194.8 (-22.7, 1.117x; infer "
              "145.2 -> 121.6); stock allocator 602.5 -> 570.9 (-31.6) and 441.5 -> 411.3 (-30.2). "
              "kv message unchanged within noise (bookkeeping +0.3 ms ring, +4-5 ms stock). Two fail-closed "
              "gates: any forward before clear_pred_cache (vendor generate(), predict() without commit()) "
              "materializes the skipped forward and disables the pass for that server; video_exec_step "
              "!= -1 declines.",
        measured_on="H100 80GB HBM3 on the 4xh100 box (in-process probe, no websocket; not the protocol box)",
    ),
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


# ---------------------------------------------------------------------------------------------------
# CURRENT DISPOSITION -- separate from the historical record above, deliberately.
#
# `RELEASED` is a LEDGER: what was released, at what tier, on what evidence, at the time. It is frozen
# and nothing below rewrites it. But "was released" and "should run today" drifted apart, and the
# repository asserted both at once -- P005 was registered as shipping at 1.380x while serve_variant's
# own --graph-blocks help called it NOT SHIPPABLE and a later measurement put capture 1.43x SLOWER.
#
# So the ledger keeps history and this table states the recommendation, with the measurement that
# justifies it. A pass can be released and not recommended; those are different facts and conflating
# them is what produced the inconsistency.
#
# THIS TABLE IS THE SINGLE SOURCE OF TRUTH FOR THE SHIPPED CONFIGURATION. The launch scripts, the
# README and tests/test_shipped_config.py all derive from `shipped_configuration()`; if they disagree
# the test fails. Add a flag here, not in four places.
# ---------------------------------------------------------------------------------------------------

SERVED = "SERVED"                    # in the default serving path
AVAILABLE = "AVAILABLE"              # released and correct; opt-in, not in the default path
NOT_RECOMMENDED = "NOT_RECOMMENDED"  # released historically; current measurement says do not enable


@dataclass(frozen=True)
class Disposition:
    pid: str
    status: str
    #: CLI flags on serve_variant.py that enable it. Empty for a pass with no serving flag.
    serving_flags: tuple[str, ...]
    #: Why it holds this status TODAY, with the measurement. Not a summary of `gates`.
    note: str


DISPOSITIONS = (
    Disposition("P001", SERVED, ("--no-fsdp", "--no-empty-cache", "--no-debug-dump"),
                "Bit-exact, no counter-indication."),
    Disposition("P002", SERVED, ("--conditioning-prefill",),
                "Bit-exact, no counter-indication."),
    Disposition("P003", SERVED, ("--ring-kv",),
                "Bit-exact. Also load-bearing beyond its own speedup: the interval addressing is what "
                "makes the block capturable AND compilable at all -- without it dynamo reports 10 "
                "graphs and 9 breaks on mask.nonzero()."),
    Disposition("P004", AVAILABLE, ("--hoist-casts",),
                "Bit-exact and correct, but never in a launch script and its 1.02x was measured on a "
                "~2.5 s cycle. Enabling it is a behaviour change that needs its own gate re-run at the "
                "current operating point; not done, so it stays opt-in rather than being switched on "
                "during a consistency cleanup."),
    Disposition("P005", NOT_RECOMMENDED, ("--graph-blocks",),
                "DO NOT ENABLE at the Fast operating point. Registered 1.380x was measured on a ~2.5 s "
                "cycle. Today: capture ON + plan buffer 503.5 ms against capture OFF 351.4 ms -- 1.43x "
                "SLOWER, because 5.3 surviving captures at ~111 ms each exceed the whole cycle "
                "(LAYER5_GRAPH_PERSISTENCE_RESULT.md). A second, independent reason: graph eviction "
                "does not return its private pool, so a 50-task run climbs to the 80 GB ceiling and "
                "OOMs. The pass is correct; the mechanism does not pay here."),
    Disposition("P006", NOT_RECOMMENDED, ("--stable-pools",),
                "Exists only so P005's graphs survive a reset -- serve_variant: 'Only has an effect "
                "with --graph-blocks'. With P005 not recommended this has no served effect. Not "
                "refuted on its own; simply has nothing to do."),
    Disposition("P007", SERVED, ("--conv-layout",),
                "The only pass that measurably moves the current cycle: 1.405x under ABBA, certified "
                "NUMERIC on 555 paired episodes (delta -0.0036, exact McNemar p = 0.897, one-sided "
                "non-inferiority p = 0.00031). Enabling it makes the served chain NUMERIC rather than "
                "bit-exact end to end -- that is the certificate's purpose and summary() says so."),
    Disposition("P009", AVAILABLE, ("--sm120-gated-residual",),
                "Correct and profitable on RTX 5090, but conditional on an explicitly built SM120 "
                "native library. The planner auto-applies it only when DeviceProfile reports "
                "sm120_kernels; it is not in the architecture-neutral shipped flag list, so H100 "
                "and machines without the extension remain unchanged."),
    Disposition("P009-A2", AVAILABLE, ("--sm120-wan-stage2",),
                "Correct and profitable on RTX 5090, but requires both the explicitly built "
                "P009-A1 library and its independent stage2 ABI. The planner auto-applies it only "
                "when both native features are present; architecture-neutral serving and machines "
                "without either extension remain unchanged."),
    Disposition("P009-A3", AVAILABLE, ("--sm120-wan-stage3",),
                "Correct and profitable on RTX 5090, but requires the A1/A2 chain and its own "
                "independent stage3 ABI. The planner applies it only when all three native "
                "features are present; other devices and builds remain unchanged."),
    Disposition("P009-A4", AVAILABLE, ("--sm120-wan-qk-rope",),
                "Correct and profitable on RTX 5090, but requires A1/A2/A3 and its independent "
                "QK-RoPE ABI. The planner applies it only with the complete native feature chain; "
                "other devices and builds remain unchanged."),
    Disposition("P009-A5", AVAILABLE, ("--sm120-wan-gemm",),
                "Correct and profitable on RTX 5090, but requires A1-A4 and its independent "
                "pinned-cuBLASLt ABI. It applies only to two certified shapes with split-K=1; "
                "all other Linear calls stay on the original PyTorch path."),
    Disposition("P009-A6", AVAILABLE, ("--sm120-wan-ring-concat",),
                "Correct and modestly profitable on RTX 5090, but requires A1-A5 and its "
                "independent dual-K/V copy ABI. It replaces only wrapped P003 ring intervals; "
                "non-wrapped intervals and all unsupported shapes retain the PyTorch path."),
    Disposition("P010", SERVED, ("--action-terminal-elision",),
                "Bit-exact through the ring wrap on both allocators and both operating points (two "
                "independent 48-cycle ABBA runs, max|delta action| = 0.000e+00 everywhere), so no "
                "closed-loop certificate is needed. Removes one DiT forward per cycle: 1 of 10 at 2V/4A, "
                "1 of 8 at 2V/2A; -23 ms/cycle on H100 (1.098x / 1.117x on the shipped chain). Fails "
                "closed on the two things it depends on: the closed-loop message order (materialize + "
                "disable on violation) and video_exec_step == -1 (decline)."),
)

_BY_PID = {d.pid: d for d in DISPOSITIONS}


def disposition_of(pid: str) -> Disposition:
    return _BY_PID[pid]


def shipped_configuration() -> list[str]:
    """The serve_variant.py flags that constitute the default serving path.

    Every launch script and the README must match this exactly; test_shipped_config.py enforces it.
    """
    out: list[str] = []
    for r in RELEASED:                      # RELEASED order, so the list is stable and reviewable
        d = _BY_PID[r.pid]
        if d.status == SERVED:
            out.extend(d.serving_flags)
    return out


def shipped_pids() -> list[str]:
    return [r.pid for r in RELEASED if _BY_PID[r.pid].status == SERVED]


def served_tier() -> Tier:
    """The tier of the SERVED chain as a whole -- the weakest link, not the best one."""
    tiers = [r.tier for r in RELEASED if _BY_PID[r.pid].status == SERVED]
    if any(t is Tier.BEHAVIORAL for t in tiers):
        return Tier.BEHAVIORAL
    if any(t is Tier.NUMERIC for t in tiers):
        return Tier.NUMERIC
    return Tier.BITEXACT


def summary() -> str:
    out = ["Released passes (frozen)"]
    for r in RELEASED:
        flag = "" if r.is_verified() else "   [GATES OWED]"
        d = _BY_PID[r.pid]
        mark = {SERVED: "SERVED         ", AVAILABLE: "available      ",
                NOT_RECOMMENDED: "NOT RECOMMENDED"}[d.status]
        speed = f"{r.step_speedup:.2f}x step"
        if r.measured_on:
            # inline, not only in the Disposition note: these rows sit beside H100-measured ones,
            # and an unlabelled foreign-hardware number reads as fleet-verified.
            speed += f" [{r.measured_on}]"
        out.append(f"  {r.pid} {r.name:22s} v{r.version}  {r.tier.name:9s} "
                   f"{speed}   {mark} [{r.evidence_kind()}]{flag}")
    owed = [r.pid for r in RELEASED if not r.is_verified()]
    if owed:
        out.append(f"  NOT FULLY VERIFIED: {', '.join(owed)}. Either gates are owed, or a "
                   f"non-BITEXACT pass is missing its certificate. See Released.gates_owed / "
                   f".certificate.")
    out.append("")
    out.append(f"  SHIPPED CONFIGURATION ({', '.join(shipped_pids())}), tier "
               f"{served_tier().name} -- this is the single source of truth:")
    out.append(f"    {' '.join(shipped_configuration())}")
    notrec = [r.pid for r in RELEASED if _BY_PID[r.pid].status == NOT_RECOMMENDED]
    if notrec:
        out.append(f"  NOT RECOMMENDED: {', '.join(notrec)} -- released historically, and current "
                   f"measurement says do not enable. The ledger above is the record of what was "
                   f"released; Disposition.note is why it should not run today.")
    out.append("")
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

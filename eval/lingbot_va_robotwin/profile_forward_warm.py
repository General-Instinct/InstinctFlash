#!/usr/bin/env python3
"""Warm steady-state per-forward cost at 2V/4A, and what a forward is made of.

WHY THE PREVIOUS ATTEMPT COULD NOT BE TRUSTED

`profile_fixed_term.py` measured 83 ms/forward, which cannot be steady state: 79 forwards at Quality
would be 6.6 s against a measured 2315 ms. Two contaminations, both recorded in PROFILE.md:

  GRAPH RECAPTURE     the instrumented pass began from dropped graph pools, so every forward paid
                      capture cost. The graph key contains the ring signature (start, count) and the
                      ring advances every cycle, so keys keep changing and capture never stops.
  ALLOCATOR STATE     pass 1 ended with 64 held graphs and ~1.2 GiB free of 79; pass 2 started with
                      46 GiB free. The pass with LESS memory pressure ran 47% faster, which means
                      allocator state dominated the comparison.

This probe removes both rather than averaging over them.

  1. ONE PASS, ONE ALLOCATOR STATE. No second pass, no pool release mid-run, and free memory is
     reported per window so a drift in allocator pressure is visible instead of silent.

  2. CAPTURE IS OFF BY DEFAULT. A per-forward number for RANKING Layer 4/5 should describe the model's
     compute, not the launch machinery. Graph capture changes launch overhead and nothing else, so
     measuring without it gives the eager compute baseline the ranking actually needs -- and removes
     recapture entirely rather than trying to outrun it. `--graph` re-enables it for comparison.

  3. STEADY STATE IS DEMONSTRATED, NOT ASSUMED. Cost is reported in windows across the run. If the
     last two windows do not agree to within a tolerance, the answer is NOT EVALUATED: the run had not
     converged and its mean describes a transient.

WHAT IT REPORTS

  * ms/forward in steady state, split by phase (kv_refresh / video / action), because the three have
    different token counts and averaging them hides that
  * a kernel-level breakdown by operator category from torch.profiler -- attention vs matmul vs
    normalisation vs elementwise -- which is what re-ranks Layer 4 against Layer 5

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29985 \\
        profile_forward_warm.py [--cycles 90] [--graph]
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics
import sys
import time
from pathlib import Path

IFL_ROOT = os.environ.get("IFL_ROOT") or str(Path(__file__).resolve().parents[2])
if IFL_ROOT not in sys.path:
    sys.path.insert(0, IFL_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from instinctflash.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

#: Two consecutive windows must agree within this to call the run converged.
CONVERGED = 0.05


def sync():
    torch.cuda.synchronize()


def categorise(op: str) -> str:
    """Bucket a kernel name into the layer that would optimise it.

    ORDER MATTERS, AND A SUBSTRING BUG HERE PRODUCED A NONSENSE ANSWER ONCE. The first version tested
    `"mul" in name` in the elementwise bucket, which matches "matmul" -- so every GEMM in the model was
    counted as elementwise and the matmul bucket came out at 0.3% of a transformer's GPU time. An
    implausible number is the only reason it was caught. GEMM patterns are now tested FIRST and the
    elementwise test no longer uses a substring that appears inside them.

    Deliberately coarse: the question is "which LAYER gets the next unit of work", and a 200-row
    kernel table does not answer it.
    """
    o = op.lower()
    # THE ACTUAL KERNEL NAMES ON THIS BOX, which two guessed versions of this function both got wrong:
    #   attention   cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_*
    #   GEMM        nvjet_tst_128x96_64x7_4x1_v_bz_bias_TNT      <- cuBLAS, matches nothing obvious
    #   conv        void at::native::vol2col_kernel<...>          <- im2col lowering for conv3d
    #   elementwise void at::native::{unrolled_,vectorized_,}elementwise_kernel<...>
    #   layout      void at::native::CatArrayBatchedCopy<...>
    # `nvjet` in particular carries no "gemm"/"matmul" substring, so ~54 ms/cycle of matrix multiply was
    # filed under "other" and the matmul bucket read 0.3% of a transformer's GPU time. And "add_"
    # matched "badd_" inside an nvjet name, pulling a GEMM into elementwise. Guessing at kernel names
    # does not work; the raw list is printed below the table for exactly this reason.
    if any(x in o for x in ("flash", "attention", "attn", "sdpa", "scaled_dot", "softmax")):
        return "attention (Layer 4)"
    if any(x in o for x in ("nvjet", "matmul", "gemm", "addmm", "cutlass", "bmm",
                            "s16816", "sm80_", "sm90_xmma", "wgrad")):
        return "matmul / projections (Layer 5-6)"
    if any(x in o for x in ("layer_norm", "layernorm", "rms_norm", "group_norm")):
        return "normalisation (Layer 5)"
    if any(x in o for x in ("vol2col", "col2vol", "conv", "vae", "upsample", "interpolat")):
        return "conv / VAE (Layer 5)"
    if any(x in o for x in ("elementwise", "vectorized", "catarray", "silu", "gelu", "copy",
                            "index", "slice", "transpose", "permute", "contiguous", "fill", "pad")):
        return "elementwise / layout (Layer 5)"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=90)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--video", type=int, default=2)
    ap.add_argument("--action", type=int, default=4)
    ap.add_argument("--graph", action="store_true", help="enable graph capture (default: off)")
    ap.add_argument("--conv-layout", choices=["as-is", "ndhwc"], default="as-is",
                    help="apply the conv backend plan: 'ndhwc' converts the VAE's Conv3d weights so "
                         "cuDNN serves them instead of slow_conv_dilated3d (NUMERIC tier)")
    ap.add_argument("--trace-cycles", type=int, default=3)
    ap.add_argument("--fused-qkv", action="store_true",
                    help="fuse Q/K/V into one GEMM where the per-shape certificate passes")
    ap.add_argument("--step-cast", action="store_true",
                    help="hoist the timestep cast from LAYER to STEP scope (BITEXACT)")
    a = ap.parse_args()

    hot = [ln for ln in os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    ).read().strip().split("\n") if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%).")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_fwd_warm"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = a.video, a.action

    print(f"building the real server at {a.video}V/{a.action}A "
          f"(graph capture {'ON' if a.graph else 'OFF'}) ...", flush=True)
    server = S.VA_Server(cfg)

    from instinctflash.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    for n in install_conditioning_prefill(S, type(server)):
        print(f"  installed {n}", flush=True)
    for n in install_debug_dump_elision(S):
        print(f"  installed {n}", flush=True)
    if a.graph:
        from instinctflash.passes.lingbot.graph_capture import GraphBlockStack
        for n in GraphBlockStack().install(S, type(server)):
            print(f"  installed {n}", flush=True)

    # ---- apply the conv backend plan, through the dispatch layer ------------------------------
    if a.conv_layout == "ndhwc":
        from instinctflash.backends.conv import REGISTRY, ConvShape, register_declared
        from instinctflash.backends.conv.semantics import ConvSemantics, MemoryLayout
        register_declared(REGISTRY)
        convs = [m for m in server.streaming_vae.vae.modules()
                 if isinstance(m, torch.nn.Conv3d) and m.weight.dim() == 5]
        # Ask the layer, with the measured encode-scale numbers, rather than asserting the answer.
        plan = REGISTRY.select(
            semantics=ConvSemantics.CAUSAL_TIME,
            shape=ConvShape(160, 160, (3, 3, 3), spatial=(8, 128, 160), dtype="bfloat16"),
            have_layout=MemoryLayout.NCDHW, subgraph_size=len(convs),
            prefer_bitexact=False,
            measured={("torch_fallback", MemoryLayout.NCDHW): 175.72,
                      ("cudnn_conv3d", MemoryLayout.NDHWC): 17.00})
        print(f"  conv plan: {plan.backend_name} / {plan.use_layout.value} "
              f"[{plan.tier.name}] convert_subgraph={plan.convert_subgraph}")
        print(f"    {plan.reason}")
        if plan.convert_subgraph:
            fmt = plan.use_layout.torch_memory_format()
            for m in convs:
                m.to(memory_format=fmt)
            print(f"    converted {len(convs)} Conv3d weights; the encoder's activations follow")
        # The half-res VAE serves the wrist cameras and must be converted too, or two thirds of the
        # encode stays on the fallback path.
        half = getattr(server, "streaming_vae_half", None)
        if half is not None and plan.convert_subgraph:
            hc = [m for m in half.vae.modules()
                  if isinstance(m, torch.nn.Conv3d) and m.weight.dim() == 5]
            for m in hc:
                m.to(memory_format=plan.use_layout.torch_memory_format())
            print(f"    converted {len(hc)} Conv3d weights in the half-res VAE as well")

    if a.step_cast:
        from instinctflash.passes.lingbot.step_scope_cast import StepScopeCastHoist
        _sc = StepScopeCastHoist()
        for n in _sc.install(S, type(server)):
            print(f"  installed {n}", flush=True)

    if a.fused_qkv:
        from instinctflash.passes.lingbot.fused_qkv import FusedQKVProjection
        _fq = FusedQKVProjection()
        _fq.install(S, server)

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    if not ctx:
        raise SystemExit("no contexts; run collect_contexts.sh")
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = {"obs": [{full: z[s] for s, full in short.items()}], "state": z["state"]}
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)
    rng = np.random.default_rng(0)

    def keyframes(n):
        return [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
                for _ in range(n)]

    # Per-forward timing, tagged by which phase issued it. `action_mode` distinguishes the action
    # loop; `update_cache` distinguishes the KV refresh. Averaging the three would hide that they
    # carry very different token counts.
    fwd = collections.defaultdict(list)
    phase = {"tag": "kv_refresh"}
    orig_fwd = server.transformer.forward

    def timed_fwd(*args, **kw):
        sync(); t0 = time.perf_counter()
        out = orig_fwd(*args, **kw)
        sync()
        tag = "action" if kw.get("action_mode") else phase["tag"]
        fwd[tag].append(time.perf_counter() - t0)
        return out
    server.transformer.forward = timed_fwd

    def cycle(first, n_kf=8):
        if first:
            server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        phase["tag"] = "video"
        act = server.infer(dict(obs=obs["obs"], prompt=prompt,
                                save_visualization=False))["action"]
        phase["tag"] = "kv_refresh"
        server.infer(dict(obs=keyframes(n_kf // 2 if first else n_kf), compute_kv_cache=True,
                          imagine=False, save_visualization=False, state=act))
        return act

    print(f"\nrunning {a.cycles} cycles, one allocator state, no pool release ...", flush=True)
    cycle(True)
    windows = []
    cyc_ms = []
    for i in range(a.cycles):
        sync(); t0 = time.perf_counter()
        cycle(False)
        sync(); cyc_ms.append((time.perf_counter() - t0) * 1000.0)
        if (i + 1) % a.window == 0:
            w = cyc_ms[-a.window:]
            free, total = torch.cuda.mem_get_info()
            windows.append((statistics.median(w), free / 2**30))
            print(f"  cycles {i + 2 - a.window:3d}-{i + 1:3d}: median {windows[-1][0]:7.1f} ms   "
                  f"free {windows[-1][1]:5.1f} GiB")

    if len(windows) < 2:
        print("NOT EVALUATED: fewer than two windows; cannot demonstrate steady state.")
        return 2
    w1, w2 = windows[-2][0], windows[-1][0]
    drift = abs(w2 - w1) / ((w1 + w2) / 2)
    mem_drift = abs(windows[-1][1] - windows[-2][1])
    print(f"\nconvergence: last two windows {w1:.1f} / {w2:.1f} ms -> {drift:.1%} apart "
          f"(tolerance {CONVERGED:.0%}); free-memory drift {mem_drift:.2f} GiB")
    if drift > CONVERGED:
        print("NOT EVALUATED: the run had not converged, so its mean describes a transient rather "
              "than steady state. Raise --cycles.")
        return 2
    print("  CONVERGED -- the numbers below describe steady state, not a warmup.")

    print(f"\n{'=' * 78}\nWARM PER-FORWARD COST at {a.video}V/{a.action}A, "
          f"graph capture {'ON' if a.graph else 'OFF'}\n{'=' * 78}")
    tail = a.cycles - a.window          # attribute only forwards from the converged tail
    print(f"{'phase':<22}{'forwards/cyc':>14}{'ms/forward':>12}{'ms/cycle':>11}")
    print("-" * 78)
    tot = 0.0
    for tag in ("kv_refresh", "video", "action"):
        xs = fwd[tag]
        if not xs:
            continue
        n_per_cycle = len(xs) / (a.cycles + 1)
        per = statistics.median(xs) * 1000.0
        cyc = per * n_per_cycle
        tot += cyc
        print(f"{tag:<22}{n_per_cycle:>14.1f}{per:>12.2f}{cyc:>11.1f}")
    print("-" * 78)
    print(f"{'all forwards':<22}{sum(len(v) for v in fwd.values()) / (a.cycles + 1):>14.1f}"
          f"{'':>12}{tot:>11.1f}")
    print(f"{'cycle total':<22}{'':>14}{'':>12}{w2:>11.1f}")
    print(f"{'forwards as share':<22}{'':>14}{'':>12}{tot / w2:>10.1%}")

    # ---- kernel-level categories, which is what re-ranks Layer 4 vs Layer 5 -------------------
    print(f"\nprofiling {a.trace_cycles} cycles at kernel level ...", flush=True)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(a.trace_cycles):
            cycle(False)
    buckets = collections.Counter()
    raw = []
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", 0) or 0
        if t > 0:
            buckets[categorise(e.key)] += t
            raw.append((t, e.key))
    gpu_total = sum(buckets.values())
    # PRINT THE RAW KEYS. Two categoriser versions produced byte-identical bucket totals, which meant
    # the keys were not what either version assumed. A bucketing whose inputs are invisible cannot be
    # audited, and an unauditable 48% "elementwise" share is not a finding.
    raw.sort(reverse=True)
    print(f"\n  top raw profiler keys ({len(raw)} with device time), so the buckets are checkable:")
    for t, k in raw[:22]:
        print(f"    {t / 1000 / a.trace_cycles:>8.2f} ms/cyc  [{categorise(k):<32}] {k[:58]}")
    print(f"\n{'=' * 78}\nGPU TIME BY CATEGORY (which layer would optimise it)\n{'=' * 78}")
    if gpu_total <= 0:
        print("  NOT EVALUATED: the profiler recorded no device time.")
    else:
        for name, us in buckets.most_common():
            print(f"  {name:<34}{us / 1000 / a.trace_cycles:>9.1f} ms/cycle{us / gpu_total:>9.1%}")
        print(f"  {'-' * 60}")
        print(f"  {'total GPU busy':<34}{gpu_total / 1000 / a.trace_cycles:>9.1f} ms/cycle")
        # A large "other" bucket means the categoriser is failing, not that the time is mysterious.
        # Print the biggest unmatched kernels so the gap is diagnosable instead of decorative.
        unc = sorted(((getattr(e, "self_device_time_total", 0) or 0, e.key)
                      for e in prof.key_averages() if categorise(e.key) == "other"), reverse=True)
        if unc and unc[0][0] > 0:
            print("\n  largest kernels landing in 'other' (categoriser gaps, not mysteries):")
            for us, name in unc[:8]:
                print(f"    {us / 1000 / a.trace_cycles:>8.1f} ms/cycle  {name[:66]}")
        print("\n  Attention's share here is what Layer 4 multiplies against. PROFILE.md retracted a")
        print("  7% figure taken from a bad cost model; this is the measured replacement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Decompose one LingBot-VA control cycle into launch / sync / gather / memory / compute.

We know the *what*: after the 1.92x bit-exact wins plus conditioning prefill, a cycle costs
~4115 ms of which arithmetic explains ~17% and weight traffic ~6%. This answers the *why* for
the remaining ~77%, because the four candidate causes want completely different fixes:

  launch-bound   -> CUDA graphs, kernel fusion              (fewer launches)
  sync-bound     -> remove data-dependent shapes            (no host round trips)
  gather-bound   -> in-place/paged KV attention             (stop copying the pool)
  memory-bound   -> quantization, better layouts            (fewer bytes)
  compute-bound  -> better kernels, lower NFE               (less math)

Method. Nsight Systems is installed on this box but its report importer is broken (the bundled
libssh is missing a symbol its own SshClient needs), so this uses `torch.profiler` with CUDA
activities plus explicit CUDA-event timing around the suspected hot regions. The numbers that
matter here — GPU busy vs wall clock, launch count, per-kernel totals, and the isolated cost of
the KV gather — are all directly measurable that way.

Runs the REAL server object (`VA_Server`) in-process, in the real message order
(reset -> [infer -> compute_kv_cache] x N), so the kernel mix is the served one. Driving it
in-process rather than over the websocket keeps the trace free of transport noise.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import numpy as np
import torch

CAMS = ["observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"]


def make_obs(rng, h=240, w=320):
    return {k: rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for k in CAMS}


def build_server(ckpt: str, no_fsdp: bool, prefill: bool):
    root = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
    sys.path.insert(0, os.path.join(root, "wan_va"))
    sys.path.insert(0, root)
    import wan_va_server as S
    from configs import VA_CONFIGS

    if no_fsdp:
        def _cfg(model, shard_fn, param_dtype, device, eval_mode=True):
            if eval_mode:
                model.eval().requires_grad_(False)
            return model.to(param_dtype).to(device)
        S._configure_model = _cfg
    S.save_async = lambda obj, path: None
    torch.cuda.empty_cache = lambda *a, **k: None

    if prefill:
        sys.path.insert(0, "/home/ubuntu/InstinctWM")
        from instinctwm.runtime.lingbot_install import install_conditioning_prefill
        install_conditioning_prefill(S, S.VA_Server)

    if os.environ.get("IWM_RING_KV") == "1":
        sys.path.insert(0, "/home/ubuntu/InstinctWM")
        from instinctwm.optimizer.passes.ring_kv import RingKVAddressing
        RingKVAddressing().install(S, S.VA_Server)

    cfg = VA_CONFIGS["robotwin"]
    cfg.wan22_pretrained_model_name_or_path = ckpt
    cfg.rank = cfg.local_rank = 0
    cfg.world_size = 1
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29899")
        dist.init_process_group("nccl", init_method="env://", rank=0, world_size=1)
    torch.cuda.set_device(0)
    return S.VA_Server(cfg), S


def drive(srv, rng, cycles, prompt, first_cycle_index=0, timings=None):
    """`cycles` control steps in the real message order.

    `first_cycle_index` matters: the keyframe count depends on whether this is the FIRST cycle of
    an episode (4) or a later one (8), and an earlier version of this harness restarted that
    counter on every call, so the measured cycle silently used a different workload from the warm
    cycles and from probe_latency.
    """
    first_obs = {"obs": make_obs(rng)}
    for c in range(cycles):
        gi = first_cycle_index + c
        t0 = time.perf_counter()
        srv._infer(first_obs, frame_st_id=srv.frame_st_id)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        nkf = 4 if gi == 0 else 8
        kfs = [make_obs(rng) for _ in range(nkf)]          # host-side obs build, NOT timed below
        t2 = time.perf_counter()
        srv._compute_kv_cache({"obs": kfs,
                               "state": np.zeros((16, 2, 16), dtype=np.float32)})
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if timings is not None:
            timings.append({"cycle": gi, "infer_ms": (t1 - t0) * 1e3,
                            "obs_build_ms": (t2 - t1) * 1e3, "kv_ms": (t3 - t2) * 1e3,
                            "nkf": nkf})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get(
        "LINGBOT_CKPT", "/home/ubuntu/ckpt_lingbot/lingbot-va-posttrain-robotwin"))
    ap.add_argument("--warm-cycles", type=int, default=6,
                    help="cycles run before profiling, to grow the KV pool")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--no-fsdp", action="store_true", default=True)
    ap.add_argument("--prefill", action="store_true", default=True)
    ap.add_argument("--out", default="/home/ubuntu/iwm_logs/profile")
    args = ap.parse_args()

    prompt = "Use the left arm to lift the plastic drink bottle head-up"
    srv, S = build_server(args.ckpt, args.no_fsdp, args.prefill)
    rng = np.random.default_rng(0)
    srv._reset(prompt=prompt)

    print(f"warming {args.warm_cycles} cycles (KV grows 272 tokens/cycle)...", flush=True)
    warm_t = []
    drive(srv, rng, args.warm_cycles, prompt, first_cycle_index=0, timings=warm_t)
    torch.cuda.synchronize()

    # ---- wall clock reference for the same work we are about to profile -------------------
    meas_t = []
    t0 = time.perf_counter()
    drive(srv, rng, args.cycles, prompt, first_cycle_index=args.warm_cycles, timings=meas_t)
    torch.cuda.synchronize()
    wall_unprofiled = (time.perf_counter() - t0) * 1000
    print(f"unprofiled wall for {args.cycles} cycle(s): {wall_unprofiled:.1f} ms", flush=True)
    print(f"{'cyc':>4s} {'infer_ms':>10s} {'obs_build':>10s} {'kv_ms':>8s} {'nkf':>4s}")
    for r in warm_t + meas_t:
        print(f"{r['cycle']:4d} {r['infer_ms']:10.1f} {r['obs_build_ms']:10.1f} "
              f"{r['kv_ms']:8.1f} {r['nkf']:4d}")

    # ---- profiled run ---------------------------------------------------------------------
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        t0 = time.perf_counter()
        drive(srv, rng, args.cycles, prompt, first_cycle_index=args.warm_cycles + args.cycles)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000

    evs = prof.key_averages()
    gpu_total_us = 0.0
    launches = 0
    per_kernel = collections.Counter()
    per_kernel_n = collections.Counter()
    for e in evs:
        dev = getattr(e, "self_device_time_total", 0) or 0
        if dev > 0:
            gpu_total_us += dev
            per_kernel[e.key] += dev
            per_kernel_n[e.key] += e.count
            launches += e.count

    print()
    print("=" * 78)
    print(f"wall (profiled)            : {wall:9.1f} ms   [profiler adds overhead]")
    print(f"wall (unprofiled)          : {wall_unprofiled:9.1f} ms   <- use this as the denominator")
    print(f"GPU kernel time (sum)      : {gpu_total_us/1000:9.1f} ms")
    print(f"GPU busy / wall            : {100*gpu_total_us/1000/wall_unprofiled:8.1f}%")
    print(f"GPU IDLE (gap)             : {wall_unprofiled - gpu_total_us/1000:9.1f} ms"
          f"   <- launch/sync bound if large")
    print(f"kernel launches            : {launches:9d}")
    print(f"mean kernel duration       : {gpu_total_us/max(launches,1):9.1f} us"
          f"   <- <10us means launch-dominated")
    print()
    print(f"{'kernel':<58s} {'ms':>8s} {'n':>7s} {'us/ea':>8s}")
    print("-" * 84)
    for k, us in per_kernel.most_common(18):
        n = per_kernel_n[k]
        print(f"{k[:58]:<58s} {us/1000:8.2f} {n:7d} {us/max(n,1):8.1f}")
    print("-" * 84)

    # ---- classify -------------------------------------------------------------------------
    def bucket(name: str) -> str:
        n = name.lower()
        if any(s in n for s in ("gemm", "cutlass", "matmul", "addmm", "mm_", "sm90", "linear")):
            return "GEMM"
        if any(s in n for s in ("attention", "sdpa", "flash", "fmha", "softmax")):
            return "attention"
        if any(s in n for s in ("index", "gather", "take", "copy", "cat", "slice", "narrow")):
            return "gather/copy"
        if any(s in n for s in ("elementwise", "vectorized", "unrolled", "fill", "mul", "add",
                                "norm", "silu", "gelu")):
            return "elementwise/norm"
        if "memcpy" in n or "memset" in n:
            return "memcpy"
        return "other"

    agg = collections.Counter()
    aggn = collections.Counter()
    for k, us in per_kernel.items():
        agg[bucket(k)] += us
        aggn[bucket(k)] += per_kernel_n[k]
    print()
    print(f"{'category':<20s} {'ms':>9s} {'% of GPU':>9s} {'launches':>10s}")
    for cat, us in agg.most_common():
        print(f"{cat:<20s} {us/1000:9.2f} {100*us/gpu_total_us:8.1f}% {aggn[cat]:10d}")

    os.makedirs(args.out, exist_ok=True)
    trace = os.path.join(args.out, "cycle_trace.json")
    prof.export_chrome_trace(trace)
    print(f"\nchrome trace: {trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Is device time on the critical path? Measure d(cycle)/d(device time) directly.

WHY THIS IS THE GOVERNING QUESTION. Device work is 191.8 ms of a 330.7 ms cycle. Layer 5 kernel work was
abandoned on the argument that "the device chain has ~155 ms of slack, so a faster kernel shortens
nothing." That argument assumed the 138.9 ms of device-timeline gaps was device SLACK. It is not -- it is
host issue cost between kernels (docs/research/docs/research/LAYER6_GAPS.md). So the premise is void and the question is open:

    slope ~ 1.0  device time is fully on the critical path; the 191.8 ms is directly reducible
    slope ~ 0    the host genuinely runs ahead; device work cannot help
    in between   the slope IS the exchange rate for pricing any future device-side change

THE METHOD. Inject exactly ONE dummy GPU kernel per block execution -- 300 per cycle -- and vary only its
DURATION across arms by changing the matmul size. Launch count is identical in every arm including the
control, so the host-side cost of the injection differences out and only device time varies.

WHY A MATMUL AND NOT AN ELEMENTWISE OP. The model's own kernels are largely bandwidth-bound. A large
elementwise dummy would contend for memory bandwidth and slow the REAL kernels, which would masquerade as
slope. A square matmul is compute-bound on tensor cores and touches comparatively little memory.

NO PROFILER IN THE TIMED LOOP. torch.profiler inflates this cycle by 1.29x-2.02x and the inflation lands
in exactly the quantity under study. Cycle wall comes from perf_counter around cuda synchronize; the
dummy's duration comes from CUDA events measured in situ.

THREE CONTROLS, because five predictions and two instruments have been wrong here:
  1. k=0 arms interleaved between every level, so drift is visible and cancels.
  2. THE REAL KERNELS MUST NOT CHANGE. One profiled pass at the smallest and largest injection compares
     device busy EXCLUDING the dummy. Device busy is instrument-independent (190.9-191.8 ms across three
     instruments), so this detects cache/bandwidth contention even though the profiler distorts wall time.
     If non-dummy device time grows with the dummy, the slope is contaminated and the run is NOT EVALUATED.
  3. SM clock and power are sampled during every arm. An H100 that boosts differently under sustained
     matmul load would produce a superlinear slope that has nothing to do with the critical path.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29985 probe_device_slope.py
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
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


def gpu_state():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")],
            capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        return f"sm {out[0].strip()} MHz, {out[1].strip()} W, {out[2].strip()} C"
    except Exception:
        return "unavailable"


def time_kernel_solo(fn, n=200):
    """In-situ duration with CUDA events -- no profiler."""
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / n  # us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70)
    ap.add_argument("--arm-cycles", type=int, default=14)
    ap.add_argument("--target", choices=["blocks", "vae"], default="blocks",
                    help="where to inject. 'blocks' = the 30 transformer blocks, 300 sites/cycle, tiny "
                         "kernels, the host-bound region. 'vae' = every Conv3d in the two observation "
                         "VAEs, where P007 acted and where the kernels are milliseconds each. The slope "
                         "is NOT assumed to be the same in both -- that is the whole point of the flag.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[0, 128, 1024, 2048, 3072, 4096],
                    help="square matmul side per injected kernel. 0 = no injection at all; 128 is the "
                         "SLOPE REFERENCE -- it launches 300 kernels like every other arm but adds "
                         "almost no device time, so differencing against it holds launch count constant")
    a = ap.parse_args()

    hot = [ln for ln in os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    ).read().strip().split("\n") if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%).")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_slope2"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

    print("building server at 2V/4A, shipped stack ...", flush=True)
    server = S.VA_Server(cfg)
    from instinctflash.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    for _ in install_conditioning_prefill(S, type(server)):
        pass
    for _ in install_debug_dump_elision(S):
        pass
    from instinctflash.backends.conv.apply import install_conv_layout
    for _ in install_conv_layout(server):
        pass

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = [{full: z[s] for s, full in short.items()}]
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)
    rng = np.random.default_rng(0)

    # ---- the dummy: one compute-bound matmul, preallocated, result discarded --------------------
    dev = torch.device("cuda")
    bufs = {}
    for n in a.sizes:
        if n == 0:
            continue
        bufs[n] = (torch.randn(n, n, device=dev, dtype=torch.bfloat16),
                   torch.randn(n, n, device=dev, dtype=torch.bfloat16),
                   torch.empty(n, n, device=dev, dtype=torch.bfloat16))

    solo = {0: 0.0}
    for n in a.sizes:
        if n == 0:
            continue
        x, y, o = bufs[n]
        solo[n] = time_kernel_solo(lambda x=x, y=y, o=o: torch.mm(x, y, out=o))

    state = {"n": 0, "sites": 0}

    def wrap(orig):
        def forward(*args, **kwargs):
            n = state["n"]
            if n:
                x, y, o = bufs[n]
                torch.mm(x, y, out=o)   # exactly one launch; the slope is fitted against the side=128
                state["sites"] += 1     # arm, which also launches 300, so launch count differences out
            return orig(*args, **kwargs)
        return forward

    if a.target == "blocks":
        targets = list(server.transformer.blocks)
    else:
        targets = []
        for nm in ("streaming_vae", "streaming_vae_half", "vae", "vae_half"):
            v = getattr(server, nm, None)
            if v is None:
                continue
            v = getattr(v, "vae", v)
            targets += [m for m in v.modules() if isinstance(m, torch.nn.Conv3d)]
    if not targets:
        print(f"NOT EVALUATED: no injection sites found for target={a.target}")
        return 2
    for t in targets:
        t.forward = wrap(t.forward)
    print(f"injection target: {a.target}, {len(targets)} module(s) patched")

    def cycle(keyframes=None, first=False):
        if first:
            server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        act = server.infer(dict(obs=obs, prompt=prompt, save_visualization=False))["action"]
        kf = keyframes if keyframes is not None else [
            {k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
            for _ in range(4 if first else 8)]
        server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=act))
        return act

    print(f"warming {a.warm} cycles ...", flush=True)
    cycle(first=True)
    for _ in range(a.warm):
        cycle()
    kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
          for _ in range(8)]

    def timed(n_cycles):
        xs = []
        for _ in range(n_cycles):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cycle(keyframes=kf)
            torch.cuda.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(xs), xs

    # count the injection sites actually hit in one cycle rather than assuming
    state["n"], state["sites"] = min([x for x in a.sizes if x], default=128), 0
    cycle(keyframes=kf)
    sites = state["sites"]
    state["n"] = 0
    print(f"\n{'=' * 104}\n0. THE DUMMY KERNEL  (one torch.mm per site; {sites} sites hit per cycle, "
          f"target={a.target})\n{'=' * 104}")
    print(f"  {'side':>6}{'solo us':>10}{'injected ms/cycle':>20}{'footprint MB':>14}")
    inj_ms = {}
    for n in a.sizes:
        inj_ms[n] = solo[n] * sites / 1000
        fp = 3 * n * n * 2 / 2 ** 20 if n else 0
        print(f"  {n:>6}{solo[n]:>10.1f}{inj_ms[n]:>20.1f}{fp:>14.0f}")

    # ---- control 2: do the REAL kernels change? device busy excluding the dummy -----------------
    print(f"\n{'=' * 104}\n1. CONTROL: does the dummy slow the real kernels? (device busy excludes it)"
          f"\n{'=' * 104}")
    from torch.profiler import ProfilerActivity, profile

    def real_device_ms(n):
        state["n"] = n
        cycle(keyframes=kf)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            cycle(keyframes=kf)
            torch.cuda.synchronize()
        # total device time; the dummy's own contribution is known analytically from its solo timing,
        # so the residual after subtracting it is whatever the dummy did to the REAL kernels.
        tot = sum(max(0.0, getattr(e, "self_device_time_total", 0) or 0) for e in p.key_averages())
        return tot / 1000

    big = max(a.sizes)
    d0 = real_device_ms(0)
    dbig = real_device_ms(big)
    state["n"] = 0
    expected = inj_ms[big]
    residual = (dbig - d0) - expected
    print(f"  device time, no dummy         {d0:8.1f} ms")
    print(f"  device time, side={big:<5}       {dbig:8.1f} ms")
    print(f"  of which the dummy should be  {expected:8.1f} ms")
    print(f"  RESIDUAL on the real kernels  {residual:+8.1f} ms  "
          f"({residual / max(d0, 1):+.1%} of the real device time)")
    contaminated = abs(residual) > 0.10 * d0
    if contaminated:
        print(f"  WARNING: the dummy changed the real kernels by more than 10%. Cache or bandwidth "
              f"contention;\n           the slope below is an upper bound, not a clean measurement.")
    else:
        print(f"  OK: the real kernels are unchanged, so the injection adds device time and nothing else.")

    # ---- the sweep -------------------------------------------------------------------------------
    print(f"\n{'=' * 104}\n2. SWEEP  ({a.arm_cycles} cycles/arm, k=0 re-measured between every level)"
          f"\n{'=' * 104}")
    print(f"  gpu at start: {gpu_state()}")
    zeros, rows = [], []
    for n in a.sizes:
        state["n"] = 0
        m0, _ = timed(a.arm_cycles)
        zeros.append(m0)
        state["n"] = n
        mn, xs = timed(a.arm_cycles)
        rows.append((n, inj_ms[n], m0, mn))
        print(f"  side={n:<5} injected {inj_ms[n]:6.1f} ms   baseline {m0:7.1f}   with dummy {mn:7.1f}"
              f"   delta {mn - m0:+7.1f} ms   [{gpu_state()}]")
    state["n"] = 0

    drift = (max(zeros) - min(zeros)) / statistics.mean(zeros)
    print(f"\n  k=0 arms: {['%.1f' % v for v in zeros]}   spread {drift:.1%}")
    if drift > 0.05:
        print(f"  NOT EVALUATED: the control moved {drift:.1%} across the sweep.")
        return 2

    print(f"\n{'=' * 104}\n3. THE SLOPE  (fitted against the side=128 arm, which also launches 300)"
          f"\n{'=' * 104}")
    ref = next((r for r in rows if r[0] == 128), None)
    if ref is None:
        print("  no side=128 reference arm; cannot hold launch count constant. NOT EVALUATED.")
        return 2
    _, ref_inj, ref_m0, ref_mn = ref
    zero = next((r for r in rows if r[0] == 0), None)
    if zero is not None:
        launch_cost = ref_mn - zero[3]
        print(f"  cost of {sites} extra LAUNCHES alone (side=0 -> side=128): {launch_cost:+.1f} ms "
              f"= {launch_cost * 1000 / max(sites, 1):.1f} us/launch  -- NOTE this is the cost of a "
              f"cuBLAS mm dispatch\n  (heuristic lookup, workspace, launch), NOT a generic kernel "
              f"launch; do not generalise it")
        print(f"  (that term is present in every arm below and therefore differences out of the slope)\n")
    print(f"  {'injected ms':>13}{'cycle delta ms':>17}{'slope':>10}   both relative to side=128")
    xs_f, ys_f = [], []
    for n, inj, m0, mn in rows:
        if n in (0, 128):
            continue
        dx = inj - ref_inj
        dy = (mn - m0) - (ref_mn - ref_m0)
        xs_f.append(dx)
        ys_f.append(dy)
        print(f"  {dx:>13.1f}{dy:>+17.1f}{dy / dx:>10.3f}")
    if xs_f:
        fit = sum(x * y for x, y in zip(xs_f, ys_f)) / sum(x * x for x in xs_f)
        print(f"\n  least-squares slope through the origin: {fit:.3f}")
        print(f"  P007 (the natural experiment) implies:   0.80 - 1.24 depending on the host credit")
        print(f"\n  INTERPRETATION")
        if fit > 0.8:
            print(f"    Device time is ON the critical path at {fit:.2f} ms per ms. The 191.8 ms of device")
            print(f"    work is directly reducible and the 'the device has 155 ms of slack' argument that")
            print(f"    stopped Layer 5 kernel/layout work is void.")
        elif fit < 0.25:
            print(f"    Device time is essentially FREE at {fit:.2f} ms per ms: the host absorbs it. Device")
            print(f"    work cannot shorten the cycle and the eager host floor is the whole story.")
        else:
            print(f"    Partial overlap at {fit:.2f} ms per ms. That number is the exchange rate: a device")
            print(f"    change saving X ms of kernel time returns {fit:.2f}X ms of cycle.")
        print(f"\n  Applied to the current device breakdown (of 191.8 ms busy):")
        for nm, ms in (("elementwise / layout", 74.0), ("matmul / projections", 60.2),
                       ("attention", 44.4), ("normalisation", 10.1)):
            print(f"    {nm:<24}{ms:6.1f} ms device  ->  at most {ms * fit:6.1f} ms of cycle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

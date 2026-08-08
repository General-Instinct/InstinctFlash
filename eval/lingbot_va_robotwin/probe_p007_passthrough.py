#!/usr/bin/env python3
"""The positive control: does a REAL device-time change pass through to the cycle?

Injecting a dummy kernel into the transformer measured a device-time slope of -0.067 -- added device
work is absorbed. But that experiment probed only host-bound segments, and a wall of the form

    W(alpha) = SUM over segments of max(H_s, alpha * D_s)

has a slope equal to the DEVICE-BOUND segments' share of the total. An injection that lands entirely in
host-bound segments is predetermined to return zero and cannot distinguish "the device is free
everywhere" from "the device is free HERE".

So this probe uses ground truth instead of a dummy. P007 (conv NDHWC layout) is a real, shipped,
device-side change whose device-time delta can be measured directly, in the region where it acts, by
toggling it in-process with the revert path that backends/conv/apply.py already provides:

    pass-through = delta(cycle wall) / delta(device busy)

    ~1.0  the VAE encode is device-bound; P007's gain IS its kernel change; further device work there
          pays at that rate, while the transformer stays host-bound and pays nothing
    ~0    P007's 1.405x came from the ~56,600 host dispatches it also removed, not from the device,
          and the device-side attribution in LAYER5.md and released.py is wrong and must be corrected
          for a second time

Both arms measure BOTH quantities on the same footing:
  cycle wall   unprofiled, median over N cycles, ABBA-ordered (base, treat, treat, base)
  device busy  interval union from a CUDA-activities-only profile. That instrument inflates WALL time
               by ~1.29x but leaves device busy alone (190.9-191.8 ms across three instruments in
               LAYER6_GAPS.md), so it is used for device busy and never for wall.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY -u \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29981 probe_p007_passthrough.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from instinctwm.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "Kernel"}


def union_len(intervals):
    if not intervals:
        return 0.0
    xs = sorted(intervals)
    merged = [list(xs[0])]
    for s, e in xs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70)
    ap.add_argument("--settle", type=int, default=12, help="cycles after each toggle before timing")
    ap.add_argument("--arm-cycles", type=int, default=30)
    a = ap.parse_args()

    hot = [ln for ln in os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    ).read().strip().split("\n") if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%).")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_p007pt"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

    print("building server at 2V/4A (ring KV + prefill; conv layout toggled below) ...", flush=True)
    server = S.VA_Server(cfg)
    from instinctwm.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    for _ in install_conditioning_prefill(S, type(server)):
        pass
    for _ in install_debug_dump_elision(S):
        pass

    from instinctwm.backends.conv.apply import (install_conv_layout, plan_for_vae, revert_conv_plan)

    def vaes():
        out = []
        for nm in ("streaming_vae", "streaming_vae_half"):
            sv = getattr(server, nm, None)
            if sv is not None:
                out.append((nm, getattr(sv, "vae", sv)))
        return out

    def set_layout(on: bool):
        if on:
            return install_conv_layout(server)
        return [f"{nm}: reverted {revert_conv_plan(v)} Conv3d weights to contiguous"
                for nm, v in vaes()]

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = [{full: z[s] for s, full in short.items()}]
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)
    rng = np.random.default_rng(0)

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

    print(f"warming {a.warm} cycles (layout OFF) ...", flush=True)
    cycle(first=True)
    for _ in range(a.warm):
        cycle()
    kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
          for _ in range(8)]

    def wall(n):
        xs = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cycle(keyframes=kf)
            torch.cuda.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(xs), (max(xs) - min(xs)) / statistics.mean(xs)

    def device_busy():
        """Interval union of all device events. The CUDA-only profiler inflates wall, not this."""
        from torch.profiler import ProfilerActivity, profile
        cycle(keyframes=kf)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            cycle(keyframes=kf)
            torch.cuda.synchronize()
        path = "/tmp/iwm_p007pt_trace.json"
        p.export_chrome_trace(path)
        with open(path) as fh:
            evs = json.load(fh)["traceEvents"]
        dev = [e for e in evs if e.get("cat") in DEVICE_CATS and e.get("ph") == "X"]
        return union_len([(e["ts"], e["ts"] + e["dur"]) for e in dev]) / 1000, len(dev)

    def settle(on):
        for line in set_layout(on):
            print(f"    {line}")
        for _ in range(a.settle):
            cycle(keyframes=kf)

    print(f"\n{'=' * 100}\nARMS: ABBA (off, on, on, off), {a.arm_cycles} cycles each\n{'=' * 100}")
    arms = {"off": [], "on": []}
    busies = {"off": [], "on": []}
    counts = {"off": [], "on": []}
    for i, name in enumerate(["off", "on", "on", "off"]):
        settle(name == "on")
        w, spread = wall(a.arm_cycles)
        b, n = device_busy()
        arms[name].append(w)
        busies[name].append(b)
        counts[name].append(n)
        print(f"  arm {i + 1}  layout {name:3}   cycle {w:7.1f} ms (spread {spread:.1%})   "
              f"device busy {b:7.1f} ms   {n} device events")

    set_layout(False)
    w_off = sum(arms["off"]) / 2
    w_on = sum(arms["on"]) / 2
    b_off = sum(busies["off"]) / 2
    b_on = sum(busies["on"]) / 2
    drift_off = abs(arms["off"][0] - arms["off"][1]) / w_off
    drift_on = abs(arms["on"][0] - arms["on"][1]) / w_on

    print(f"\n{'=' * 100}\nRESULT\n{'=' * 100}")
    print(f"  cycle wall    off {arms['off'][0]:7.1f} / {arms['off'][1]:7.1f} -> {w_off:7.1f} ms   "
          f"drift {drift_off:.1%}")
    print(f"                on  {arms['on'][0]:7.1f} / {arms['on'][1]:7.1f} -> {w_on:7.1f} ms   "
          f"drift {drift_on:.1%}")
    print(f"  device busy   off {b_off:7.1f} ms      on {b_on:7.1f} ms")
    print(f"  device events off {counts['off'][0]:6d}         on {counts['on'][0]:6d}")
    d_wall = w_off - w_on
    d_dev = b_off - b_on
    print(f"\n  delta cycle wall   {d_wall:+7.1f} ms   ({w_off / w_on:.3f}x)")
    print(f"  delta device busy  {d_dev:+7.1f} ms")
    if max(drift_off, drift_on) > 0.05:
        print(f"  NOT EVALUATED: drift {max(drift_off, drift_on):.1%} > 5%.")
        return 2
    if abs(d_dev) < 5:
        print(f"  NOT EVALUATED: the device-time change is only {d_dev:.1f} ms; too small to divide by.")
        return 2
    pt = d_wall / d_dev
    print(f"\n  PASS-THROUGH = {d_wall:.1f} / {d_dev:.1f} = {pt:.2f} ms of cycle per ms of device time")
    print(f"\n  For comparison, the transformer-region injection measured -0.07 "
          f"(probe_device_slope.py).")
    if pt > 0.7:
        print(f"\n  => The VAE encode is DEVICE-BOUND. P007's speedup is its kernel change, the "
              f"device-side\n     attribution stands, and device work in THAT region pays at "
              f"{pt:.2f}. The transformer\n     remains host-bound at ~0. The cycle has two regimes and "
              f"they need different levers.")
    elif pt < 0.3:
        print(f"\n  => P007's device-time reduction did NOT pass through. Its 1.405x came from the "
              f"~56,600\n     host dispatches it also removed. The device-side attribution is wrong "
              f"and LAYER5.md,\n     LAYER5_CRITICAL_PATH.md and released.py must be corrected again.")
    else:
        print(f"\n  => Partial pass-through at {pt:.2f}. Both terms contributed; neither attribution "
              f"is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

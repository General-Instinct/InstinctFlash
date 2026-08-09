#!/usr/bin/env python3
"""The transformer below-knee slope, re-measured with the harness out of the way.

WHY AGAIN. probe_device_slope.py ran its profiled contamination control BEFORE the timed sweep, and a
torch.profiler CUDA context leaves permanent per-launch residue after it exits. Its k=0 arms read
396-403 ms where the same cycle reads 330.7 ms unprofiled. Extra HOST time is extra absorption capacity,
one for one, so that residue biases the below-knee slope TOWARD ZERO -- in the direction that makes the
"device work is worthless in the transformer" conclusion look stronger than it is.

That conclusion is load-bearing: it sets the ceiling on all remaining Layer 5 work. At slope 0.207 the
ceiling is 54 ms of cycle; at 0.45 it reaches the 100 ms bar. So the number has to be clean.

WHAT CHANGED
  1. NO PROFILER ANYWHERE. Not before, not during, not after. The x-axis comes from the dummy's clock
     count, not from a profiled duration.
  2. The dummy is torch.cuda._sleep(n) -- one launch, duration exactly linear in n, no memory traffic,
     no tensor cores, no L2 footprint, no power ramp. The matmul dummy confounded duration with
     bandwidth, cache and 656 W of power draw; this one cannot.
  3. The reference arm sleeps ~1 us rather than not launching, so launch count is identical in EVERY arm
     including the reference.
  4. HARD GATE: the first k=0 arm must land within 3% of 330.7 ms, the known unprofiled cycle. If the
     harness has inflated the baseline, the run is NOT EVALUATED rather than reported.
  5. Arms are counterbalanced (forward then reverse) so a monotonic drift cannot masquerade as slope.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY -u \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29977 probe_slope_clean.py
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
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

REFERENCE_CYCLE_MS = 330.7   # probe_device_gaps.py, unprofiled, median of 12


def sm_clock_hz():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits",
                              "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return float(out) * 1e6
    except Exception:
        return 1.98e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70)
    ap.add_argument("--arm-cycles", type=int, default=20)
    ap.add_argument("--us", type=float, nargs="+", default=[1, 200, 500, 1000, 1500],
                    help="microseconds of sleep per block execution; the first is the reference")
    a = ap.parse_args()

    hot = [ln for ln in os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    ).read().strip().split("\n") if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%).")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_slopeclean"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

    print("building server at 2V/4A, shipped stack ...", flush=True)
    server = S.VA_Server(cfg)
    from instinctwm.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    for _ in install_conditioning_prefill(S, type(server)):
        pass
    for _ in install_debug_dump_elision(S):
        pass
    from instinctwm.backends.conv.apply import install_conv_layout
    for _ in install_conv_layout(server):
        pass

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = [{full: z[s] for s, full in short.items()}]
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)
    rng = np.random.default_rng(0)

    hz = sm_clock_hz()
    state = {"cycles": 0, "sites": 0}
    blocks = server.transformer.blocks

    def wrap(orig):
        def fwd(*ar, **kw):
            n = state["cycles"]
            if n:
                torch.cuda._sleep(n)
                state["sites"] += 1
            return orig(*ar, **kw)
        return fwd

    for b in blocks:
        b.forward = wrap(b.forward)

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

    def timed(n):
        xs = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cycle(keyframes=kf)
            torch.cuda.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(xs), (max(xs) - min(xs)) / statistics.mean(xs)

    # ---- gate 1: is the harness itself clean? ---------------------------------------------------
    state["cycles"] = 0
    base0, spread0 = timed(a.arm_cycles)
    err = abs(base0 - REFERENCE_CYCLE_MS) / REFERENCE_CYCLE_MS
    print(f"\n{'=' * 100}\nGATE: harness baseline vs the known unprofiled cycle\n{'=' * 100}")
    print(f"  k=0 baseline {base0:.1f} ms (spread {spread0:.1%}) vs reference {REFERENCE_CYCLE_MS} ms"
          f"  -> {err:+.1%}")
    if err > 0.03:
        print(f"  NOT EVALUATED: the harness itself is {err:.1%} off the served cycle. Any slope measured "
              f"here\n  would describe the harness. (The previous probe read 396-403 ms here, +20%.)")
        print(f"  This gate is the whole point of the re-run; it is reporting a real problem, not a flake.")
        return 2
    print(f"  OK: within 3%, so the sweep below measures the runtime rather than the instrument.")

    # count sites actually hit
    state["cycles"], state["sites"] = 1000, 0
    cycle(keyframes=kf)
    sites = state["sites"]
    state["cycles"] = 0
    print(f"\n  injection sites hit per cycle: {sites}   SM clock {hz / 1e9:.2f} GHz")

    def cyc_for(us):
        return int(us * 1e-6 * hz)

    print(f"\n{'=' * 100}\nSWEEP  ({a.arm_cycles} cycles/arm, counterbalanced, k=0 between every level)"
          f"\n{'=' * 100}")
    order = list(a.us) + list(reversed(a.us))          # forward then reverse
    rows, zeros = [], []
    for us in order:
        state["cycles"] = 0
        m0, _ = timed(a.arm_cycles)
        zeros.append(m0)
        state["cycles"] = cyc_for(us)
        mk, sp = timed(a.arm_cycles)
        inj = us * sites / 1000.0
        rows.append((us, inj, m0, mk))
        print(f"  sleep {us:6.0f} us/block  injected {inj:7.1f} ms   base {m0:7.1f}   with {mk:7.1f}"
              f"   delta {mk - m0:+7.1f} ms   (spread {sp:.1%})")
    state["cycles"] = 0

    drift = (max(zeros) - min(zeros)) / statistics.mean(zeros)
    print(f"\n  k=0 arms: {['%.1f' % v for v in zeros]}  spread {drift:.1%}")
    if drift > 0.03:
        print(f"  NOT EVALUATED: control drift {drift:.1%} > 3%.")
        return 2

    # average the forward and reverse pass for each level
    agg = {}
    for us, inj, m0, mk in rows:
        agg.setdefault(us, []).append((inj, mk - m0))
    ref_us = a.us[0]
    ref_inj = statistics.mean(x for x, _ in agg[ref_us])
    ref_d = statistics.mean(d for _, d in agg[ref_us])

    print(f"\n{'=' * 100}\nMARGINAL SLOPE  (relative to the {ref_us:.0f} us reference arm, which also "
          f"launches {sites}x)\n{'=' * 100}")
    pts = [(0.0, 0.0)]
    for us in a.us[1:]:
        inj = statistics.mean(x for x, _ in agg[us]) - ref_inj
        dy = statistics.mean(d for _, d in agg[us]) - ref_d
        pts.append((inj, dy))
    print(f"  {'injected ms':>13}{'cycle delta ms':>17}{'marginal slope':>17}{'absorbed ms':>14}")
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        print(f"  {x1:>13.1f}{y1:>+17.1f}{(y1 - y0) / (x1 - x0):>17.3f}{x1 - y1:>14.1f}")

    below = [(x, y) for x, y in pts[1:] if x <= 160]
    if below:
        s = sum(y for _, y in below) / sum(x for x, _ in below)
        print(f"\n  BELOW-KNEE SLOPE (arms up to 160 ms injected): {s:.3f}")
        print(f"  previous, contaminated measurement: 0.207")
        tf, vae = 179.2, 16.6
        print(f"\n  CEILING on all remaining device-side work at this slope:")
        print(f"    transformer {tf:.1f} ms x {s:.3f} = {tf * s:.1f} ms")
        print(f"    VAE         {vae:.1f} ms x 1.0   = {vae:.1f} ms")
        print(f"    TOTAL {tf * s + vae:.1f} ms of a {REFERENCE_CYCLE_MS:.0f} ms cycle"
              f"   ({'BELOW' if tf * s + vae < 100 else 'AT OR ABOVE'} the 100 ms bar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

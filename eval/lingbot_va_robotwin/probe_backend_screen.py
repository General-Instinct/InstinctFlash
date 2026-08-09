#!/usr/bin/env python3
"""Screen the post-P007 execution stack for P007-class opportunities. No implementation.

P007 was found by asking why an operator was on a fallback path at all. This asks that question of
everything left, and prices each answer against the REGIME, because a device-side win is worth its
marginal slope and nothing more (LAYER6_REGIMES.md):

    transformer   ~156 ms device, marginal slope <= 0.2   ->  50 ms of device time buys 10 ms of cycle
    VAE            ~36 ms device, marginal slope ~ 1.0    ->  10 ms of device time buys 10 ms of cycle
    ceiling if EVERY device kernel became free            ->  ~67 ms of a 331 ms cycle

So the screen reports, per kernel: device ms, the region it runs in, and the CYCLE ms it could return if
it were eliminated entirely. Anything under ~10 ms of cycle is below the bar by construction.

WHY THE PROFILER IS SAFE HERE. It inflates wall time 1.3-2.0x but leaves DEVICE time alone -- device busy
read 190.9-191.8 ms across three instruments. This probe reports only device time, so scopes are free.

WHAT IT LOOKS FOR, the P007 question list:
  fallback kernels     names like slow_*, *vol2col*, *im2col*, unrolled_* -- a library declined
  library selection    GEMMs not on cuBLASLt (nvjet_*), attention not on cuDNN/flash
  memory layout        a kernel whose name encodes a layout the rest of the graph does not use
  algorithm choice     e.g. an O(n^2) path where a fused one exists
  duplicated execution the same kernel+shape run more than the algorithm requires
  eager elementwise    many small kernels doing one pass each, where the work is one fused pass

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY -u \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29978 probe_backend_screen.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from instinctwm.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "Kernel"}
# regime multipliers, measured in LAYER6_REGIMES.md
SLOPE = {"vae": 1.0, "transformer": 0.2, "other": 0.5}

FALLBACK_MARKERS = ("slow_", "vol2col", "im2col", "col2im", "unrolled", "_naive", "generic_")
LIB_MARKERS = {"cuBLASLt (nvjet)": ("nvjet",), "cuBLAS legacy": ("gemm_", "sgemm", "cutlass"),
               "cuDNN": ("cudnn",), "flash": ("flash",)}


def classify_region(scope: str) -> str:
    s = (scope or "").lower()
    if "vae" in s or "encode_obs" in s or "decode" in s:
        return "vae"
    if any(k in s for k in ("block", "attn", "ffn", "transformer")):
        return "transformer"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_screen"
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
    for line in install_conv_layout(server):
        print(f"  {line}")

    print(f"\n  config: guidance_scale={cfg.guidance_scale}, "
          f"action_guidance_scale={cfg.action_guidance_scale}, "
          f"use_cfg={(cfg.guidance_scale > 1) or (cfg.action_guidance_scale > 1)}")

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

    print(f"warming {a.warm} cycles ...", flush=True)
    cycle(first=True)
    for _ in range(a.warm):
        cycle()
    kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
          for _ in range(8)]

    # scopes: device time is instrument-independent, so annotating costs nothing that matters here
    undo = []

    def wrap(obj, attr, label):
        orig = getattr(obj, attr, None)
        if orig is None:
            return
        def w(*ar, **kw):
            with record_function(f"iwm::{label}"):
                return orig(*ar, **kw)
        setattr(obj, attr, w)
        undo.append(lambda o=obj, at=attr, fn=orig: setattr(o, at, fn))

    for attr in ("_encode_obs", "_infer", "_compute_kv_cache", "decode_one_video",
                 "postprocess_action", "_prepare_latent_input"):
        wrap(server, attr, attr.lstrip("_"))
    for nm in ("streaming_vae", "streaming_vae_half"):
        v = getattr(server, nm, None)
        if v is not None:
            wrap(getattr(v, "vae", v), "encode", f"{nm}.encode")
    wrap(server.transformer, "forward", "transformer.forward")
    for blk in server.transformer.blocks:
        wrap(blk, "forward", "block.forward")

    cycle(keyframes=kf)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        with record_function("iwm::cycle"):
            cycle(keyframes=kf)
        torch.cuda.synchronize()
    path = "/tmp/iwm_screen_trace.json"
    p.export_chrome_trace(path)
    for f in reversed(undo):
        f()

    with open(path) as fh:
        evs = json.load(fh)["traceEvents"]
    dev = [e for e in evs if e.get("cat") in DEVICE_CATS and e.get("ph") == "X"]
    ann = sorted([e for e in evs if e.get("cat") == "user_annotation" and e.get("ph") == "X"],
                 key=lambda e: e["ts"])
    rt = {(e.get("args") or {}).get("correlation"): e
          for e in evs if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"}

    # attribute each device event to the innermost scope containing its LAUNCH (host-side)
    starts = [e["ts"] for e in ann]
    import bisect

    def scope_of(t):
        i = bisect.bisect_right(starts, t)
        best = None
        for e in reversed(ann[max(0, i - 3000):i]):
            if e["ts"] + e.get("dur", 0) >= t and (best is None or e["ts"] > best["ts"]):
                best = e
        return best["name"].replace("iwm::", "") if best else "(unscoped)"

    by_kernel = collections.defaultdict(lambda: [0.0, 0, collections.Counter()])
    total = 0.0
    for e in dev:
        d = e.get("dur", 0) or 0
        total += d
        r = rt.get((e.get("args") or {}).get("correlation"))
        sc = scope_of(r["ts"]) if r else "(unscoped)"
        k = e["name"]
        rec = by_kernel[k]
        rec[0] += d
        rec[1] += 1
        rec[2][classify_region(sc)] += d

    print(f"\n{'=' * 122}\nDEVICE TIME BY KERNEL, post-P007  (total {total / 1000:.1f} ms; "
          f"cycle ~331 ms)\n{'=' * 122}")
    print(f"  {'ms':>7}{'calls':>7}{'us/call':>9}  {'region':<13}{'cycle ms if removed':>20}  kernel")
    rows = sorted(by_kernel.items(), key=lambda kv: -kv[1][0])
    screened = []
    for name, (ms, n, regions) in rows[:a.top]:
        ms /= 1000
        reg = regions.most_common(1)[0][0]
        share = regions.most_common(1)[0][1] / max(sum(regions.values()), 1)
        cyc = sum(v / 1000 * SLOPE[k] for k, v in regions.items())
        screened.append((name, ms, n, reg, cyc))
        print(f"  {ms:>7.1f}{n:>7}{ms * 1000 / max(n, 1):>9.1f}  {reg + ('' if share > 0.9 else '*'):<13}"
              f"{cyc:>20.1f}  {name[:60]}")

    print(f"\n{'=' * 122}\nP007-CLASS SCREEN\n{'=' * 122}")
    fb = [(k, v[0] / 1000) for k, v in by_kernel.items()
          if any(m in k.lower() for m in FALLBACK_MARKERS)]
    print(f"  FALLBACK KERNELS (slow_*, vol2col, im2col, unrolled, naive, generic):")
    if fb:
        for k, ms in sorted(fb, key=lambda x: -x[1]):
            print(f"    {ms:7.1f} ms  {k[:88]}")
    else:
        print(f"    NONE. This is the P007 question asked again and answered: no operator in the cycle "
              f"is on a\n    library fallback path any more.")

    print(f"\n  LIBRARY COVERAGE of the compute-heavy kernels:")
    for lib, marks in LIB_MARKERS.items():
        sel = [(k, v[0] / 1000) for k, v in by_kernel.items()
               if any(m in k.lower() for m in marks)]
        if sel:
            print(f"    {sum(m for _, m in sel):7.1f} ms across {len(sel):3d} kernels on {lib}")

    eager = [(k, v[0] / 1000, v[1]) for k, v in by_kernel.items()
             if "elementwise" in k.lower() or "vectorized" in k.lower()]
    e_ms = sum(m for _, m, _ in eager)
    e_n = sum(n for _, _, n in eager)
    print(f"\n  EAGER ELEMENTWISE: {e_ms:.1f} ms across {e_n} launches in {len(eager)} distinct kernels")
    print(f"    a fusing compiler is the only thing that removes these, and torch.compile is rejected "
          f"(LAYER6.md D)")

    dup = [(k, v[0] / 1000, v[1]) for k, v in by_kernel.items() if v[1] >= 600]
    print(f"\n  HIGHEST-COUNT KERNELS (>=600 launches/cycle) -- candidates for 'the same work repeated':")
    for k, ms, n in sorted(dup, key=lambda x: -x[2])[:8]:
        print(f"    {n:6d} launches {ms:7.1f} ms  {k[:74]}")

    print(f"\n{'=' * 122}\nCEILING\n{'=' * 122}")
    tf = sum(v[2]["transformer"] for v in by_kernel.values()) / 1000
    va = sum(v[2]["vae"] for v in by_kernel.values()) / 1000
    ot = sum(v[2]["other"] for v in by_kernel.values()) / 1000
    print(f"  device time by region:  transformer {tf:6.1f} ms   vae {va:6.1f} ms   other {ot:6.1f} ms")
    print(f"  x regime slope:         transformer {tf * 0.2:6.1f} ms   vae {va * 1.0:6.1f} ms   "
          f"other {ot * 0.5:6.1f} ms")
    print(f"  MAXIMUM cycle recoverable if EVERY device kernel became free: "
          f"{tf * 0.2 + va + ot * 0.5:.1f} ms of ~331 ms")
    print(f"\n  Any single candidate is a fraction of that. The 10 ms/cycle screening bar therefore "
          f"requires\n  removing {10 / 0.2:.0f} ms of transformer device time or {10 / 1.0:.0f} ms of "
          f"VAE device time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

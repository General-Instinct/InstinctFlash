#!/usr/bin/env python3
"""How many of the ~105k host ops per cycle fall inside the region a captured graph replaces?

That is the ceiling on what a persistent graph can remove, and it is the number the whole
graph-persistence design rests on. `graph_capture._stack` wraps the 30-block loop, so the ops a replay
eliminates are exactly the ones dispatched inside it.

Also reports the ring state per commit, so the saturation point is measured rather than derived from a
slots-per-cycle estimate.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from instinctwm.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70)
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_gscope"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4
    print("building server ...", flush=True)
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

    inside = {"v": False}
    n_in = collections.Counter()
    n_out = collections.Counter()

    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.on = False

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if self.on:
                nm = str(func).split(".")[-2] if "." in str(func) else str(func)
                (n_in if inside["v"] else n_out)[nm] += 1
            return func(*args, **kwargs)

    # Wrap each block's forward: everything dispatched within one is inside the graph's region.
    for blk in server.transformer.blocks:
        orig = blk.forward

        def wrapped(*ar, _o=orig, **kw):
            prev = inside["v"]
            inside["v"] = True
            try:
                return _o(*ar, **kw)
            finally:
                inside["v"] = prev
        blk.forward = wrapped

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = {"obs": [{full: z[s] for s, full in short.items()}], "state": z["state"]}
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)
    rng = np.random.default_rng(0)
    first = {"v": True}

    def cycle():
        if first["v"]:
            server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        act = server.infer(dict(obs=obs["obs"], prompt=prompt,
                                save_visualization=False))["action"]
        kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
              for _ in range(4 if first["v"] else 8)]
        server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=act))
        first["v"] = False

    # ---- ring saturation, measured -------------------------------------------------------------
    a0 = server.transformer.blocks[0].attn1
    print("\n=== ring progression: when does count reach total? ===", flush=True)
    first["v"] = True
    cycle()
    sat_at = None
    prev = None
    for i in range(1, a.warm + 1):
        cycle()
        sig = a0._iwm_ring_signature(server.cache_name)
        if sig is None:
            continue
        start, count = sig
        r = a0.attn_caches[server.cache_name]
        total = int(getattr(server, "kv_slots", 0)) or None
        if i in (1, 2, 5, 10, 20, 30, 32, 34, 36, 40, 50, 64, 70):
            print(f"  cycle {i:>3}  start={start:>5}  count={count:>5}")
        if prev is not None and count == prev[1] and sat_at is None:
            sat_at = i
            print(f"  => count STOPPED growing at cycle {i} (count={count}); "
                  f"start now advances instead")
        prev = (start, count)

    print("\n=== host ops inside vs outside the block stack (one warm cycle) ===", flush=True)
    m = Count()
    with m:
        m.on = True
        cycle()
        m.on = False
    tin, tout = sum(n_in.values()), sum(n_out.values())
    tot = tin + tout
    print(f"  inside the 30-block stack  {tin:>8}  {tin / tot:6.1%}   <- a persistent graph removes these")
    print(f"  outside it                 {tout:>8}  {tout / tot:6.1%}   <- VAE, schedulers, bookkeeping")
    print(f"  total                      {tot:>8}")
    print(f"\n  top ops INSIDE:  " + ", ".join(f"{k}={v}" for k, v in n_in.most_common(8)))
    print(f"  top ops OUTSIDE: " + ", ".join(f"{k}={v}" for k, v in n_out.most_common(8)))
    print(f"\n  At ~3.2 us/op, removing the inside set saves ~{tin * 3.2 / 1000:.0f} ms of a 338 ms")
    print(f"  cycle. Device work is 196 ms, so the floor is max(196, {tout * 3.2 / 1000:.0f} + replay).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

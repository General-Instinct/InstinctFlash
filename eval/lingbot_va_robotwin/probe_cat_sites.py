#!/usr/bin/env python3
"""What is `aten::cat` materializing, and can the consumer read it without a contiguous copy?

THE NUMBER. 19.68 ms/cycle, 172 calls, 114.4 us each, ONE shape signature -- the most concentrated
line left in the profile after the conv-layout fix, and unchanged by it.

THE HYPOTHESIS, AND WHY IT IS SUSPECT. `ring_kv.py:197-199` materializes the live KV window when the
ring interval wraps:

    key_all = torch.cat([kp[:, :end], kp[:, start:]], dim=1)      # two spans -> one contiguous buffer

At the measured attention shape [2, 24, 9792, 128] bf16 that is 120 MB per tensor and 240 MB for the
K/V pair -- about 120 us at this box's bandwidth, which matches 114.4 us almost exactly.

But the same function takes a VIEW in both other cases: a plain slice before the ring wraps, and the
whole pool once `count >= total`. So the cat branch should be reachable only during the wrap
transition, which is a handful of cycles out of 64 -- and that is inconsistent with 172 calls EVERY
cycle. A hypothesis that predicts its own rarity and is contradicted by the call count is probably
wrong, so this probe counts rather than assumes.

WHAT IT ANSWERS, in order:
  1. how many cats there are per cycle, where, and how many BYTES each moves
  2. specifically, how many are the ring-wrap path vs everything else
  3. whether the consumer could take two spans instead: attention is called as
     `self.attn_op(query, key_all, value_all)`, so if the cat is real and hot, the question is whether
     attn_op can be handed (span_a, span_b) -- which is a backend capability question, not a kernel.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29991 probe_cat_sites.py
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import traceback
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70, help="past ring saturation (~cycle 64)")
    ap.add_argument("--conv-layout", choices=["as-is", "ndhwc"], default="ndhwc")
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_cat"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4
    print("building server ...", flush=True)
    server = S.VA_Server(cfg)
    from instinctflash.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    for n in install_conditioning_prefill(S, type(server)):
        print(f"  installed {n}", flush=True)
    for n in install_debug_dump_elision(S):
        print(f"  installed {n}", flush=True)
    if a.conv_layout == "ndhwc":
        from instinctflash.backends.conv.apply import install_conv_layout
        for line in install_conv_layout(server):
            print(f"  {line}", flush=True)

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    if not ctx:
        raise SystemExit("no contexts; run collect_contexts.sh")
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = {"obs": [{full: z[s] for s, full in short.items()}], "state": z["state"]}
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)
    rng = np.random.default_rng(0)

    def cycle(first):
        if first:
            server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        act = server.infer(dict(obs=obs["obs"], prompt=prompt,
                                save_visualization=False))["action"]
        kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
              for _ in range(4 if first else 8)]
        server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=act))
        return act

    print(f"warming {a.warm} cycles (ring saturates ~64) ...", flush=True)
    cycle(True)
    for _ in range(a.warm):
        cycle(False)

    # `torch.cat` IS Python-visible, unlike aten::copy_ which is emitted inside C++ ops -- that is why
    # patching copy_ counted zero and patching cat works.
    sites = collections.Counter()
    nbytes = collections.Counter()
    orig = torch.cat
    ON = {"v": False}

    def site():
        for f in reversed(traceback.extract_stack()[:-2]):
            fn = f.filename
            if "/torch/" in fn or "probe_cat_sites" in fn:
                continue
            tag = ("iwm" if "/instinctflash/" in fn else
                   "lingbot" if "/wan_va/" in fn or "/lingbot" in fn else
                   "diffusers" if "diffusers" in fn else "other")
            return f"[{tag}] {Path(fn).name}:{f.lineno} {f.name}"
        return "?"

    def counted(tensors, *args, **kw):
        if ON["v"]:
            s = site()
            sites[s] += 1
            try:
                nbytes[s] += sum(t.numel() * t.element_size() for t in tensors)
            except Exception:
                pass
        return orig(tensors, *args, **kw)

    torch.cat = counted
    ON["v"] = True
    cycle(False)
    ON["v"] = False
    torch.cat = orig

    total = sum(sites.values())
    tot_mb = sum(nbytes.values()) / 2**20
    print(f"\n{'=' * 104}\naten::cat CALL SITES, one warm cycle: {total} calls, "
          f"{tot_mb:.1f} MiB read\n{'=' * 104}")
    print(f"{'calls':>7}{'MiB read':>11}{'MiB/call':>10}  site")
    print("-" * 104)
    for s, n in sites.most_common(18):
        mb = nbytes[s] / 2**20
        print(f"{n:>7}{mb:>11.1f}{mb / max(n, 1):>10.2f}  {s}")
    print("-" * 104)
    ring = {s: n for s, n in sites.items() if "ring_kv" in s}
    print(f"\n  ring-wrap materialisations: {sum(ring.values())} of {total} calls, "
          f"{sum(nbytes[s] for s in ring) / 2**20:.1f} MiB")
    if not ring:
        print("  => the ring-wrap hypothesis is WRONG. The cat cost is somewhere else entirely, and")
        print("     'let the consumer read two spans' would optimise a path that is not taken.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

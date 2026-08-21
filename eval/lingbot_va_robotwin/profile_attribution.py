#!/usr/bin/env python3
"""The Layer 5 report: every dominant operator, grouped by callsite.

Operator totals have twice pointed at the wrong work. `aten::copy_` was the largest line in the Fast
profile and 82% of it was `slow_conv_dilated3d`'s `vol2col` lowering -- fixed by a layout decision two
layers away, with no copy kernel written. Before that, a RoPE kernel was written against a site that
turned out to be ~3% of the copies, because the sample it was chosen from covered 12% of them.

So this reports (operator, callsite, calls, bytes, shapes, exclusive time) and, for every operator,
what fraction of its calls were actually attributed. An operator below the coverage threshold is
printed with [PARTIAL, NOT RANKABLE] and must not be used to choose work.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29992 \\
        profile_attribution.py [--warm 70] [--conv-layout ndhwc]
"""
from __future__ import annotations

import argparse
import os
import sys
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
from instinctflash.verify.attribution import attribute  # noqa: E402

#: The operators the warm profile shows as dominant. Watching everything would work and be very slow;
#: these are the ones a Layer 5 decision would be made about.
WATCH_SETS = {
    # The Layer 5 set: operators that move or compute bytes.
    "device": ("copy_", "cat", "addmm", "add", "mul", "fill_", "clone", "contiguous", "_to_copy",
               "empty", "index_put_", "slice", "select", "view", "reshape"),
    # The Layer 6 set: operators that launch NO kernel and exist only to describe a tensor. This is
    # the largest population in the cycle by count (47,020 of 105,123 aten events, 44.7%) and the
    # same rule applies to it as to any other -- a callsite distribution before a decision.
    "metadata": ("as_strided", "view", "transpose", "slice", "reshape", "t", "narrow", "squeeze",
                 "unsqueeze", "flatten", "unflatten", "expand", "permute", "select", "detach"),
    # Allocation: 4.0 us/op, the most expensive class after real kernels.
    "allocation": ("empty", "empty_strided", "empty_like", "lift_fresh", "clone", "zeros",
                   "ones", "full", "scalar_tensor", "_to_copy"),
}
WATCH = WATCH_SETS["device"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70, help="past ring saturation (~cycle 64)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--watch-set", choices=sorted(WATCH_SETS), default="device",
                    help="'device' moves bytes (Layer 5); 'metadata' launches no kernel (Layer 6); "
                         "'allocation' creates storage")
    ap.add_argument("--conv-layout", choices=["as-is", "ndhwc"], default="ndhwc")
    ap.add_argument("--video", type=int, default=2)
    ap.add_argument("--action", type=int, default=4)
    a = ap.parse_args()

    hot = [ln for ln in os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    ).read().strip().split("\n") if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%).")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_attr"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = a.video, a.action

    print(f"building server at {a.video}V/{a.action}A ...", flush=True)
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
        return act

    print(f"warming {a.warm} cycles (ring saturates ~64) ...", flush=True)
    for _ in range(a.warm):
        cycle()

    watch = WATCH_SETS[a.watch_set]
    print(f"attributing {a.repeats} warm cycles across {len(watch)} watched operators "
          f"(set={a.watch_set}) ...", flush=True)
    rep = attribute(cycle, watch=watch, repeats=a.repeats)

    print(f"\n{'=' * 118}\nOPERATOR x CALLSITE  (2V/4A warm, conv-layout={a.conv_layout})\n"
          f"{'=' * 118}")
    print(rep.format_table(top=6))
    print(f"\n{rep.coverage_warnings()}")

    print("\n" + "=" * 118)
    print("WHICH KIND OF OPTIMIZATION EACH DOMINANT CALLSITE CALLS FOR")
    print("=" * 118)
    print("  Read the table above, then classify. The classes are not interchangeable and the")
    print("  cheapest one that applies wins:")
    print("    backend dispatch          the operator is on a fallback path and a better backend")
    print("                              already exists  (the conv fix: 1.49x, no kernel)")
    print("    layout planning           the fast backend exists but declines this layout")
    print("    materialization removal   the tensor is built only to be consumed once; the consumer")
    print("                              could read the parts  (candidate: the ring KV window)")
    print("    operator fusion           several ops over one tensor, each re-reading it")
    print("    custom kernel             none of the above applies and the arithmetic is the cost")
    print("  Nothing here justifies a kernel until the three cheaper classes are ruled out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Exact Python call sites for the 34,710 copy_ / 29,681 fill_ calls per warm cycle.

WHY NOT THE PROFILER. `with_stack=True` returned empty stacks for these events, and `record_shapes`
reports `aten::cat`'s TensorList as `[[], []]`. Neither can name the source line. But a kernel cannot
be written against "aten::copy_, 105 shapes" -- it has to be written against a call site.

So this wraps `torch.Tensor.copy_` and `torch.Tensor.fill_` and counts by CALLER, using
`traceback.extract_stack`. That is far too slow to time anything, and timing is not the point: the
device cost is already known (66.25 ms and 1.69 ms per cycle). What is missing is WHERE, and a count
per call site answers it exactly.

The distinction that decides the kernel design:

  ONE site, many calls   -> a batched/grouped kernel replacing N launches with 1. Reusable.
  MANY sites, few each   -> no kernel helps; the caller is the problem.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29987 probe_copy_sites.py
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
    ap.add_argument("--warm", type=int, default=34)
    ap.add_argument("--video", type=int, default=2)
    ap.add_argument("--action", type=int, default=4)
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_copy_sites"
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

    print(f"warming {a.warm} cycles ...", flush=True)
    cycle(True)
    for _ in range(a.warm):
        cycle(False)

    # ---- count by caller ----------------------------------------------------------------------
    sites = collections.Counter()
    shapes = collections.defaultdict(collections.Counter)
    orig_copy = torch.Tensor.copy_
    orig_fill = torch.Tensor.fill_
    ON = {"v": False}

    def site_of():
        """The innermost frame outside torch itself -- the code that asked for the copy."""
        for f in reversed(traceback.extract_stack()[:-2]):
            if "/torch/" in f.filename or f.filename.endswith("probe_copy_sites.py"):
                continue
            return f"{Path(f.filename).name}:{f.lineno} in {f.name}"
        return "?"

    def counted_copy(self, other, *args, **kw):
        if ON["v"]:
            s = site_of()
            sites[("copy_", s)] += 1
            shapes[("copy_", s)][tuple(self.shape)] += 1
        return orig_copy(self, other, *args, **kw)

    def counted_fill(self, value):
        if ON["v"]:
            s = site_of()
            sites[("fill_", s)] += 1
            shapes[("fill_", s)][tuple(self.shape)] += 1
        return orig_fill(self, value)

    torch.Tensor.copy_ = counted_copy
    torch.Tensor.fill_ = counted_fill
    print("counting one warm cycle (instrumented, SLOW -- counts only, not timings) ...", flush=True)
    ON["v"] = True
    cycle(False)
    ON["v"] = False
    torch.Tensor.copy_ = orig_copy
    torch.Tensor.fill_ = orig_fill

    total = sum(sites.values())
    print(f"\n{'=' * 100}\nCALL SITES, one warm cycle: {total} counted calls\n{'=' * 100}")
    print(f"{'op':<7}{'calls':>8}{'share':>8}  site / dominant shape")
    print("-" * 100)
    for (op, s), n in sites.most_common(22):
        dom, dn = shapes[(op, s)].most_common(1)[0]
        nsh = len(shapes[(op, s)])
        print(f"{op:<7}{n:>8}{n / total:>7.1%}  {s}")
        print(f"{'':<23}  dominant shape {dom} ({dn}/{n}), {nsh} distinct")
    print("-" * 100)
    top = sites.most_common(1)[0] if sites else None
    if top:
        print(f"\nCONCENTRATION: the single largest site is {top[1] / total:.1%} of all counted "
              f"calls.\n  {top[0][1]}")
        print("  A grouped kernel is worth writing when one site dominates; if the top site is a few")
        print("  percent, the launches are spread and the caller has to change instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

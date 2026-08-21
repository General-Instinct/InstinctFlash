#!/usr/bin/env python3
"""Localize the 34,710 copy_ and 172 cat calls per warm cycle, via TorchDispatchMode.

FOURTH ATTEMPT, and the previous three are recorded because each failed differently and each failure
narrowed the search:

  1. `profile(with_stack=True)` then reading `ev.stack`   -> empty for these ops
  2. `key_averages(group_by_stack_n=6)`                   -> empty frame lists
  3. monkeypatching `torch.Tensor.copy_` / `.fill_`       -> ZERO calls counted, which is the
     informative one: these copies are not Python `.copy_()` calls at all. They are emitted INSIDE
     C++ ops -- slice assignment, `.to()`, `contiguous()`, conv fallbacks -- so only the dispatcher
     sees them.

`TorchDispatchMode` intercepts at the dispatcher, below Python-level operator sugar and above the
kernel, which is exactly the layer where an `aten::copy_` emitted by `x[a:b] = y` becomes visible. The
Python stack at that point names the code that *asked* for the work, which is what a kernel has to be
written against.

Counts only. This mode is far too slow to time anything, and timing is already known from
profile_layer5_ops.py: copy_ 66.42 ms/cycle over 34,710 calls (1.9 us each -- launch-bound), cat 21.60
ms/cycle over 172 calls (125.6 us each -- bandwidth-bound).

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29988 probe_dispatch_sites.py
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
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from instinctflash.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

WATCH = ("copy_", "cat", "fill_", "empty", "zeros", "contiguous", "clone", "_to_copy", "pad")


class CountSites(TorchDispatchMode):
    """Count watched aten ops by the Python frame that caused them."""

    def __init__(self):
        super().__init__()
        self.sites = collections.Counter()
        self.shapes = collections.defaultdict(collections.Counter)
        self.enabled = False

    def _frame(self):
        for f in reversed(traceback.extract_stack()[:-3]):
            fn = f.filename
            if "/torch/" in fn or "probe_dispatch_sites" in fn or "_python_dispatch" in fn:
                continue
            tag = "lingbot" if "/wan_va/" in fn or "/lingbot" in fn else (
                "iwm" if "/instinctflash/" in fn else (
                    "diffusers" if "diffusers" in fn else "other"))
            return f"[{tag}] {Path(fn).name}:{f.lineno} {f.name}"
        return "?"

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func).split(".")[-2] if "." in str(func) else str(func)
        if self.enabled and any(name == w for w in WATCH):
            s = self._frame()
            self.sites[(name, s)] += 1
            try:
                shp = tuple(args[0].shape) if args and hasattr(args[0], "shape") else ()
                self.shapes[(name, s)][shp] += 1
            except Exception:
                pass
        return func(*args, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=34)
    ap.add_argument("--video", type=int, default=2)
    ap.add_argument("--action", type=int, default=4)
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_dispatch"
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

    def cycle(first, mode=None):
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

    print("counting one warm cycle under TorchDispatchMode (slow) ...", flush=True)
    m = CountSites()
    with m:
        m.enabled = True
        cycle(False)
        m.enabled = False

    total = sum(m.sites.values())
    print(f"\n{'=' * 104}\nDISPATCH-LEVEL CALL SITES, one warm cycle: {total} watched calls"
          f"\n{'=' * 104}")
    if total == 0:
        print("  NOT EVALUATED: TorchDispatchMode saw none of the watched ops. Four methods have now")
        print("  failed to attribute them; do not write a kernel against an unlocalized pattern.")
        return 2
    print(f"{'op':<12}{'calls':>8}{'share':>8}  site")
    print("-" * 104)
    for (op, s), n in m.sites.most_common(24):
        sh = m.shapes[(op, s)]
        dom = sh.most_common(1)[0] if sh else ((), 0)
        print(f"{op:<12}{n:>8}{n / total:>7.1%}  {s}")
        print(f"{'':<28}dominant shape {dom[0]} x{dom[1]}, {len(sh)} distinct")
    by_op = collections.Counter()
    for (op, _), n in m.sites.items():
        by_op[op] += n
    print("-" * 104)
    print("  totals by op: " + ", ".join(f"{o}={n}" for o, n in by_op.most_common()))
    top = m.sites.most_common(1)[0]
    print(f"\n  TOP SITE holds {top[1] / total:.1%} of watched calls: {top[0][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

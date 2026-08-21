#!/usr/bin/env python3
"""How many dispatcher ops survive when one transformer block is compiled? The ceiling, measured.

Layer 6's top candidate is "one persistent execution object per block": replace 121 dispatcher calls
(and ~350 profiler-visible aten events) with a single compiled callable. The CEILING is the metadata +
allocation + bookkeeping share; the ACTUAL number needs a compile.

mode="default" ON PURPOSE. `reduce-overhead` enables CUDA graphs, which is out of scope by instruction
and separately measured as unprofitable here. So this isolates dispatch elimination via fusion from
graph replay -- they are different mechanisms and only one of them is being asked about.

Counts at BOTH levels, because they answer different questions:
  dispatcher level (TorchDispatchMode) -- what the ATen dispatcher sees
  profiler level (aten:: events)       -- what the host-cost model was calibrated against
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
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


def count_dispatch(fn):
    c = collections.Counter()

    class M(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            nm = str(func).split(".")[-2] if "." in str(func) else str(func)
            c[nm] += 1
            return func(*args, **kwargs)
    with M():
        fn()
    return sum(c.values()), c


def count_profiler(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as p:
        fn()
        torch.cuda.synchronize()
    n = sum(e.count for e in p.key_averages() if e.key.startswith("aten::"))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=40)
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_cdisp"
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

    print(f"warming {a.warm} cycles ...", flush=True)
    for _ in range(a.warm):
        cycle()

    # Capture one real block call's arguments.
    blk = server.transformer.blocks[0]
    orig = blk.forward
    grab = {"v": None}

    def snoop(*ar, **kw):
        if grab["v"] is None:
            grab["v"] = (ar, kw)
        return orig(*ar, **kw)
    blk.forward = snoop
    cycle()
    blk.forward = orig
    ar, kw = grab["v"]
    print(f"  captured a block call: {len(ar)} positional, {sorted(kw)} keyword")

    n_eager, c_eager = count_dispatch(lambda: orig(*ar, **kw))
    p_eager = count_profiler(lambda: orig(*ar, **kw))
    print(f"\n  EAGER    dispatcher {n_eager:>5} ops   profiler {p_eager:>5} aten events")

    print("\n  compiling (mode=default, no cuda graphs) ...", flush=True)
    try:
        comp = torch.compile(orig, mode="default", dynamic=False)
        for _ in range(3):
            comp(*ar, **kw)       # warm the compile
        torch.cuda.synchronize()
        n_c, c_c = count_dispatch(lambda: comp(*ar, **kw))
        p_c = count_profiler(lambda: comp(*ar, **kw))
        print(f"  COMPILED dispatcher {n_c:>5} ops   profiler {p_c:>5} aten events")
        print(f"\n  dispatcher reduction {n_eager} -> {n_c} = {1 - n_c / max(n_eager,1):.0%}")
        print(f"  profiler   reduction {p_eager} -> {p_c} = {1 - p_c / max(p_eager,1):.0%}")
        print(f"\n  extrapolated to 300 block executions per cycle:")
        print(f"    dispatcher ops/cycle inside blocks: {n_eager*300} -> {n_c*300} "
              f"({(n_eager-n_c)*300} removed)")
        print(f"    profiler ops/cycle inside blocks:   {p_eager*300} -> {p_c*300} "
              f"({(p_eager-p_c)*300} removed)")
        print(f"    at ~3.2 us per profiler op: {(p_eager-p_c)*300*3.2/1000:.0f} ms of a 351 ms cycle")
        # Exactness is a separate question, but report the delta so the tier is not a guess.
        with torch.no_grad():
            ref = orig(*ar, **kw)
            got = comp(*ar, **kw)
        if isinstance(ref, torch.Tensor) and isinstance(got, torch.Tensor):
            nd = int((ref.contiguous().view(torch.int16)
                      != got.contiguous().view(torch.int16)).sum()) if ref.dtype == torch.bfloat16 \
                 else -1
            d = float((ref.float() - got.float()).abs().max())
            print(f"\n  numerics: max|delta| = {d:.3e}, differing bf16 words = "
                  f"{nd if nd >= 0 else 'n/a'} of {ref.numel()}")
            print(f"  => {'BITEXACT' if d == 0.0 else 'NUMERIC'} for this block, this shape. Fusion "
                  f"changes rounding boundaries, so a NUMERIC result here is expected, not a defect.")
        top_removed = collections.Counter()
        for k2, v in c_eager.items():
            top_removed[k2] = v - c_c.get(k2, 0)
        print(f"\n  ops most reduced: " +
              ", ".join(f"{k2}={v}" for k2, v in top_removed.most_common(10) if v > 0))
    except Exception as e:
        print(f"  NOT EVALUATED: torch.compile raised {type(e).__name__}: {str(e)[:200]}")
        print("  That is itself a finding: if the block does not compile, the top-ranked Layer 6")
        print("  candidate is blocked and the reason belongs in the proposal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

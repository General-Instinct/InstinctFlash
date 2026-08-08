#!/usr/bin/env python3
"""What does ONE host operation actually cost the cycle? Measure the slope, do not divide.

The `~3.2 us/op` figure that has driven four rounds of work was obtained by DIVIDING cycle time by
operation count. That is circular: it assumes the cycle is the sum of per-operation host costs, which is
the very thing in question. `probe_prebound_projection.py` then removed 12,190 of 105,123 operations
(11.6%), bit-exactly, and the cycle got 3.4 ms SLOWER -- so the assumption is wrong somewhere.

This probe measures the derivative instead. Inject K extra no-kernel dispatches into every block
execution, sweep K, and fit d(cycle)/d(ops). The injected operation is `as_strided` on an existing
tensor: pure stride arithmetic, no kernel, no allocation, no numerics -- the same class as the 47,020
metadata operations that dominate the cycle by count.

  slope ~ 3.2 us/op  -> the cycle IS host-throughput-bound and dispatch removal is worth the arithmetic
  slope ~ 0          -> the host runs ahead of the device and dispatch removal buys nothing;
                        the 155 ms of device "idle" is dependency stalls, not host starvation

The result decides whether Layer 6 exists as a direction. Nothing is optimized here; the injection
makes the runtime strictly SLOWER on purpose.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29998 probe_host_op_slope.py
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70)
    ap.add_argument("--arm-cycles", type=int, default=14)
    ap.add_argument("--inject", type=int, nargs="+", default=[0, 10, 25, 50, 100],
                    help="extra no-kernel dispatches per block execution")
    a = ap.parse_args()

    hot = [ln for ln in os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    ).read().strip().split("\n") if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%). This probe is a "
              f"latency measurement and cannot run on a contended device.")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_slope"
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

    # ---- the injection --------------------------------------------------------------------------
    blocks = server.transformer.blocks
    N_BLOCK_EXEC = len(blocks) * (cfg.num_inference_steps + cfg.action_num_inference_steps
                                  + 4)  # reported, not relied on; the real count is measured below
    state = {"k": 0}
    originals = [b.forward for b in blocks]

    def wrap(orig):
        def forward(hidden_states, *args, **kwargs):
            k = state["k"]
            if k:
                sz = hidden_states.shape
                st = hidden_states.stride()
                for _ in range(k):
                    # one dispatch, no kernel, no allocation, result discarded
                    torch.as_strided(hidden_states, sz, st)
            return orig(hidden_states, *args, **kwargs)
        return forward

    for b, orig in zip(blocks, originals):
        b.forward = wrap(orig)

    def cycle(rng, first=False):
        if first:
            server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        act = server.infer(dict(obs=obs, prompt=prompt, save_visualization=False))["action"]
        kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
              for _ in range(4 if first else 8)]
        server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=act))
        return act

    def warm(n, seed=7):
        rng = np.random.default_rng(seed)
        cycle(rng, first=True)
        for _ in range(n):
            cycle(rng)
        return rng

    def timed(n, rng):
        xs = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cycle(rng)
            torch.cuda.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(xs), xs

    def count_ops(rng):
        from torch.profiler import ProfilerActivity, profile
        cycle(rng)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            cycle(rng)
            torch.cuda.synchronize()
        return sum(e.count for e in p.key_averages() if e.key.startswith("aten::"))

    print(f"warming {a.warm} cycles ...", flush=True)
    rng = warm(a.warm)

    # measure the true op count at each injection level, then the wall time
    print(f"\n{'=' * 100}\nOP COUNT vs INJECTION  (verifying the injection lands where intended)"
          f"\n{'=' * 100}")
    counts = {}
    for k in a.inject:
        state["k"] = k
        counts[k] = count_ops(rng)
        print(f"  inject {k:4d}/block -> {counts[k]:7d} aten events/cycle "
              f"({counts[k] - counts[a.inject[0]]:+7d} vs k={a.inject[0]})")
    state["k"] = 0

    # ABBA-style: return to k=0 between every level so drift is visible and cancels
    print(f"\n{'=' * 100}\nCYCLE TIME vs INJECTION  ({a.arm_cycles} cycles/arm, k=0 re-measured "
          f"between levels)\n{'=' * 100}")
    rng = warm(15)
    rows = []
    zero_arms = []
    for k in a.inject:
        state["k"] = 0
        m0, _ = timed(a.arm_cycles, rng)
        zero_arms.append(m0)
        state["k"] = k
        mk, xs = timed(a.arm_cycles, rng)
        rows.append((k, counts[k], m0, mk))
        print(f"  k={k:4d}  baseline {m0:7.1f} ms   injected {mk:7.1f} ms   "
              f"delta {mk - m0:+7.1f} ms   min/max {min(xs):.1f}/{max(xs):.1f}")
    state["k"] = 0

    drift = (max(zero_arms) - min(zero_arms)) / statistics.mean(zero_arms)
    print(f"\n  k=0 arms: {['%.1f' % v for v in zero_arms]}  spread {drift:.1%}")
    if drift > 0.05:
        print(f"  NOT EVALUATED: the k=0 baseline moved {drift:.1%} across the sweep.")
        return 2

    print(f"\n{'=' * 100}\nTHE SLOPE\n{'=' * 100}")
    print(f"  {'extra ops/cycle':>16}{'delta ms':>10}{'us per host op':>16}")
    slopes = []
    for k, c, m0, mk in rows:
        extra = c - counts[a.inject[0]]
        if extra <= 0:
            continue
        us = (mk - m0) * 1000 / extra
        slopes.append(us)
        print(f"  {extra:>16d}{mk - m0:>+10.1f}{us:>16.3f}")
    if slopes:
        # least-squares through the origin over all levels
        xs = [c - counts[a.inject[0]] for k, c, m0, mk in rows if c - counts[a.inject[0]] > 0]
        ys = [(mk - m0) * 1000 for k, c, m0, mk in rows if c - counts[a.inject[0]] > 0]
        fit = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
        print(f"\n  least-squares slope through the origin: {fit:.3f} us per host operation")
        print(f"  the model in LAYER5_CRITICAL_PATH.md assumes:  3.200 us per host operation")
        print(f"  ratio measured/assumed: {fit / 3.2:.2f}x")
        print(f"\n  What 12,190 removed operations are therefore worth: "
              f"{12190 * fit / 1000:+.1f} ms")
        print(f"  What probe_prebound_projection.py actually measured:  -3.4 ms "
              f"(i.e. 3.4 ms slower)")
        if fit < 1.0:
            print(f"\n  VERDICT: the host is NOT the constraint. At {fit:.2f} us/op, removing every "
                  f"single\n           no-kernel dispatch in the cycle (77,497 of them) would buy "
                  f"{77497 * fit / 1000:.0f} ms.\n           Dispatch elimination is not a viable "
                  f"direction at this operating point.")
        else:
            print(f"\n  VERDICT: the host contributes {fit:.2f} us/op. Dispatch removal is worth "
                  f"pursuing,\n           and the prebound-projection null result needs a different "
                  f"explanation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

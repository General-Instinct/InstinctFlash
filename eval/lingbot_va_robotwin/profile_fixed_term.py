#!/usr/bin/env python3
"""Decompose the FIXED term of the cost model into named components, at the Fast operating point.

WHY THIS EXISTS

The warm cost model at Fast (2 video / 4 action) is

    cycle = FIXED 1164 ms + 15.5 ms/forward        (R^2 = 0.994, 5 operating points)

so at 6 forwards per cycle, 93 ms is denoising and ~1164 ms is something else. Every optimization
ranking we hold addresses the per-forward term -- graph capture, attention backends, fused kernels --
and at Fast that term is 7% of the cycle. The next optimization should come from THIS breakdown, not
from the layer roadmap.

`FIXED` is a regression intercept. It is not a component. This probe turns it into components.

HOW

The latency numbers we quote come from `probe_episode.py`, a websocket client, so they include
transport and serialisation as well as server compute. This probe runs the real server IN-PROCESS --
no websocket -- and wraps each phase of a cycle, so the sum of parts can be compared against both:

    sum(components)        what the server actually spends, attributed
    server cycle total     the same cycle measured end to end in-process
    client cycle total     what probe_episode reports  (transport = client - server)

An unattributed remainder is reported explicitly rather than folded into the nearest bucket, because
a decomposition that always sums to 100% is usually hiding its residual.

MEASUREMENT HYGIENE. CUDA is asynchronous, so every phase boundary synchronises -- that perturbs the
absolute total slightly and is the only honest way to attribute time to a phase. The whole-cycle
figure is measured separately, WITHOUT inner synchronisation, so the total is not inflated by the
instrumentation. Both are reported. Early cycles are discarded: graph capture warms up, and the ring
grows for the first ~36 cycles.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29983 \\
        profile_fixed_term.py [--cycles 40] [--video 2] [--action 4]
"""
from __future__ import annotations

import argparse
import collections
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


def sync():
    torch.cuda.synchronize()


class Timers:
    """EXCLUSIVE wall-clock per named phase, with CUDA synchronised at each boundary.

    EXCLUSIVE, NOT INCLUSIVE, AND THAT WAS A REAL BUG. The phases nest: `_compute_kv_cache` calls
    `_encode_obs` and then two transformer forwards. Timing each independently and summing counted the
    encode and those forwards twice, so the first run of this probe attributed 1508 ms of a 1961 ms
    cycle with `kv_refresh` at 454 ms while `vae_encode_obs` (183 ms of it) sat on its own line. Any
    "decomposition" whose parts overlap is a list, not a decomposition.

    So each timer subtracts the time consumed by timers that ran inside it. The parent's line then
    reads "time spent here that is not spent in a child", which is the only quantity that can be
    summed.
    """

    def __init__(self):
        self.t = collections.defaultdict(list)
        self.n = collections.Counter()
        self._stack: list[float] = []      # child time accumulated per open frame

    def record(self, name, dt, count=1):
        self.t[name].append(dt)
        self.n[name] += count

    def wrap(self, obj, attr, name):
        """Replace `obj.attr` with an exclusively-timed version."""
        orig = getattr(obj, attr)

        def timed(*a, **k):
            sync()
            t0 = time.perf_counter()
            self._stack.append(0.0)
            try:
                out = orig(*a, **k)
            finally:
                sync()
                gross = time.perf_counter() - t0
                child = self._stack.pop()
                if self._stack:
                    self._stack[-1] += gross          # tell the parent how long we took
                self.record(name, gross - child)
            return out
        setattr(obj, attr, timed)
        return orig


def summarise(timers, cycles, label, total_ms):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    rows = []
    for name, xs in timers.t.items():
        # Per-cycle mean: total time in this phase divided by the number of cycles measured.
        per_cycle = sum(xs) * 1000.0 / cycles
        calls = len(xs) / cycles
        rows.append((per_cycle, name, calls, statistics.median(xs) * 1000.0))
    rows.sort(reverse=True)
    attributed = sum(r[0] for r in rows if "PROBE-ONLY" not in r[1])
    print(f"{'phase':<34}{'ms/cycle':>10}{'share':>8}{'calls/cyc':>11}{'ms/call':>10}")
    print("-" * 78)
    for per_cycle, name, calls, med in rows:
        print(f"{name:<34}{per_cycle:>10.1f}{per_cycle / total_ms:>7.1%}{calls:>11.1f}{med:>10.2f}")
    print("-" * 78)
    print(f"{'ATTRIBUTED':<34}{attributed:>10.1f}{attributed / total_ms:>7.1%}")
    print(f"{'unattributed remainder':<34}{total_ms - attributed:>10.1f}"
          f"{(total_ms - attributed) / total_ms:>7.1%}")
    print(f"{'CYCLE TOTAL (uninstrumented)':<34}{total_ms:>10.1f}{1.0:>7.1%}")
    return {name: per_cycle for per_cycle, name, _, _ in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8, help="cycles discarded before measuring")
    ap.add_argument("--video", type=int, default=2)
    ap.add_argument("--action", type=int, default=4)
    a = ap.parse_args()

    busy = [ln for ln in os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    ).read().strip().split("\n") if ln.strip() and int(ln.split(",")[1]) >= 15]
    if busy:
        print(f"NOT EVALUATED: the fleet is busy ({'; '.join(x.strip() for x in busy)}%). A latency "
              f"decomposition on a contended device attributes the neighbour's time to our phases.")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_fixed_term"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1

    # The shipped Fast chain: substrate elision + prefill + ring KV + graph capture.
    install_fsdp_elision(S)
    # substrate_elision, as served. Keep a handle: this probe runs TWO passes in one process, and
    # graph capture holds several GiB in private pools, so between passes the real allocator has to be
    # let go of that memory or the second pass OOMs in the VAE's conv3d. The elision stays in force
    # DURING each measured pass, which is what makes the numbers representative.
    _real_empty_cache = torch.cuda.empty_cache
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps = a.video
    cfg.action_num_inference_steps = a.action

    print(f"building the real server at {a.video}V/{a.action}A ...", flush=True)
    server = S.VA_Server(cfg)

    from instinctwm.passes.lingbot.graph_capture import GraphBlockStack
    from instinctwm.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    # `ConditioningPrefill` is the declarative pass object; the direct installer is what serve_variant
    # uses and what every measured number was taken with.
    for n in install_conditioning_prefill(S, type(server)):
        print(f"  installed {n}", flush=True)
    # --no-debug-dump, as served: save_async writes latents/actions per cycle otherwise.
    for n in install_debug_dump_elision(S):
        print(f"  installed {n}", flush=True)
    gp = GraphBlockStack()
    for n in gp.install(S, type(server)):
        print(f"  installed {n}", flush=True)

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    if not ctx:
        raise SystemExit("no contexts; run collect_contexts.sh")
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = {"obs": [{full: z[s] for s, full in short.items()}], "state": z["state"]}
    prompt = str(z["prompt"])

    # A CONTROL CYCLE IS BOTH HALVES, and the first version of this probe measured only the first.
    # probe_episode.py times: infer(obs) -> _infer(...), then infer(obs=8 keyframes,
    # compute_kv_cache=True) -> _compute_kv_cache(...). The VAE encode of those keyframes never ran
    # in the earlier version, so `vae_encode_obs` and `kv_refresh` recorded nothing and the whole
    # decomposition was 99.8% unattributed.
    rng = np.random.default_rng(0)
    cams = list(cfg.obs_cam_keys)

    def keyframes(n):
        return [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
                for _ in range(n)]

    def cycle(first, n_kf=8):
        """Exactly probe_episode.py's cycle, minus the websocket.

        GOING THROUGH `server.infer()` MATTERS. An earlier version of this probe called `_infer` and
        `_compute_kv_cache` directly, which skips the bookkeeping `infer()` does between them --
        notably advancing `frame_st_id`. With that frozen, the ring never advanced as it does in
        service, KV accumulated, and the probe OOM'd at 76 GiB ALLOCATED (not merely reserved, which
        is why empty_cache could not rescue it) in the keyframe VAE encode. The lesson is the general
        one: measure through the entry point the system actually uses.
        """
        if first:
            server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        a = server.infer(dict(obs=obs["obs"], prompt=prompt,
                              save_visualization=False))["action"]
        kfs = keyframes(n_kf // 2 if first else n_kf)
        server.infer(dict(obs=kfs, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=a))
        return a

    T = Timers()          # used by the guarded reclaim in both passes
    # ---- pass 1: whole-cycle totals, NO inner instrumentation ---------------------------------
    print(f"\npass 1: {a.cycles} cycles, uninstrumented (the honest total) ...", flush=True)
    cycle(True)
    for _ in range(a.warmup):
        cycle(False)
    sync()
    per_cycle = []
    for _ in range(a.cycles):
        sync(); t0 = time.perf_counter()
        cycle(False)
        sync(); per_cycle.append(time.perf_counter() - t0)
    total_ms = statistics.median(per_cycle) * 1000.0
    spread = (max(per_cycle) - min(per_cycle)) / statistics.mean(per_cycle)
    ms = [x * 1000.0 for x in per_cycle]
    print(f"  cycle: median {total_ms:.1f} ms   spread {spread:.1%}   "
          f"min {min(ms):.1f}  max {max(ms):.1f}")
    # A wide spread here is not noise to be averaged away -- graph capture re-captures whenever the
    # ring signature changes, and the pool grows every cycle, so slow and fast cycles are two
    # populations rather than one distribution. Print the series; a median over a bimodal sample is
    # a number that describes no cycle that actually happened.
    if spread > 0.25:
        lo = [x for x in ms if x < (min(ms) + max(ms)) / 2]
        hi = [x for x in ms if x >= (min(ms) + max(ms)) / 2]
        print(f"  BIMODAL: {len(lo)} cycles near {statistics.median(lo):.0f} ms, "
              f"{len(hi)} near {statistics.median(hi):.0f} ms -- reporting both, not their mean")
        print("  series: " + " ".join(f"{x:.0f}" for x in ms))

    # ---- between passes: release the graph pools -----------------------------------------------
    if hasattr(gp, "drop_graphs"):
        gp.drop_graphs("probe: between passes")
    else:
        for attr in ("graphs", "_graphs"):
            if hasattr(gp, attr):
                getattr(gp, attr).clear()
    _real_empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"  released graph pools: {free / 2**30:.1f} GiB free of {total / 2**30:.1f}")

    # ---- pass 2: instrumented -----------------------------------------------------------------
    print(f"\npass 2: {a.cycles} cycles, phase-instrumented ...", flush=True)
    T.t.clear(); T.n.clear()
    T.wrap(server, "_encode_obs", "vae_encode_obs")
    T.wrap(server, "_compute_kv_cache", "kv_refresh (encode+2 fwd)")
    T.wrap(server, "preprocess_action", "preprocess_action")
    T.wrap(server, "postprocess_action", "postprocess_action")
    T.wrap(server, "_prepare_latent_input", "prepare_latent_input")
    T.wrap(server.scheduler, "step", "scheduler.step (video)")
    T.wrap(server.action_scheduler, "step", "scheduler.step (action)")
    T.wrap(server.scheduler, "set_timesteps", "set_timesteps")
    T.wrap(server.action_scheduler, "set_timesteps", "set_timesteps")
    if hasattr(S, "save_async"):
        T.wrap(S, "save_async", "save_async (debug dump)")

    # The transformer is the per-forward term. Time it separately from everything else so the
    # FIXED/PER_STEP split is measured rather than inferred from a regression.
    # Wrap `forward`, NOT `__call__`. Python resolves special methods on the TYPE, so assigning
    # `instance.__call__` is never consulted by `module(...)` -- the first version of this probe did
    # that and recorded 0.0 ms over 0.0 forwards, which is the giveaway.
    orig_fwd = server.transformer.forward

    def timed_fwd(*args, **kw):
        sync(); t0 = time.perf_counter()
        T._stack.append(0.0)
        try:
            out = orig_fwd(*args, **kw)
        finally:
            sync()
            gross = time.perf_counter() - t0
            child = T._stack.pop()
            if T._stack:
                T._stack[-1] += gross
            T.record("transformer forwards", gross - child)
        return out
    server.transformer.forward = timed_fwd

    cycle(True)
    for _ in range(a.warmup):
        cycle(False)
    T.t.clear(); T.n.clear()
    inst = []
    for _ in range(a.cycles):
        sync(); t0 = time.perf_counter()
        cycle(False)
        sync(); inst.append(time.perf_counter() - t0)
    inst_ms = statistics.median(inst) * 1000.0

    comp = summarise(T, a.cycles, f"FIXED-TERM DECOMPOSITION at {a.video}V/{a.action}A", inst_ms)
    print(f"\ncycle total, pass 1 (uninstrumented, graph pools warm): {total_ms:.1f} ms")
    print(f"cycle total, pass 2 (instrumented, pools dropped first) : {inst_ms:.1f} ms  "
          f"({inst_ms / total_ms - 1:+.1%})")
    print("  Shares are against PASS 2's own total, because that is the pass the components come "
          "from.\n  The two passes are NOT interchangeable: pass 1 carries graph-capture churn that\n"
          "  pass 2 does not, having started from dropped pools. Using pass 1's total as the\n"
          "  denominator understated every share by ~45%.")

    fwd = comp.get("transformer forwards", 0.0)
    n_fwd = T.n["transformer forwards"] / a.cycles
    total_ms = inst_ms
    print(f"\nPER_STEP vs FIXED, measured directly rather than regressed:")
    print(f"  transformer forwards : {fwd:7.1f} ms/cycle over {n_fwd:.1f} forwards "
          f"= {fwd / max(n_fwd, 1):.1f} ms/forward")
    print(f"  everything else      : {total_ms - fwd:7.1f} ms/cycle "
          f"({(total_ms - fwd) / total_ms:.0%} of the cycle)")
    print("\nThe regression put FIXED at 1164 ms and PER_STEP at 15.5 ms/forward. Compare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

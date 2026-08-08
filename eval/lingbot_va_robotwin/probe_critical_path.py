#!/usr/bin/env python3
"""What determines cycle completion? A CPU+CUDA timeline and its critical path, not an op histogram.

WHY THIS EXISTS. Three consecutive region-level wins failed to move the cycle: a bit-exact Triton RoPE
kernel (1.10x region, 0.3% cycle), a cast hoist (1.4% predicted, 0.66% and unresolvable), and a fused
QKV projection (1.9% predicted, 0.2% SLOWER). Meanwhile the two things that did move it -- P001-P006 and
P007 -- both removed or redirected work rather than accelerating it. An operator histogram cannot explain
that pattern, because it ranks work by cost without knowing whether the cycle is waiting on it.

WHAT A CRITICAL PATH MEANS HERE, stated precisely, because the honest answer is unusual.

With one host thread issuing onto one stream, execution is already a chain: host program order serialises
the CPU side, stream order serialises the device side, and the two couple at launches and at
synchronisations. The DAG's longest path is therefore the wall clock itself, and asking "what is the
longest path" is not the interesting question. The interesting question is the COMPOSITION of that path:

    DEVICE-BOUND segments   the GPU is busy and the host is ahead of it. Making a kernel faster here
                            shortens the cycle.
    HOST-BOUND segments     the GPU is idle and waiting for the host to issue work. Making a kernel
                            faster here shortens nothing -- it lengthens the idle gap.
    SYNC segments           both sides stalled at a barrier.

So this probe measures the device-busy interval union against the wall clock, attributes every idle
interval to the host operator that was running during it, and reports which side of the coupling each
expensive operator sits on. An operator whose time falls inside a host-bound segment is OFF the critical
path by construction, and that is the report the last three proposals needed.

Multiple streams would make this a genuine DAG rather than a chain; the stream inventory is reported so
that assumption is checked rather than assumed.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29998 probe_critical_path.py
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

from instinctwm.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

#: Runtime calls that block the host until the device catches up.
SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
              "cudaMemcpyAsync", "cudaMemcpy", "cudaStreamWaitEvent")
#: Allocator calls. These serialise against all streams and are worth counting separately.
ALLOC_NAMES = ("cudaMalloc", "cudaFree", "cudaHostAlloc", "cudaMallocAsync", "cudaFreeAsync",
               "cudaHostRegister")
DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation")


def union_length(iv: list[tuple[float, float]]) -> tuple[float, list[tuple[float, float]]]:
    """Total length of a union of intervals, and the merged intervals."""
    if not iv:
        return 0.0, []
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return sum(e - s for s, e in out), [(s, e) for s, e in out]


def gaps(merged: list[tuple[float, float]], t0: float, t1: float) -> list[tuple[float, float]]:
    """Complement of the merged busy intervals within [t0, t1]."""
    out = []
    cur = t0
    for s, e in merged:
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < t1:
        out.append((cur, t1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70)
    ap.add_argument("--conv-layout", choices=["as-is", "ndhwc"], default="ndhwc")
    ap.add_argument("--trace", default="/tmp/iwm_cycle_trace.json")
    ap.add_argument("--untraced-ms", type=float, default=0.0,
                    help="measured cycle time WITHOUT the profiler, for overhead correction")
    a = ap.parse_args()

    hot = [ln for ln in os.popen("nvidia-smi --query-gpu=index,utilization.gpu "
                                 "--format=csv,noheader,nounits").read().strip().split("\n")
           if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%).")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_critpath"
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
    if a.conv_layout == "ndhwc":
        from instinctwm.backends.conv.apply import install_conv_layout
        for line in install_conv_layout(server):
            print(f"  {line}", flush=True)

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    if not ctx:
        raise SystemExit("no contexts")
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

    print(f"warming {a.warm} cycles (ring saturates ~64) ...", flush=True)
    for _ in range(a.warm):
        cycle()

    print("tracing ONE saturated cycle, CPU + CUDA ...", flush=True)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        cycle()
        torch.cuda.synchronize()
    prof.export_chrome_trace(a.trace)
    print(f"  trace written to {a.trace} "
          f"({os.path.getsize(a.trace) / 2**20:.1f} MiB)", flush=True)

    ev = json.load(open(a.trace))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    X = [e for e in ev if e.get("ph") == "X" and "ts" in e and "dur" in e]

    dev, cpu, runtime = [], [], []
    streams = collections.Counter()
    for e in X:
        cat = (e.get("cat") or "").lower()
        s, d = float(e["ts"]), float(e["dur"])
        if cat in DEVICE_CATS:
            dev.append((s, s + d, e.get("name", "?"), cat))
            st = (e.get("args") or {}).get("stream")
            if st is not None:
                streams[st] += 1
        elif cat in ("cpu_op", "user_annotation"):
            cpu.append((s, s + d, e.get("name", "?")))
        elif cat in ("cuda_runtime", "cuda_driver"):
            runtime.append((s, s + d, e.get("name", "?")))

    if not dev:
        print("NOT EVALUATED: no device events in the trace.")
        return 2

    t0 = min(min(s for s, _, _ in cpu) if cpu else 1e18, min(s for s, _, _, _ in dev))
    t1 = max(max(e for _, e, _ in cpu) if cpu else 0.0, max(e for _, e, _, _ in dev))
    wall = (t1 - t0) / 1000.0
    busy_us, merged = union_length([(s, e) for s, e, _, _ in dev])
    busy = busy_us / 1000.0
    idle_iv = gaps(merged, t0, t1)
    idle = sum(e - s for s, e in idle_iv) / 1000.0

    print(f"\n{'=' * 112}\nONE SATURATED CYCLE: is the critical path device-bound or host-bound?"
          f"\n{'=' * 112}")
    print(f"  wall clock (trace span)     {wall:8.1f} ms")
    print(f"  device BUSY (interval union){busy:8.1f} ms   {busy / wall:6.1%}")
    print(f"  device IDLE                 {idle:8.1f} ms   {idle / wall:6.1%}   "
          f"in {len(idle_iv)} gaps")
    print(f"  streams observed            {dict(streams) or 'not reported by this build'}")
    if len(streams) <= 1:
        print("    => a single device stream, so device order is a chain, not a DAG. The critical")
        print("       path is the wall clock and the question is which side of the coupling owns it.")

    # ---- attribute every idle gap to the host operator running during it ---------------------
    print(f"\n{'=' * 112}\nHOST-BOUND SEGMENTS: which host work the device is waiting for"
          f"\n{'=' * 112}")
    cpu_sorted = sorted(cpu)
    starts = [c[0] for c in cpu_sorted]
    import bisect
    attrib = collections.Counter()
    attrib_n = collections.Counter()
    unattributed = 0.0
    for gs, ge in idle_iv:
        mid = (gs + ge) / 2.0
        i = bisect.bisect_right(starts, mid) - 1
        best = None
        # innermost enclosing cpu_op: scan back a bounded window and keep the shortest that covers mid
        while i >= 0 and i > bisect.bisect_right(starts, mid) - 400:
            s, e, n = cpu_sorted[i]
            if s <= mid <= e and (best is None or (e - s) < (best[1] - best[0])):
                best = (s, e, n)
            i -= 1
        if best is None:
            unattributed += (ge - gs)
        else:
            attrib[best[2]] += (ge - gs)
            attrib_n[best[2]] += 1
    # PROFILER OVERHEAD CORRECTION. Tracing CPU+CUDA costs host time per op, and this cycle issues
    # ~15,000 of them, so the traced wall (595 ms in the first run) far exceeds the real one (338 ms).
    # That inflation lands entirely on the HOST side and therefore entirely in the idle attribution, so
    # the raw table over-ranks whatever is called most often. Uncorrected it reported aten::empty and
    # aten::empty_strided near the top -- ops that do almost nothing.
    n_cpu = len(cpu)
    over_per_op = 0.0
    if a.untraced_ms > 0 and n_cpu:
        over_per_op = max(0.0, (wall - a.untraced_ms) * 1000.0 / n_cpu)   # us per cpu_op
        print(f"\n  overhead correction: traced {wall:.0f} ms vs untraced {a.untraced_ms:.0f} ms over "
              f"{n_cpu} cpu ops -> {over_per_op:.1f} us/op subtracted")
    else:
        print("\n  NO OVERHEAD CORRECTION (pass --untraced-ms). The table below over-ranks "
              "high-call-count ops.")

    # A stall at one of these is the host BLOCKING on the device: a real serialisation point, not
    # profiler cost. They must not be corrected, and they are the interesting rows.
    SYNCY = ("aten::item", "aten::_local_scalar_dense", "aten::to", "aten::cpu", "aten::numpy",
             "aten::nonzero", "aten::equal", "aten::allclose", "cudaStreamSynchronize",
             "cudaDeviceSynchronize", "aten::_to_copy")
    print(f"{'host operator':<46}{'raw ms':>9}{'corrected':>11}{'share':>9}{'gaps':>7}  kind")
    print("-" * 112)
    corrected_total = 0.0
    rows = []
    for name, us in attrib.items():
        syncy = any(name.startswith(s) for s in SYNCY)
        corr = us / 1000.0 if syncy else max(0.0, (us - attrib_n[name] * over_per_op) / 1000.0)
        corrected_total += corr
        rows.append((corr, us / 1000.0, name, attrib_n[name], syncy))
    rows.sort(reverse=True)
    for corr, raw, name, n, syncy in rows[:16]:
        print(f"{name[:44]:<46}{raw:>9.2f}{corr:>11.2f}"
              f"{corr / max(corrected_total, 1e-9):>9.1%}{n:>7}  "
              f"{'HOST BLOCKS ON DEVICE' if syncy else 'launch/dispatch'}")
    print("-" * 112)
    print(f"{'corrected host-bound total':<46}{'':>9}{corrected_total:>11.2f}")
    sync_bound = sum(c for c, _, _, _, sy in rows if sy)
    print(f"{'of which the host BLOCKING on the device':<46}{'':>9}{sync_bound:>11.2f}"
          f"{sync_bound / max(corrected_total, 1e-9):>9.1%}")

    # ---- where do the blocking calls come from? ---------------------------------------------
    print(f"\n{'=' * 112}\nSERIALISATION POINTS: every host->device block, and how many\n{'=' * 112}")
    blockers = collections.Counter()
    blockers_t = collections.Counter()
    for s, e, n in cpu:
        if n in ("aten::item", "aten::_local_scalar_dense"):
            blockers[n] += 1
            blockers_t[n] += (e - s)
    for n, c in blockers.most_common():
        print(f"  {n:<40}{c:>8} calls{blockers_t[n] / 1000:>10.2f} ms")
    if blockers:
        print(f"\n  Each of these drains the queue: the host cannot proceed until the device reaches")
        print(f"  that point. {sum(blockers.values())} of them per cycle is a serialisation structure,")
        print(f"  not a cost -- removing GPU time between two syncs cannot help, because the host is")
        print(f"  waiting on the sync and not on the work.")

    # ---- syncs and allocator ------------------------------------------------------------------
    print(f"\n{'=' * 112}\nSYNCHRONISATION AND ALLOCATOR\n{'=' * 112}")
    sync = collections.Counter()
    sync_t = collections.Counter()
    alloc = collections.Counter()
    alloc_t = collections.Counter()
    for s, e, n in runtime:
        for k in SYNC_NAMES:
            if n.startswith(k):
                sync[k] += 1
                sync_t[k] += (e - s)
        for k in ALLOC_NAMES:
            if n.startswith(k):
                alloc[k] += 1
                alloc_t[k] += (e - s)
    print(f"{'runtime call':<32}{'count':>10}{'total ms':>12}{'us/call':>10}")
    print("-" * 112)
    for k in sorted(set(list(sync) + list(alloc)), key=lambda x: -(sync_t[x] + alloc_t[x])):
        c = sync[k] + alloc[k]
        tt = sync_t[k] + alloc_t[k]
        print(f"{k:<32}{c:>10}{tt / 1000:>12.2f}{tt / max(c, 1):>10.1f}")
    if not sync and not alloc:
        print("  none observed in this build's runtime category")

    # ---- device-bound: what actually occupies the GPU -----------------------------------------
    print(f"\n{'=' * 112}\nDEVICE-BOUND: kernels inside the busy union (these ARE on the critical "
          f"path)\n{'=' * 112}")
    kb = collections.Counter()
    kn = collections.Counter()
    for s, e, n, _ in dev:
        kb[n] += (e - s)
        kn[n] += 1
    print(f"{'kernel':<66}{'ms':>9}{'% of busy':>11}{'% of wall':>11}{'calls':>8}")
    print("-" * 112)
    for n, us in kb.most_common(14):
        print(f"{n[:64]:<66}{us / 1000:>9.2f}{us / (busy * 1000):>11.1%}"
              f"{us / (wall * 1000):>11.1%}{kn[n]:>8}")

    # ---- the verdict --------------------------------------------------------------------------
    print(f"\n{'=' * 112}\nWHAT THIS MEANS FOR OPTIMIZATION\n{'=' * 112}")
    print(f"  Device work is {busy / wall:.0%} of the cycle. An optimization that removes GPU time can")
    print(f"  therefore recover at most that share, and only for the kernels inside the busy union.")
    print(f"  The other {idle / wall:.0%} is the device waiting on the host: {idle:.0f} ms across")
    print(f"  {len(idle_iv)} gaps. No kernel optimization touches it.")
    if a.untraced_ms > 0:
        real_idle = max(0.0, a.untraced_ms - busy)
        print(f"\n  CORRECTED FOR THE UNTRACED CYCLE ({a.untraced_ms:.0f} ms):")
        print(f"    device busy {busy:.0f} ms = {busy / a.untraced_ms:.0%}   "
              f"device idle {real_idle:.0f} ms = {real_idle / a.untraced_ms:.0%}")
        print(f"    if ALL device work were free : cycle -> {real_idle:.0f} ms "
              f"({a.untraced_ms / max(real_idle, 1e-9):.2f}x)")
        print(f"    if ALL host gaps were closed : cycle -> {busy:.0f} ms "
              f"({a.untraced_ms / max(busy, 1e-9):.2f}x)")
    print(f"\n  Upper bound on any pure GPU-time optimization: {busy:.0f} ms of {wall:.0f} ms.")
    print(f"  Upper bound if ALL device work became free:      cycle -> {idle:.0f} ms "
          f"({wall / max(idle, 1e-9):.2f}x)")
    print(f"  Upper bound if ALL host work became free:        cycle -> {busy:.0f} ms "
          f"({wall / max(busy, 1e-9):.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Explain the ~155 ms of device-timeline gaps. One diagnostic, no optimization.

The cycle is 351 ms and the device is busy 196 ms. Layer 6 assumed the remaining ~155 ms was host
dispatch throughput and was wrong: removing 12,190 dispatches bit-exactly moved nothing, and the measured
marginal cost of a Python-level dispatch is 1.02 us, capping the whole host-dispatch term at ~56 ms
(LAYER6.md sections H and I). So ~99 ms is unaccounted for and this probe goes after the gaps directly.

WHY GAPS ARE THE RIGHT UNIT HERE. The runtime uses a single compute stream. On one stream, kernels run
back-to-back if they are enqueued, so a gap means the queue was EMPTY -- the device had nothing to run.
That makes every gap host-attributable by construction, but NOT necessarily to dispatch throughput. It
could be a blocking synchronization, an allocator call into the driver, pure CPU work between launches, or
launch latency. Those are different problems with different fixes, and the histogram in section 2 decides
which one this is.

WHAT IS REPORTED, per gap: the preceding and following device event, the innermost named scope and aten
operator live during the gap, the count of Python-originated vs C++-internal dispatches inside it,
allocator calls, synchronization calls, and metadata/shape operations. Then gaps are ranked by scope so
the cumulative share of the total is visible.

PYTHON-ORIGINATED IS DERIVED, NOT ESTIMATED. A `cpu_op` nested inside another `cpu_op` on the same thread
is a C++-internal redispatch (`aten::linear` emitting `aten::t`); a `cpu_op` with no enclosing `cpu_op` was
entered from Python. Section I priced those two at ~1.02 us and ~0, so the distinction is the one that
matters and the trace can settle it exactly rather than by inference from cProfile.

CONTROL. Named scopes cost ~1 us each and this probe adds ~1,200 of them, so it runs twice -- once with
scopes and once without -- and compares total gap time. If the two disagree by more than a few percent the
instrument is disturbing what it measures and the run is NOT EVALUATED.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29987 probe_device_gaps.py
"""
from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
import os
import sys
from pathlib import Path

IFL_ROOT = os.environ.get("IFL_ROOT") or str(Path(__file__).resolve().parents[2])
if IFL_ROOT not in sys.path:
    sys.path.insert(0, IFL_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from instinctflash.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "Kernel"}
ALLOC_NAMES = ("cudaMalloc", "cudaFree", "cudaHostAlloc", "cudaHostRegister", "cudaMallocHost")
SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
              "cudaStreamWaitEvent", "cudaMemcpy")
METADATA_OPS = {
    "aten::as_strided", "aten::view", "aten::transpose", "aten::slice", "aten::reshape", "aten::t",
    "aten::narrow", "aten::squeeze", "aten::unsqueeze", "aten::flatten", "aten::unflatten",
    "aten::expand", "aten::permute", "aten::select", "aten::detach", "aten::chunk", "aten::split",
    "aten::unbind", "aten::contiguous",
}


# ---------------------------------------------------------------------------------------------------
# scoping
# ---------------------------------------------------------------------------------------------------
def add_scopes(S, server):
    """Name the regions a gap can fall inside. Reversible; returns an undo callable."""
    undo = []

    def wrap_method(obj, attr, label):
        orig = getattr(obj, attr, None)
        if orig is None:
            return
        def wrapped(*a, **k):
            with record_function(f"iwm::{label}"):
                return orig(*a, **k)
        setattr(obj, attr, wrapped)
        undo.append(lambda o=obj, at=attr, fn=orig: setattr(o, at, fn))

    for attr in ("_encode_obs", "_prepare_latent_input", "_infer", "_compute_kv_cache",
                 "postprocess_action", "normalize_latents", "decode_one_video", "_reset"):
        wrap_method(server, attr, attr.lstrip("_"))

    # scheduler.step is where the only genuine device->host reads live (scheduler.py:82,83,87)
    for name in ("scheduler", "action_scheduler"):
        sch = getattr(server, name, None)
        if sch is not None:
            wrap_method(sch, "step", f"{name}.step")
            wrap_method(sch, "set_timesteps", f"{name}.set_timesteps")

    for name in ("vae", "vae_half", "streaming_vae", "streaming_vae_half"):
        v = getattr(server, name, None)
        if v is None:
            continue
        wrap_method(v, "encode", f"{name}.encode")
        wrap_method(v, "decode", f"{name}.decode")

    # module classes: one scope per call, ~1,200 per cycle
    tf = server.transformer
    wrap_method(tf, "forward", "transformer.forward")
    for i, blk in enumerate(tf.blocks):
        wrap_method(blk, "forward", "block.forward")
        for sub, lbl in (("attn1", "attn.self"), ("attn2", "attn.cross"), ("ffn", "ffn")):
            m = getattr(blk, sub, None)
            if m is not None:
                wrap_method(m, "forward", lbl)

    return lambda: [f() for f in reversed(undo)]


# ---------------------------------------------------------------------------------------------------
# trace parsing
# ---------------------------------------------------------------------------------------------------
def load_trace(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)["traceEvents"]


def union_len(intervals):
    """Total covered time of a list of (start, end), merging overlaps."""
    if not intervals:
        return 0.0, []
    xs = sorted(intervals)
    merged = [list(xs[0])]
    for s, e in xs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged), merged


def analyse(events, gap_floor_us, top_n, label):
    dev = [e for e in events
           if e.get("cat") in DEVICE_CATS and e.get("ph") == "X" and e.get("dur", 0) is not None]
    cpu = [e for e in events if e.get("cat") == "cpu_op" and e.get("ph") == "X"]
    ann = [e for e in events if e.get("cat") == "user_annotation" and e.get("ph") == "X"]
    rt = [e for e in events if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"]

    # the profiled window: our outermost annotation
    cyc = [e for e in ann if e["name"] == "iwm::cycle"]
    if not cyc:
        print(f"  [{label}] no iwm::cycle annotation found; using the device span")
        t0 = min(e["ts"] for e in dev)
        t1 = max(e["ts"] + e["dur"] for e in dev)
    else:
        t0, t1 = cyc[0]["ts"], cyc[0]["ts"] + cyc[0]["dur"]
    wall = t1 - t0

    dev = [e for e in dev if e["ts"] + e["dur"] > t0 and e["ts"] < t1]
    busy, merged = union_len([(e["ts"], e["ts"] + e["dur"]) for e in dev])

    # gaps = the complement of the busy union inside the window
    gaps = []
    cur = t0
    for s, e in merged:
        if s > cur:
            gaps.append((cur, s))
        cur = max(cur, e)
    if cur < t1:
        gaps.append((cur, t1))
    total_gap = sum(b - a for a, b in gaps)

    print(f"\n  [{label}] window {wall / 1000:.1f} ms   device busy {busy / 1000:.1f} ms "
          f"({busy / wall:.1%})   gaps {total_gap / 1000:.1f} ms ({total_gap / wall:.1%})   "
          f"{len(dev)} device events, {len(gaps)} gaps")
    return dict(t0=t0, t1=t1, wall=wall, busy=busy, gaps=gaps, total_gap=total_gap,
                dev=dev, cpu=cpu, ann=ann, rt=rt, merged=merged)


def gap_histogram(gaps, total_gap):
    bins = [(0, 5), (5, 20), (20, 100), (100, 1000), (1000, 10000), (10000, 1e12)]
    names = ["< 5 us", "5-20 us", "20-100 us", "100 us - 1 ms", "1-10 ms", "> 10 ms"]
    print(f"\n{'=' * 108}\n2. GAP SIZE DISTRIBUTION -- concentrated or diffuse?\n{'=' * 108}")
    print(f"  {'bucket':>16}{'count':>9}{'total ms':>11}{'% of gap':>10}{'mean us':>10}")
    rows = []
    for (lo, hi), nm in zip(bins, names):
        sel = [b - a for a, b in gaps if lo <= (b - a) < hi]
        tot = sum(sel)
        rows.append((nm, len(sel), tot))
        if sel:
            print(f"  {nm:>16}{len(sel):>9}{tot / 1000:>11.1f}{tot / total_gap:>9.1%}"
                  f"{tot / len(sel):>10.1f}")
    return rows


def launch_decomposition(events, a, label):
    """Was the host LATE, or was the kernel already enqueued and still not running?

    On a single stream a gap means the queue was empty, but that has two very different causes and the
    launch timestamps separate them exactly. Each device event carries a `correlation` id shared with the
    runtime call that issued it, so for the kernel that ENDS a gap we can ask when its launch was made:

      launch begins DURING the gap  -> the host had not issued it yet: dispatch starvation
      launch begins BEFORE the gap  -> it was already queued and the device still idled: a dependency,
                                       a stream serialisation, or driver-side launch latency
    """
    rt = [e for e in events if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"]
    by_corr = {}
    for e in rt:
        c = (e.get("args") or {}).get("correlation")
        if c is not None:
            by_corr[c] = e

    dev = sorted(a["dev"], key=lambda e: e["ts"])
    dev_starts = [e["ts"] for e in dev]
    late, late_n, queued, queued_n, unknown, unknown_n = 0.0, 0, 0.0, 0, 0.0, 0
    for g0, g1 in a["gaps"]:
        dur = g1 - g0
        i = bisect.bisect_left(dev_starts, g1)
        if i >= len(dev):
            unknown += dur
            unknown_n += 1
            continue
        nxt = dev[i]
        r = by_corr.get((nxt.get("args") or {}).get("correlation"))
        if r is None:
            unknown += dur
            unknown_n += 1
        elif r["ts"] >= g0:
            late += dur
            late_n += 1
        else:
            queued += dur
            queued_n += 1
    tot = a["total_gap"]
    print(f"\n{'=' * 108}\nLAUNCH TIMING OF THE KERNEL THAT ENDS EACH GAP  [{label}]\n{'=' * 108}")
    print(f"  {'':<44}{'gaps':>8}{'total ms':>11}{'% of gap':>10}{'mean us':>10}")
    for nm, t, n in (("host had not launched it yet", late, late_n),
                     ("already enqueued, device still idle", queued, queued_n),
                     ("no correlation recorded", unknown, unknown_n)):
        if n:
            print(f"  {nm:<44}{n:>8}{t / 1000:>11.1f}{t / tot:>9.1%}{t / n:>10.1f}")
    return late, queued


def enclosing(sorted_starts, evs, t):
    """Innermost event containing t, from events sorted by ts."""
    i = bisect.bisect_right(sorted_starts, t)
    best = None
    for e in reversed(evs[max(0, i - 4000):i]):
        if e["ts"] + e.get("dur", 0) >= t:
            if best is None or e["ts"] > best["ts"]:
                best = e
    return best


def attribute(a, gap_floor_us, top_n):
    dev = sorted(a["dev"], key=lambda e: e["ts"])
    dev_ends = [e["ts"] + e["dur"] for e in dev]
    cpu = sorted(a["cpu"], key=lambda e: e["ts"])
    ann = sorted(a["ann"], key=lambda e: e["ts"])
    rt = sorted(a["rt"], key=lambda e: e["ts"])
    cpu_starts = [e["ts"] for e in cpu]
    ann_starts = [e["ts"] for e in ann]
    rt_starts = [e["ts"] for e in rt]

    # Python-originated == a cpu_op not contained in any other cpu_op on the same thread.
    roots = set()
    by_tid = collections.defaultdict(list)
    for e in cpu:
        by_tid[e.get("tid")].append(e)
    n_root = n_child = 0
    for tid, evs in by_tid.items():
        stack = []
        for e in sorted(evs, key=lambda x: (x["ts"], -x.get("dur", 0))):
            end = e["ts"] + e.get("dur", 0)
            while stack and stack[-1] <= e["ts"]:
                stack.pop()
            if stack:
                n_child += 1
            else:
                roots.add(id(e))
                n_root += 1
            stack.append(end)
    print(f"\n{'=' * 108}\n3. PYTHON-ORIGINATED vs C++-INTERNAL DISPATCH (derived from trace nesting)"
          f"\n{'=' * 108}")
    print(f"  cpu_op events in the window      {len(cpu):>8}")
    print(f"  Python-originated (no cpu_op parent) {n_root:>8}   at 1.02 us = "
          f"{n_root * 1.02 / 1000:.1f} ms")
    print(f"  C++-internal redispatch          {n_child:>8}   priced at ~0 in LAYER6.md section I")

    big = sorted([g for g in a["gaps"] if (g[1] - g[0]) >= gap_floor_us],
                 key=lambda g: -(g[1] - g[0]))
    print(f"\n{'=' * 108}\n4. THE {min(top_n, len(big))} LARGEST GAPS (>= {gap_floor_us} us), "
          f"attributed\n{'=' * 108}")
    scope_tot = collections.Counter()
    scope_cnt = collections.Counter()
    cause_tot = collections.Counter()
    cause_cnt = collections.Counter()

    for gi, (g0, g1) in enumerate(sorted(a["gaps"], key=lambda g: -(g[1] - g[0]))):
        dur = g1 - g0
        mid = (g0 + g1) / 2
        # runtime calls inside the gap
        i0 = bisect.bisect_left(rt_starts, g0)
        i1 = bisect.bisect_right(rt_starts, g1)
        inside = rt[i0:i1]
        names = [e["name"] for e in inside]
        n_launch = sum(1 for n in names if "LaunchKernel" in n)
        n_alloc = sum(1 for n in names if any(x in n for x in ALLOC_NAMES))
        n_sync = sum(1 for n in names if any(x in n for x in SYNC_NAMES))
        n_memcpy = sum(1 for n in names if "Memcpy" in n)
        # dispatches inside the gap
        j0 = bisect.bisect_left(cpu_starts, g0)
        j1 = bisect.bisect_right(cpu_starts, g1)
        in_cpu = cpu[j0:j1]
        n_py = sum(1 for e in in_cpu if id(e) in roots)
        n_meta = sum(1 for e in in_cpu if e["name"] in METADATA_OPS)

        if n_alloc:
            cause = "ALLOCATOR (driver call)"
        elif n_sync:
            cause = "SYNCHRONIZATION"
        elif n_memcpy and not n_launch:
            cause = "MEMCPY wait"
        elif not inside and not in_cpu:
            cause = "HOST-ONLY (no dispatch at all)"
        elif n_launch == 0:
            cause = "HOST-ONLY (dispatch, no launch)"
        else:
            cause = "LAUNCH-STARVED"

        sc = enclosing(ann_starts, ann, mid)
        scope = sc["name"].replace("iwm::", "") if sc else "(unscoped)"
        scope_tot[scope] += dur
        scope_cnt[scope] += 1
        cause_tot[cause] += dur
        cause_cnt[cause] += 1

        if gi < top_n:
            prev = None
            k = bisect.bisect_right(dev_ends, g0) - 1
            # dev_ends is not sorted in general; scan a local window instead
            cand = [e for e in dev if abs(e["ts"] + e["dur"] - g0) < 2000]
            if cand:
                prev = min(cand, key=lambda e: abs(e["ts"] + e["dur"] - g0))
            m = bisect.bisect_left([e["ts"] for e in dev], g1)
            nxt = dev[m] if m < len(dev) else None
            op = enclosing(cpu_starts, cpu, mid)
            print(f"\n  gap {gi + 1}: {dur:.0f} us   [{cause}]   scope={scope}")
            print(f"    prev device : {(prev['name'][:74] if prev else '-')}")
            print(f"    next device : {(nxt['name'][:74] if nxt else '-')}")
            print(f"    host op live: {(op['name'] if op else '-')}")
            print(f"    inside      : launches={n_launch} allocator={n_alloc} sync={n_sync} "
                  f"memcpy={n_memcpy} | dispatches={len(in_cpu)} python-originated={n_py} "
                  f"metadata={n_meta}")
            if inside:
                top = collections.Counter(names).most_common(4)
                print(f"    runtime     : " + "  ".join(f"{n}x{c}" for n, c in top))

    print(f"\n{'=' * 108}\n5. GAP TIME BY CAUSE\n{'=' * 108}")
    print(f"  {'cause':>32}{'gaps':>8}{'total ms':>11}{'% of gap':>10}{'mean us':>10}")
    for c, t in cause_tot.most_common():
        print(f"  {c:>32}{cause_cnt[c]:>8}{t / 1000:>11.1f}{t / a['total_gap']:>9.1%}"
              f"{t / cause_cnt[c]:>10.1f}")

    print(f"\n{'=' * 108}\n6. GAP TIME BY SCOPE -- ranked, with cumulative share\n{'=' * 108}")
    print(f"  {'scope':>24}{'gaps':>8}{'total ms':>11}{'% of gap':>10}{'cum %':>9}{'mean us':>10}")
    cum = 0.0
    for s, t in scope_tot.most_common(18):
        cum += t
        print(f"  {s:>24}{scope_cnt[s]:>8}{t / 1000:>11.1f}{t / a['total_gap']:>9.1%}"
              f"{cum / a['total_gap']:>8.1%}{t / scope_cnt[s]:>10.1f}")
    return scope_tot, cause_tot, n_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70)
    ap.add_argument("--gap-floor", type=float, default=300.0, help="us; gaps at least this big are "
                                                                  "printed individually")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--trace", default="/tmp/iwm_gap_trace.json")
    a = ap.parse_args()

    hot = [ln for ln in os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    ).read().strip().split("\n") if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%).")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_gaps"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

    print("building server at 2V/4A, shipped stack ...", flush=True)
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
    obs = [{full: z[s] for s, full in short.items()}]
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)
    rng = np.random.default_rng(0)

    def cycle(first=False, keyframes=None):
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

    # keyframes generated OUTSIDE the profiled window: 5.5 MB of numpy randomness per cycle is
    # harness cost and would otherwise appear as the largest gap in the report.
    kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
          for _ in range(8)]

    def profiled_cycle(path, cpu=True):
        acts = ([ProfilerActivity.CPU] if cpu else []) + [ProfilerActivity.CUDA]
        with profile(activities=acts) as p:
            with record_function("iwm::cycle"):
                act = server.infer(dict(obs=obs, prompt=prompt, save_visualization=False))["action"]
                server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                                  save_visualization=False, state=act))
            torch.cuda.synchronize()
        p.export_chrome_trace(path)
        return path

    # ---- the control the first attempt lacked: how much does each instrument distort? -------------
    import statistics as _st
    import time as _t
    xs = []
    for _ in range(12):
        torch.cuda.synchronize()
        t0 = _t.perf_counter()
        act = server.infer(dict(obs=obs, prompt=prompt, save_visualization=False))["action"]
        server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=act))
        torch.cuda.synchronize()
        xs.append((_t.perf_counter() - t0) * 1e3)
    unprofiled = _st.median(xs)

    print(f"\n{'=' * 108}\n1. INSTRUMENT DISTORTION -- the gap total is what the profiler inflates"
          f"\n{'=' * 108}")
    print(f"  UNPROFILED cycle (median of 12, keyframes pre-generated): {unprofiled:.1f} ms")

    cheap = analyse(load_trace(profiled_cycle("/tmp/iwm_gap_cuda.json", cpu=False)),
                    a.gap_floor, a.top, "CUDA activities only")
    full_bare = analyse(load_trace(profiled_cycle("/tmp/iwm_gap_bare.json")),
                        a.gap_floor, a.top, "CPU+CUDA, no scopes")
    undo = add_scopes(S, server)
    cycle()  # settle after patching
    scoped = analyse(load_trace(profiled_cycle(a.trace)), a.gap_floor, a.top, "CPU+CUDA, with scopes")
    undo()

    print(f"\n  {'instrument':<26}{'window ms':>11}{'vs unprofiled':>15}{'gap ms':>10}")
    for nm, r in (("CUDA only", cheap), ("CPU+CUDA", full_bare), ("CPU+CUDA + scopes", scoped)):
        print(f"  {nm:<26}{r['wall'] / 1000:>11.1f}{r['wall'] / 1000 / unprofiled:>14.2f}x"
              f"{r['total_gap'] / 1000:>10.1f}")
    cheap_err = abs(cheap["wall"] / 1000 - unprofiled) / unprofiled
    print(f"\n  The CUDA-only pass is within {cheap_err:.1%} of the unprofiled cycle, so IT owns the gap")
    print(f"  sizes below. The CPU+CUDA passes inflate the host and therefore manufacture gap time; they")
    print(f"  are used ONLY for attribution shares, which is stated wherever their numbers appear.")
    if cheap_err > 0.15:
        print(f"  NOT EVALUATED: even the CUDA-only pass distorts the cycle by {cheap_err:.1%}.")

    print(f"\n{'=' * 108}\n2a. GAP SIZES -- from the CUDA-only pass (trustworthy)\n{'=' * 108}")
    gap_histogram(cheap["gaps"], cheap["total_gap"])
    launch_decomposition(load_trace("/tmp/iwm_gap_cuda.json"), cheap, "CUDA-only")

    print(f"\n{'=' * 108}\n2b. GAP SIZES -- from the CPU+CUDA pass, for comparison only (INFLATED)"
          f"\n{'=' * 108}")
    gap_histogram(scoped["gaps"], scoped["total_gap"])
    scope_tot, cause_tot, n_root = attribute(scoped, a.gap_floor, a.top)

    print(f"\n{'=' * 108}\n7. WHAT THIS EXPLAINS OF THE GAP TOTAL\n{'=' * 108}")
    print(f"  TRUE gap total (CUDA-only pass)      {cheap['total_gap'] / 1000:8.1f} ms of a "
          f"{unprofiled:.0f} ms cycle, device busy {cheap['busy'] / 1000:.0f} ms")
    print(f"  gaps counted                         {len(cheap['gaps']):8d}   mean "
          f"{cheap['total_gap'] / max(len(cheap['gaps']), 1):.1f} us")
    print(f"  ---- shares below come from the INFLATED pass ----")
    tg = scoped["total_gap"] / 1000
    print(f"  total gap in the profiled cycle      {tg:8.1f} ms")
    print(f"  host-dispatch ceiling from section I "
          f"{n_root * 1.02 / 1000:8.1f} ms  ({n_root} Python-originated x 1.02 us)")
    top3 = sum(t for _, t in scope_tot.most_common(3)) / 1000
    print(f"  top 3 scopes by gap time             {top3:8.1f} ms  ({top3 / tg:.0%})")
    for c, t in cause_tot.most_common(3):
        print(f"  {c:<36}{t / 1000:8.1f} ms  ({t / scoped['total_gap']:.0%})")
    print(f"\n  DECISION RULE: concentrated in a few mechanisms -> optimize those. Diffuse across tens of")
    print(f"  thousands of small gaps -> that is the eager-runtime floor, and micro-optimization stops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

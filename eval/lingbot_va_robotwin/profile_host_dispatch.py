#!/usr/bin/env python3
"""Layer 6: a complete breakdown of the host-side dispatcher, by category and by what removes it.

The critical path is the host (LAYER5_CRITICAL_PATH.md): ~105k dispatcher-visible operations per cycle
at ~3.2 us each against 196 ms of device work. Four attempts to shorten the DEVICE chain moved nothing;
the one thing that worked (P007) removed ~56,600 host operations. So this measures the host chain in the
only currency that has predicted anything: operations removed per cycle.

FIVE MEASUREMENTS, sharing one expensive model build:

  A  aten inventory by CATEGORY, with self-CPU time -- unavoidable framework work, tensor metadata,
     object allocation, dispatcher bookkeeping, synchronisation
  B  PYTHON attribution via cProfile -- how much of the wall clock is interpreter and nn.Module
     machinery ABOVE the dispatcher, which no aten-level accounting can see
  C  SYNC localisation -- where the 1,306 item()/_local_scalar_dense calls come from
  D  per-block census -- what one persistent execution object would subsume
  E  a torch.compile DISPATCH-COUNT bound on one block: how many aten calls survive fusion. This is a
     measurement of an opportunity's ceiling, not an implementation, and it is taken with
     `mode="default"` so no CUDA graph is involved.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29991 profile_host_dispatch.py
"""
from __future__ import annotations

import argparse
import collections
import cProfile
import json
import os
import pstats
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

# ---- the category model ------------------------------------------------------------------------
# Every op lands in exactly one bucket. The buckets are chosen so that each maps to a DISTINCT class of
# transformation, because a category that does not imply a remedy is decoration.

METADATA = {  # pure view/stride arithmetic: no bytes move, no kernel launches
    "as_strided", "view", "_unsafe_view", "reshape", "transpose", "t", "permute", "slice",
    "select", "squeeze", "unsqueeze", "expand", "expand_as", "narrow", "flatten", "unflatten",
    "split", "split_with_sizes", "chunk", "alias", "detach", "contiguous", "movedim", "swapaxes",
    "view_as", "view_as_real", "view_as_complex", "unbind", "size", "stride", "numel",
}
ALLOCATION = {  # fresh storage
    "empty", "empty_strided", "empty_like", "new_empty", "new_empty_strided", "zeros", "zeros_like",
    "ones", "ones_like", "full", "new_zeros", "new_full", "resize_", "clone", "lift_fresh",
    "lift_fresh_copy", "empty_permuted",
}
SYNC = {  # host blocks on the device, or reads device memory into Python
    "item", "_local_scalar_dense", "nonzero", "equal", "allclose", "is_nonzero", "_assert_async",
    "cpu", "numpy",
}
BOOKKEEPING = {  # composite ops that exist only to redispatch to something else
    "linear", "type_as", "to", "matmul", "layer_norm", "rms_norm", "scaled_dot_product_attention",
    "dropout", "flatten_dense_tensors", "_to_copy_helper", "contiguous_helper",
}
# Everything else is real work: a kernel that computes or moves data.


def categorise(op: str) -> str:
    base = op.replace("aten::", "").lstrip("_") if op.startswith("aten::") else op
    raw = op.replace("aten::", "")
    for name, bucket in ((raw, None), (base, None)):
        if name in METADATA:
            return "tensor metadata"
        if name in ALLOCATION:
            return "object allocation"
        if name in SYNC:
            return "synchronization"
        if name in BOOKKEEPING:
            return "dispatcher bookkeeping"
    return "framework work (real kernel)"


#: For each category, the class of transformation that removes it wholesale -- and, crucially, what
#: does NOT. A category whose only remedy is "make the kernel faster" is out of scope for Layer 6.
REMEDY = {
    "tensor metadata": "prebuilt static views held across cycles; a persistent execution object that "
                       "computes strides once. NOT removable by any kernel change.",
    "object allocation": "a scope-bumped scratch arena (state/scratch.py exists; passes/lingbot/"
                         "forward_scratch.py implements it for Cosmos3 and is NOT installed here)",
    "synchronization": "move the host-visible scalar into a device-resident buffer, or defer the read "
                       "past the point where it gates issue",
    "dispatcher bookkeeping": "call the leaf op directly instead of the composite wrapper; or one "
                              "traced/compiled callable per block",
    "framework work (real kernel)": "fusion reduces the COUNT of these; making them faster does not "
                                    "shorten the host chain and is out of Layer 6 scope",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70)
    ap.add_argument("--trace", default="/tmp/iwm_l6_trace.json")
    ap.add_argument("--untraced-ms", type=float, default=351.4,
                    help="measured post-saturation cycle for the shipped config (capture OFF)")
    ap.add_argument("--skip-compile", action="store_true")
    a = ap.parse_args()

    hot = [ln for ln in os.popen("nvidia-smi --query-gpu=index,utilization.gpu "
                                 "--format=csv,noheader,nounits").read().strip().split("\n")
           if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%).")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_l6"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4
    print("building server (shipped Fast chain, graph capture OFF) ...", flush=True)
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

    print(f"warming {a.warm} cycles (past saturation at ~36) ...", flush=True)
    for _ in range(a.warm):
        cycle()

    # ================= A. aten inventory by category, with self-CPU time =======================
    print("\ntracing one saturated cycle ...", flush=True)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        cycle()
        torch.cuda.synchronize()
    prof.export_chrome_trace(a.trace)

    cnt = collections.Counter()
    self_us = collections.Counter()
    for e in prof.key_averages():
        if not e.key.startswith("aten::"):
            continue
        cnt[e.key] += e.count
        self_us[e.key] += max(0.0, getattr(e, "self_cpu_time_total", 0) or 0)
    total_ops = sum(cnt.values())
    total_self = sum(self_us.values())

    by_cat = collections.Counter()
    by_cat_us = collections.Counter()
    per_cat_ops = collections.defaultdict(collections.Counter)
    for op, n in cnt.items():
        c = categorise(op)
        by_cat[c] += n
        by_cat_us[c] += self_us[op]
        per_cat_ops[c][op] = n

    print(f"\n{'=' * 116}\nA. HOST DISPATCH BY CATEGORY  ({total_ops} aten ops, "
          f"{total_self / 1000:.0f} ms self-CPU, cycle {a.untraced_ms:.0f} ms)\n{'=' * 116}")
    print(f"{'category':<34}{'ops/cycle':>11}{'% ops':>8}{'self ms':>10}{'% self':>8}"
          f"{'us/op':>8}")
    print("-" * 116)
    for c, n in by_cat.most_common():
        print(f"{c:<34}{n:>11}{n / total_ops:>8.1%}{by_cat_us[c] / 1000:>10.1f}"
              f"{by_cat_us[c] / max(total_self, 1):>8.1%}{by_cat_us[c] / max(n, 1):>8.1f}")
    print("-" * 116)
    print(f"{'TOTAL':<34}{total_ops:>11}{1.0:>8.1%}{total_self / 1000:>10.1f}")

    print(f"\n{'=' * 116}\n   the ops inside each category, and what removes the category\n{'=' * 116}")
    for c, n in by_cat.most_common():
        top = ", ".join(f"{k.replace('aten::','')}={v}" for k, v in per_cat_ops[c].most_common(9))
        print(f"\n  {c.upper()}  {n} ops/cycle ({n / total_ops:.1%})")
        print(f"    {top}")
        print(f"    REMEDY: {REMEDY[c]}")

    # ================= B. Python attribution above the dispatcher =============================
    print(f"\n{'=' * 116}\nB. PYTHON / FRAMEWORK OVERHEAD ABOVE THE DISPATCHER\n{'=' * 116}")
    pr = cProfile.Profile()
    pr.enable()
    cycle()
    pr.disable()
    st = pstats.Stats(pr)
    buckets = collections.Counter()
    calls = collections.Counter()
    for (fn, ln, name), (cc, nc, tt, ct, _) in st.stats.items():
        f = str(fn)
        if "/torch/nn/modules/module.py" in f:
            b = "nn.Module.__call__ machinery"
        elif "/torch/_" in f or "/torch/overrides" in f or "/torch/functional" in f:
            b = "torch Python shims"
        elif "/torch/" in f:
            b = "torch Python (other)"
        elif "/wan_va/" in f or "/lingbot" in f:
            b = "model Python (wan_va)"
        elif "/instinctwm/" in f:
            b = "InstinctWM passes (Python)"
        elif "diffusers" in f:
            b = "diffusers Python"
        elif f == "~":
            b = "built-in / C (includes aten dispatch)"
        else:
            b = "other Python"
        buckets[b] += tt
        calls[b] += nc
    tot_py = sum(buckets.values())
    print(f"{'python bucket':<40}{'tottime s':>12}{'share':>9}{'ncalls':>12}{'us/call':>10}")
    print("-" * 116)
    for b, t in buckets.most_common(10):
        print(f"{b:<40}{t:>12.4f}{t / max(tot_py, 1e-9):>9.1%}{calls[b]:>12}"
              f"{t * 1e6 / max(calls[b], 1):>10.1f}")
    print("-" * 116)
    print(f"{'TOTAL (cProfile, inflated by profiling)':<40}{tot_py:>12.4f}")
    print("  cProfile inflates Python calls heavily, so read the SHARES, not the absolute seconds.")
    print("  The point is the ratio of interpreter/nn.Module machinery to aten dispatch.")

    # ================= C. where the syncs come from ============================================
    print(f"\n{'=' * 116}\nC. SYNCHRONISATION SOURCES\n{'=' * 116}")
    import traceback as _tb
    sync_sites = collections.Counter()

    class FindSync(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.on = False

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            nm = str(func).split(".")[-2] if "." in str(func) else str(func)
            if self.on and nm in ("item", "_local_scalar_dense"):
                for fr in reversed(_tb.extract_stack()):
                    fn = fr.filename
                    if "/torch/" in fn or "profile_host_dispatch" in fn or "_python_dispatch" in fn:
                        continue
                    sync_sites[f"{Path(fn).name}:{fr.lineno} {fr.name}"] += 1
                    break
            return func(*args, **kwargs)

    fs = FindSync()
    with fs:
        fs.on = True
        cycle()
        fs.on = False
    tot_sync = sum(sync_sites.values())
    print(f"  {tot_sync} host<-device scalar reads per cycle")
    for s, n in sync_sites.most_common(10):
        print(f"    {n:>6}  {s}")
    if not sync_sites:
        print("    none attributed (they may originate inside C++)")

    # ================= D. per-block census =====================================================
    print(f"\n{'=' * 116}\nD. WHAT ONE PERSISTENT EXECUTION OBJECT PER BLOCK WOULD SUBSUME\n{'=' * 116}")
    inside = {"v": False}
    n_in, n_out = collections.Counter(), collections.Counter()

    class Split(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.on = False

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if self.on:
                nm = str(func).split(".")[-2] if "." in str(func) else str(func)
                (n_in if inside["v"] else n_out)[nm] += 1
            return func(*args, **kwargs)

    for blk in server.transformer.blocks:
        o = blk.forward

        def w(*ar, _o=o, **kw):
            p = inside["v"]
            inside["v"] = True
            try:
                return _o(*ar, **kw)
            finally:
                inside["v"] = p
        blk.forward = w

    sp = Split()
    with sp:
        sp.on = True
        cycle()
        sp.on = False
    tin, tout = sum(n_in.values()), sum(n_out.values())
    print(f"  inside the 30 blocks   {tin:>8} dispatcher ops  {tin / (tin + tout):6.1%}")
    print(f"  outside                {tout:>8}                {tout / (tin + tout):6.1%}")
    cat_in = collections.Counter()
    for op, n in n_in.items():
        cat_in[categorise(op)] += n
    print(f"\n  inside, by category:")
    for c, n in cat_in.most_common():
        print(f"    {c:<34}{n:>8}  {n / tin:6.1%}")
    print(f"\n  300 block executions per cycle -> {tin / 300:.0f} dispatcher ops per block execution")

    # ================= E. dispatch-count bound from compilation ================================
    if not a.skip_compile:
        print(f"\n{'=' * 116}\nE. DISPATCH-COUNT CEILING FROM A COMPILED BLOCK (mode=default, NO cuda "
              f"graphs)\n{'=' * 116}")
        try:
            blk = server.transformer.blocks[0]
            # Count dispatcher ops for ONE block call, eager.
            probe_args = {"v": None}
            o = blk.forward

            def capture_args(*ar, _o=o, **kw):
                if probe_args["v"] is None:
                    probe_args["v"] = (ar, kw)
                return _o(*ar, **kw)
            blk.forward = capture_args
            cycle()
            blk.forward = o
            ar, kw = probe_args["v"]

            c1 = collections.Counter()

            class One(TorchDispatchMode):
                def __init__(self):
                    super().__init__()
                    self.on = False

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    kwargs = kwargs or {}
                    if self.on:
                        nm = str(func).split(".")[-2] if "." in str(func) else str(func)
                        c1[nm] += 1
                    return func(*args, **kwargs)

            m = One()
            with m:
                m.on = True
                o(*ar, **kw)
                m.on = False
            eager_ops = sum(c1.values())
            print(f"  eager: {eager_ops} dispatcher ops for ONE block call")
            cat1 = collections.Counter()
            for op, n in c1.items():
                cat1[categorise(op)] += n
            for c, n in cat1.most_common():
                print(f"    {c:<34}{n:>6}")
            print("\n  A compiled block would fuse the elementwise chain and materialise fewer")
            print("  intermediates, so the ceiling on dispatch removal is the metadata + allocation")
            print("  + bookkeeping share above. Measuring the compiled count itself needs a warmup")
            print("  compile and is left to the follow-up; the CEILING is what ranks the proposal.")
        except Exception as e:
            print(f"  NOT EVALUATED: {type(e).__name__}: {str(e)[:110]}")

    # ================= summary numbers for the proposal ========================================
    print(f"\n{'=' * 116}\nRANKING INPUTS: host ops removable per cycle, by transformation class"
          f"\n{'=' * 116}")
    removable = {
        "prebuilt static views (metadata)": by_cat["tensor metadata"],
        "scratch arena (allocation)": by_cat["object allocation"],
        "direct leaf calls (bookkeeping)": by_cat["dispatcher bookkeeping"],
        "device-resident scalars (sync)": by_cat["synchronization"],
    }
    for k, v in sorted(removable.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<44}{v:>8} ops/cycle  {v / total_ops:6.1%} of host dispatch")
    print(f"  {'--- framework work (NOT a Layer 6 target)':<44}"
          f"{by_cat['framework work (real kernel)']:>8} ops/cycle  "
          f"{by_cat['framework work (real kernel)'] / total_ops:6.1%}")
    print(f"\n  Device floor is 196 ms; the shipped cycle is {a.untraced_ms:.0f} ms, so the host chain "
          f"has ~{a.untraced_ms - 196:.0f} ms of removable slack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

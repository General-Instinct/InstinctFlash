#!/usr/bin/env python3
"""Which OPERATORS produce the elementwise/layout bucket, with shapes and call counts.

PROFILE.md establishes that elementwise and layout kernels are 45.9% of GPU time in the warm 2V/4A
cycle -- 112 ms, the largest category, 2.5x attention. That is a bucket, not a target. A kernel cannot
be written against `void at::native::elementwise_kernel<128, 4, ...>`; it has to be written against a
source-level pattern with known shapes and a known call count.

So this aggregates by ATEN OPERATOR and by INPUT SHAPE, not by kernel name, and reports device time.
The output is the candidate list for a fusion region: an op that costs a lot AND is called many times
AND always sees the same shapes is fusible; one that costs a lot across a hundred different shapes is
not, however large its total.

MEASURED WARM ONLY. Cycles 1-30 run at 1385 ms and cycles 31+ at 490 ms (PROFILE.md), so anything
profiled before cycle 31 describes the transient. Tracing starts after the run has converged.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29986 \\
        profile_layer5_ops.py [--warm 40] [--trace 3]
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

from instinctflash.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

LAYOUT_OPS = ("cat", "copy", "clone", "contiguous", "permute", "transpose", "reshape", "view",
              "expand", "repeat", "stack", "index", "slice", "pad", "to", "empty", "zeros")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=40, help="cycles before tracing (transient ends ~31)")
    ap.add_argument("--trace", type=int, default=3)
    ap.add_argument("--video", type=int, default=2)
    ap.add_argument("--action", type=int, default=4)
    ap.add_argument("--top", type=int, default=28)
    ap.add_argument("--stacks", action="store_true", help="attribute cat/copy_ to source")
    ap.add_argument("--conv-layout", choices=["as-is", "ndhwc"], default="as-is")
    a = ap.parse_args()

    hot = [ln for ln in os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    ).read().strip().split("\n") if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%).")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_l5_ops"
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

    # ---- apply the conv backend plan, through the dispatch layer ------------------------------
    if a.conv_layout == "ndhwc":
        from instinctflash.backends.conv import REGISTRY, ConvShape, register_declared
        from instinctflash.backends.conv.semantics import ConvSemantics, MemoryLayout
        register_declared(REGISTRY)
        convs = [m for m in server.streaming_vae.vae.modules()
                 if isinstance(m, torch.nn.Conv3d) and m.weight.dim() == 5]
        # Ask the layer, with the measured encode-scale numbers, rather than asserting the answer.
        plan = REGISTRY.select(
            semantics=ConvSemantics.CAUSAL_TIME,
            shape=ConvShape(160, 160, (3, 3, 3), spatial=(8, 128, 160), dtype="bfloat16"),
            have_layout=MemoryLayout.NCDHW, subgraph_size=len(convs),
            prefer_bitexact=False,
            measured={("torch_fallback", MemoryLayout.NCDHW): 175.72,
                      ("cudnn_conv3d", MemoryLayout.NDHWC): 17.00})
        print(f"  conv plan: {plan.backend_name} / {plan.use_layout.value} "
              f"[{plan.tier.name}] convert_subgraph={plan.convert_subgraph}")
        print(f"    {plan.reason}")
        if plan.convert_subgraph:
            fmt = plan.use_layout.torch_memory_format()
            for m in convs:
                m.to(memory_format=fmt)
            print(f"    converted {len(convs)} Conv3d weights; the encoder's activations follow")
        # The half-res VAE serves the wrist cameras and must be converted too, or two thirds of the
        # encode stays on the fallback path.
        half = getattr(server, "streaming_vae_half", None)
        if half is not None and plan.convert_subgraph:
            hc = [m for m in half.vae.modules()
                  if isinstance(m, torch.nn.Conv3d) and m.weight.dim() == 5]
            for m in hc:
                m.to(memory_format=plan.use_layout.torch_memory_format())
            print(f"    converted {len(hc)} Conv3d weights in the half-res VAE as well")

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

    print(f"warming {a.warm} cycles (transient ends ~31) ...", flush=True)
    cycle(True)
    for _ in range(a.warm):
        cycle(False)

    print(f"tracing {a.trace} warm cycles with shapes ...", flush=True)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=a.stacks) as prof:
        for _ in range(a.trace):
            cycle(False)

    if a.stacks:
        # `group_by_stack_n` is the SUPPORTED way to attribute an aten op to Python frames. Two
        # earlier attempts failed and are worth recording: iterating prof.events() and reading
        # `ev.stack` returned empty, and monkeypatching torch.Tensor.copy_ counted ZERO calls --
        # because these copies are emitted inside C++ ops (slice assignment, .to(), conv fallbacks),
        # never from a Python `.copy_()`. Only the dispatcher sees them.
        print(f"\n{'=' * 100}\nCALL SITES by Python stack (aten::copy_ / fill_ / cat)\n{'=' * 100}")
        rows = []
        for e in prof.key_averages(group_by_stack_n=6):
            if e.key not in ("aten::copy_", "aten::fill_", "aten::cat"):
                continue
            st = [f for f in (getattr(e, "stack", None) or [])
                  if "/torch/" not in f and "profile_layer5" not in f]
            rows.append((e.count, e.key, (getattr(e, "self_device_time_total", 0) or 0), st))
        rows.sort(reverse=True)
        if not rows:
            print("  NOT EVALUATED: no stacks captured for these ops.")
        for count, key, us, st in rows[:14]:
            print(f"\n  {key:<12} {count / a.trace:>8.0f} calls/cyc  {us / 1000 / a.trace:>7.2f} ms/cyc")
            for f in st[:4]:
                print(f"      {f[-96:]}")

    # ---- by operator, ignoring shape ---------------------------------------------------------
    by_op = collections.Counter()
    calls = collections.Counter()
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", 0) or 0
        if t > 0 and e.key.startswith("aten::"):
            by_op[e.key] += t
            calls[e.key] += e.count
    tot = sum(by_op.values())
    print(f"\n{'=' * 96}\nDEVICE TIME BY ATEN OPERATOR (warm, {a.trace} cycles)\n{'=' * 96}")
    print(f"{'operator':<34}{'ms/cycle':>10}{'share':>8}{'calls/cyc':>11}{'us/call':>10}")
    print("-" * 96)
    layout_total = 0.0
    for op, us in by_op.most_common(a.top):
        c = calls[op] / a.trace
        ms = us / 1000 / a.trace
        short_name = op.replace("aten::", "")
        is_layout = any(short_name.startswith(x) or short_name == x for x in LAYOUT_OPS)
        if is_layout:
            layout_total += ms
        print(f"{op:<34}{ms:>10.2f}{us / tot:>7.1%}{c:>11.1f}"
              f"{us / max(calls[op], 1):>10.1f}{'   <-- layout' if is_layout else ''}")
    print("-" * 96)
    print(f"{'total device time in aten ops':<34}{tot / 1000 / a.trace:>10.2f}")
    print(f"{'of which layout/copy ops':<34}{layout_total:>10.2f}"
          f"{layout_total / (tot / 1000 / a.trace):>7.1%}")

    # ---- by operator AND shape: fusibility needs shape stability ------------------------------
    print(f"\n{'=' * 96}\nTOP (OPERATOR, SHAPE) PAIRS -- a fusion target needs ONE shape, not many"
          f"\n{'=' * 96}")
    by_shape = collections.Counter()
    shape_calls = collections.Counter()
    for e in prof.key_averages(group_by_input_shape=True):
        t = getattr(e, "self_device_time_total", 0) or 0
        if t > 0 and e.key.startswith("aten::"):
            k = (e.key, str(getattr(e, "input_shapes", ""))[:58])
            by_shape[k] += t
            shape_calls[k] += e.count
    print(f"{'operator':<24}{'ms/cyc':>8}{'calls/cyc':>10}  shapes")
    print("-" * 96)
    for (op, shp), us in by_shape.most_common(a.top):
        print(f"{op.replace('aten::', ''):<24}{us / 1000 / a.trace:>8.2f}"
              f"{shape_calls[(op, shp)] / a.trace:>10.1f}  {shp}")

    # ---- how concentrated is each op across shapes? -------------------------------------------
    print(f"\n{'=' * 96}\nSHAPE CONCENTRATION of the top layout ops (fusible <=> few distinct shapes)"
          f"\n{'=' * 96}")
    per_op_shapes = collections.defaultdict(list)
    for (op, shp), us in by_shape.items():
        per_op_shapes[op].append((us, shp))
    for op, us in by_op.most_common(12):
        sn = op.replace("aten::", "")
        if not any(sn.startswith(x) or sn == x for x in LAYOUT_OPS):
            continue
        shapes = sorted(per_op_shapes.get(op, []), reverse=True)
        top_share = shapes[0][0] / sum(s[0] for s in shapes) if shapes else 0.0
        print(f"  {sn:<20} {len(shapes):>3} distinct shapes, "
              f"largest holds {top_share:>5.1%} of its time  "
              f"({us / 1000 / a.trace:.2f} ms/cyc total)")
    print("\n  An op with few shapes and one dominant shape is a kernel. An op spread over dozens of")
    print("  shapes is a design problem in the caller, and fusing it would need dynamic shapes --")
    print("  which also forfeits CUDA graph capture. Prefer the concentrated ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

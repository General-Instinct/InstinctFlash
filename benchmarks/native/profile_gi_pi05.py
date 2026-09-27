"""Kernel breakdown of the GI pi05 engine on a parity bundle's first input."""
import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    from benchmarks.native.pi05_parity import _active, _checkpoint
    from instinctflash.engines.pi05.engine import Pi05Engine
    from instinctflash.engines.pi05.weights import load_pi05_weights
    data = torch.load(args.bundle)
    dev = torch.device("cuda")
    weights = load_pi05_weights(_checkpoint(), dev)
    x, n = data["inputs"][0], data["noise"][0]
    views = torch.stack(_active(x)).to(dev)
    eng = Pi05Engine(weights, views.shape[0], dev)
    eng.set_prompt(x["tokens"][x["masks"]].numpy())
    for _ in range(3):
        eng.infer(views, n.to(dev))
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10):
        eng.infer(views, n.to(dev))
    e.record()
    torch.cuda.synchronize()
    print(f"infer (graph replay incl. input copies): {s.elapsed_time(e) / 10:.2f} ms")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.reps):
            eng.infer(views, n.to(dev))
        torch.cuda.synchronize()
    totals, counts = defaultdict(float), defaultdict(int)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            key = ev.name.split("(")[0][:110]
            totals[key] += ev.device_time_total
            counts[key] += 1
    total = sum(totals.values())
    print(f"GPU kernel time per infer: {total / args.reps / 1e3:.3f} ms")
    for name, us in sorted(totals.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"{us / args.reps / 1e3:8.3f} ms {counts[name] // args.reps:6d}x {100 * us / total:5.1f}%  {name}")


if __name__ == "__main__":
    main()

"""Event-timed GI pi05 sections (vision / prefix / expert / full), each its own CUDA graph."""
import argparse
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--reps", type=int, default=20)
    args = p.parse_args()
    from benchmarks.native.pi05_parity import _active, _checkpoint
    from instinctflash.engines.pi05.engine import Pi05Engine
    from instinctflash.engines.pi05.weights import load_pi05_weights
    data = torch.load(args.bundle)
    dev = torch.device("cuda")
    x, n = data["inputs"][0], data["noise"][0]
    views = torch.stack(_active(x)).to(dev)
    eng = Pi05Engine(load_pi05_weights(_checkpoint(), dev), views.shape[0], dev)
    eng.set_prompt(x["tokens"][x["masks"]].numpy())
    eng.img.copy_(views)
    eng.noise.copy_(n)
    M = eng._rows
    sections = {"vision": eng._vision, "prefix": lambda: eng._prefix(M), "expert": eng._expert,
                "full": lambda: eng._forward(M)}
    graphs = {}
    for name, fn in sections.items():
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        graphs[name] = g
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        graphs["full"].replay()
    torch.cuda.synchronize()
    for name, g in graphs.items():
        times = []
        for _ in range(args.reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        print(f"{name:8s} p50 {times[len(times) // 2]:7.2f} ms  min {times[0]:7.2f}", flush=True)


if __name__ == "__main__":
    main()

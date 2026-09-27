"""Per-op timing inside a captured GI engine graph via recorded CUDA events.

Each kernel launcher used by the engine module is wrapped to record an event
after it; the graph is captured with those event nodes, replayed with warmed
clocks, and consecutive event deltas are attributed to the op (grouped by
section and op name). Unlike CUPTI tracing on Jetson, this keeps the real L2
and clock state of an ordinary replay.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import torch


class Marker:
    def __init__(self):
        self.events = []
        self.section = "?"
        self.enabled = False

    def mark(self, name):
        if self.enabled:
            ev = torch.cuda.Event(enable_timing=True, external=True)
            ev.record()
            self.events.append((self.section, name, ev))

    def wrap(self, module, names):
        for n in names:
            fn = getattr(module, n)

            def wrapped(*a, _fn=fn, _n=n, **k):
                out = _fn(*a, **k)
                self.mark(_n)
                return out
            setattr(module, n, wrapped)


def profile_engine(engine, forward_sections, module, names, reps=20):
    m = Marker()
    m.wrap(module, names)
    for fn in forward_sections.values():
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    m.enabled = True
    with torch.cuda.graph(graph):
        start = torch.cuda.Event(enable_timing=True, external=True)
        start.record()
        for sec, fn in forward_sections.items():
            m.section = sec
            fn()
            m.mark("<torch/other>")
    m.enabled = False
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        graph.replay()
    torch.cuda.synchronize()
    acc = defaultdict(list)
    for _ in range(reps):
        graph.replay()
        torch.cuda.synchronize()
        prev = start
        per = defaultdict(float)
        cnt = defaultdict(int)
        for sec, name, ev in m.events:
            per[(sec, name)] += prev.elapsed_time(ev)
            cnt[(sec, name)] += 1
            prev = ev
        for k, v in per.items():
            acc[k].append((v, cnt[k]))
    rows = []
    for k, vals in acc.items():
        ts = sorted(v for v, _ in vals)
        rows.append((ts[len(ts) // 2], vals[0][1], k))
    rows.sort(key=lambda r: -r[0])
    total = sum(r[0] for r in rows)
    print(f"total {total:.2f} ms")
    for t, c, (sec, name) in rows:
        print(f"{sec:8s} {name:22s} {t:8.3f} ms {c:5d}x {1e3 * t / c:8.2f} us/call {100 * t / total:5.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    args = p.parse_args()
    from benchmarks.native.pi05_parity import _active, _checkpoint
    import instinctflash.engines.pi05.engine as E
    from instinctflash.engines.pi05.weights import load_pi05_weights
    data = torch.load(args.bundle)
    dev = torch.device("cuda")
    x, n = data["inputs"][0], data["noise"][0]
    views = torch.stack(_active(x)).to(dev)
    eng = E.Pi05Engine(load_pi05_weights(_checkpoint(), dev), views.shape[0], dev)
    eng.set_prompt(x["tokens"][x["masks"]].numpy())
    eng.img.copy_(views)
    eng.noise.copy_(n)
    M = eng._rows
    names = ["gemm", "attention", "residual_norm_quant", "row_quant", "gated_act_quant", "qkv_split", "dequant_rows"]
    profile_engine(eng, {"vision": eng._vision, "prefix": lambda: eng._prefix(M), "expert": eng._expert}, E, names)


if __name__ == "__main__":
    main()

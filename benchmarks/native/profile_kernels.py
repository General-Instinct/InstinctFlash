"""Kernel-time breakdown of a pi05 engine generation (CUPTI via torch.profiler).

    python -m benchmarks.native.profile_kernels --engine gi --top 60
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main():
    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile
    from benchmarks.native.pi05_engine_ab import CHECKPOINT, REVISION, _loop, _observations
    from huggingface_hub import snapshot_download
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="gi")
    p.add_argument("--inputs", default=str(__import__("pathlib").Path.home() / "gi_native/recorded_inputs_v1.npz"))
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--json")
    args = p.parse_args()
    path = snapshot_download(CHECKPOINT, revision=REVISION, local_files_only=True)
    loop = _loop(args.engine, path)
    obs = _observations(args.inputs, 4)
    loop.reset(prompt=obs[0]["prompt"])
    for o in obs:
        loop._queue.clear()
        loop.predict(o)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(args.reps):
            loop._queue.clear()
            np.random.seed(0)
            torch.manual_seed(0)
            loop.predict(obs[1])
        torch.cuda.synchronize()
    totals, counts = defaultdict(float), defaultdict(int)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            totals[e.name] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
            counts[e.name] += 1
    total = sum(totals.values())
    rows = sorted(totals.items(), key=lambda kv: -kv[1])
    print(f"total GPU kernel time per generation: {total / args.reps / 1e3:.3f} ms")
    for name, us in rows[:args.top]:
        print(f"{us / args.reps / 1e3:8.3f} ms  {counts[name] // args.reps:5d}x  {100 * us / total:5.1f}%  {name[:150]}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{"name": n, "ms": us / args.reps / 1e3, "count": counts[n] // args.reps}
                       for n, us in rows], f, indent=1)
    loop.close()


if __name__ == "__main__":
    main()

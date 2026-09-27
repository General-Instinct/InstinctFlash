"""Paired pi05 engine measurement through the Runtime's own engine loop.

Each arm runs the GI engine loop with the checkpoint's native processors and
the recorded public inputs (e.g. two source revisions, or two machines). Every measured
call is a full generation (the action queue is cleared first), and the numpy
and torch RNGs are reseeded before each generation so repeated runs draw
identical initial noise.

    python -m benchmarks.native.pi05_engine_ab --output a.npz     # revision A
    python -m benchmarks.native.pi05_engine_ab --output b.npz     # revision B
    python -m benchmarks.native.pi05_engine_ab --compare a.npz b.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"
REVISION = "8e174154ef5f6c60a8da12ae99c303d8963138c1"


def _observations(archive, count):
    import numpy as np
    with np.load(archive, allow_pickle=False) as data:
        frames = data["frames"].copy()
    out = []
    for i in range(count):
        views = frames[1 + i % 12]
        out.append({
            "observation.images.image": views[0].transpose(2, 0, 1).astype(np.float32) / 255,
            "observation.images.image2": views[1].transpose(2, 0, 1).astype(np.float32) / 255,
            "observation.state": np.zeros(8, np.float32),
            "prompt": "pick up the object" if (i // 4) % 2 == 0 else "place the object down",
        })
    return out


def _loop(engine, path):
    checkpoint = SimpleNamespace(path=path)
    if engine == "gi":
        from instinctflash.engines.pi05 import build_pi05_engine_loop
        return build_pi05_engine_loop(checkpoint, device="cuda:0")
    raise ValueError(engine)


def run(args):
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    path = args.checkpoint or snapshot_download(CHECKPOINT, revision=REVISION, local_files_only=True)
    t0 = time.perf_counter()
    loop = _loop(args.engine, path)
    build_s = time.perf_counter() - t0
    observations = _observations(args.inputs, args.warmup + args.iters)
    loop.reset(prompt=observations[0]["prompt"])
    actions, latencies = [], []
    first_s = None
    for i, obs in enumerate(observations):
        loop._queue.clear()
        np.random.seed(1000 + i)
        torch.manual_seed(1000 + i)
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = loop.predict(obs)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) * 1e3
        if i == 0:
            first_s = dt / 1e3
        chunk = np.stack([loop._post(a).squeeze(0).cpu().numpy() for a in loop._queue]
                         ) if loop._queue else np.zeros((0,))
        actions.append(np.concatenate([out["action"][None], chunk]) if chunk.size else out["action"][None])
        if i >= args.warmup:
            latencies.append(dt)
    lat = np.asarray(latencies)
    summary = {
        "engine": args.engine, "build_s": build_s, "first_call_s": first_s,
        "p50_ms": float(np.percentile(lat, 50)), "p90_ms": float(np.percentile(lat, 90)),
        "mean_ms": float(lat.mean()), "min_ms": float(lat.min()), "iters": len(lat),
        "max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    print(json.dumps(summary, indent=1))
    if args.output:
        np.savez(args.output, actions=np.stack(actions), latencies=lat,
                 summary=json.dumps(summary))
    loop.close()


def compare(a, b):
    import numpy as np
    x, y = np.load(a), np.load(b)
    ax, ay = x["actions"].astype(np.float64), y["actions"].astype(np.float64)
    diff = np.abs(ax - ay)
    scale = np.abs(ax).max()
    print(json.dumps({
        "a": json.loads(str(x["summary"])), "b": json.loads(str(y["summary"])),
        "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
        "max_abs_ref": float(scale),
        "per_call_max_abs_diff": [float(v) for v in diff.reshape(diff.shape[0], -1).max(1)],
        "bit_identical": bool((x["actions"] == y["actions"]).all()),
    }, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("gi",), default="gi")
    p.add_argument("--checkpoint")
    p.add_argument("--inputs", default=str(Path.home() / "gi_native/recorded_inputs_v1.npz"))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--output")
    p.add_argument("--compare", nargs=2)
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()

"""Measure the GR00T N1.7 upstream-BF16 Runtime baseline.

The built-in observation is a valid synthetic OXE DROID input. It measures the
serving path, not task quality; use recorded robot observations for policy evals.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from instinctflash import Runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=None,
        help="N1.7 weights dir; defaults to $GR00T_N17_CHECKPOINT or the package's "
             "base_weights pointer (HF cache)",
    )
    parser.add_argument(
        "--source-root", default=None,
        help="Isaac-GR00T checkout; defaults to $GR00T_ROOT or the documented conventions",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--backend", choices=("eager", "cuda_graph"), default="cuda_graph")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    package = Path(__file__).resolve().parent
    if args.source_root:
        os.environ["GR00T_ROOT"] = str(Path(args.source_root).expanduser().resolve())
    if args.checkpoint:
        os.environ["GR00T_N17_CHECKPOINT"] = str(Path(args.checkpoint).expanduser().resolve())
    os.environ["IFL_GROOT_STATIC_CAPTURE"] = "1" if args.backend == "cuda_graph" else "0"
    runtime = Runtime.from_pretrained(
        package,
        device=args.device,
        placement="in_process",
        nfe={"action": args.nfe},
    )
    runtime.reset(prompt="pick up the object")
    observation = _synthetic_observation()

    for _ in range(args.warmup):
        runtime.predict(observation)
    torch.cuda.synchronize()

    samples = []
    for _ in range(args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = runtime.predict(observation)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)

    values = np.asarray(samples, dtype=np.float64)
    stats = getattr(runtime._backend._impl, "backend_stats", {})
    print(json.dumps({
        "backend": stats.get("backend", args.backend),
        "gpu": torch.cuda.get_device_name(),
        "nfe": args.nfe,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "mean_ms": float(values.mean()),
        "min_ms": float(values.min()),
        "action_shape": list(result["action"].shape),
        "finite": bool(np.isfinite(result["action"]).all()),
        "captured": bool(stats.get("captured", False)),
        "graph_captures": int(stats.get("graph_captures", 0)),
        "graph_replays": int(stats.get("graph_replays", 0)),
        "cpu_threads": stats.get("cpu_threads"),
        "fast_decode": bool(stats.get("fast_decode", False)),
        "backbone_fastpath": bool(stats.get("backbone_fastpath", False)),
        "backbone_cache_hits": int(stats.get("backbone_cache_hits", 0)),
        "backbone_cache_misses": int(stats.get("backbone_cache_misses", 0)),
    }, indent=2))
    runtime.close()
    return 0


def _synthetic_observation() -> dict:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    state = np.array(
        [
            0.5, 0.0, 0.3,
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.4,
            0.0, 0.3, 0.0, -1.9, 0.0, 2.2, 0.1,
        ],
        dtype=np.float32,
    )
    return {"images": [image, image], "state": state}


if __name__ == "__main__":
    raise SystemExit(main())

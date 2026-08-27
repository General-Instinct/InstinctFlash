#!/usr/bin/env python3
"""Measure a Cosmos3 DROID policy server that speaks the openpi msgpack websocket protocol
(NVIDIA's stock robolab server). Same canonical observation as measure_predict.py, so the
two arms are measured by identical bytes."""
import argparse
import json
import statistics
import time

import numpy as np

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ImportError:  # the RoboTwin eval checkout carries the same client
    import sys

    sys.path.insert(0, "/home/ubuntu/RoboTwin")
    from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    obs = {
        "prompt": "pick up the banana and place it in the bowl",
        "observation/image": rng.integers(0, 256, size=(540, 640, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros((2, 7), dtype=np.float32),
        "observation/gripper_position": np.zeros((2, 1), dtype=np.float32),
    }
    cli = WebsocketClientPolicy(host=a.host, port=a.port)

    lat = []
    for i in range(a.warmup + a.n):
        t0 = time.perf_counter()
        ret = cli.infer(obs)
        dt = (time.perf_counter() - t0) * 1000
        if i >= a.warmup:
            lat.append(dt)
        if i == 0:
            act = np.asarray(ret["action"])
            print(f"action {act.shape}  finite {bool(np.isfinite(act).all())}")

    res = {"label": a.label, "n": a.n,
           "wall_ms_p50": round(statistics.median(lat), 1),
           "wall_ms_mean": round(statistics.mean(lat), 1),
           "wall_ms_p90": round(float(np.percentile(lat, 90)), 1)}
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

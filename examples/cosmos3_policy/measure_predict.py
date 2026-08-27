#!/usr/bin/env python3
"""Measure one arm of the Cosmos3 policy benchmark over POST /predict.

Canonical request per the published protocol: one 540x640 image, an action chunk of
[16, 8], 4 denoise steps. The server returns per-request timing; wall latency is measured
client-side on localhost so transport cost is negligible.
"""
import argparse
import base64
import io
import json
import statistics
import time
import urllib.request

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="arm")
    a = ap.parse_args()

    from PIL import Image
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, size=(540, 640, 3), dtype=np.uint8))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    req_body = {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "prompt": "pick up the banana and place it in the bowl",
        "state": [0.0] * 8,
    }
    url = f"http://127.0.0.1:{a.port}/predict"

    lat, server_ms = [], []
    for i in range(a.warmup + a.n):
        t0 = time.perf_counter()
        r = urllib.request.urlopen(urllib.request.Request(
            url, data=json.dumps(req_body).encode(),
            headers={"Content-Type": "application/json"}), timeout=1800)
        resp = json.loads(r.read())
        dt = (time.perf_counter() - t0) * 1000
        if i >= a.warmup:
            lat.append(dt)
            tm = resp.get("timing") or {}
            if "total_ms" in tm:
                server_ms.append(tm["total_ms"])
        act = np.asarray(resp["action"])
        if i == 0:
            print(f"action {act.shape}  finite {bool(np.isfinite(act).all())}")

    res = {"label": a.label, "n": a.n,
           "wall_ms_p50": round(statistics.median(lat), 1),
           "wall_ms_mean": round(statistics.mean(lat), 1),
           "wall_ms_p90": round(float(np.percentile(lat, 90)), 1),
           "server_total_ms_p50": round(statistics.median(server_ms), 1) if server_ms else None}
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

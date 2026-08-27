#!/usr/bin/env python3
"""Measure one arm of the DreamZero benchmark over the official websocket server.

Official observation schema (serve_dreamzero_wan22.py): 2 exterior + 1 wrist camera at
160x320, prompt, session_id. Causal protocol: the FIRST call of the session sends one frame
per camera (cache warm), every later call sends four. The first call is part of warmup and
never lands in the reported latencies.
"""
import argparse
import json
import statistics
import time

import numpy as np
import websockets.sync.client

from openpi_client import msgpack_numpy


class _NoKeepaliveClient:
    """The openpi msgpack protocol, with the keepalive ping DISABLED.

    Same bytes as openpi_client.WebsocketClientPolicy — one packed observation out, one packed
    reply in — but connected with ping_interval=None. The stock client pings every 20 s, and the
    single-threaded upstream server cannot answer while a request computes, so any call past the
    ping timeout kills the connection mid-measurement: DreamZero's second dynamic-arm call
    (torch.compile recompiles on the new shapes) measured >40 s and died exactly there."""

    def __init__(self, host: str, port: int):
        self._packer = msgpack_numpy.Packer()
        while True:
            try:
                self._ws = websockets.sync.client.connect(
                    f"ws://{host}:{port}", compression=None, max_size=None, ping_interval=None)
                msgpack_numpy.unpackb(self._ws.recv())           # server metadata
                return
            except ConnectionRefusedError:
                time.sleep(5)

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)


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

    def frames(n):
        return rng.integers(0, 256, size=(n, 160, 320, 3), dtype=np.uint8)

    cli = _NoKeepaliveClient(host=a.host, port=a.port)
    lat = []
    for i in range(a.warmup + a.n):
        nf = 1 if i == 0 else 4
        obs = {
            "observation/exterior_image_0_left": frames(nf),
            "observation/exterior_image_1_left": frames(nf),
            "observation/wrist_image_left": frames(nf),
            "prompt": "pick up the banana and place it in the bowl",
            "session_id": "bench-1",
            "endpoint": "infer",
        }
        t0 = time.perf_counter()
        ret = cli.infer(obs)
        dt = (time.perf_counter() - t0) * 1000
        if i >= a.warmup:
            lat.append(dt)
        if i <= 1:
            arr = np.asarray(ret["action"] if isinstance(ret, dict) else ret)
            print(f"call {i}: action {arr.shape} finite {bool(np.isfinite(arr).all())}  {dt:.0f} ms")

    res = {"label": a.label, "n": a.n,
           "wall_ms_p50": round(statistics.median(lat), 1),
           "wall_ms_mean": round(statistics.mean(lat), 1),
           "wall_ms_p90": round(float(np.percentile(lat, 90)), 1)}
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The Rerun dashboard observes serving; it must never be able to slow it down.

Two claims:

  1. a loopback session with `viz` attached lands the lerobot-convention entity paths
     (`observation.images.<cam>`, `observation.state`, `action`) plus the serving-only ones
     (`serving/latency_ms`, `serving/infer_ms`, `serving/prompt`, `serving/episode`) in a
     headless .rrd recording, one logged record per predict/episode and zero drops;
  2. when the sink stalls, the predict path keeps its nanosecond enqueue — records are DROPPED
     and counted, the emit calls never block, and close() returns within its timeout.

Needs rerun-sdk (the `viz` extra) plus the serve wire deps; the torch-free core run SKIPs.

    python tests/test_serve_viz.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import msgpack  # noqa: F401
import numpy as np
import rerun  # noqa: F401  (skip-if-absent: the whole file needs the viz extra)
import websockets.sync.client as ws_client

from instinctflash.serving import WebsocketPolicyServer
from instinctflash.serving import msgpack_numpy as mn
from instinctflash.serving.viz import RerunViz

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


class StubEpisode:
    def __init__(self, runtime):
        self.runtime = runtime

    def predict(self, observation, *, executed_action=None):
        return {"action": np.arange(6, dtype=np.float32).reshape(2, 3)}  # a 2-step chunk of dim 3

    def close(self):
        pass


class StubRuntime:
    model_id = "stub-org/toy-wam"
    observation = None

    def episode(self, **conditioning):
        return StubEpisode(self)


def test_loopback_lands_lerobot_paths_in_the_rrd():
    print("\n=== 1. a served episode lands lerobot's entity paths in a headless .rrd ===")
    rrd = os.path.join(tempfile.mkdtemp(), "serve_viz.rrd")
    viz = RerunViz(session_name="instinctflash-serve-test", sink=rrd)
    server = WebsocketPolicyServer(StubRuntime(), host="127.0.0.1", port=0, viz=viz)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.wait_ready(10)

    conn = ws_client.connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None)
    conn.recv(timeout=15)                                        # metadata frame
    obs = {
        "image": {"cam_high": np.zeros((32, 32, 3), np.uint8)},
        "state": np.zeros(7, np.float32),
        "prompt": "stack the bowls",
    }
    for _ in range(3):
        conn.send(mn.packb(obs))
        reply = mn.unpackb(conn.recv(timeout=15))
        assert "action" in reply
    conn.send(mn.packb({"reset": True, "prompt": "second task"}))
    conn.recv(timeout=15)
    conn.close()
    server.shutdown()

    deadline = time.monotonic() + 10                             # 2 episode markers + 3 predicts
    while viz.logged < 5 and time.monotonic() < deadline:
        time.sleep(0.05)
    viz.close()

    check(viz.logged == 5, "3 predicts + 2 episode boundaries -> 5 logged records",
          f"logged={viz.logged}")
    check(viz.dropped == 0, "nothing dropped at this rate", f"dropped={viz.dropped}")
    data = open(rrd, "rb").read()
    check(len(data) > 0, f"the .rrd recorded headless ({len(data)} bytes)")
    for path in (b"observation.images.cam_high", b"observation.state", b"action",
                 b"serving/latency_ms", b"serving/infer_ms", b"serving/prompt",
                 b"serving/episode"):
        check(path in data, f"entity path {path.decode()} is in the recording")
    for needle in (b"stack the bowls", b"second task"):
        check(needle in data, f"the prompt text {needle.decode()!r} is in the recording")
    for timeline in (b"step", b"action_step"):
        check(timeline in data, f"the {timeline.decode()} timeline exists")


def test_backpressure_drops_instead_of_blocking():
    print("\n=== 2. a stalled sink drops and counts; the predict path never blocks ===")
    rrd = os.path.join(tempfile.mkdtemp(), "stalled.rrd")
    viz = RerunViz(session_name="instinctflash-viz-backpressure", sink=rrd, queue_size=2)
    release = threading.Event()
    viz._log_record = lambda record: release.wait(10)            # the sink wedges completely

    obs = {"state": np.zeros(4, np.float32), "prompt": "p"}
    resp = {"action": np.zeros(3, np.float32), "server_timing": {"infer_ms": 1.0}}
    t0 = time.monotonic()
    for _ in range(200):
        viz.emit_predict(obs, resp, wall_ms=1.0)
    elapsed = time.monotonic() - t0
    check(elapsed < 0.5, f"200 emits against a wedged sink took {elapsed * 1000:.1f} ms "
                         "(enqueue-only, never a wait)")
    check(viz.dropped >= 190, "the overflow was dropped and counted", f"dropped={viz.dropped}")

    t0 = time.monotonic()
    viz.close(timeout=1.0)
    closed = time.monotonic() - t0
    check(closed < 5.0, f"close() returns within its timeout even wedged ({closed:.2f}s)")
    release.set()


def main() -> int:
    test_loopback_lands_lerobot_paths_in_the_rrd()
    test_backpressure_drops_instead_of_blocking()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the dashboard observes serving and cannot slow it down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

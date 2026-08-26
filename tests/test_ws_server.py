#!/usr/bin/env python3
"""The network serving mode speaks openpi's wire protocol EXACTLY, and fails loudly.

Four claims, each of which would break a real robot client if false:

  1. our vendored msgpack-numpy shim produces byte-identical frames to the `openpi-client`
     wheel as shipped on pypi (0.1.1) — compatibility is measured, not asserted;
  2. a loopback client sees the openpi session shape: metadata frame first, obs-dict in,
     action-dict + server_timing out, `executed_action` passed through to the episode;
  3. episode boundaries exist for stateful models: a changed prompt rotates the episode, an
     explicit `{"reset": True, ...}` opens one with conditioning, an unchanged prompt does not;
  4. a server-side exception becomes the 0.1.1 error contract (text frame -> client RuntimeError,
     close 1011) and the SERVER SURVIVES — plus a busy port refuses at startup, because the
     client retries a dead port forever (the eval README's documented trap).

No GPU, no weights: the runtime is a stub that records episode/predict calls. Needs numpy,
msgpack and websockets (the `[serve]` extra), so the core torch-free run reports SKIP.
`openpi_client` sections skip individually when the wheel is absent.

    python tests/test_ws_server.py
"""
from __future__ import annotations

import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import msgpack  # noqa: F401  (the wire dependency; missing => this file SKIPs)
import numpy as np
import websockets.sync.client as ws_client
from websockets.exceptions import ConnectionClosed

from instinctflash.serving import WebsocketPolicyServer, default_metadata
from instinctflash.serving import msgpack_numpy as mn

try:
    from openpi_client import msgpack_numpy as openpi_mn
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    HAVE_OPENPI = True
except ImportError:
    HAVE_OPENPI = False

FAILED: list[str] = []
RECV_TIMEOUT = 15


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)
    return cond


# -- a stub runtime: the episode API, none of the weights ------------------------------------------
class StubEpisode:
    def __init__(self, runtime, conditioning):
        self.runtime, self.conditioning, self.closed = runtime, dict(conditioning), False

    def predict(self, observation, *, executed_action=None):
        self.runtime.predicts.append((dict(observation), executed_action, self))
        if self.runtime.fail_next:
            self.runtime.fail_next = False
            raise ValueError("stub predict exploded on purpose")
        if self.runtime.return_raw:
            return np.arange(4, dtype=np.float32)                # NOT a dict: server must wrap it
        return {"action": np.full((2, 3), len(self.runtime.episodes), np.float32)}

    def close(self):
        self.closed = True


class StubRuntime:
    model_id = "stub-org/toy-wam"
    observation = None

    def __init__(self):
        self.episodes: list[StubEpisode] = []
        self.predicts: list[tuple] = []
        self.fail_next = False
        self.return_raw = False

    def episode(self, **conditioning):
        ep = StubEpisode(self, conditioning)
        self.episodes.append(ep)
        return ep


def start_server(runtime):
    server = WebsocketPolicyServer(runtime, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.wait_ready(10)
    return server, port


# -- a minimal client with exactly openpi-client 0.1.1's semantics, plus timeouts ------------------
def connect(port):
    conn = ws_client.connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None)
    metadata = mn.unpackb(conn.recv(timeout=RECV_TIMEOUT))
    return conn, metadata


def infer(conn, obs):
    conn.send(mn.packb(obs))
    response = conn.recv(timeout=RECV_TIMEOUT)
    if isinstance(response, str):
        # the 0.1.1 contract: a text frame is the error channel
        raise RuntimeError(f"Error in inference server:\n{response}")
    return mn.unpackb(response)


# ===================================================================================================
def test_wire_bytes_match_openpi_client():
    print("\n=== 1. our frames are byte-identical to openpi-client 0.1.1's ===")
    if not HAVE_OPENPI:
        print("  SKIP  openpi_client not installed (pip install openpi-client)")
        return
    payloads = [
        {"prompt": "put the bottle in the dustbin", "state": np.random.rand(1, 8).astype(np.float32)},
        {"image": {"base_0_rgb": np.random.randint(0, 256, (3, 224, 224), np.uint8)},
         "obs": [np.zeros((240, 320, 3), np.uint8)] * 2, "reset": True, "n": 3, "f": 2.5,
         "flag": False, "none": None},
        {"scalar": np.float32(1.5), "i64": np.int64(-7), "empty": {}, "list": [1, "two", 3.0]},
        {},
    ]
    for i, p in enumerate(payloads):
        ours, theirs = mn.packb(p), openpi_mn.packb(p)
        check(ours == theirs, f"payload {i}: identical bytes", f"{len(ours)}B vs {len(theirs)}B")
        rt = openpi_mn.unpackb(ours)                             # they unpack ours
        rt2 = mn.unpackb(theirs)                                 # we unpack theirs
        check(_deep_eq(rt, p) and _deep_eq(rt2, p), f"payload {i}: cross round-trip equal")
    for bad in (np.zeros(2, dtype=np.complex64), np.array([object()])):
        mine = their = None
        try:
            mn.packb({"x": bad})
        except ValueError as e:
            mine = str(e)
        try:
            openpi_mn.packb({"x": bad})
        except ValueError as e:
            their = str(e)
        check(mine is not None and their is not None,
              f"dtype {bad.dtype}: both refuse at pack time", f"{mine!r} / {their!r}")


def _deep_eq(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b)) and np.asarray(a).dtype == np.asarray(b).dtype
    return a == b


def test_loopback_session():
    print("\n=== 2. loopback: metadata first, obs in, action + server_timing out ===")
    rt = StubRuntime()
    server, port = start_server(rt)
    try:
        conn, metadata = connect(port)
        check(metadata.get("model_id") == "stub-org/toy-wam", "metadata advertises model_id",
              str(metadata.get("model_id")))
        check(metadata.get("protocol", {}).get("reset_extension") is True,
              "metadata advertises the reset extension")
        check(metadata.get("protocol", {}).get("prompt_rotation") is True,
              "metadata advertises prompt rotation")

        obs = {"state": np.zeros((1, 8), np.float32), "prompt": "p1"}
        out = infer(conn, obs)
        check("action" in out, "a dict prediction passes through unwrapped", str(sorted(out)))
        check("infer_ms" in out.get("server_timing", {}), "server_timing.infer_ms present")
        seen, executed, _ = rt.predicts[-1]
        check("prompt" in seen, "prompt is NOT stripped from the observation (pi0-class reads it)")
        check(executed is None, "no executed_action -> None")

        out = infer(conn, {**obs, "executed_action": np.ones(4, np.float32)})
        _, executed, _ = rt.predicts[-1]
        check(isinstance(executed, np.ndarray) and executed.sum() == 4,
              "executed_action reaches episode.predict as a kwarg")
        check("executed_action" not in rt.predicts[-1][0], "and is removed from the observation")
        check("prev_total_ms" in out["server_timing"], "second reply carries prev_total_ms")

        rt.return_raw = True
        out = infer(conn, obs)
        rt.return_raw = False
        check(isinstance(out.get("actions"), np.ndarray),
              "a non-dict prediction is wrapped under openpi's 'actions' key")

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as resp:
            check(resp.status == 200, "GET /healthz answers 200 on the same port")
        conn.close()
    finally:
        server.shutdown()


def test_prompt_rotation_and_reset():
    print("\n=== 3. episode boundaries: prompt change rotates, reset opens, same prompt keeps ===")
    rt = StubRuntime()
    server, port = start_server(rt)
    try:
        conn, _ = connect(port)
        obs = lambda p: {"state": np.zeros(2, np.float32), "prompt": p}  # noqa: E731

        infer(conn, obs("pick up the cup"))
        infer(conn, obs("pick up the cup"))
        check(len(rt.episodes) == 1, "same prompt twice -> one episode", str(len(rt.episodes)))

        infer(conn, obs("place it on the shelf"))
        check(len(rt.episodes) == 2, "a changed prompt -> a second episode", str(len(rt.episodes)))
        check(rt.episodes[0].closed, "the first episode was closed before the second opened")
        check(rt.episodes[1].conditioning == {"prompt": "place it on the shelf"},
              "the new episode is conditioned on the new prompt")

        reply = infer(conn, {"reset": True, "prompt": "third task", "save_visualization": False})
        check(len(rt.episodes) == 3, "explicit reset -> a third episode")
        check(rt.episodes[2].conditioning == {"prompt": "third task", "save_visualization": False},
              "reset conditioning passes through minus the flag",
              str(rt.episodes[2].conditioning))
        check(set(reply) == {"server_timing"}, "reset replies {} (wan_va parity) + timing only",
              str(sorted(reply)))

        infer(conn, {"state": np.zeros(2, np.float32)})          # no prompt key at all
        check(len(rt.episodes) == 3, "a prompt-less observation keeps the current episode")
        conn.close()
    finally:
        server.shutdown()


def test_error_frame_not_hang():
    print("\n=== 4. a server exception is a typed error frame + close 1011, never a hang ===")
    rt = StubRuntime()
    server, port = start_server(rt)
    try:
        conn, _ = connect(port)
        rt.fail_next = True
        err = None
        try:
            infer(conn, {"state": np.zeros(2, np.float32), "prompt": "x"})
        except RuntimeError as e:
            err = str(e)
        check(err is not None and "stub predict exploded" in err,
              "client raises RuntimeError carrying the server traceback")
        code = None
        try:
            conn.recv(timeout=RECV_TIMEOUT)
        except ConnectionClosed as e:
            code = e.rcvd.code if e.rcvd else None
        check(code == 1011, "the connection closes with INTERNAL_ERROR (1011)", str(code))

        conn2, metadata = connect(port)                          # the port must still serve
        out = infer(conn2, {"state": np.zeros(2, np.float32), "prompt": "y"})
        check("action" in out and metadata.get("model_id") == "stub-org/toy-wam",
              "the server survives one bad request and keeps serving")
        conn2.close()
    finally:
        server.shutdown()


def test_busy_port_fails_loudly():
    print("\n=== 5. a busy port refuses at startup instead of half-starting ===")
    rt = StubRuntime()
    server, port = start_server(rt)
    try:
        second = WebsocketPolicyServer(StubRuntime(), host="127.0.0.1", port=port)
        thread = threading.Thread(target=second.serve_forever, daemon=True)
        thread.start()
        msg = ""
        try:
            second.wait_ready(10)
        except RuntimeError as e:
            msg = str(e)
        check("cannot bind" in msg and "forever" in msg,
              "bind failure names the port and the retry-forever trap", msg[:100])
    finally:
        server.shutdown()


def test_openpi_client_end_to_end():
    print("\n=== 6. the real openpi-client 0.1.1 connects unchanged ===")
    if not HAVE_OPENPI:
        print("  SKIP  openpi_client not installed (pip install openpi-client)")
        return
    rt = StubRuntime()
    server, port = start_server(rt)
    try:
        client = WebsocketClientPolicy(host="127.0.0.1", port=port)
        check(client.get_server_metadata().get("model_id") == "stub-org/toy-wam",
              "get_server_metadata sees our metadata frame")
        out = client.infer({"state": np.zeros((1, 8), np.float32), "prompt": "p1"})
        check("action" in out and "server_timing" in out, "infer round-trips", str(sorted(out)))
        rt.fail_next = True
        raised = False
        try:
            client.infer({"state": np.zeros((1, 8), np.float32), "prompt": "p1"})
        except RuntimeError as e:
            raised = "stub predict exploded" in str(e)
        check(raised, "its error path raises RuntimeError from our text frame")
    finally:
        server.shutdown()


def main() -> int:
    test_wire_bytes_match_openpi_client()
    test_loopback_session()
    test_prompt_rotation_and_reset()
    test_error_frame_not_hang()
    test_busy_port_fails_loudly()
    test_openpi_client_end_to_end()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the serving mode speaks openpi's wire protocol and fails loudly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

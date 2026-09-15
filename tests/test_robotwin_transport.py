"""Exercise the actual msgpack/websocket transport, including wrong-server and timeout paths."""
from __future__ import annotations

import contextlib
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from websockets.sync.server import serve
from instinctflash.serving.msgpack_numpy import Packer, unpackb
from benchmarks.vla.remote_policy import RemotePolicy
from benchmarks.vla.util import ConfigurationError, sha256_json
sys.path.insert(0, str(ROOT / "tests"))
from run_tests import run_module_tests


@contextlib.contextmanager
def server(metadata, respond):
    release = threading.Event()
    def handler(ws):
        packer = Packer()
        ws.send(packer.pack(metadata))
        try:
            for data in ws:
                response = respond(unpackb(data), release)
                ws.send(response if isinstance(response, str) else packer.pack(response))
        except Exception:
            pass  # the client intentionally closes early in refusal and timeout tests
    with serve(handler, "127.0.0.1", 0) as ws_server:
        thread = threading.Thread(target=ws_server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{ws_server.socket.getsockname()[1]}"
        finally:
            release.set()
            ws_server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_real_wire_seed_ack_and_roundtrip_timing():
    identity = {"test_identity": "fixture"}
    def respond(value, release):
        if value.get("reset"):
            return {"benchmark_seed": value["benchmark_seed"],
                    "benchmark_identity_sha256": value["benchmark_identity_sha256"]}
        return {"answer": 1}
    with server({"benchmark_identity": identity}, respond) as endpoint:
        remote = RemotePolicy(endpoint, identity, timeout=2)
        try:
            assert remote.reset_episode("task", 42)["benchmark_seed"] == 42
            assert remote.infer({"obs": []}) == {"answer": 1}
            assert [x["phase"] for x in remote.timings] == ["reset", "infer"]
            assert all(x["roundtrip_ms"] > 0 for x in remote.timings)
        finally:
            remote.close()


def test_empty_metadata_and_unacknowledged_seed_are_refused():
    with server({}, lambda value, release: {}) as endpoint:
        try:
            RemotePolicy(endpoint, {"expected": True}, timeout=1)
        except ConfigurationError as error:
            assert "identity" in str(error)
        else:
            raise AssertionError("unidentified server was accepted")
    identity = {"expected": True}
    with server({"benchmark_identity": identity}, lambda value, release: {}) as endpoint:
        remote = RemotePolicy(endpoint, identity, timeout=1)
        try:
            try:
                remote.reset_episode("task", 7)
            except ConfigurationError as error:
                assert "seed" in str(error)
            else:
                raise AssertionError("unseeded response was accepted")
        finally:
            remote.close()


def test_silent_server_times_out_instead_of_retrying_forever():
    identity = {"expected": True}
    def respond(value, release):
        release.wait(timeout=5)
        return {}
    with server({"benchmark_identity": identity}, respond) as endpoint:
        remote = RemotePolicy(endpoint, identity, timeout=0.05)
        try:
            try:
                remote.infer({"obs": []})
            except TimeoutError:
                pass
            else:
                raise AssertionError("silent server did not time out")
        finally:
            remote.close()


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))

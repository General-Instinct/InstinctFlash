"""Bounded openpi-wire transport for benchmark drivers, independent of simulator/family.

The live server identity is pinned in the plan. An empty upstream metadata frame cannot
attest weights, execution settings or deterministic reset support and is refused.
"""
from __future__ import annotations

import time
import math
from urllib.parse import urlsplit

from .util import ConfigurationError, sha256_json


class RemotePolicy:
    def __init__(self, endpoint: str, identity: dict, *, timeout: float = 120.0, record_dir=None):
        from websockets.sync.client import connect
        from instinctflash.serving.msgpack_numpy import Packer, unpackb

        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname or parsed.username or parsed.password:
            raise ConfigurationError("endpoint must be a ws(s) URL without credentials")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ConfigurationError("remote timeout must be positive")
        self.timeout = timeout
        self.identity = identity
        self.timings = []
        self.recording = None
        self._packer, self._unpack = Packer(), unpackb
        self._ws = connect(endpoint, compression=None, max_size=None, open_timeout=timeout,
                           close_timeout=5, ping_interval=None)
        try:
            metadata = self._receive()
            if metadata.get("benchmark_identity") != identity:
                raise ConfigurationError("live benchmark_identity differs from the frozen plan")
            if record_dir is not None:
                from .policy_trace import PolicyTrace
                self.recording = PolicyTrace(record_dir, identity)
        except BaseException:
            self.close()
            raise

    def _receive(self):
        frame = self._ws.recv(timeout=self.timeout)
        if isinstance(frame, str):
            raise RuntimeError(f"remote inference failed: {frame}")
        value = self._unpack(frame)
        if not isinstance(value, dict):
            raise ConfigurationError("remote response must be a mapping")
        return value

    def infer(self, observation):
        start = time.monotonic()
        packet = self._packer.pack(observation)
        self._ws.send(packet)
        result = self._receive()
        if self.recording is not None:
            self.recording.append(packet, self._packer.pack(result))
        self.timings.append({"phase": "reset" if observation.get("reset") else
                             "commit" if observation.get("compute_kv_cache") else "infer",
                             "roundtrip_ms": (time.monotonic() - start) * 1000})
        return result

    def reset_episode(self, prompt: str, seed: int):
        response = self.infer({"reset": True, "prompt": prompt, "benchmark_seed": seed,
                               "benchmark_identity_sha256": sha256_json(self.identity),
                               "save_visualization": False})
        if response.get("benchmark_seed") != seed:
            raise ConfigurationError("server did not acknowledge the requested episode noise seed")
        if response.get("benchmark_identity_sha256") != sha256_json(self.identity):
            raise ConfigurationError("server did not acknowledge the frozen identity")
        return response

    def close(self):
        self._ws.close()
        if self.recording is not None:
            self.recording.close()

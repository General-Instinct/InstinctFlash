"""Live serving dashboard on Rerun — lerobot's viewer, lerobot's entity paths.

`instinctflash serve <model-id> --serve.viz=true` streams what the server is doing to a Rerun
viewer: camera frames under `observation.images.<cam>`, the state vector under
`observation.state`, the returned action chunk as one per-dimension `rr.Scalars` batch under
`action` — exactly the paths `lerobot-dataset-viz` and lerobot's control loop use, so a lerobot
user's muscle memory works here — plus what only a server knows: `serving/latency_ms`,
`serving/infer_ms`, and episode boundaries (`serving/episode`, `serving/prompt`) on the `step`
timeline. Action chunk rows advance their own `action_step` timeline, one tick per low-level
action, so a 32-step chunk reads as 32 points rather than one smear.

VIZ MUST NEVER DISTORT THE NUMBER BEING SERVED. The predict path only ever *enqueues* (after the
reply has left the socket), a daemon thread does all Rerun work, and a full queue drops the record
and counts it (`dropped`) rather than waiting — a stalled viewer costs frames on the dashboard,
never milliseconds on the wire. Benchmark-grade timing should still run with viz off: the enqueue
is nanoseconds, but a measurement arm that differs from the shipped arm in ANY way is how
mismeasured optimizations happen.

Requires the `viz` extra (`pip install 'instinctflash[viz]'`); constructing `RerunViz` without
`rerun-sdk` raises the ImportError, which the CLI turns into its UNSUPPORTED CAPABILITY report.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_SENTINEL = object()


class RerunViz:
    """A drop-on-backpressure Rerun sink for the websocket policy server.

    `sink` routes the stream: "" spawns a local viewer (lerobot's default UX), a path ending in
    `.rrd` records headless to that file, and a `rerun+http://...` URL connects to a running
    viewer/server via gRPC.
    """

    def __init__(self, session_name: str = "instinctflash-serve", sink: str = "",
                 queue_size: int = 256):
        import rerun as rr                                       # the [viz] extra; may raise
        self._rr = rr
        rr.init(session_name)
        if sink.endswith(".rrd"):
            rr.save(sink)
        elif sink.startswith("rerun+"):
            rr.connect_grpc(url=sink)
        elif sink:
            raise ValueError(f"serve.viz_sink must be '', '*.rrd' or 'rerun+http://...'; "
                             f"got {sink!r}")
        else:
            rr.spawn()

        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self.logged = 0                                          #: records the drain actually sent
        self.dropped = 0                                         #: records lost to backpressure
        self._step = 0
        self._action_step = 0
        self._episode = 0
        self._warned = False
        self._thread = threading.Thread(target=self._drain, daemon=True,
                                        name="instinctflash-viz")
        self._thread.start()

    # -- the predict path calls ONLY these two; both are non-blocking by construction -------------
    def emit_predict(self, observation: Mapping[str, Any], response: Mapping[str, Any],
                     wall_ms: float) -> None:
        self._put(("predict", observation, response, wall_ms, time.time()))

    def emit_episode(self, conditioning: Mapping[str, Any]) -> None:
        self._put(("episode", dict(conditioning), time.time()))

    def _put(self, record) -> None:
        try:
            self._q.put_nowait(record)
        except queue.Full:
            self.dropped += 1                                    # the dashboard pays, not the robot

    def close(self, timeout: float = 5.0) -> None:
        try:
            self._q.put(_SENTINEL, timeout=timeout)
        except queue.Full:
            pass                                                 # drain is wedged; join() below caps the wait
        self._thread.join(timeout=timeout)
        try:
            self._rr.rerun_shutdown()                            # flushes the .rrd sink
        except Exception:                                        # noqa: BLE001 - shutdown is best-effort
            pass

    # -- everything below runs on the drain thread ------------------------------------------------
    def _drain(self) -> None:
        while True:
            record = self._q.get()
            if record is _SENTINEL:
                return
            try:
                self._log_record(record)
                self.logged += 1
            except Exception:                                    # noqa: BLE001 - viz must not take serving down
                if not self._warned:
                    self._warned = True
                    logger.exception("rerun logging failed; further failures are silent")

    def _log_record(self, record) -> None:
        if record[0] == "episode":
            self._log_episode(record[1], record[2])
        else:
            self._log_predict(record[1], record[2], record[3], record[4])

    def _log_episode(self, conditioning: dict, captured: float) -> None:
        rr = self._rr
        self._episode += 1
        rr.set_time("step", sequence=self._step)
        rr.set_time("timestamp", timestamp=captured)
        rr.log("serving/episode", rr.Scalars(float(self._episode)))
        prompt = conditioning.get("prompt")
        if prompt is not None:
            rr.log("serving/prompt", rr.TextLog(str(prompt)))

    def _log_predict(self, observation: Mapping[str, Any], response: Mapping[str, Any],
                     wall_ms: float, captured: float) -> None:
        rr = self._rr
        rr.set_time("step", sequence=self._step)
        rr.set_time("timestamp", timestamp=captured)
        self._step += 1

        for key, value in observation.items():
            self._log_obs(str(key), value)

        rr.log("serving/latency_ms", rr.Scalars(float(wall_ms)))
        infer_ms = (response.get("server_timing") or {}).get("infer_ms")
        if infer_ms is not None:
            rr.log("serving/infer_ms", rr.Scalars(float(infer_ms)))

        action = response.get("action", response.get("actions"))
        for row in _action_rows(action):
            rr.set_time("action_step", sequence=self._action_step)
            self._action_step += 1
            rr.log("action", rr.Scalars(row))

    def _log_obs(self, key: str, value: Any) -> None:
        import numpy as np
        rr = self._rr
        if isinstance(value, Mapping):                           # openpi's {"image": {cam: arr}}
            for cam, frame in value.items():
                self._log_obs(f"{key}.{cam}" if key != "image" else str(cam), frame)
            return
        if isinstance(value, (list, tuple)):                     # LingBot-VA's frames-under-one-key
            if value and isinstance(value[-1], (Mapping, np.ndarray)):
                self._log_obs(key, value[-1])                    # the newest frame is the dashboard
            return
        if isinstance(value, np.ndarray):
            arr = np.squeeze(value)                              # drop batch dims
            if arr.ndim >= 2:                                    # an image, in lerobot's paths
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))           # CHW -> HWC, as lerobot does
                cam = key.removeprefix("observation.").removeprefix("images.")
                rr.log(f"observation.images.{cam}", rr.Image(arr))
            elif arr.ndim == 1:
                rr.log(f"observation.{key.removeprefix('observation.')}",
                       rr.Scalars(arr.astype(float)))
            else:
                rr.log(f"observation.{key.removeprefix('observation.')}",
                       rr.Scalars(float(arr)))
        # strings (prompt) are logged at the episode boundary, not per call


def _action_rows(action) -> list:
    """An action chunk as rows on the action_step timeline: (chunk, dim) -> chunk rows of dim."""
    import numpy as np
    if action is None:
        return []
    arr = np.squeeze(np.asarray(action)).astype(float, copy=False)
    if arr.ndim == 0:
        return [arr.reshape(1)]
    if arr.ndim == 1:
        return [arr]
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return list(arr)

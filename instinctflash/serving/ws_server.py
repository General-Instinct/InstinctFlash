"""Serve a `Runtime` over the network, on the openpi wire protocol.

    from instinctflash import Runtime
    from instinctflash.serving import WebsocketPolicyServer

    runtime = Runtime.from_pretrained("robbyant/lingbot-va-posttrain-robotwin")
    WebsocketPolicyServer(runtime).serve_forever()

WHY THIS PROTOCOL. The pi0/openpi ecosystem already ships a robot-side client
(`pip install openpi-client`): one websocket, msgpack with a small numpy extension
(`serving/msgpack_numpy.py`, vendored byte-for-byte), a metadata dict as the first server frame,
then obs-dict-in / action-dict-out. LingBot-VA's own upstream server is a vendored copy of the
same thing. Speaking it exactly means every existing client — openpi's, wan_va's, and our eval
harness's — connects unchanged; inventing a better-typed protocol would mean shipping and
maintaining a client nobody has.

EPISODE SEMANTICS, because the wire has none. openpi-client 0.1.1 has no reset verb (its
`reset()` sends nothing), which is fine for pi0-class models — per-call stateless, prompt inside
every obs — and wrong for VA-class models whose KV cache IS the episode. Two rules close the gap:

  * a `{"reset": True, <conditioning...>}` message opens a new `runtime.episode(**conditioning)`
    and returns `{}` — the same in-band shape wan_va's server mode uses, so VA-aware clients work
    unchanged;
  * a changed `"prompt"` in an ordinary observation rotates the episode server-side, so a pure
    openpi client that never heard of reset still gets a correct episode boundary wherever a new
    rollout has a new prompt.

Stateless models never notice either rule: the prompt is not stripped from the obs, and rotation
costs a no-op reset.

FAIL LOUDLY. `WebsocketClientPolicy._wait_for_server` retries a refused connection every 5 s
forever, so a server that half-starts (or dies and leaves a script looping) is worse than one that
crashes — the documented dead-port trap in eval/lingbot_va_robotwin/README.md. Consequences here:
the caller loads the model BEFORE the port binds (a doomed checkpoint exits with the loader's
error, port never open); a busy port raises at startup instead of wedging; a request that throws
sends the traceback as a text frame (exactly what 0.1.1 turns into `RuntimeError`) and closes that
connection 1011 — and the server outlives it, so one bad request does not turn the port into the
trap for every other client.

ONE EPISODE PER SERVER PROCESS. `Runtime.episode()` resets the backend's one implicit episode, so
concurrent episodes do not exist below this line either. Connections are accepted concurrently
(healthz, monitors), inference is serialized on one worker thread — which also keeps the event
loop answering pings during a multi-second VA-class control cycle, where the upstream design
(inference on the loop) gets killed by the client's own 20 s ping timeout.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import http
import logging
import threading
import time
import traceback
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: distinguishes "no prompt key at all" from prompt=None; never leaves this module
_UNSET = object()


def default_metadata(runtime) -> dict:
    """What the first server frame advertises: identity and declared execution facts.

    Everything here comes from the checkpoint declaration or the adapter's observation contract —
    the same honesty level as openpi, whose metadata is whatever the policy chooses to carry.
    Action shape rides in the declaration's `extra`/`output_projection` when the checkpoint
    declares it. Additive only: 0.1.1 clients ignore keys they do not know.
    """
    md: dict[str, Any] = {
        "protocol": {
            "wire": "openpi-websocket-msgpack-numpy",
            "reset_extension": True,     # {"reset": True, ...} opens a new episode
            "prompt_rotation": True,     # a changed "prompt" opens a new episode
        },
    }
    try:
        from instinctflash import __version__
        md["instinctflash_version"] = __version__
    except Exception:                                            # noqa: BLE001 - metadata is best-effort
        pass
    model_id = getattr(runtime, "model_id", None)
    if model_id:
        md["model_id"] = model_id
    ex = getattr(getattr(runtime, "checkpoint", None), "execution", None)
    if ex is not None:
        md["backbone"] = getattr(ex, "backbone", None)
        md["nfe"] = dict(getattr(ex, "nfe", None) or {})
        md["guidance"] = dict(getattr(ex, "guidance", None) or {})
        extra = dict(getattr(ex, "extra", None) or {})
        if extra:
            md["extra"] = extra
    try:
        obs = getattr(runtime, "observation", None)
        if obs is not None and getattr(obs, "fields", ()):
            md["observation"] = obs.describe()
            src = getattr(runtime, "observation_source", None)
            if src:
                md["observation_source"] = src
    except Exception:                                            # noqa: BLE001 - metadata is best-effort;
        pass                                                     # the load path raises the real error
    return md


def _same_prompt(a, b) -> bool:
    if a is b:
        return True
    try:
        eq = a == b
        if isinstance(eq, bool):
            return eq
        return bool(getattr(eq, "all", lambda: eq)())            # ndarray/list-shaped prompts
    except Exception:                                            # noqa: BLE001 - incomparable => different
        return False


def _health_check(connection, request):
    """HTTP GET /healthz answers 200 before the websocket upgrade — same as openpi's server."""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class WebsocketPolicyServer:
    """One `Runtime`, one port, the openpi wire protocol. See the module docstring for the rules."""

    def __init__(self, runtime, host: str = "0.0.0.0", port: int = 8000,
                 metadata: Mapping[str, Any] | None = None, viz=None):
        self._runtime = runtime
        self._host, self._port = host, port
        self._metadata = dict(metadata) if metadata is not None else default_metadata(runtime)
        #: optional observability sink (serving/viz.py). The contract is emit_predict /
        #: emit_episode, both non-blocking: viz is never allowed to add wire latency, so it is
        #: fed strictly AFTER the reply frame has been sent.
        self._viz = viz
        self._episode = None
        self._episode_prompt: Any = _UNSET
        # max_workers=1 is the serialization: one implicit episode in the backend means one
        # in-flight predict, and the event loop stays free for pings and /healthz meanwhile.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="instinctflash-serve")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        #: the actually-bound port (differs from `port` when port=0 asks the OS)
        self.bound_port: int | None = None

    # -- lifecycle --------------------------------------------------------------------------------
    def serve_forever(self) -> None:
        """Bind and serve until interrupted. Raises on a failed bind rather than half-starting."""
        asyncio.run(self.run())

    async def run(self) -> None:
        import websockets.asyncio.server as _server

        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        try:
            try:
                ctx = _server.serve(
                    self._handler, self._host, self._port,
                    compression=None, max_size=None, process_request=_health_check,
                )
                async with ctx as server:
                    self.bound_port = next(
                        (s.getsockname()[1] for s in server.sockets), self._port)
                    self._ready.set()
                    logger.info("serving %r on ws://%s:%d",
                                self._metadata.get("model_id", "?"), self._host, self.bound_port)
                    await self._stop.wait()
            except OSError as e:
                # The dead-port trap works both ways: a process that cannot bind but keeps
                # running looks exactly like a healthy slow server to a client that retries
                # forever.
                raise RuntimeError(
                    f"cannot bind ws://{self._host}:{self._port}: {e}. Refusing to continue — "
                    f"openpi clients retry a dead port forever, so a half-started server is "
                    f"worse than a crashed one. Free the port or pass a different --serve.port."
                ) from e
        except BaseException as e:
            # Record BEFORE releasing wait_ready(), or a waiter can observe "not bound" with no
            # reason attached (a measured race, not a hypothetical).
            self._startup_error = e
            raise
        finally:
            self._ready.set()
            self._executor.shutdown(wait=False)

    def wait_ready(self, timeout: float | None = None) -> int:
        """Block until the port is bound (returns it) or startup failed (raises). For tests."""
        self._ready.wait(timeout)
        if self.bound_port is None:
            raise RuntimeError(f"server did not come up: {self._startup_error or 'not bound yet'}")
        return self.bound_port

    def shutdown(self) -> None:
        """Thread-safe stop; `serve_forever` returns after in-flight handlers unwind."""
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)

    # -- the wire ---------------------------------------------------------------------------------
    async def _handler(self, websocket) -> None:
        import websockets
        import websockets.frames

        from instinctflash.serving import msgpack_numpy

        logger.info("connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                is_reset = isinstance(obs, dict) and bool(obs.get("reset"))

                infer_time = time.monotonic()
                response = await asyncio.get_running_loop().run_in_executor(
                    self._executor, self._step, obs)
                infer_time = time.monotonic() - infer_time

                response["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time
                if self._viz is not None and not is_reset:
                    # after the reply is on the wire, and enqueue-only: a stalled dashboard
                    # drops frames, it never adds serving latency (episode markers come from
                    # _open_episode).
                    self._viz.emit_predict(obs, response, wall_ms=prev_total_time * 1000)
            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:                                    # noqa: BLE001 - becomes the error frame
                # The 0.1.1 error contract, verbatim: a TEXT frame where a binary frame was
                # expected is what `WebsocketClientPolicy.infer` turns into RuntimeError. Unlike
                # openpi's reference server we do NOT re-raise: this connection dies 1011, the
                # port keeps serving everyone else instead of becoming the dead-port trap.
                err = traceback.format_exc()
                logger.error("request failed; error frame sent to %s\n%s",
                             websocket.remote_address, err)
                try:
                    await websocket.send(err)
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.")
                except Exception:                                # noqa: BLE001 - already tearing down
                    pass
                break

    # -- episode mapping (runs on the executor thread) --------------------------------------------
    def _step(self, obs: Any) -> dict:
        if not isinstance(obs, dict):
            raise TypeError(
                f"expected a msgpack map (an observation dict); got {type(obs).__name__}")

        if obs.pop("reset", False):
            # In-band episode boundary — wire-compatible with wan_va server mode's
            # {"reset": True, "prompt": ...}; everything but the flag is episode conditioning.
            self._open_episode(obs)
            return {}

        prompt = obs.get("prompt", _UNSET)
        if self._episode is None or (
                prompt is not _UNSET and not _same_prompt(prompt, self._episode_prompt)):
            self._open_episode({} if prompt is _UNSET else {"prompt": prompt})

        executed_action = obs.pop("executed_action", None)
        result = self._episode.predict(obs, executed_action=executed_action)
        if isinstance(result, Mapping):
            return dict(result)                                  # LingBot-VA's {"action": ...} et al.
        import numpy as np
        return {"actions": np.asarray(result)}                   # openpi's conventional key

    def _open_episode(self, conditioning: dict) -> None:
        if self._episode is not None:
            self._episode.close()
        self._episode = self._runtime.episode(**conditioning)
        self._episode_prompt = conditioning.get("prompt", _UNSET)
        logger.info("new episode: conditioning keys %s", sorted(conditioning) or "(none)")
        if self._viz is not None:
            self._viz.emit_episode(conditioning)

"""Where a checkpoint executes. Deliberately not part of the model abstraction.

THE PRODUCT QUESTION THIS ANSWERS, and the measurement that settled it.

LingBot-VA serves behind a websocket because two environments on this box cannot be merged. That is
not a legacy accident; it is bidirectional and it was re-verified on 2026-08-09:

    server interpreter (.venv-lingbot, torch 2.9.0+cu126, diffusers 0.36.0)
        sapien       MISSING      mplib   MISSING       -> cannot host the simulator
    client interpreter (RoboTwin/.venv, torch 2.4.1+cu121, sapien 3.0.0b1)
        diffusers    MISSING      transformers MISSING  -> cannot host the model

So the boundary stays. But notice WHAT it separates: the SIMULATOR from the MODEL. It does not
separate the *user* from the model. Anyone who installs the serving environment can host the model in
their own process; only a caller whose environment cannot import the model stack -- the RoboTwin
client being the live example -- needs a worker.

That is the whole design:

    Runtime picks a PLACEMENT at load time. In-process when this interpreter can host the model,
    a managed worker when it cannot. `predict()` is identical either way, and no public type,
    argument or return value mentions a socket.

Placement is a property of the DEPLOYMENT, not of the checkpoint. Nothing here is read from, or
written to, the execution declaration -- a checkpoint that had to say "I am a websocket model" would
be a checkpoint describing its transport, which is exactly the coupling the platform forbids.
"""

from __future__ import annotations

import atexit
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Protocol


class ExecutionBackend(Protocol):
    """What `Runtime` needs from a placement. Three methods, none of them transport-shaped."""

    def predict(self, observation: Mapping[str, Any]) -> Any: ...
    def reset(self, **conditioning: Any) -> None: ...
    def close(self) -> None: ...


def model_stack_importable() -> tuple[bool, str]:
    """Can THIS interpreter host the model? Returns (yes, reason).

    Checked by import rather than by version pin: a pin would go stale, and the only question that
    matters is whether the modules load here and now.
    """
    missing = []
    for m in ("torch", "diffusers", "transformers", "safetensors"):
        try:
            __import__(m)
        except Exception:                                     # noqa: BLE001 - any failure disqualifies
            missing.append(m)
    if missing:
        return False, f"this interpreter cannot import {', '.join(missing)}"
    return True, "the model stack imports here"


class InProcessBackend:
    """Host the model in the caller's process.

    Chosen when the model stack imports. The adapter builds and drives the server object directly;
    there is no socket, no subprocess and no serialisation.
    """

    def __init__(self, adapter, checkpoint, plan, *, device: str | None = None,
                 nfe: Mapping[str, int] | None = None):
        self._adapter, self._checkpoint, self._plan = adapter, checkpoint, plan
        self._device, self._nfe = device, dict(nfe or {})
        self._impl = None

    def _ensure(self):
        if self._impl is None:
            build = getattr(self._adapter, "build_in_process", None)
            if build is None:
                raise NotImplementedError(
                    f"adapter for {self._checkpoint.execution.backbone!r} does not implement "
                    f"build_in_process(checkpoint, plan, device=, nfe=). Either implement it, or "
                    f"load with placement='worker' so the model runs in a managed subprocess.")
            self._impl = build(self._checkpoint, self._plan, device=self._device, nfe=self._nfe)
        return self._impl

    def predict(self, observation):
        return self._ensure().infer(dict(observation))

    def reset(self, **conditioning):
        impl = self._ensure()
        reset = getattr(impl, "reset", None)
        if reset is not None:
            reset(**conditioning)
        else:
            impl.infer(dict(reset=True, **conditioning))

    def close(self):
        self._impl = None


class WorkerBackend:
    """Host the model in a managed subprocess and talk to it over the existing transport.

    Chosen when the caller's interpreter cannot import the model stack. The subprocess is started,
    waited for, and torn down by this object; the caller never sees a port, a URL or a payload.

    THE TRANSPORT IS AN IMPLEMENTATION DETAIL AND IS TREATED AS ONE. It reuses the websocket server
    the project already ships and gates, rather than inventing a second serving path that would then
    need its own bit-exactness evidence.
    """

    def __init__(self, adapter, checkpoint, plan, *, device: str | None = None,
                 nfe: Mapping[str, int] | None = None, port: int | None = None,
                 python: str | None = None, startup_timeout_s: float = 900.0):
        self._adapter, self._checkpoint, self._plan = adapter, checkpoint, plan
        self._device, self._nfe = device, dict(nfe or {})
        self._port = port or _free_port()
        self._python = python or _serving_interpreter()
        self._timeout = startup_timeout_s
        self._proc: subprocess.Popen | None = None
        self._client = None

    # -- lifecycle -------------------------------------------------------------------------------
    def _spawn(self):
        launch = getattr(self._adapter, "worker_command", None)
        if launch is None:
            raise NotImplementedError(
                f"adapter for {self._checkpoint.execution.backbone!r} does not implement "
                f"worker_command(checkpoint, plan, port=, python=, device=, nfe=), so InstinctWM "
                f"cannot start a worker for it. Implement it, or run in an interpreter that can "
                f"import the model stack so placement='in_process' is available.")
        cmd, env = launch(self._checkpoint, self._plan, port=self._port, python=self._python,
                          device=self._device, nfe=self._nfe)
        self._proc = subprocess.Popen(cmd, env={**os.environ, **(env or {})},
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        atexit.register(self.close)
        self._wait_for_port()

    def _wait_for_port(self):
        deadline = time.time() + self._timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                out = (self._proc.stdout.read() if self._proc.stdout else "") or ""
                raise RuntimeError(
                    f"the model worker exited with code {self._proc.returncode} before it began "
                    f"serving.\n--- worker output (last 4000 chars) ---\n{out[-4000:]}")
            with socket.socket() as s:
                s.settimeout(0.5)
                if s.connect_ex(("127.0.0.1", self._port)) == 0:
                    return
            time.sleep(0.5)
        self.close()
        raise TimeoutError(
            f"the model worker did not begin serving within {self._timeout:.0f}s. Large checkpoints "
            f"take minutes to load; raise startup_timeout_s if this is a cold 10 GB load.")

    def _ensure(self):
        if self._client is None:
            if self._proc is None:
                self._spawn()
            self._client = _connect_client(self._port)
        return self._client

    # -- the same three methods ------------------------------------------------------------------
    def predict(self, observation):
        return self._ensure().infer(dict(observation))

    def reset(self, **conditioning):
        self._ensure().infer(dict(reset=True, **conditioning))

    def close(self):
        self._client = None
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()


# -- helpers -------------------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _serving_interpreter() -> str:
    """The interpreter that can host the model.

    `IWM_SERVER_PY` is the project's existing name for it and wins. Otherwise fall back to this
    interpreter, which is right whenever the caller could have hosted the model anyway.
    """
    p = os.environ.get("IWM_SERVER_PY")
    if p and shutil.which(p) or (p and Path(p).exists()):
        return p
    return sys.executable


def _connect_client(port: int):
    """The websocket client the project already ships. Imported lazily and only by the worker path."""
    root = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
    for cand in (f"{root}/wan_va/utils/Simple_Remote_Infer/deploy", f"{root}/evaluation/robotwin"):
        if Path(cand, "websocket_client_policy.py").exists():
            if cand not in sys.path:
                sys.path.insert(0, cand)
            from websocket_client_policy import WebsocketClientPolicy  # noqa: PLC0415
            return WebsocketClientPolicy(host="127.0.0.1", port=port)
    raise RuntimeError(
        f"no websocket client found under {root}. Set LINGBOT_ROOT, or load with "
        f"placement='in_process' in an interpreter that can host the model.")


def choose_backend(placement: str, adapter, checkpoint, plan, **kw) -> tuple[ExecutionBackend, str]:
    """Pick a placement. Returns (backend, one-line reason) so `explain()` can report it."""
    ok, why = model_stack_importable()
    if placement == "auto":
        placement = "in_process" if ok else "worker"
        why = f"auto -> {placement}: {why}"
    elif placement == "in_process" and not ok:
        raise RuntimeError(
            f"placement='in_process' was requested but {why}. Use placement='auto' to run the model "
            f"in a managed worker instead, or install the serving environment here.")
    else:
        why = f"placement={placement!r} (explicit)"
    backend = (InProcessBackend if placement == "in_process" else WorkerBackend)(
        adapter, checkpoint, plan, **kw)
    return backend, why

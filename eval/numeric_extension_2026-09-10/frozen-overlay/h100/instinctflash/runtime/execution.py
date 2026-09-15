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
from collections import deque
import threading
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


def imports_available(modules) -> tuple[bool, str]:
    """Do these modules load in THIS interpreter? Returns (yes, reason).

    Checked by import rather than by version pin: a pin would go stale, and the only question that
    matters is whether the modules load here and now. The MODULE LIST comes from the adapter, because
    only the adapter knows what its model needs -- see `can_host_in_process`.
    """
    missing = []
    for m in modules:
        try:
            __import__(m)
        except Exception:                                     # noqa: BLE001 - any failure disqualifies
            missing.append(m)
    if missing:
        return False, f"this interpreter cannot import {', '.join(missing)}"
    return True, "the model stack imports here"


def _seed_rngs(seed: int) -> None:
    """Seed every RNG a hosted model might draw noise from. Called per EPISODE.

    Per-episode is the floor that makes two seeded runtimes comparable given the same call
    sequence; an adapter that accepts `seed=` in `build_in_process` threads it deeper and seeds
    per request (wan_va re-seeds every `_infer` draw as seed+frame_st_id, preserving the stock
    within-episode noise distribution). Both run when both exist — re-seeding twice cannot
    disagree with itself.
    """
    import random

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _accepts_seed(fn) -> bool:
    import inspect
    try:
        return "seed" in inspect.signature(fn).parameters
    except (TypeError, ValueError):                              # builtins/partials without signatures
        return False


def can_host_in_process(adapter) -> tuple[bool, str]:
    """Ask the ADAPTER whether this interpreter can host its model.

    This used to be a hardcoded list -- torch, diffusers, transformers, safetensors -- which is
    LingBot-VA's dependency set, not anyone else's. A pi05 VLA needs lerobot and does not need
    diffusers, so in an interpreter with lerobot but no diffusers the runtime concluded "cannot host"
    and fell through to a worker the pi05 adapter has no reason to implement. The error blamed the
    adapter for a question the runtime had answered on the wrong model's behalf.

    An adapter may implement `can_host_in_process() -> (bool, str)`. If it does not, an adapter that
    implements `build_in_process` is taken at its word: it will raise its own specific reason if its
    stack is missing, which is a better diagnostic than a guess made from another model's imports.
    """
    fn = getattr(adapter, "can_host_in_process", None)
    if fn is not None:
        try:
            return fn()
        except Exception as e:                                    # noqa: BLE001
            return False, f"the adapter's own host check failed: {type(e).__name__}: {e}"
    if getattr(adapter, "build_in_process", None) is not None:
        return True, "the adapter implements build_in_process"
    return False, "the adapter implements neither can_host_in_process nor build_in_process"


class InProcessBackend:
    """Host the model in the caller's process.

    Chosen when the model stack imports. The adapter builds and drives the server object directly;
    there is no socket, no subprocess and no serialisation.
    """

    def __init__(self, adapter, checkpoint, plan, *, device: str | None = None,
                 nfe: Mapping[str, int] | None = None, seed: int | None = None):
        self._adapter, self._checkpoint, self._plan = adapter, checkpoint, plan
        self._device, self._nfe, self._seed = device, dict(nfe or {}), seed
        self._impl = None

    def _ensure(self):
        if self._impl is None:
            build = getattr(self._adapter, "build_in_process", None)
            if build is None:
                raise NotImplementedError(
                    f"adapter for {self._checkpoint.execution.backbone!r} does not implement "
                    f"build_in_process(checkpoint, plan, device=, nfe=). Either implement it, or "
                    f"load with placement='worker' so the model runs in a managed subprocess.")
            # The seed reaches the adapter only when its build declares seed= — an adapter that
            # threads it deeper (per request) opts in; everyone else gets the per-episode floor
            # from reset() below, so an older adapter signature is never called with a keyword
            # it does not know.
            kw = {"seed": self._seed} if self._seed is not None and _accepts_seed(build) else {}
            self._impl = build(self._checkpoint, self._plan, device=self._device, nfe=self._nfe,
                               **kw)
            self._report_unapplied()
        return self._impl

    def _report_unapplied(self) -> None:
        """Say so when a plan APPLIES passes the adapter cannot install.

        A plan is a claim about what will happen to this model. An adapter without `install` cannot
        make any of it happen, so a plan reporting APPLY while nothing is installed is the plan lying
        -- and it lied quietly: pi05's plan says `conditioning_prefill` APPLIES, its adapter has no
        install method, and the optimization simply did not occur. Reporting beats asserting, because
        an adapter that deliberately builds an already-optimized object is legitimate; what is not
        legitimate is the caller being unable to tell which case they are in.
        """
        if getattr(self._adapter, "install", None) is not None:
            return
        applied = [r.name for r in getattr(self._plan, "results", []) if getattr(r, "applies", False)]
        if applied:
            print(f"InstinctFlash: the plan applies {applied}, but the adapter for "
                  f"{self._checkpoint.execution.backbone!r} implements no install(), so NONE of them "
                  f"were installed. The model runs unoptimized. Implement install(server_module, plan) "
                  f"to act on a plan, or treat the plan as advisory for this backbone.")

    def predict(self, observation, *, executed_action=None):
        """One control cycle. Every internal phase the model needs happens inside this call.

        THE IMPL CONTRACT, which used to be `infer(dict)` and nothing else. `infer` is LingBot-VA's
        *server* verb, and hardcoding it meant an external adapter that implemented the obvious
        `predict` got `AttributeError: '_Impl' object has no attribute 'infer'` from inside the
        runtime -- with nothing in the documented adapter protocol to predict that. `predict` is now
        preferred and `infer` is the compatibility path, so a new model family implements the verb
        the rest of the API already uses.

        `commit` is optional and private to the model. A model whose control cycle updates state
        after producing an action (LingBot-VA advances a KV ring; an autoregressive model may append
        tokens) implements it and the runtime drives it here. That is what makes `predict` loopable
        WITHOUT the caller learning that phases exist.
        An optional `validate_executed_action(action)` hook rejects unsupported
        feedback before prediction changes the model's state.
        """
        impl = self._ensure()
        validate_feedback = getattr(impl, "validate_executed_action", None)
        if validate_feedback is not None:
            # Reject unsupported feedback before prediction mutates model history.
            validate_feedback(executed_action)
        fn = getattr(impl, "predict", None) or getattr(impl, "infer", None)
        if fn is None:
            raise NotImplementedError(
                f"the object returned by build_in_process for "
                f"{self._checkpoint.execution.backbone!r} implements neither predict(observation) "
                f"nor infer(dict). One of them is how a control cycle runs.")
        out = fn(dict(observation))

        commit = getattr(impl, "commit", None)
        if commit is not None:
            action = out.get("action") if isinstance(out, dict) else out
            # `executed_action` is what the robot ACTUALLY did -- a safety filter or a low-level
            # controller may not have executed what we predicted. Defaulting to the prediction is
            # the honest fallback, not a claim that they are always equal.
            commit(dict(observation), executed_action if executed_action is not None else action)
        return out

    def reset(self, **conditioning):
        impl = self._ensure()
        if self._seed is not None:
            _seed_rngs(self._seed)                               # per-episode determinism floor
        reset = getattr(impl, "reset", None)
        if reset is not None:
            reset(**conditioning)
        else:
            impl.infer(dict(reset=True, **conditioning))

    def close(self):
        impl, self._impl = self._impl, None
        close = getattr(impl, "close", None) if impl is not None else None
        if close is not None:
            close()


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
                 python: str | None = None, startup_timeout_s: float = 900.0,
                 seed: int | None = None):
        self._adapter, self._checkpoint, self._plan = adapter, checkpoint, plan
        self._device, self._nfe, self._seed = device, dict(nfe or {}), seed
        self._port = port or _free_port()
        self._python = python or _serving_interpreter()
        self._timeout = startup_timeout_s
        self._proc: subprocess.Popen | None = None
        self._client = None
        self._transport_client = None
        self._log_chunks = deque(maxlen=16)
        self._log_thread = None

    # -- lifecycle -------------------------------------------------------------------------------
    def _spawn(self):
        launch = getattr(self._adapter, "worker_command", None)
        if launch is None:
            raise NotImplementedError(
                f"adapter for {self._checkpoint.execution.backbone!r} does not implement "
                f"worker_command(checkpoint, plan, port=, python=, device=, nfe=), so InstinctFlash "
                f"cannot start a worker for it. Implement it, or run in an interpreter that can "
                f"import the model stack so placement='in_process' is available.")
        kw = {}
        if self._seed is not None:
            # A worker draws its noise in another process, so the per-episode floor in this one
            # cannot reach it: either the adapter carries the seed onto the worker command line,
            # or the caller is refused. Spawning an UNSEEDED worker under a caller who asked for
            # determinism is the lie a seed flag exists to prevent.
            if not _accepts_seed(launch):
                raise RuntimeError(
                    f"seed={self._seed} was requested, but the adapter for "
                    f"{self._checkpoint.execution.backbone!r} does not accept seed= in "
                    f"worker_command, so its worker would serve unseeded noise while the caller "
                    f"believes otherwise. Add seed= there, or serve with placement='in_process'.")
            kw["seed"] = self._seed
        cmd, env = launch(self._checkpoint, self._plan, port=self._port, python=self._python,
                          device=self._device, nfe=self._nfe, **kw)
        self._proc = subprocess.Popen(cmd, env={**os.environ, **(env or {})},
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # Model loaders and progress bars can fill a PIPE before the listener starts,
        # or midway through an episode. Drain continuously and retain bounded diagnostics.
        pipe = self._proc.stdout
        def drain():
            try:
                for chunk in iter(lambda: pipe.read(4096), ''):
                    self._log_chunks.append(chunk)
            finally:
                pipe.close()
        self._log_thread = threading.Thread(target=drain, daemon=True)
        self._log_thread.start()
        atexit.register(self.close)
        self._wait_for_port()

    def _wait_for_port(self):
        deadline = time.time() + self._timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                if self._log_thread is not None:
                    self._log_thread.join(timeout=1)
                out = ''.join(self._log_chunks)
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
            client = _connect_client(self._port)
            wrap = getattr(self._adapter, "wrap_worker_client", None)
            try:
                self._client = wrap(client, self._checkpoint) if wrap is not None else client
            except Exception:
                close = getattr(client, "close", None)
                if close is not None:
                    close()
                raise
            self._transport_client = client
        return self._client

    # -- the same three methods ------------------------------------------------------------------
    def predict(self, observation, *, executed_action=None):
        client = self._ensure()
        predict = getattr(client, "predict", None) or client.infer
        out = predict(dict(observation))
        commit = getattr(client, "commit", None)
        if commit is not None:
            action = out.get("action") if isinstance(out, dict) else out
            commit(dict(observation), executed_action if executed_action is not None else action)
        return out

    def reset(self, **conditioning):
        client = self._ensure()
        reset = getattr(client, "reset", None)
        if reset is not None:
            reset(**conditioning)
        else:
            client.infer(dict(reset=True, **conditioning))

    def close(self):
        self._client = None
        transport, self._transport_client = self._transport_client, None
        if transport is not None:
            close = getattr(transport, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass  # A failed socket close must not prevent terminating our child.
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
        if self._log_thread is not None:
            self._log_thread.join(timeout=1)


# -- helpers -------------------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _serving_interpreter() -> str:
    """The interpreter that can host the model.

    `IFL_SERVER_PY` is the project's existing name for it and wins. Otherwise fall back to this
    interpreter, which is right whenever the caller could have hosted the model anyway.
    """
    p = os.environ.get("IFL_SERVER_PY")
    if p and shutil.which(p) or (p and Path(p).exists()):
        return p
    return sys.executable


def _connect_client(port: int):
    """Use the shipped wire codec; transport must not depend on a model checkout's imports."""
    from websockets.sync.client import connect
    from instinctflash.serving.msgpack_numpy import Packer, unpackb

    class Client:
        def __init__(self):
            self.packer = Packer()
            self.connection = connect(f"ws://127.0.0.1:{port}", compression=None,
                                      max_size=None, ping_interval=None, close_timeout=5)
            try:
                self.metadata = unpackb(self.connection.recv())
            except Exception:
                self.connection.close()
                raise

        def infer(self, observation):
            self.connection.send(self.packer.pack(observation))
            message = self.connection.recv()
            if isinstance(message, str):
                raise RuntimeError(f"model worker refused prediction: {message[:4000]}")
            return unpackb(message)

        def close(self):
            self.connection.close()

    return Client()


def choose_backend(placement: str, adapter, checkpoint, plan, **kw) -> tuple[ExecutionBackend, str]:
    """Pick a placement. Returns (backend, one-line reason) so `explain()` can report it."""
    from instinctflash.runtime.precision import constrain_precision, resolve_precision, require_fp8_plan

    precision = kw.pop("precision", "native")
    resolve_precision(precision, None, placement)
    constrain_precision(plan, precision)
    if precision == "fp8":
        from instinctflash.runtime.engine_backend import (
            EngineBackend, engine_available,
        )
        eng_result = require_fp8_plan(plan, checkpoint.execution.backbone)
        eng_ok, eng_why = (engine_available(checkpoint.execution.backbone)
                           if checkpoint.execution.backbone in ("cosmos3_policy", "dreamzero")
                           else engine_available())
        if not eng_ok:
            raise RuntimeError(f"precision='fp8' unavailable: {eng_why}")
        if kw.get("seed") is not None:
            raise RuntimeError(
                "seed= is not supported by the FP8 engine. Use precision='native' for seeded execution.")
        engine_kw = dict(kw)
        engine_kw.pop("startup_timeout_s", None)
        engine_kw.pop("seed", None)
        backend = EngineBackend(adapter, checkpoint, plan, **engine_kw)
        _mark_plan_engine_executed(plan, eng_result)
        return backend, ("engine: " + eng_why + " -> explicit FP8 execution; chain tier "
                         "NUMERIC (live execution uncertified)")
    ok, why = can_host_in_process(adapter)
    if placement == "auto":
        placement = "in_process" if ok else "worker"
        why = f"auto -> {placement}: {why}"
    elif placement == "in_process" and not ok:
        raise RuntimeError(
            f"placement='in_process' was requested but {why}. Use placement='auto' to run the model "
            f"in a managed worker instead, or install the serving environment here.")
    else:
        why = f"placement={placement!r} (explicit)"
    backend_cls = InProcessBackend if placement == "in_process" else WorkerBackend
    backend_kw = dict(kw)
    if backend_cls is InProcessBackend:
        backend_kw.pop("startup_timeout_s", None)
    backend = backend_cls(adapter, checkpoint, plan, **backend_kw)
    return backend, why


def _mark_plan_engine_executed(plan, eng_result) -> None:
    """Report the selected FP8 engine and demote torch passes it does not execute."""
    results = getattr(plan, "results", None)
    if not isinstance(results, list):
        return
    from instinctflash.planners.planner import PassResult

    # H100 wraps the native Torch adapter; its installed passes remain in effect.
    # Only the separate fused engine replaces that graph.
    retains_torch_passes = eng_result.params.get("executor") == "h100_torch_fp8"
    for i, r in enumerate(results):
        if r.name == "engine_offload":
            if getattr(r, "excluded", False):
                # A caller exclusion is never rewritten to APPLY. choose_backend refuses to build
                # the engine for an excluded plan, so reaching this line means a bug upstream --
                # leaving the entry untouched keeps explain() truthful either way.
                continue
            # Strip the optimizer's ceiling-demotion prefix (planner.compile writes
            # "legal but tier ... exceeds ceiling ...: <original reason>") so the restored
            # entry reads as the pass's own verdict, not as a contradiction.
            reason = r.reason
            if reason.startswith("legal but tier ") and ": " in reason:
                reason = reason.split(": ", 1)[1]
            results[i] = PassResult(
                name=r.name, applies=True, tier=r.tier,
                reason=f"engine placement selected: {reason}",
                params=r.params, expected_win=r.expected_win)
        elif r.applies and not retains_torch_passes:
            results[i] = PassResult(
                name=r.name, applies=False, tier=r.tier,
                reason=(f"engine placement: execution goes through the fused engine pipeline, "
                        f"which is not a torch module graph, so there is nothing for this pass "
                        f"to install into. It was legal here: {r.reason}"),
                params=r.params, expected_win=r.expected_win)

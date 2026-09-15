"""Bind shared prediction reuse to the audited native DreamZero denoising loop.

The native loop still owns both solvers, CFG, random draws and observation KV.
Only its decision hook and the ownership of retained predictions are replaced.
The source gate deliberately rejects a changed loop until it is re-audited.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import textwrap
import threading
from types import MethodType

from instinctflash.runtime.step_cache import StepCacheConfig, StepCacheController


_METHOD_HASHES = {
    "_run_diffusion_steps": "739a3017e0c07371223ce0c831dfc8a5a00ede1aaffca5de6abe102ab28bbf1c",
    "should_run_model": "7a7f6026dba7a5f1320586f00d4220e65e972cace7d5d270bcb322a99a223b60",
    "lazy_joint_video_action": "e89f8491903e0214eb5606453d68c555cbb947a2dfc1700781b7d740044aae62",
}
_MISSING = object()


def _verify_native_methods(head):
    if int(head.num_inference_steps) != 16 or int(getattr(head, "ip_size", 1)) != 1:
        raise ValueError("DreamZero dynamic cache requires the audited 16-slot, single-GPU loop")
    if not head.dynamic_cache_schedule:
        raise ValueError("DreamZero dynamic cache must be explicitly selected before installation")
    hashes = {}
    for name, expected in _METHOD_HASHES.items():
        method = getattr(head, name)
        if not inspect.ismethod(method) or method.__self__ is not head:
            raise ValueError(f"DreamZero dynamic cache requires the native bound method {name}")
        try:
            source = textwrap.dedent(inspect.getsource(method)).strip()
        except (OSError, TypeError, SyntaxError) as error:
            raise ValueError(f"Cannot verify DreamZero native method {name}") from error
        # Source text stays stable across Python AST schema changes (3.12 added
        # type_params); Thor and CPU CI use different Python versions.
        actual = hashlib.sha256(source.encode()).hexdigest()
        if actual != expected:
            raise ValueError(f"DreamZero {name} changed; re-audit dynamic-cache integration")
        hashes[name] = actual
    return hashes


class DreamZeroDynamicCache:
    """Owned per-head hooks, with one independent cache lifetime per generation."""

    def __init__(self, head):
        self.source_methods = _verify_native_methods(head)
        self.controller = StepCacheController(StepCacheConfig())
        self._head = head
        self._lock = threading.Lock()
        self._active = False
        self._pending = False
        self._history = None
        self._last = None
        self._counts = None
        self._originals = {name: getattr(head, name) for name in _METHOD_HASHES}
        self._instance_values = {name: vars(head).get(name, _MISSING) for name in _METHOD_HASHES}

        def generate(bound_head, *args, **kwargs):
            return self._generate(*args, **kwargs)

        def decide(bound_head, index, current_timestep, prev_predictions):
            return self._decide(index, current_timestep, prev_predictions)

        def forward(bound_head, *args, **kwargs):
            output = self._originals["_run_diffusion_steps"](*args, **kwargs)
            if self._active:
                if not isinstance(output, list) or len(output) != 2:
                    raise RuntimeError("DreamZero native CFG must return two prediction branches")
                commit = (kwargs.get("kv_cache_metadata") or {}).get("update_kv_cache")
                if commit is True:
                    self._counts["kv_update_calls"] += 1
                    self._counts["kv_update_branch_forwards"] += len(output)
                elif commit is False:
                    self._counts["denoiser_calls"] += 1
                    self._counts["denoiser_branch_forwards"] += len(output)
                else:
                    raise RuntimeError("DreamZero forward has no explicit KV mutation declaration")
            return output

        self._installed = {
            "lazy_joint_video_action": MethodType(generate, head),
            "should_run_model": MethodType(decide, head),
            "_run_diffusion_steps": MethodType(forward, head),
        }
        for name, method in self._installed.items():
            object.__setattr__(head, name, method)

    def _record_pending(self):
        if not self._pending:
            return
        if not self._history or len(self._history[-1]) != 3:
            raise RuntimeError("DreamZero did not record its computed prediction")
        _, video, action = self._history[-1]
        self.controller.record_prediction({"video": video, "action": action}, signal_key="video")
        self._pending = False

    def _decide(self, index, current_timestep, history):
        if not self._active:
            raise RuntimeError("DreamZero step-cache decision outside its generation")
        if index == 0:
            if history or self._history is not None:
                raise RuntimeError("DreamZero reused prediction history across generations")
            self._history = history
        elif history is not self._history:
            raise RuntimeError("DreamZero replaced prediction history during generation")
        self._record_pending()
        compute = self.controller.decide(index)
        if compute:
            self._pending = True
        else:
            prediction = self.controller.reuse()
            # Native skip consumes the most recent tuple. Give its solver owned
            # copies so replay or in-place consumers cannot corrupt our history.
            timestamp = history[-1][0]
            history[-1] = (timestamp, prediction["video"], prediction["action"])
        return compute

    def _generate(self, *args, **kwargs):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A DreamZero head cannot run concurrent generations")
        try:
            if self._head is None:
                raise RuntimeError("DreamZero dynamic cache is closed")
            if not self._head.dynamic_cache_schedule or self._head.num_inference_steps != 16:
                raise RuntimeError("DreamZero cache configuration changed after installation")
            self.controller.begin_generation(16)
            self._active = True
            self._pending = False
            self._history = None
            self._counts = dict(denoiser_calls=0, denoiser_branch_forwards=0,
                                kv_update_calls=0, kv_update_branch_forwards=0)
            try:
                result = self._originals["lazy_joint_video_action"](*args, **kwargs)
                self._record_pending()
                current = self.controller.report()["current_generation"]
                if current["computed_steps"] != self._counts["denoiser_calls"]:
                    raise RuntimeError("DreamZero cache decisions disagree with native denoiser calls")
                generation = self.controller.end_generation()
                self._last = dict(generation, **self._counts)
                return result
            except BaseException:
                self.controller.abort_generation()
                self._last = dict(status="aborted", **self._counts)
                raise
            finally:
                self._active = self._pending = False
                self._history = None
        finally:
            self._lock.release()

    def reset(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot reset DreamZero cache during a generation")
        try:
            self.controller.abort_generation()
            self._history = None
            self._pending = False
        finally:
            self._lock.release()

    def report(self):
        return {"installed": self._head is not None, "profile": "dreamzero_velocity_v1",
                "decision_signal": "cfg_video", "reused_outputs": ["cfg_video", "conditional_action"],
                "source_methods": dict(self.source_methods),
                "controller": self.controller.report(), "last_generation": copy.deepcopy(self._last),
                "quality_certificate": None}

    @property
    def closed(self):
        return self._head is None

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close DreamZero cache during a generation")
        try:
            if self._head is None:
                return
            conflicts = []
            for name, method in self._installed.items():
                if vars(self._head).get(name) is not method:
                    conflicts.append(name)
                    continue
                value = self._instance_values[name]
                if value is _MISSING:
                    object.__delattr__(self._head, name)
                else:
                    object.__setattr__(self._head, name, value)
            self.controller.close()
            self._head = None
            self._history = None
            self._originals.clear()
            self._installed.clear()
            self._instance_values.clear()
            if conflicts:
                raise RuntimeError("DreamZero methods were replaced after cache installation: "
                                   + ", ".join(conflicts) + "; owned cache released")
        finally:
            self._lock.release()


def install(head):
    """Install once on an explicitly selected, source-verified native head."""
    return DreamZeroDynamicCache(head)

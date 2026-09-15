"""Generation-scoped approximate denoiser reuse; Torch is imported only on use.

This controller preserves the DreamZero video-cosine decision rule on valid
inputs. Reusing a prediction changes computation: neither native arithmetic nor
successful controller checks establish action quality or a speed benefit.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral, Real
from threading import RLock


MAX_GENERATION_STEPS = 4096
MAX_SKIP_COUNT = 1024


@dataclass(frozen=True)
class StepCacheConfig:
    profile: str = "dreamzero_velocity_v1"
    thresholds: tuple[float, ...] = (0.95, 0.93)
    skip_counts: tuple[int, ...] = (4, 2)
    max_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        if self.profile != "dreamzero_velocity_v1":
            raise ValueError(f"Unsupported step-cache profile: {self.profile!r}")
        try:
            thresholds, skips = tuple(self.thresholds), tuple(self.skip_counts)
        except TypeError as error:
            raise ValueError("Thresholds and skip counts must be sequences") from error
        if not thresholds or len(thresholds) != len(skips):
            raise ValueError("Thresholds and skip counts must have equal nonzero length")
        if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
               or not -1.0 <= x <= 1.0 for x in thresholds):
            raise ValueError("Cosine thresholds must be finite real numbers in [-1, 1]")
        thresholds = tuple(float(x) for x in thresholds)
        if any(a <= b for a, b in zip(thresholds, thresholds[1:])):
            raise ValueError("Cosine thresholds must be strictly descending")
        if any(isinstance(x, bool) or not isinstance(x, Integral)
               or not 1 <= x <= MAX_SKIP_COUNT for x in skips):
            raise ValueError(f"Skip counts must be integers in [1, {MAX_SKIP_COUNT}]")
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, Integral) or self.max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "skip_counts", tuple(int(x) for x in skips))
        object.__setattr__(self, "max_bytes", int(self.max_bytes))

    def to_dict(self):
        return {"profile": self.profile, "thresholds": list(self.thresholds),
                "skip_counts": list(self.skip_counts), "max_bytes": self.max_bytes}


class StepCacheController:
    """Own at most two predictions, isolated to one sequential generation.

    A compute decision must be followed by ``record_prediction``; a reuse
    decision must be followed by ``reuse``. Every scheduler slot still belongs
    to the adapter. CFG, scheduler updates, RNG and KV commits are not managed
    here. CUDA use explicitly requires one device and stream per generation;
    retaining or consuming predictions under another stream is refused.

    ``max_bytes`` bounds retained prediction storage. Returned copies belong to
    the caller and are not counted in that bound. Metadata retains only the
    current and most recent generation, each limited to 4096 slots.
    """

    def __init__(self, config: StepCacheConfig = StepCacheConfig()):
        if not isinstance(config, StepCacheConfig):
            raise TypeError("config must be a StepCacheConfig")
        self._config = config
        self._lock = RLock()
        self._closed = False
        self._generation_id = 0
        self._current = None
        self._last = None
        self._aggregate = dict(generations_started=0, generations_completed=0,
                               generations_aborted=0, computed_steps=0, reused_steps=0)
        self._history = deque()
        self._signature = None
        self._signal_key = None
        self._device = None
        self._stream = None
        self._pending = None
        self._next_index = 0
        self._countdown = 0
        self._bytes = 0

    @property
    def config(self) -> StepCacheConfig:
        return self._config

    def _open(self):
        if self._closed:
            raise RuntimeError("Step-cache controller is closed")

    def _active(self):
        self._open()
        if self._current is None:
            raise RuntimeError("No active step-cache generation")

    def _context(self, device=None):
        import torch

        if torch.is_grad_enabled():
            raise RuntimeError("Step caching requires inference/no_grad")
        if device is not None and device.type == "cuda":
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Step caching must run outside CUDA graph capture")
                stream = (device.index, torch.cuda.current_stream(device).cuda_stream)
            if self._stream is not None and stream != self._stream:
                raise RuntimeError("Step caching requires the same CUDA stream for the generation")
            return stream
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Step caching must run outside CUDA graph capture")
        return None

    def begin_generation(self, total_steps: int):
        with self._lock:
            self._open()
            if self._current is not None:
                raise RuntimeError("End or abort the active generation before starting another")
            if isinstance(total_steps, bool) or not isinstance(total_steps, Integral) or not 1 <= total_steps <= MAX_GENERATION_STEPS:
                raise ValueError(f"total_steps must be an integer in [1, {MAX_GENERATION_STEPS}]")
            self._context()
            self._generation_id += 1
            self._aggregate["generations_started"] += 1
            self._clear_predictions()
            self._current = dict(generation_id=self._generation_id, status="active",
                                 total_steps=int(total_steps), computed_steps=0, reused_steps=0,
                                 peak_cache_bytes=0, compute_mask=[], trace=[])

    def decide(self, index: int) -> bool:
        with self._lock:
            self._active()
            self._context(self._device)
            if self._pending is not None:
                raise RuntimeError(f"The preceding {self._pending} decision must be consumed first")
            if (isinstance(index, bool) or not isinstance(index, Integral)
                    or index != self._next_index or index >= self._current["total_steps"]):
                raise ValueError(f"Expected sequential slot {self._next_index} within the generation")
            before, countdown, similarity = self._countdown, self._countdown, None
            if len(self._history) < 2:
                compute, reason = True, "insufficient_history"
            elif countdown > 1:
                compute, reason, countdown = False, "countdown_reuse", countdown - 1
            elif countdown == 1:
                compute, reason, countdown = True, "countdown_refresh", 0
            else:
                import torch

                last = self._history[-1][self._signal_key].flatten(1).float()
                previous = self._history[-2][self._signal_key].flatten(1).float()
                similarity_tensor = torch.nn.functional.cosine_similarity(last, previous, dim=1).mean()
                similarity = float(similarity_tensor.item())
                if not math.isfinite(similarity):
                    raise ValueError("Nonfinite FP32 cosine similarity; abort the generation")
                compute, reason = True, "similarity_refresh"
                for threshold, skip in zip(self.config.thresholds, self.config.skip_counts):
                    # Preserve upstream scalar-tensor threshold rounding. A
                    # Python float comparison differs at FP32 .93/.95 equality.
                    if similarity_tensor > threshold:
                        compute, reason, countdown = False, "similarity_reuse", skip
                        break
            self._countdown = countdown
            self._pending = "compute" if compute else "reuse"
            self._next_index += 1
            self._current["compute_mask"].append(compute)
            self._current["trace"].append(dict(index=int(index), compute=compute, reason=reason,
                                                 countdown_before=before, countdown_after=countdown,
                                                 similarity=similarity, consumed=False))
            return compute

    def record_prediction(self, outputs: Mapping, *, signal_key: str = "video"):
        with self._lock:
            self._active()
            if self._pending != "compute":
                raise RuntimeError("record_prediction requires a pending compute decision")
            self._context(self._device)
            import torch

            if not isinstance(outputs, Mapping) or not outputs:
                raise ValueError("Prediction outputs must be a nonempty tensor mapping")
            if any(not isinstance(key, str) or not key for key in outputs):
                raise ValueError("Prediction keys must be nonempty strings")
            if not isinstance(signal_key, str) or signal_key not in outputs:
                raise ValueError("signal_key must name a prediction tensor")
            if self._signal_key is not None and signal_key != self._signal_key:
                raise ValueError("The decision signal cannot change within a generation")
            device, signature, size = None, {}, 0
            for key, value in outputs.items():
                if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                        or not value.is_floating_point() or value.numel() == 0
                        or value.device.type not in ("cpu", "cuda")):
                    raise ValueError("Predictions must be nonempty real floating CPU/CUDA strided tensors")
                if value.requires_grad:
                    raise ValueError("Cannot retain gradient-bearing predictions")
                device = value.device if device is None else device
                if value.device != device:
                    raise ValueError("Prediction outputs must share one device")
                signature[key] = (tuple(value.shape), value.dtype, value.device)
                size += value.numel() * value.element_size()
            signal = outputs[signal_key]
            if signal.ndim < 2 or signal.shape[0] == 0:
                raise ValueError("The decision signal requires a batch dimension and at least two dimensions")
            if self._signature is not None and signature != self._signature:
                raise ValueError("Prediction keys, shape, dtype and device must remain stable")
            stream = self._context(device)
            if min(self._current["total_steps"], 2) * size > self.config.max_bytes:
                raise ValueError("Prediction history exceeds the step-cache max_bytes budget")
            if any(not bool(torch.isfinite(value).all().item()) for value in outputs.values()):
                raise ValueError("Nonfinite prediction; abort the generation")
            # Retire the oldest entry before cloning; never own three histories.
            if len(self._history) == 2:
                old = self._history.popleft()
                self._bytes -= sum(t.numel() * t.element_size() for t in old.values())
                del old
            owned = {key: value.detach().clone() for key, value in outputs.items()}
            self._history.append(owned)
            self._bytes += size
            self._signature, self._signal_key = signature, signal_key
            self._device, self._stream = device, stream
            self._current["peak_cache_bytes"] = max(self._current["peak_cache_bytes"], self._bytes)
            self._consume("computed_steps")

    def reuse(self) -> dict:
        with self._lock:
            self._active()
            if self._pending != "reuse":
                raise RuntimeError("reuse requires a pending reuse decision")
            self._context(self._device)
            outputs = {key: value.clone() for key, value in self._history[-1].items()}
            self._consume("reused_steps")
            return outputs

    def _consume(self, field):
        self._current[field] += 1
        self._aggregate[field] += 1
        self._current["trace"][-1]["consumed"] = True
        self._pending = None

    def end_generation(self) -> dict:
        with self._lock:
            self._active()
            self._context(self._device)
            if self._pending is not None:
                raise RuntimeError(f"The final {self._pending} decision has not been consumed")
            if self._next_index != self._current["total_steps"]:
                raise RuntimeError("Not all scheduler slots have been consumed")
            return self._finish("completed")

    def _finish(self, status):
        self._current["status"] = status
        self._current["pending_decision"] = self._pending
        self._last = deepcopy(self._current)
        self._aggregate["generations_" + status] += 1
        self._current = None
        self._clear_predictions()
        return deepcopy(self._last)

    def abort_generation(self):
        # Cleanup must remain possible after an exception, stream change or grad-mode exit.
        with self._lock:
            if self._current is not None:
                self._finish("aborted")

    def _clear_predictions(self):
        self._history.clear()
        self._signature = self._signal_key = self._device = self._stream = self._pending = None
        self._next_index = self._countdown = self._bytes = 0

    def close(self):
        with self._lock:
            self.abort_generation()
            self._closed = True

    def report(self) -> dict:
        with self._lock:
            return {"kind": "approximate_denoiser_prediction_reuse", "config": self.config.to_dict(),
                    "transform_tier": "BEHAVIORAL", "quality_certified": False,
                    "active": self._current is not None, "closed": self._closed,
                    "generation_id": self._generation_id, "history_entries": len(self._history),
                    "cache_bytes": self._bytes, "pending_decision": self._pending,
                    "aggregate": dict(self._aggregate), "current_generation": deepcopy(self._current),
                    "last_generation": deepcopy(self._last)}

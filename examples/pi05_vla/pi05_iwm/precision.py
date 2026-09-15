"""Process-global PyTorch matmul precision leases for pi0.5 runtimes.

PyTorch's TF32 controls are process-global. Pretending otherwise lets a TF32 runtime silently change
an FP32 runtime sharing the interpreter. A lease makes that limitation explicit: same-mode runtimes
share by reference count, mixed modes are refused, and the last close restores the caller's state.
"""
from __future__ import annotations

import threading


_LOCK = threading.RLock()
_TOKENS: dict[int, str] = {}
_MODE: str | None = None
_SAVED: tuple[str, bool] | None = None
_NEXT_TOKEN = 0


class Pi05PrecisionLease:
    def __init__(self, torch_module, mode: str):
        if mode not in {"fp32", "tf32"}:
            raise ValueError(f"pi0.5 precision mode must be fp32 or tf32, got {mode!r}")
        self._torch = torch_module
        self.mode = mode
        self._token: int | None = None
        self._acquire()

    def _acquire(self) -> None:
        global _MODE, _SAVED, _NEXT_TOKEN
        with _LOCK:
            if _TOKENS and _MODE != self.mode:
                raise RuntimeError(
                    f"cannot host pi0.5 {_MODE.upper()} and {self.mode.upper()} operating points "
                    "in one Python process: torch matmul precision is process-global. Put one "
                    "runtime in a worker/dedicated process."
                )
            if not _TOKENS:
                _SAVED = (
                    self._torch.get_float32_matmul_precision(),
                    bool(self._torch.backends.cuda.matmul.allow_tf32),
                )
                _MODE = self.mode
                self._torch.set_float32_matmul_precision(
                    "high" if self.mode == "tf32" else "highest"
                )
                self._torch.backends.cuda.matmul.allow_tf32 = self.mode == "tf32"
            _NEXT_TOKEN += 1
            self._token = _NEXT_TOKEN
            _TOKENS[self._token] = self.mode

    def close(self) -> None:
        global _MODE, _SAVED
        with _LOCK:
            token, self._token = self._token, None
            if token is None or token not in _TOKENS:
                return
            del _TOKENS[token]
            if _TOKENS:
                return
            saved, _SAVED = _SAVED, None
            _MODE = None
            if saved is not None:
                precision, allow_tf32 = saved
                self._torch.set_float32_matmul_precision(precision)
                self._torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def _state_for_tests() -> tuple[str | None, int]:
    with _LOCK:
        return _MODE, len(_TOKENS)


__all__ = ["Pi05PrecisionLease"]

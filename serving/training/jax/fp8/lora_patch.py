"""Monkey-patch ``openpi.models.lora`` to route base GEMMs through FP8.

Two call sites in ``openpi-jax-mlir/src/openpi/models/lora.py`` carry
the LoRA-bearing matmuls of pi0.5:

* :class:`openpi.models.lora.Einsum` — attention QKV / O via
  ``jnp.einsum(eqn, x, self.w)`` (line 57).
* :class:`openpi.models.lora.FeedForward` — FFN gate / up / down via
  ``jnp.dot(x, w)`` inside ``_dot`` (line 145).

Both are patched so that, when ``self.lora_config`` is non-None
(i.e. only on layers that actually train) **and** an env flag is
set, the BASE matmul is rerouted through
``training.jax.fp8.fp8_jax.fp8_gemm_bf16_out``. The LoRA A/B path
itself stays in BF16 (rank-≤32 matmuls — FP8 wouldn't help and
quantizing them back-and-forth costs more than it saves).

Activation thresholding
-----------------------
The ``flashvla-ft`` prototype topped out at 1.20× wall-clock partly
because per-GEMM FFI calls broke XLA's 18-layer scan-fusion. Small-M
shapes (attention QKV at single-token decode, etc.) pay the FFI
overhead without earning the FP8 win. We add an ``M`` threshold
(default 64, tunable via ``FLASHVLA_JAX_FP8_MIN_M``): below it, the
patch falls back to the original ``jnp.einsum`` / ``jnp.dot``.

Activation scale
----------------
For Phase 2 we compute the activation scale dynamically per call
(``jnp.max(jnp.abs(x)) / 448``). This is what the prototype did and
what our parity test does. It matches the meaning of the PyTorch
path's "static after calibration" only on inputs whose distribution
is stable at training time; it is NOT CUDA-Graph compatible (XLA
graphs do not capture amax sync). That is acceptable for Phase 2 —
calibration-driven static scales land in Phase 3 alongside the
training driver. The patch is correct numerically either way.

Usage
-----
::

    from training.jax.fp8.lora_patch import enable, disable, get_stats
    enable()                                 # idempotent
    # ... build model, run training ...
    print(get_stats())                       # {'einsum_routed': N, ...}
    disable()                                # restore original methods

Environment flags
-----------------
* ``FLASHVLA_JAX_FP8`` (default ``"1"`` after ``enable()``) — call-site
  on/off without re-enabling. Set to ``"0"`` to bypass the FP8 path
  while keeping the patch installed (useful for A/B parity tests
  inside the same process).
* ``FLASHVLA_JAX_FP8_MIN_M`` (default ``64``) — minimum collapsed
  activation M (product of non-contracted axes of x) for the patch
  to route through the FP8 FFI.
"""

from __future__ import annotations

import inspect
import os
import re
import threading
from collections import defaultdict
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from .fp8_jax import (
    FP8_E4M3_MAX,
    FP8_SCALE_FLOOR,
    fp8_gemm_bf16_out,
    quantize_fp8_static,
)


# ── State ──────────────────────────────────────────────────────────

_LOCK = threading.Lock()
_INSTALLED = False
_ORIGINALS: dict[str, Callable] = {}  # name → original method
_STATS = defaultdict(int)


def _flag(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _is_enabled() -> bool:
    return _flag("FLASHVLA_JAX_FP8", "1") == "1"


def _min_m() -> int:
    try:
        return int(_flag("FLASHVLA_JAX_FP8_MIN_M", "64"))
    except ValueError:
        return 64


def get_stats() -> dict[str, int]:
    """Return per-call-site routing counters."""
    return dict(_STATS)


def reset_stats() -> None:
    _STATS.clear()


# ── 2-D FP8 dot (rank-2 case, used by FeedForward._dot) ────────────


def _fp8_dot_2d(x_bf16: jnp.ndarray, w_bf16: jnp.ndarray) -> jnp.ndarray:
    """``y = x @ w`` with FP8 forward + BF16 backward. ``w`` is 2-D.

    Activation scale + weight scale are computed dynamically from
    per-tensor amax. The custom_vjp internally quantizes ``x`` and
    runs the FFI GEMM; the BF16 mirror of ``w`` is used in backward.
    """
    if x_bf16.dtype != jnp.bfloat16:
        x_bf16 = x_bf16.astype(jnp.bfloat16)
    if w_bf16.dtype != jnp.bfloat16:
        w_bf16 = w_bf16.astype(jnp.bfloat16)
    if w_bf16.ndim != 2:
        raise ValueError(
            f"_fp8_dot_2d: w must be rank-2; got {w_bf16.shape}"
        )
    K = w_bf16.shape[0]
    N = w_bf16.shape[1]
    if x_bf16.shape[-1] != K:
        raise ValueError(
            f"_fp8_dot_2d: x last-dim {x_bf16.shape[-1]} != w first-dim {K}"
        )

    M_shape = x_bf16.shape[:-1]
    x_2d = x_bf16.reshape(-1, K)

    # Dynamic per-tensor scales. clip(...).max(eps) avoids div-by-zero.
    x_scale = jnp.maximum(
        jnp.max(jnp.abs(x_bf16)).astype(jnp.float32) / FP8_E4M3_MAX,
        FP8_SCALE_FLOOR,
    ).reshape((1,))
    w_scale = jnp.maximum(
        jnp.max(jnp.abs(w_bf16)).astype(jnp.float32) / FP8_E4M3_MAX,
        FP8_SCALE_FLOOR,
    ).reshape((1,))

    # Quantize w to fp8 bytes — held briefly, fed into the FP8 GEMM.
    w_fp8 = quantize_fp8_static(w_bf16, w_scale)

    y_2d = fp8_gemm_bf16_out(x_2d, w_bf16, w_fp8, x_scale, w_scale)
    return y_2d.reshape(*M_shape, N)


# ── Generic einsum → FP8 2-D GEMM mapper ───────────────────────────

_EQN_RE = re.compile(r"^(.*?)\s*,\s*(.*?)\s*->\s*(.*?)$")


def _parse_eqn(eqn: str) -> tuple[str, str, str]:
    m = _EQN_RE.match(eqn)
    if not m:
        raise ValueError(f"unsupported einsum equation: {eqn!r}")
    return m.group(1), m.group(2), m.group(3)


def _einsum_to_2d_plan(
    eqn: str, x_shape: tuple[int, ...], w_shape: tuple[int, ...]
) -> dict[str, Any] | None:
    """Compute the transpose / reshape plan that turns ``eqn`` into a 2-D GEMM.

    Returns ``None`` if the equation contains uncommon patterns
    (broadcast axes, axes appearing only in lhs but kept in out,
    repeated labels) — caller falls back to ``jnp.einsum``.
    """
    lhs, rhs, out = _parse_eqn(eqn)
    if any(s.count(c) != 1 for s in (lhs, rhs, out) for c in s):
        return None  # repeated labels — diag, trace, etc., not a plain GEMM
    lhs_set, rhs_set, out_set = set(lhs), set(rhs), set(out)

    # K = labels shared by lhs and rhs that disappear in out.
    K_set = (lhs_set & rhs_set) - out_set
    if not K_set:
        return None
    # Refuse cases where a label appears in lhs+out but not rhs (or
    # rhs+out but not lhs) — this is broadcast/copy semantics, which
    # the simple 2-D GEMM doesn't model.
    extras_lhs = (lhs_set - rhs_set) - K_set    # labels only in lhs
    extras_rhs = (rhs_set - lhs_set) - K_set    # labels only in rhs
    # extras_lhs must all be in out (M-side) and extras_rhs must all be in out (N-side).
    if not extras_lhs.issubset(out_set) or not extras_rhs.issubset(out_set):
        return None

    # Order K labels by their position in lhs; rhs is permuted to match.
    K_in_lhs = [c for c in lhs if c in K_set]
    M_labels = [c for c in lhs if c not in K_set]
    N_labels = [c for c in rhs if c not in K_set]

    # Sanity: the 2-D output's natural label order is M_labels + N_labels.
    natural_labels = M_labels + N_labels
    if set(natural_labels) != out_set:
        return None
    perm_out = [natural_labels.index(c) for c in out]
    perm_x = [lhs.index(c) for c in M_labels] + [lhs.index(c) for c in K_in_lhs]
    perm_w = [rhs.index(c) for c in K_in_lhs] + [rhs.index(c) for c in N_labels]

    M_dims = [x_shape[lhs.index(c)] for c in M_labels]
    K_dims = [x_shape[lhs.index(c)] for c in K_in_lhs]
    N_dims = [w_shape[rhs.index(c)] for c in N_labels]
    return {
        "perm_x": perm_x,
        "perm_w": perm_w,
        "M_dims": M_dims,
        "K_dims": K_dims,
        "N_dims": N_dims,
        "perm_out": perm_out,
        "M_labels": M_labels,
        "N_labels": N_labels,
    }


def _fp8_einsum_via_2d(
    eqn: str,
    x_bf16: jnp.ndarray,
    w_bf16: jnp.ndarray,
) -> jnp.ndarray | None:
    """Run an einsum-equivalent matmul through the FP8 2-D GEMM.

    Returns ``None`` if the equation cannot be mapped to a clean
    2-D GEMM — caller should fall back to ``jnp.einsum``.
    """
    plan = _einsum_to_2d_plan(eqn, tuple(x_bf16.shape), tuple(w_bf16.shape))
    if plan is None:
        return None

    x_perm = jnp.transpose(x_bf16, plan["perm_x"])
    w_perm = jnp.transpose(w_bf16, plan["perm_w"])
    M_total = int(np.prod(plan["M_dims"])) if plan["M_dims"] else 1
    K_total = int(np.prod(plan["K_dims"]))
    N_total = int(np.prod(plan["N_dims"])) if plan["N_dims"] else 1
    x_2d = x_perm.reshape(M_total, K_total)
    w_2d = w_perm.reshape(K_total, N_total)

    y_2d = _fp8_dot_2d(x_2d, w_2d)        # (M_total, N_total) bf16
    y_natural = y_2d.reshape(*plan["M_dims"], *plan["N_dims"])
    return jnp.transpose(y_natural, plan["perm_out"])


def _activation_M(eqn: str, x_shape: tuple[int, ...]) -> int:
    """Return M = product of non-contracted axes of x. Returns 0 on parse failure."""
    try:
        lhs, rhs, out = _parse_eqn(eqn)
    except ValueError:
        return 0
    if any(s.count(c) != 1 for s in (lhs, rhs, out) for c in s):
        return 0
    K_set = (set(lhs) & set(rhs)) - set(out)
    if not K_set:
        return 0
    M_labels = [c for c in lhs if c not in K_set]
    if not M_labels:
        return 1
    try:
        return int(np.prod([x_shape[lhs.index(c)] for c in M_labels]))
    except IndexError:
        return 0


# ── Patches ────────────────────────────────────────────────────────


def _patched_einsum_call(self, eqn: str, x):
    """Drop-in replacement for ``openpi.models.lora.Einsum.__call__``."""
    dtype = x.dtype

    # The original Einsum.__call__ registers "w" as a parameter; we
    # rely on @nn.compact wiring around this method to keep that
    # machinery unchanged. ``self.w`` stays available regardless of
    # routing decisions below.
    take_fp8 = (
        self.lora_config is not None
        and _is_enabled()
        and _activation_M(eqn, tuple(x.shape)) >= _min_m()
    )

    if take_fp8:
        result = _fp8_einsum_via_2d(eqn, x, self.w.astype(dtype))
        if result is None:
            # Equation didn't map to a clean 2-D GEMM (broadcast / repeat).
            _STATS["einsum_unmappable"] += 1
            result = jnp.einsum(eqn, x, self.w.astype(dtype))
        else:
            _STATS["einsum_routed"] += 1
            if result.dtype != dtype:
                result = result.astype(dtype)
    else:
        if self.lora_config is None:
            _STATS["einsum_no_lora"] += 1
        elif not _is_enabled():
            _STATS["einsum_disabled"] += 1
        else:
            _STATS["einsum_below_threshold"] += 1
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

    if config := self.lora_config:
        # LoRA delta path stays in BF16 — small matmuls, FP8 quantize
        # would lose precision without payoff.
        eqn_a, eqn_b = self._make_lora_eqns(eqn)
        lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
        lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
        result = result + lora * config.scaling_value

    return result


def _patched_feedforward_dot(self, x, w, lora_weights):
    """Drop-in replacement for ``openpi.models.lora.FeedForward._dot``."""
    dtype = x.dtype
    take_fp8 = (
        self.lora_config is not None
        and _is_enabled()
        and int(np.prod(x.shape[:-1])) >= _min_m()
    )

    if take_fp8 and w.ndim == 2:
        base = _fp8_dot_2d(x.astype(dtype), w.astype(dtype))
        if base.dtype != dtype:
            base = base.astype(dtype)
        _STATS["dot_routed"] += 1
    else:
        if self.lora_config is None:
            _STATS["dot_no_lora"] += 1
        elif not _is_enabled():
            _STATS["dot_disabled"] += 1
        elif w.ndim != 2:
            _STATS["dot_higher_rank_w"] += 1
        else:
            _STATS["dot_below_threshold"] += 1
        base = jnp.dot(x, w.astype(dtype))

    if lora_weights is None:
        return base
    return base + jnp.dot(
        jnp.dot(x, lora_weights[0].astype(dtype)),
        lora_weights[1].astype(dtype),
    )


def _assert_signature(fn: Callable, expected_params: list[str], where: str) -> None:
    """Fail fast if the upstream signature drifts away from what we patch."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return  # Decorator-wrapped: trust prototype-equivalent shape.
    params = list(sig.parameters.keys())
    if params[: len(expected_params)] != expected_params:
        raise RuntimeError(
            f"openpi.{where} signature drift detected: expected leading "
            f"params {expected_params!r}, got {params!r}. "
            "Update training/jax/fp8/lora_patch.py before re-enabling."
        )


def enable() -> None:
    """Install the FP8 patches on ``openpi.models.lora``. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        # Local import so users without openpi installed still get a
        # clean import-time error message, not a NameError on enable.
        from openpi.models import lora as openpi_lora
        import flax.linen as nn

        _assert_signature(
            openpi_lora.Einsum.__call__,
            ["self", "eqn", "x"],
            "Einsum.__call__",
        )
        _assert_signature(
            openpi_lora.FeedForward._dot,
            ["self", "x", "w", "lora_weights"],
            "FeedForward._dot",
        )

        _ORIGINALS["Einsum.__call__"] = openpi_lora.Einsum.__call__
        _ORIGINALS["FeedForward._dot"] = openpi_lora.FeedForward._dot

        # Wrap the einsum patch with @nn.compact so Flax recognises it
        # as the module's compact body (matches the original definition
        # at lora.py:54-65, which has the decorator).
        @nn.compact
        def _einsum_compact(self, eqn: str, x):
            return _patched_einsum_call(self, eqn, x)

        openpi_lora.Einsum.__call__ = _einsum_compact

        # _dot is a regular method (no @nn.compact in the upstream).
        openpi_lora.FeedForward._dot = _patched_feedforward_dot

        _INSTALLED = True


def disable() -> None:
    """Uninstall the FP8 patches, restoring the original openpi methods."""
    global _INSTALLED
    if not _INSTALLED:
        return
    with _LOCK:
        if not _INSTALLED:
            return
        from openpi.models import lora as openpi_lora
        openpi_lora.Einsum.__call__ = _ORIGINALS.pop("Einsum.__call__")
        openpi_lora.FeedForward._dot = _ORIGINALS.pop("FeedForward._dot")
        _INSTALLED = False


def is_installed() -> bool:
    return _INSTALLED

"""The `AttentionBackend` interface, and the profitability model that ranks candidates.

FIVE QUESTIONS, THE SAME FIVE

An attention backend is a pass with a narrower job, so it answers the same five questions
`passes/contract.py` asks of everything else:

    1. DETECTION      the adapter publishes ATTENTION sites; the backend does not go looking
    2. APPLICABILITY  `capabilities()` + `legality()` -- pure, no GPU
    3. CORRECTNESS    tier DERIVED from declared numerics, then verified against the real model
    4. PERFORMANCE    `measure()` on the site's real shapes, never a reputation ranking
    5. HARDWARE       `capabilities().hardware`

WHAT A BACKEND MAY NOT DO

  * import a model module, or name one. It receives shapes and layouts, never symbols.
  * decide when it is used. It declares; the planner selects.
  * change the function being computed. See `semantics.py`.

THE PROFITABILITY MODEL IS WHERE THE INTERESTING FAILURE LIVES

Attention is a PER_STEP cost, so the naive model is `saving = forwards_per_cycle * delta_per_forward`.
That model is wrong in two ways we have already been bitten by, and `expected_delta_ms()` exists to
make both explicit rather than discovering them after a kernel is written.

FIRST: the operating point sets the denominator. Graph capture (P005) is profitable at Quality (75
forwards/cycle) and a REGRESSION at Fast (6), because it trades ~17 ms/forward for ~700 ms/cycle of
fixed capture cost and breaks even near 41 forwards. Any backend with a `host_setup_us` term has the
same shape of behaviour, and the crossover must be computed, not assumed.

SECOND: attention's share is small, and shrinks. It is 7% of GPU-busy time at Quality. The measured
warm cost model at Fast is `FIXED 1164 ms + 15.5 ms/forward`, so at 6 forwards/cycle 93% of latency is
fixed overhead that no attention kernel touches. A backend that halves attention time buys roughly
3.5% of the cycle at Quality and 0.5% at Fast. Both are real; neither is what the layer's reputation
suggests. This is the arithmetic that should stop a kernel being written, and it belongs in the
planner rather than in a reviewer's head.

THIRD, and worst: a backend can be a net loss while being strictly faster at attention. If it is not
`capture_safe`, selecting it forfeits graph capture -- 1.205x at Quality. A 15% attention win against
a 20% plan-wide loss is a regression that every per-kernel microbenchmark reports as a success. The
plan-level term is therefore part of the model, not a caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from instinctwm.backends.attention.capabilities import AttentionCapabilities
from instinctwm.backends.attention.semantics import AttentionShape
from instinctwm.passes.contract import DeviceProfile, Tier


@dataclass(frozen=True)
class AttentionMeasurement:
    """One measured backend on one site's real shapes.

    `spread` and `not_evaluated` are not optional extras. A latency ranking taken on a contended box
    describes the neighbour: a 48%-busy GPU once turned a genuine 1.20x into a reported regression.
    A measurement that could not be trusted must be able to say so rather than returning a number.
    """

    backend: str
    us_per_call: float
    spread: float = 0.0
    not_evaluated: str = ""          # non-empty means NO verdict is available from this sample
    max_abs_delta: float | None = None   # vs the site's reference implementation
    tier_achieved: Tier | None = None

    def usable(self) -> bool:
        return not self.not_evaluated


@dataclass(frozen=True)
class AttentionBinding:
    """What a backend hands back: a callable plus what the executor must know to install it.

    Deliberately NOT a torch module. The executor installs this through the adapter's own rewrite
    handle (`RewriteKind.WRAP` or `SET` on an ATTENTION site), which is what keeps the backend free
    of model symbols and the adapter free of backend knowledge.
    """

    call: Callable[..., Any]
    #: Set when the binding must be rebuilt: shapes left the declared envelope, or the KV ring
    #: wrapped. The executor keys its cache on this, the way graph capture keys on (start, count).
    validity_key: tuple = ()
    #: True if `call` allocates or synchronises on first use, so a warmup call is required before
    #: any capture attempt.
    needs_warmup: bool = False
    note: str = ""


@runtime_checkable
class AttentionBackend(Protocol):
    """One implementation of one or more attention semantics.

    Registering a backend must not require touching a planner or an adapter. That is the whole test
    of this interface: `registry.register(MyBackend())` and nothing else changes.
    """

    name: str
    version: str

    def capabilities(self) -> AttentionCapabilities:
        """Declared envelope. Static, cheap, and safe to call with no GPU present."""
        ...

    def expected_delta_ms(self, shape: AttentionShape, forwards_per_cycle: int,
                          device: DeviceProfile) -> float:
        """Predicted per-CYCLE saving, positive meaning faster. A formula, not a rank.

        Implementations should charge `host_setup_us * forwards_per_cycle` against their own win and
        must not model the plan-level capture term -- the planner owns that, because only the planner
        knows whether capture is in the plan at all. See `plan_penalty_ms`.
        """
        ...

    def measure(self, shape: AttentionShape, device: DeviceProfile) -> AttentionMeasurement:
        """Measured cost on the site's real shapes, plus the delta against the reference.

        Required to withhold a verdict on a contended device rather than return a number, and
        required to report `max_abs_delta` so the tier is verified rather than claimed.
        """
        ...

    def bind(self, shape: AttentionShape, **site_attrs) -> AttentionBinding:
        """Produce the callable. Receives shapes and declared attributes only -- never a module."""
        ...


def plan_penalty_ms(caps: AttentionCapabilities, *, capture_in_plan: bool,
                    capture_speedup: float, cycle_ms: float) -> float:
    """The cost of choosing this backend that is invisible to a per-kernel benchmark.

    A capture-hostile backend forfeits graph capture. That penalty belongs to the PLAN, not to the
    backend, because it is zero when capture is not in the plan and large when it is -- and a
    microbenchmark of the kernel cannot see it either way.

    Returns milliseconds per cycle to SUBTRACT from the backend's own predicted win.
    """
    if caps.capture_safe or not capture_in_plan or capture_speedup <= 1.0:
        return 0.0
    # Losing a k-times speedup on the whole cycle costs cycle_ms * (1 - 1/k).
    return cycle_ms * (1.0 - 1.0 / capture_speedup)

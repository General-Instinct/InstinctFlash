"""The optimization pass contract.

Every optimization in InstinctWM is a pass, and every pass answers five questions. This module
is the contract; `passes/lingbot/` are the implementations.

    1. DETECTION      can the optimizer find the opportunity by itself?
    2. APPLICABILITY  is it legal for this model, on this hardware, right now?
    3. CORRECTNESS    what does it do to the outputs, and how is that proven?
    4. PERFORMANCE    does it actually make this model faster, measured?
    5. HARDWARE       where does it run?

The two gates are separate on purpose, because they fail independently. A pass can be perfectly
accuracy-neutral and still be a regression: on pi-0's real shapes, swapping eager attention for
SDPA while keeping the mask measures 133.5 -> 144-184 us. A correctness-only gate certifies the
numerics of the slower variant and ships it. So `verify()` and `benchmark()` are both required,
and a pass that does not improve its declared cost term is rejected regardless of its tier.

The cost model has two terms because ranking by software layer is wrong. Cosmos3-Edge measures
p99 = 94.6 ms FIXED + 31.76 ms x NFE; a stack that only reduces per-step cost has nothing to
offer the one model with a measured deadline problem, and will not say so unless the ranking
function can see the difference. Hence `CostTerm` and `expected_delta_ms(nfe)`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from instinctwm.adapters.base import AdapterSpec


class Tier(enum.IntEnum):
    """Ordered weakest-claim-last, so `max()` over a plan yields the plan's tier."""

    BITEXACT = 0
    NUMERIC = 1
    BEHAVIORAL = 2


class CostTerm(enum.Enum):
    """Which term of `latency = fixed + nfe * per_step` a pass reduces."""

    FIXED = "fixed"
    PER_STEP = "per_step"
    BOTH = "both"


class Discovery(enum.Enum):
    """How the opportunity is found. AUTO is the product; the others are honest fallbacks."""

    AUTO = "auto"              # optimizer detects it: module tree, trace, profile, differential test
    DECLARED = "declared"      # needs an adapter fact that cannot be safely inferred
    CHECKPOINT = "checkpoint"  # needs new weights; the runtime can only host it


@dataclass(frozen=True)
class HardwareReq:
    """Where a pass can run. Empty fields mean 'anywhere'."""

    min_capability: tuple[int, int] | None = None   # e.g. (9, 0) for Hopper
    requires: frozenset[str] = frozenset()          # 'fp8', 'nvfp4', 'cuda_graphs', 'triton'
    excludes: frozenset[str] = frozenset()

    def __post_init__(self):
        # Accept any iterable of names. A tuple is the obvious thing to write, it type-checks against
        # nothing at runtime, and it got as far as plan time before failing with
        # "unsupported operand type(s) for -: 'tuple' and 'frozenset'" -- an error about operators,
        # from deep inside the planner, for what is really a typo in a pass declaration. Every field
        # in this class is set once at class-definition time by pass authors, so normalising here is
        # free and turns a latent crash into no bug at all.
        for field in ("requires", "excludes"):
            value = getattr(self, field)
            if not isinstance(value, frozenset):
                object.__setattr__(self, field, frozenset(value or ()))
        unknown = (self.requires | self.excludes) - KNOWN_FEATURES
        if unknown:
            raise ValueError(
                f"HardwareReq names feature(s) no probe can emit: {sorted(unknown)}. "
                f"Known: {sorted(KNOWN_FEATURES)}. A requirement the probe cannot name is "
                f"unsatisfiable on every device, which is how P007's requires={{'cudnn'}} came to be "
                f"dormant-broken -- so it is refused at declaration time instead.")

    def satisfied_by(self, device: "DeviceProfile") -> tuple[bool, str]:
        if self.min_capability and device.capability < self.min_capability:
            return False, f"needs sm_{self.min_capability[0]}{self.min_capability[1]}, " \
                          f"device is sm_{device.capability[0]}{device.capability[1]}"
        missing = self.requires - device.features
        if missing:
            return False, f"device lacks {sorted(missing)}"
        clash = self.excludes & device.features
        if clash:
            return False, f"excluded on devices with {sorted(clash)}"
        return True, "ok"


def _cudnn_available() -> bool:
    import torch
    return bool(torch.backends.cudnn.is_available())


#: Every feature name `probe()` can emit. A backend may only require names from this set: a
#: requirement the probe cannot name is unsatisfiable on every device, which is how P007's
#: `requires={"cudnn"}` came to be dormant-broken. Enforced by tests/test_hardware_probe.py.
KNOWN_FEATURES = frozenset({
    "cpu", "cuda", "cuda_graphs", "triton", "fp8", "nvfp4", "wgmma", "tma", "cudnn", "cublas",
})


@dataclass(frozen=True)
class DeviceProfile:
    """What the target can do. Probed once, cached."""

    name: str
    capability: tuple[int, int]
    total_memory: int
    features: frozenset[str]
    #: measured, not assumed -- passes rank against these
    launch_overhead_us: float = 0.0
    hbm_bandwidth_gbps: float = 0.0

    @staticmethod
    def probe() -> "DeviceProfile":
        import torch
        # CPU IS A HARDWARE TARGET, and treating it as "no device" was wrong. This raised
        # `RuntimeError: No CUDA GPUs are available` on a GPU-less machine, the facade swallowed it,
        # and the planner then reported every hardware requirement as UNCHECKED -- when the truthful
        # answer was available and specific: this is a CPU, and cuda_graphs, cudnn and fp8 are
        # genuinely absent, so passes needing them must decline rather than be left undecided.
        # It is also the machine most external users have, so it is the one where an honest plan
        # matters most.
        if not torch.cuda.is_available():
            import platform
            return DeviceProfile(
                name=f"CPU ({platform.machine()})", capability=(0, 0),
                total_memory=0, features=frozenset({"cpu"}))
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        cap = (p.major, p.minor)
        # "cuda" is vacuously true here -- probe() cannot run without a CUDA device -- but
        # `graph_capture` declares `requires={"cuda"}`, and a name the probe never emits is
        # unsatisfiable everywhere. Emitting it is honest and keeps the vocabulary closed.
        feats = {"cuda", "cuda_graphs", "triton"}
        if cap >= (8, 9):
            feats.add("fp8")
        if cap >= (10, 0):
            feats.add("nvfp4")      # Blackwell only -- a pass gated on it does not apply on H100
        if cap >= (9, 0):
            feats.add("wgmma")
            feats.add("tma")
        # VENDOR LIBRARIES, and their absence here was a live latent bug. `CuDNNConv3d`
        # (backends/conv/reference.py) declares `requires={"cudnn"}`, this probe never emitted it,
        # and the two only failed to contradict each other because nothing in planners/ calls
        # `probe()`. Wiring the probe would have made P007 -- the shipped 1.405x NUMERIC pass --
        # silently inapplicable while the plan still reported a legal selection. A capability the
        # probe cannot name is a capability no backend may require, so the vocabulary has to be
        # closed on both sides; `tests/test_hardware_probe.py` now asserts that it is.
        for name, avail in (("cudnn", _cudnn_available), ("cublas", lambda: True)):
            try:
                if avail():
                    feats.add(name)
            except Exception:                                    # noqa: BLE001  never fail a probe
                pass
        return DeviceProfile(name=p.name, capability=cap, total_memory=p.total_memory,
                             features=frozenset(feats))


@dataclass
class VerifyResult:
    """Outcome of the correctness gate."""

    passed: bool
    tier_achieved: Tier
    max_abs_delta: float
    detail: str = ""


@dataclass
class BenchResult:
    """Outcome of the performance gate."""

    passed: bool
    before_ms: float
    after_ms: float
    detail: str = ""

    @property
    def speedup(self) -> float:
        return self.before_ms / self.after_ms if self.after_ms else float("nan")


@dataclass
class Applicability:
    """Why a pass will or will not fire."""

    applies: bool
    reason: str
    discovery: Discovery = Discovery.DECLARED
    cost_term: CostTerm = CostTerm.PER_STEP
    claimed_tier: Tier = Tier.BITEXACT
    params: dict = field(default_factory=dict)


@runtime_checkable
class OptimizationPass(Protocol):
    """The five questions."""

    name: str
    hardware: HardwareReq

    # 1 + 2. Detection and applicability, from declarations and the device alone.
    def applicability(self, spec: AdapterSpec, device: DeviceProfile) -> Applicability: ...

    # How much it should help, as a formula over the cost model rather than a hand-written rank.
    def expected_delta_ms(self, spec: AdapterSpec, device: DeviceProfile) -> float: ...

    # Apply to a live serving object.
    def install(self, server_module, server_cls) -> None: ...

    # 3. Correctness gate. Must be run against the real model, not asserted.
    def verify(self, harness) -> VerifyResult: ...

    # 4. Performance gate. A pass that does not improve its declared cost term is rejected.
    def benchmark(self, harness) -> BenchResult: ...


def gate(pass_: OptimizationPass, verify: VerifyResult, bench: BenchResult,
         claimed_tier: Tier) -> tuple[bool, str]:
    """Both gates, applied. Returns (accept, reason).

    Ordering matters: a pass that is wrong is rejected before we care whether it is fast, and a
    pass that is right but slower is still rejected. Neither gate is advisory.
    """
    if not verify.passed:
        return False, (f"CORRECTNESS FAIL: max|delta| = {verify.max_abs_delta:.3e}, "
                       f"achieved {verify.tier_achieved.name} < claimed {claimed_tier.name}. "
                       f"{verify.detail}")
    if verify.tier_achieved > claimed_tier:
        return False, (f"TIER DOWNGRADE: claimed {claimed_tier.name}, achieved "
                       f"{verify.tier_achieved.name}. Re-declare the tier or fix the pass.")
    if not bench.passed:
        return False, (f"PERFORMANCE FAIL: {bench.before_ms:.1f} -> {bench.after_ms:.1f} ms "
                       f"({bench.speedup:.2f}x). Accuracy-neutral is necessary, not sufficient. "
                       f"{bench.detail}")
    return True, (f"accept: {bench.before_ms:.1f} -> {bench.after_ms:.1f} ms "
                  f"({bench.speedup:.2f}x), {verify.tier_achieved.name}, "
                  f"max|delta| = {verify.max_abs_delta:.3e}")

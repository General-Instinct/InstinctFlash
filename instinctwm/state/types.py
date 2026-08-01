"""L3 State/Memory — the type system.

L3 owns where state lives and how it is addressed. The organising principle:

    A live-set query is DISCOVERY when its answer is computed from DEVICE data at read time.
    It is CONSTRUCTION when the answer is already known in HOST integers and merely restated.

Everything here exists to let a pass decide, from declarations alone, whether a given piece of
state is being discovered when it could be constructed — and to make the answer checkable.

Two design choices are worth stating because the obvious alternatives were tried and rejected:

  * A LiveSet is anchored to a TOKEN PARTITION, not to a KV arena. Anchoring to an arena means a
    model with no persistent KV is reported as having nothing to address, which is how Cosmos3-Edge
    (whose defect is a re-materialised index over a packed sequence, with no KV pool involved at
    all) fell through an earlier design.
  * Causality and order-significance are PER SEGMENT. Cosmos3's `two_way_attention` runs a causal
    pass over und-only keys and a full pass whose KV is und UNION gen, inside ONE buffer. An
    arena-scoped flag cannot express one buffer with two causalities.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence


class Scope(enum.Enum):
    """When state dies. A scope boundary must be an O(1) pointer reset (invariant I5)."""

    FORWARD = "forward"    # one network forward; LingBot has 79 per control step
    STEP = "step"          # one control step / chunk
    WINDOW = "window"      # N frames, hard reset at a boundary (DreamZero)
    EPISODE = "episode"    # one rollout (LingBot's KV pool)
    SESSION = "session"    # across episodes (weights-adjacent caches)


class Addressing(enum.Enum):
    """How the live set is located."""

    NONE = "none"                   # there is no persistent set (GR00T)
    DENSE = "dense"                 # the whole buffer is live
    PREFIX = "prefix"               # [0, n) with n a host int (pi-0's VLM prefix)
    RING_INTERVAL = "ring"          # [start, start+count) mod capacity (LingBot)
    PAGED = "paged"                 # block table (DreamZero)
    STATIC_PARTITION = "partition"  # fixed index sets from declared geometry (Cosmos3 und/gen)


class Order(enum.Enum):
    """The order the live set is enumerated in.

    Load-bearing, not cosmetic: softmax is permutation-invariant in maths and NOT in floating
    point, so re-ordering keys changes the reduction and breaks bit-exactness. Checked with
    elementwise equality on the index vector, never set equality (invariant I3).
    """

    PHYSICAL_ASCENDING = "physical"   # ascending slot index -- what a boolean mask produces
    LOGICAL_POSITION = "logical"      # chronological / positional
    UNORDERED = "unordered"           # the model asserts order does not affect its output


class Residency(enum.Enum):
    MANAGED = "managed"              # L3 allocates and accounts for it
    CALLER_OWNED = "caller_owned"    # arrives per request; never cached, never content-validated
    RECOMPUTED = "recomputed"        # not materialised; carries a recompute cost instead of bytes
    UNMANAGED = "unmanaged"          # exists, L3 does not manage it, but it IS reported


class CommitMode(enum.Enum):
    TRANSIENT = "transient"          # write then roll back within one forward
    PROVISIONAL = "provisional"      # survives the forward, dropped at the next commit boundary
    CONFIRMED = "confirmed"          # permanent for the arena's scope


class Discovery(enum.Enum):
    """The four signatures found in the corpus. A pass declares which one it detects."""

    D1_BOOLEAN_SCAN = "d1"           # (~mask).nonzero / mask.any / argsort        -- LingBot
    D2_GROWING_CAT = "d2"            # torch.cat rebuilds the set on the read path -- pi-0, Cosmos3
    D3_STATIC_INDEX = "d3"           # index rebuilt per forward from host data    -- Cosmos3 GEN
    D4_METADATA_REBUILD = "d4"       # per-forward metadata reconstruction         -- DreamZero
    NONE = "none"


@dataclass(frozen=True)
class Extent:
    """A host-evaluable bound on a slice's size.

    Must NAME the field that bounds it. An unnamed bound lets the reader pick, and the corpus has
    a case where 4096 is a prompt-truncation cap in one port and a dataloader knob in another while
    the declared position cap is 131072.
    """

    tokens: int
    bounding_field: str
    per_call_bound: bool = False     # if True this is an admission parameter, not a static bound


@dataclass(frozen=True)
class Segment:
    """One contiguous run of the live set, with its own causality.

    Per-segment because Cosmos3's single buffer carries a causal und-only pass and a full
    und-union-gen pass; an arena-scoped flag cannot say that.
    """

    name: str
    reads_from: tuple[str, ...] = ()
    causal: bool = False
    order_is_semantic: bool = True


@dataclass(frozen=True)
class LiveSet:
    """How a partition of tokens is addressed. `backing` may be None: a live set need not be KV."""

    name: str
    addressing: Addressing
    order: Order
    segments: tuple[Segment, ...] = ()
    backing: str | None = None       # arena name, or None for a pure token partition
    commit_period: int | None = None # tokens committed per boundary; needed for RING validity (I6)
    capacity: int | None = None


@dataclass(frozen=True)
class SliceSpec:
    """One named piece of state."""

    name: str
    scope: Scope
    residency: Residency
    bytes_per_episode: int = 0
    recompute_ms: float = 0.0
    commit_mode: CommitMode = CommitMode.CONFIRMED
    extent: Extent | None = None
    discovery: Discovery = Discovery.NONE
    note: str = ""


@dataclass(frozen=True)
class ArenaSpec:
    """A physically contiguous allocation. Every slice in it dies at the same event (I5)."""

    name: str
    scope: Scope
    #: host-evaluable; must not depend on device data (I7)
    extent_tokens: int
    bytes_per_token: int
    slices: tuple[str, ...] = ()

    @property
    def nbytes(self) -> int:
        return self.extent_tokens * self.bytes_per_token


@dataclass(frozen=True)
class StateManifest:
    """Everything L3 knows about one model's state. Produced by the Backend Adapter."""

    model_id: str
    arenas: tuple[ArenaSpec, ...] = ()
    slices: tuple[SliceSpec, ...] = ()
    live_sets: tuple[LiveSet, ...] = ()
    #: number of host syncs L3 permits on the critical path. CI asserts EQUALITY, not improvement.
    sync_budget: int = 0
    forwards_per_step: int = 1

    # ---- budget terms. Three, not one: a byte-only budget makes the highest-value state in the
    # corpus invisible, because it does not exist yet (LingBot's withheld cross-attention K/V is
    # 89 of 226 TFLOP per cycle and 0 bytes).
    def e_materialized(self) -> int:
        return sum(s.bytes_per_episode for s in self.slices
                   if s.residency is Residency.MANAGED)

    def e_recompute_ms(self) -> float:
        return sum(s.recompute_ms for s in self.slices
                   if s.residency is Residency.RECOMPUTED)

    def e_unmanaged(self) -> int:
        return sum(s.bytes_per_episode for s in self.slices
                   if s.residency is Residency.UNMANAGED)

    def has_state(self) -> bool:
        """False for a genuinely stateless model. Drives the enforced no-op."""
        return bool(self.arenas) or any(
            s.scope is not Scope.FORWARD for s in self.slices
            if s.residency is not Residency.CALLER_OWNED)


@dataclass(frozen=True)
class Capacity:
    """Episodes per GPU. A minimum over all declared resources, with the binding term NAMED (I10).

    An earlier formula divided by bytes alone: it raised ZeroDivisionError on GR00T, returned 4961
    for pi-0, and returned 8 for LingBot on a runner that serves one episode at a time -- because
    it had no time dimension.
    """

    n: int
    binding: str
    n_memory: float
    n_deadline: float
    n_serving: int

    @staticmethod
    def compute(manifest: StateManifest, *, hbm_bytes: int, weight_bytes: int,
                reserve_bytes: int, forward_peak_bytes: int,
                cycle_ms: float, deadline_ms: float, serving_concurrency: int) -> "Capacity":
        e = manifest.e_materialized()
        if e <= 0:
            n_mem = float("inf")          # no per-episode state: memory is not the binding term
        else:
            free = hbm_bytes - weight_bytes - reserve_bytes - forward_peak_bytes
            n_mem = max(0.0, free / e)
        n_dl = max(0.0, deadline_ms / cycle_ms) if cycle_ms > 0 else float("inf")
        cands = {"memory": n_mem, "deadline": n_dl, "serving": float(serving_concurrency)}
        binding = min(cands, key=lambda k: cands[k])
        n = int(min(cands.values())) if min(cands.values()) != float("inf") else serving_concurrency
        return Capacity(n=max(0, n), binding=binding, n_memory=n_mem, n_deadline=n_dl,
                        n_serving=serving_concurrency)


# ---------------------------------------------------------------------------------------------
# The uniform applicability predicate. Every L3 pass uses THIS -- not its own prose -- so that a
# model is refused by one code path instead of four write-ups. GR00T's no-op is a mechanism.
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class L3Applicability:
    applies: bool
    reason: str
    detects: Discovery = Discovery.NONE
    targets: tuple[str, ...] = ()


def applies_to(manifest: StateManifest, *, detects: Discovery,
               predicate: Callable[[StateManifest], tuple[bool, str, tuple[str, ...]]]
               ) -> L3Applicability:
    """Shared gate: no state means no L3, full stop, before any pass-specific logic runs."""
    if not manifest.has_state():
        return L3Applicability(
            False,
            "model declares no state outliving a forward; L3 installs nothing "
            "(this is the enforced no-op, not an oversight)",
            detects=detects)
    ok, why, targets = predicate(manifest)
    return L3Applicability(ok, why, detects=detects, targets=targets)

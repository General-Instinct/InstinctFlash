"""The pass interface: adapters locate, passes decide.

WHY THIS EXISTS

Every pass we have works and none of them generalizes. `ring_kv`, `graph_block_stack`,
`stable_pools` and `hoist_invariant_casts` all begin the same way:

    import modules.model as M
    Attn = M.WanAttention
    Attn.forward = my_replacement

That line is the whole problem. It fuses two responsibilities that have different owners:

    WHERE   which callable, which tensor, which loop -- a fact about one model
    WHAT    what to do there -- a fact about an optimization

Because they are fused, the passes are adapters wearing pass clothing: on Cosmos3-Edge all three
are no-ops, not because the optimizations are inapplicable but because the symbols are missing.

THE SPLIT

An adapter publishes SITES. A site is a located, named thing carrying declared properties, plus a
handle the adapter knows how to rewrite. A pass never imports a model module; it asks for sites of
a kind, reads their properties, and returns REWRITES. The runner applies them through the adapter.

    adapter.sites(SiteKind.INVARIANT_CONDITIONING)  ->  [Site(...), ...]
    pass.plan_rewrites(sites, device)               ->  [Rewrite(...), ...]
    runner.apply(adapter, rewrites)                 ->  installed, recorded, gated

The pass sees `site.attrs["scope"] == Scope.MODEL` and decides to hoist. It never learns that the
site happens to be `FP32LayerNorm.weight` on layer 7.

WHAT THIS DOES NOT FIX YET

Locating is only half of a rewrite. A pass that must restructure a loop body (P003 replacing a
boolean-mask gather with a ring slice) needs more than "wrap this callable" -- it needs the site to
expose the addressing decision itself. `SiteKind.STATE_ADDRESSING` names that need; it does not yet
satisfy it. Being explicit about the gap is the point: the interface should make the next
optimization naturally generic, not retroactively claim the last four were.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol


class SiteKind(Enum):
    """Kinds of location an adapter can publish. One per question a pass asks."""

    #: a callable region of execution (a block, a layer stack, a denoise phase)
    EXECUTION_REGION = "execution_region"
    #: a region the adapter believes is capturable as one graph, with its arguments
    CAPTURE_UNIT = "capture_unit"
    #: a value that is constant over some scope but recomputed inside a tighter one
    INVARIANT_CONDITIONING = "invariant_conditioning"
    #: where state is located -- how a live set is addressed, not where it is stored
    STATE_ADDRESSING = "state_addressing"
    #: where device buffers are created, so their lifetime can be moved
    ALLOCATION = "allocation"


class Scope(Enum):
    """Binding time. The engine's one optimization is evaluating things at their true scope."""
    HARDWARE = 0
    MODEL = 1
    PLAN = 2
    EPISODE = 3
    CYCLE = 4
    PHASE = 5
    STEP = 6
    LAYER = 7

    def __lt__(self, other):
        return self.value < other.value


@dataclass(frozen=True)
class Site:
    """A located thing, plus whatever the adapter can honestly say about it.

    `attrs` is the pass's entire view of the model. If a pass needs a property that is not in
    `attrs`, the answer is to add the property to the vocabulary -- not to import the model.
    """
    kind: SiteKind
    id: str
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def scope(self) -> Scope | None:
        return self.attrs.get("scope")

    def evaluated_at(self) -> Scope | None:
        return self.attrs.get("evaluated_at")

    def is_hoistable(self) -> bool:
        s, e = self.scope(), self.evaluated_at()
        return s is not None and e is not None and s < e


class RewriteKind(Enum):
    WRAP = "wrap"           # wrap the site's callable: new = f(old)
    SET = "set"             # set a named property on the site
    DEFER = "defer"         # move an effect out of the site, to be run by the caller


@dataclass(frozen=True)
class Rewrite:
    """What a pass wants done, expressed without naming a model symbol."""
    site_id: str
    kind: RewriteKind
    payload: Any
    note: str = ""


class AdapterSurface(Protocol):
    """What a Backend Adapter must publish for passes to work on it.

    Deliberately two methods. Anything more and adapters start encoding optimization policy, which
    is the thing being separated out.
    """

    model_id: str

    def sites(self, kind: SiteKind) -> Iterable[Site]: ...

    def apply(self, rewrite: Rewrite) -> None: ...


class EnginePass(Protocol):
    """A pass: needs site kinds, returns rewrites, carries its own gates."""

    name: str

    def sites_required(self) -> tuple[SiteKind, ...]: ...

    def plan_rewrites(self, sites: Mapping[SiteKind, list[Site]], device) -> list[Rewrite]: ...


@dataclass
class PassResult:
    pass_name: str
    model_id: str
    applied: tuple[str, ...]
    skipped_reason: str | None = None

    @property
    def fired(self) -> bool:
        return bool(self.applied)

    def __str__(self) -> str:
        if not self.fired:
            return f"{self.pass_name:24s} on {self.model_id:16s} -> no-op ({self.skipped_reason})"
        return (f"{self.pass_name:24s} on {self.model_id:16s} -> {len(self.applied)} rewrite(s): "
                f"{list(self.applied)[:3]}{' ...' if len(self.applied) > 3 else ''}")


def run_pass(p: EnginePass, adapter: AdapterSurface, device=None) -> PassResult:
    """Apply a pass to an adapter. Neither knows the other's type.

    A pass that finds no sites is a clean no-op with a REASON, which is the difference between
    "this model does not have that structure" and "this pass could not find the symbol it wanted".
    The old passes could not tell those apart.
    """
    found: dict[SiteKind, list[Site]] = {}
    for kind in p.sites_required():
        found[kind] = list(adapter.sites(kind))

    if not any(found.values()):
        kinds = ", ".join(k.value for k in p.sites_required())
        return PassResult(p.name, adapter.model_id, (),
                          skipped_reason=f"adapter publishes no sites of kind: {kinds}")

    rewrites = p.plan_rewrites(found, device)
    if not rewrites:
        return PassResult(p.name, adapter.model_id, (),
                          skipped_reason="sites found but none matched the pass's criteria")

    for r in rewrites:
        adapter.apply(r)
    return PassResult(p.name, adapter.model_id, tuple(r.site_id for r in rewrites))

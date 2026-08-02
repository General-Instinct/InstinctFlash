"""Automatically derive what a capture unit depends on.

WHY THIS EXISTS

Three episode-scoped dependencies have now bitten this engine, and every one of them was a case of
a hand-maintained list covering a subset of what the region actually touched:

  1. the ring bookkeeping mutated host state inside the captured region  -> max|d| = 1.398
  2. the graph key omitted (start, count)                                -> 6 graphs, stale replays
  3. P002's cross-attention K/V was rebuilt each episode                 -> nan on episode 2

Each was fixed by remembering one more thing. That does not converge. The fix is to stop
remembering: run the region under a dispatch tracer and let it tell us what it touched.

WHAT IS DERIVED, AND HOW

  device reads   Every op input whose storage was NOT produced inside the region. These are the
                 buffers a captured graph reads from fixed addresses, so every one of them must be
                 address-stable across a reset or the graph must be dropped.

  device writes  Every op output that aliases one of those external storages. These are the
                 region's side effects on device memory. They replay correctly (they are GPU work),
                 but they are what makes the region order-sensitive.

  host mutations Snapshot host state, run, snapshot again, diff. Anything that changed is Python
                 that graph replay will NOT re-execute -- the class of bug that produced (1).

  key fields     Perturb one host integer at a time, re-trace, and compare the trace signature
                 (op sequence + shapes + which external buffer each access lands in, at what
                 offset). If perturbing a field changes the signature, that field is baked into the
                 capture and MUST be in the graph cache key. This is what (2) needed, derived
                 rather than recalled.

Everything here runs at plan time, never on the replay path. Perturbation costs one extra trace per
candidate field; that is cheap against an episode.

WHAT IT CANNOT DO

Tracing observes one execution. A dependency that only appears at a ring wraparound, or on the
third call, will not show up. So this is a necessary condition for a safe capture, not a sufficient
one, and it does not replace the bit-exact gate -- it replaces the *list*.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from instinctwm.engine.effects import snapshot_host_state


def _storages(x) -> list[tuple[int, int]]:
    """(storage base pointer, byte offset) for every tensor in a pytree-ish structure.

    The STORAGE base, not `data_ptr()`: a slice of the KV pool has a different `data_ptr()` per
    offset but is the same buffer, and "which buffer" is the question that matters for address
    stability.
    """
    out = []
    if isinstance(x, torch.Tensor):
        if x.device.type == "cuda" and x.untyped_storage().size() > 0:
            base = x.untyped_storage().data_ptr()
            out.append((base, x.data_ptr() - base))
    elif isinstance(x, (list, tuple)):
        for y in x:
            out += _storages(y)
    elif isinstance(x, dict):
        for y in x.values():
            out += _storages(y)
    return out


class DependencyTracer(TorchDispatchMode):
    """Records external reads, external writes, and a structural signature of the trace."""

    def __init__(self):
        self.produced: set[int] = set()          # storages created inside the region
        self.reads: set[int] = set()
        self.writes: set[int] = set()
        self.ops: list[str] = []
        self._sig: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        ins = _storages(args) + _storages(kwargs)
        for base, _off in ins:
            if base not in self.produced:
                self.reads.add(base)

        # Which arguments does this op actually MUTATE? Ask the schema rather than inferring it
        # from storage aliasing: every view op (reshape, transpose, slice, unflatten) shares
        # storage with its input, so "output aliases an input" marks views as writes. That false
        # positive made the read-only encoder input show up as a written buffer.
        mutated_bases: set[int] = set()
        try:
            schema = func._schema
            flat = list(args) + [kwargs[a.name] for a in schema.arguments
                                 if a.name in kwargs]
            for a, v in zip(schema.arguments, flat):
                if a.alias_info is not None and a.alias_info.is_write:
                    mutated_bases.update(b for b, _ in _storages(v))
        except (AttributeError, RuntimeError):
            mutated_bases = set()

        in_bases = {b for b, _ in ins}
        out = func(*args, **kwargs)

        for base in mutated_bases:
            if base not in self.produced:
                self.writes.add(base)            # in-place on an external buffer
        # Only GENUINELY NEW allocations are internal. A view of an external buffer shares its
        # storage; recording that as "produced" made the buffer look internal and masked the
        # later copy_ into it -- `pool[:, sl] = key` lowers to slice + copy_, so the whole KV
        # write set silently disappeared.
        for base, _off in _storages(out):
            if base not in in_bases:
                self.produced.add(base)

        # Structural signature: what ran, on what shapes, at what offsets into which external
        # buffer. Offsets matter -- they are exactly what a captured graph bakes in.
        shapes = [tuple(a.shape) for a in (args or ()) if isinstance(a, torch.Tensor)]
        ext = sorted((b, o) for b, o in ins if b in self.reads)
        self._sig.append(f"{name}|{shapes}|{[o for _b, o in ext]}")
        self.ops.append(name)
        return out

    @property
    def signature(self) -> str:
        return hashlib.sha1("\n".join(self._sig).encode()).hexdigest()[:16]


@dataclass
class DependencySignature:
    """What a capture unit depends on. Generated, never hand-written."""
    n_ops: int
    reads: tuple[str, ...]              # named where possible, else hex address
    writes: tuple[str, ...]
    host_mutations: tuple[str, ...]
    key_fields: tuple[str, ...]
    trace_hash: str
    unnamed_reads: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def capturable(self) -> tuple[bool, str]:
        if self.host_mutations:
            return False, (f"mutates {len(self.host_mutations)} host key(s) graph replay will not "
                           f"re-execute: {list(self.host_mutations)[:4]}")
        return True, "no host mutation detected in the traced execution"

    def __str__(self) -> str:
        L = [f"DependencySignature[{self.trace_hash}]  {self.n_ops} ops",
             f"  device reads  ({len(self.reads)}): {list(self.reads)[:6]}"
             + (" ..." if len(self.reads) > 6 else ""),
             f"  device writes ({len(self.writes)}): {list(self.writes)[:6]}"
             + (" ..." if len(self.writes) > 6 else ""),
             f"  host mutations: {list(self.host_mutations)[:6] or 'none'}",
             f"  graph key fields: {list(self.key_fields) or 'none'}"]
        if self.unnamed_reads:
            L.append(f"  WARNING: {self.unnamed_reads} read buffer(s) could not be named -- they "
                     f"are outside the declared state, so nothing is checking their stability")
        L += [f"  note: {n}" for n in self.notes]
        return "\n".join(L)


def build_name_map(model, extra: Mapping[str, torch.Tensor] | None = None) -> dict[int, str]:
    """storage base pointer -> a name a human can act on."""
    m: dict[int, str] = {}

    def add(t, name):
        if isinstance(t, torch.Tensor) and t.device.type == "cuda" \
                and t.untyped_storage().size() > 0:
            m.setdefault(t.untyped_storage().data_ptr(), name)

    for n, p in getattr(model, "named_parameters", lambda: [])():
        add(p, f"param:{n}")
    for n, b in getattr(model, "named_buffers", lambda: [])():
        add(b, f"buffer:{n}")
    for i, blk in enumerate(getattr(model, "blocks", []) or []):
        for attr in ("attn1", "attn2"):
            a = getattr(blk, attr, None)
            if a is None:
                continue
            for cname, c in (getattr(a, "attn_caches", None) or {}).items():
                if isinstance(c, dict):
                    for k, t in c.items():
                        add(t, f"kv[{i}].{attr}.{cname}.{k}")
            for j, t in enumerate(getattr(a, "_iwm_cross_kv", None) or ()):
                add(t, f"cross_kv[{i}].{attr}[{j}]")
        # P004 caches fp32 views of loop-invariant parameters on the modules themselves. They are
        # read inside the captured region and `hoist_invariant_casts._reset` deletes them, so they
        # are reallocated every episode. Naming them is what turned 90 anonymous addresses into an
        # identified dependency.
        for name, mod in (blk.named_modules() if hasattr(blk, "named_modules") else []):
            for attr in ("_iwm_w32", "_iwm_b32", "_iwm_sst32"):
                add(getattr(mod, attr, None), f"hoisted[{i}].{name or 'block'}.{attr}")
    for k, t in (extra or {}).items():
        add(t, k)
    return m


def derive_signature(fn: Callable[[], Any], *, model, roots: Iterable[Any],
                     host_fields: Mapping[str, Callable[[], int]] | None = None,
                     extra_names: Mapping[str, torch.Tensor] | None = None,
                     perturb: Mapping[str, Callable[[int], None]] | None = None,
                     ) -> DependencySignature:
    """Trace `fn` once, diff host state, and perturb candidate host fields to find key fields.

    `host_fields` maps a name to a getter; `perturb` maps the same name to a setter. A field is a
    key field iff changing it changes the structural trace signature.
    """
    names = build_name_map(model, extra_names)
    roots = list(roots)

    before = snapshot_host_state(roots)
    tr = DependencyTracer()
    with tr, torch.no_grad():
        fn()
    after = snapshot_host_state(roots)
    mutations = tuple(sorted(k for k in after if before.get(k) != after[k]))

    def nm(b):
        return names.get(b, f"0x{b:x}")

    reads = tuple(sorted(nm(b) for b in tr.reads))
    writes = tuple(sorted(nm(b) for b in tr.writes))
    unnamed = sum(1 for b in tr.reads if b not in names)

    key_fields: list[str] = []
    if host_fields and perturb:
        base_sig = tr.signature
        for fname, get in host_fields.items():
            setter = perturb.get(fname)
            if setter is None:
                continue
            old = get()
            try:
                setter(old + 1)
                probe = DependencyTracer()
                with probe, torch.no_grad():
                    fn()
                if probe.signature != base_sig:
                    key_fields.append(fname)
            except Exception:
                # A field that cannot be perturbed safely is assumed to matter: failing closed is
                # the whole point of this module.
                key_fields.append(f"{fname}(unperturbable, assumed key)")
            finally:
                setter(old)

    notes = ()
    if unnamed:
        notes = ("unnamed reads are buffers no declared state covers; that is exactly how the "
                 "cross-attention K/V was missed",)
    return DependencySignature(
        n_ops=len(tr.ops), reads=reads, writes=writes, host_mutations=mutations,
        key_fields=tuple(key_fields), trace_hash=tr.signature, unnamed_reads=unnamed, notes=notes)

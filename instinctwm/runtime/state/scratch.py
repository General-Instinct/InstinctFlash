"""Scope-bumped scratch arena — reuse buffers across scopes, never within one.

The problem this solves, concretely. Cosmos3-Edge's `get_all_seq` allocates a fresh
`new_zeros([N_all, H, D])` and scatters twice, and it is called from `attention.py:203-204`:

    full_res = attention(
        full_q.unsqueeze(0),
        get_all_seq(packed_key_normalized).unsqueeze(0),   # call A
        get_all_seq(packed_value_states).unsqueeze(0),     # call B
        ...)

**A and B are live simultaneously as arguments to the same call.** Handing both the same reusable
buffer would alias K and V — the attention would compute `softmax(QK^T)V` with `V == K`, silently,
producing plausible wrong actions. That is the single most dangerous shape of bug in this project,
so the design has to make it *impossible*, not unlikely.

The rule
--------
A **bump pointer that only ever advances within a scope**. Every `acquire()` inside one scope
returns a DISTINCT buffer; the pointer resets only at a scope boundary. Aliasing within a scope is
therefore impossible by construction rather than by capacity planning: there is no wraparound to
get wrong, no "K is big enough" argument to be falsified by a third caller, and no assertion that
could be compiled out.

The arena grows to the high-water mark of simultaneous live buffers in a scope and then stops
allocating. For Cosmos3-Edge that is 2 buffers of [567, H, D] bf16, about 4.6 MB total, replacing
896 allocations and 2.08 GB of alloc-and-scatter traffic per control step.

What makes cross-scope reuse safe
---------------------------------
A buffer may be reused in a later scope only if no result outlives its scope. Here the results are
consumed as arguments to `attention(...)` inside the same expression, and attention returns a new
tensor rather than a view of K or V. The scope boundary is therefore placed at entry to the
attention call that consumes them. If a future caller retains a `get_all_seq` result past that
boundary, `assert_no_escapes()` in the debug harness catches it by tagging buffers and checking
identity at the next reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class _Pool:
    buffers: list[torch.Tensor] = field(default_factory=list)
    bump: int = 0


class ScratchArena:
    """Forward-scoped scratch, bump-allocated.

    Keyed by (shape, dtype, device) so differently-shaped requests never share storage even by
    accident. `high_water` is reported so the performance gate can confirm the arena stopped
    growing — a still-growing arena means the scope boundary is in the wrong place.
    """

    def __init__(self, name: str = "scratch"):
        self.name = name
        self._pools: dict[tuple, _Pool] = {}
        self.n_acquires = 0
        self.n_allocations = 0        # should stop rising once the high-water mark is reached
        self.n_scopes = 0

    def begin_scope(self) -> None:
        """Reset every bump pointer. The ONLY place a buffer becomes reusable."""
        for p in self._pools.values():
            p.bump = 0
        self.n_scopes += 1

    def acquire(self, shape: tuple[int, ...], dtype: torch.dtype,
                device: torch.device) -> torch.Tensor:
        """A buffer distinct from every other acquired in this scope."""
        key = (tuple(shape), dtype, str(device))
        pool = self._pools.get(key)
        if pool is None:
            pool = self._pools[key] = _Pool()
        if pool.bump >= len(pool.buffers):
            # Grow rather than wrap. Wrapping is the only way aliasing could occur, so the
            # structure simply does not offer it.
            pool.buffers.append(torch.empty(shape, dtype=dtype, device=device))
            self.n_allocations += 1
        buf = pool.buffers[pool.bump]
        pool.bump += 1
        self.n_acquires += 1
        return buf

    def high_water(self) -> dict[tuple, int]:
        return {k: len(p.buffers) for k, p in self._pools.items()}

    def nbytes(self) -> int:
        return sum(b.numel() * b.element_size()
                   for p in self._pools.values() for b in p.buffers)

    def live_ids_this_scope(self) -> set[int]:
        return {id(p.buffers[i]) for p in self._pools.values() for i in range(p.bump)}

    def stats(self) -> str:
        hw = ", ".join(f"{list(k[0])}x{v}" for k, v in self.high_water().items())
        return (f"{self.name}: {self.n_scopes} scopes, {self.n_acquires} acquires, "
                f"{self.n_allocations} allocations, high-water [{hw}], "
                f"{self.nbytes()/1e6:.2f} MB resident")


def assert_distinct_within_scope(arena: ScratchArena, n: int) -> tuple[bool, str]:
    """Property test: n acquires in one scope must return n distinct storages.

    This is the test that would catch an aliasing regression, and it is cheap enough to run in CI
    on every merge.
    """
    arena.begin_scope()
    shape, dtype, dev = (8, 4), torch.float32, torch.device("cpu")
    bufs = [arena.acquire(shape, dtype, dev) for _ in range(n)]
    ptrs = [b.data_ptr() for b in bufs]
    if len(set(ptrs)) != n:
        return False, f"{n} acquires produced {len(set(ptrs))} distinct storages -- ALIASING"
    # and across a scope boundary they SHOULD be reused
    arena.begin_scope()
    again = [arena.acquire(shape, dtype, dev) for _ in range(n)]
    if [b.data_ptr() for b in again] != ptrs:
        return False, "buffers were not reused across a scope boundary; the arena is not working"
    return True, f"{n} distinct within scope, reused across scopes"

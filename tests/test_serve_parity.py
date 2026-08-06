#!/usr/bin/env python3
"""The serving path must produce the SAME numbers without the training library.

WHY THIS FILE EXISTS, AND WHY IT DID NOT BEFORE

`runtime/block_heads.py` claimed its grid equivalence was "verified in tests/test_pdd_serve_parity.py
rather than asserted". That file did not exist. AUDIT.md then repeated the claim. So the one
equivalence the serving path rests on -- that folded heads reproduce a single scheduler step -- was
argued in a docstring and checked nowhere.

This is the gate for AUDIT.md Stage 1. `runtime/schedule.py` replaced an `instinct_pdd.Grid`, obtained
by building a PDD *training oracle* over a live server, with two lines of arithmetic over the
scheduler's own sigmas. That is only safe if the arithmetic is IDENTICAL, so this compares the new
runtime path against the retired Grid path directly and demands 0.00e+00 -- not "close".

The test imports `instinct_pdd` deliberately: it is the reference being compared against. What must
not import it is anything under `runtime/`, `planners/`, or `executors/`, and
`tests/test_runtime_boundary.py` enforces that separately.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from instinctwm.runtime.block_heads import fold_heads  # noqa: E402
from instinctwm.runtime.schedule import (  # noqa: E402
    block_start_timesteps,
    block_weights,
    conditioning_timesteps,
    interval_widths,
)

try:
    from instinct_pdd import Grid
    HAVE_PDD = True
except ImportError:  # pragma: no cover
    HAVE_PDD = False

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


#: The same fixture tests/test_pdd_parity.py uses: a descending schedule with the terminal 0 the
#: server pads on.
SIGMAS = [1.0, 0.82, 0.61, 0.44, 0.29, 0.17, 0.08, 0.03, 0.0]
SCALE = 1000.0
N = len(SIGMAS) - 1


def retired_grid(block: int):
    """Exactly what the removed code built: `t = 1 - sigma`, cond() still reporting `sigma * 1000`."""
    return Grid.from_times([1.0 - s for s in SIGMAS], block=block, scale=-SCALE, offset=SCALE)


def test_widths_and_conditioning():
    print("\n=== 1. the two derived quantities, against the retired Grid ===")
    if not HAVE_PDD:
        print("  NOT EVALUATED -- instinct_pdd unavailable, so there is no reference to compare "
              "against. This is the only test here that needs it.")
        return
    g = retired_grid(block=2)
    h_new = interval_widths(SIGMAS)
    dh = max(abs(h_new[l] - g.h(l)) for l in range(N))
    check(dh == 0.0, "interval widths are BIT-IDENTICAL to Grid.h(l)", f"max|delta| = {dh:.2e}")

    c_new = conditioning_timesteps(SIGMAS, SCALE)
    dc = max(abs(c_new[i] - g.cond(i)) for i in range(N + 1))
    check(dc == 0.0, "conditioning timesteps are BIT-IDENTICAL to Grid.cond(i)",
          f"max|delta| = {dc:.2e}")

    check(all(w > 0 for w in h_new),
          "widths are positive -- sigma descends, so a sign error would flip every head at once")


def test_block_weights_match_the_old_normalisation():
    print("\n=== 2. per-block fold weights ===")
    for block in (1, 2, 4, 8):
        n_blocks = N // block
        w_new = block_weights(SIGMAS, block, n_blocks, N)
        if HAVE_PDD:
            g = retired_grid(block)
            worst = 0.0
            for b in range(n_blocks):
                idx = list(range(b * block, min(b * block + block, N)))
                old = torch.tensor([g.h(l) for l in idx], dtype=torch.float64)
                old = old / old.sum()
                new = torch.tensor(w_new[b], dtype=torch.float64)
                worst = max(worst, float((old - new).abs().max()))
            check(worst == 0.0, f"L={block}: fold weights bit-identical to the retired path",
                  f"max|delta| = {worst:.2e}")
        s = [abs(sum(wb) - 1.0) for wb in w_new]
        check(max(s) < 1e-12, f"L={block}: every block's weights sum to 1", f"worst {max(s):.2e}")


def test_folded_map_equals_one_scheduler_step():
    print("\n=== 3. THE INVARIANT THE SERVING PATH RESTS ON ===")
    # sum_l h_l over a block IS sigma[n] - sigma[n+L], the dsigma the sampler applies. So the
    # normalised weighted mean of the heads, times that dsigma, equals the block step exactly. If this
    # fails, folding is not a valid substitution for L head evaluations.
    for block in (1, 2, 4):
        n_blocks = N // block
        w = block_weights(SIGMAS, block, n_blocks, N)
        h = interval_widths(SIGMAS)
        worst = 0.0
        for b in range(n_blocks):
            idx = list(range(b * block, min(b * block + block, N)))
            tot = sum(h[l] for l in idx)
            dsigma = SIGMAS[idx[0]] - SIGMAS[idx[-1] + 1]
            worst = max(worst, abs(tot - dsigma))
        check(worst < 1e-15, f"L={block}: sum of block widths == the sampler's dsigma",
              f"max|delta| = {worst:.2e}")
        check(all(abs(sum(wb) - 1.0) < 1e-15 for wb in w),
              f"L={block}: so the normalised weights are a convex combination")


def test_fold_heads_is_unchanged_by_the_refactor():
    print("\n=== 4. fold_heads: new signature vs the retired Grid signature ===")
    if not HAVE_PDD:
        print("  NOT EVALUATED -- no reference implementation available.")
        return
    torch.manual_seed(0)
    in_f, out_f, block = 6, 4, 2
    n_blocks = N // block
    sd = {}
    for l in range(N):
        sd[f"{l}.weight"] = torch.randn(out_f, in_f, dtype=torch.float32)
        sd[f"{l}.bias"] = torch.randn(out_f, dtype=torch.float32)
    template = torch.nn.Linear(in_f, out_f)

    # NEW path.
    got = fold_heads(sd, block_weights(SIGMAS, block, n_blocks, N), block, n_blocks, template)

    # RETIRED path, reproduced exactly: weights straight off Grid.h(l), same fp64 accumulation, same
    # store to template dtype. Comparing stored-to-stored is the only comparison that means anything.
    g = retired_grid(block)
    worst = 0.0
    for b in range(n_blocks):
        idx = list(range(b * block, min(b * block + block, N)))
        ww = torch.tensor([g.h(l) for l in idx], dtype=torch.float64)
        ww = ww / ww.sum()
        W = torch.zeros(out_f, in_f, dtype=torch.float64)
        B = torch.zeros(out_f, dtype=torch.float64)
        for wi, l in zip(ww, idx):
            W += sd[f"{l}.weight"].double() * wi
            B += sd[f"{l}.bias"].double() * wi
        ref_w = W.to(template.weight.dtype)
        ref_b = B.to(template.bias.dtype)
        worst = max(worst, float((got[b].weight - ref_w).abs().max()))
        worst = max(worst, float((got[b].bias - ref_b).abs().max()))
    check(worst == 0.0, "folded weights and biases BIT-IDENTICAL to the retired Grid path",
          f"max|delta| = {worst:.2e}")
    check(len(got) == n_blocks, f"one folded map per served step ({n_blocks})")
    check(all(not p.requires_grad for m in got for p in m.parameters()), "folded maps are frozen")


def test_scheduler_state_is_restored():
    print("\n=== 5. reading the schedule must not mutate the server's scheduler ===")

    class FakeScheduler:
        """set_timesteps mutates, as the real one does. Leaving it set to another grid would silently
        change what the next _infer does -- so the reader has to restore it."""
        num_train_timesteps = 1000.0

        def __init__(self):
            self.sigmas = torch.tensor([1.0, 0.5, 0.0])
            self.timesteps = self.sigmas * self.num_train_timesteps
            self.training = False

        def set_timesteps(self, n):
            self.sigmas = torch.linspace(1.0, 0.0, n + 1)
            self.timesteps = self.sigmas * self.num_train_timesteps

    from instinctwm.runtime.schedule import sigmas_from_scheduler
    sch = FakeScheduler()
    before = sch.sigmas.clone()
    out = sigmas_from_scheduler(sch, 8)
    check(torch.equal(sch.sigmas, before), "scheduler sigmas restored after reading",
          f"{list(before)} -> {list(sch.sigmas)}")
    check(len(out) == 9, "returns N+1 grid points", f"{len(out)}")
    check(abs(out[-1]) < 1e-12, "terminal sigma is 0, so the last interval is not lost")

    class NoTerminal(FakeScheduler):
        def set_timesteps(self, n):
            self.sigmas = torch.linspace(1.0, 0.1, n + 1)
    out2 = sigmas_from_scheduler(NoTerminal(), 4)
    check(abs(out2[-1]) < 1e-12 and len(out2) == 6,
          "a schedule that does not end at 0 gets the terminal point padded on", f"{len(out2)} pts")


def main() -> int:
    test_widths_and_conditioning()
    test_block_weights_match_the_old_normalisation()
    test_folded_map_equals_one_scheduler_step()
    test_fold_heads_is_unchanged_by_the_refactor()
    test_scheduler_state_is_restored()
    print("\n" + "=" * 72)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the serving schedule is bit-identical without the training library")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

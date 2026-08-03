#!/usr/bin/env python3
"""PDD core: the algorithm must be right, and the package must stay splittable.

Two independent claims.

  1. THE BOUNDARY. `instinctwm/train/pdd/**` is destined to become `instinct-pdd`, so it may not
     import from `instinctwm`. Checked by parsing every file's imports, not by reading the docstring
     and hoping. This is the test that makes the future split a directory move.

  2. THE ALGORITHM. Checked against closed forms rather than "the loss went down":
       * sampling a constant field reproduces the exact analytic endpoint, for ANY block size,
         which is what proves the block-step composition (Eq 10) is right;
       * a student initialised to the teacher's velocity has near-zero loss on a short interval;
       * the loss is zero exactly when the student already predicts the teacher's mean velocity;
       * gradients reach ONLY the supervised head -- the stop-gradient on X_k is load-bearing and
         its absence would be invisible in the loss value.

    python tests/test_pdd_core.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from instinctwm.train.pdd import (  # noqa: E402
    Grid, advance, block_sample, mean_velocity_euler, mean_velocity_midpoint, pdd_loss, shift_time,
)

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


# -- 1. the boundary ----------------------------------------------------------------------------

def test_pdd_package_does_not_import_instinctwm():
    print("\n=== 1. instinctwm/train/pdd is free of instinctwm imports (stays splittable) ===")
    pkg = ROOT / "instinctwm" / "train" / "pdd"
    files = sorted(pkg.rglob("*.py"))
    check(len(files) >= 4, f"found {len(files)} modules to check")
    offenders = []
    for f in files:
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                # Self-references inside the package are fine and become relative on split.
                if m.startswith("instinctwm") and not m.startswith("instinctwm.train.pdd"):
                    offenders.append(f"{f.relative_to(ROOT)}: {m}")
    check(not offenders, "no module imports instinctwm outside the package",
          "; ".join(offenders) if offenders else "")
    # ...and nothing LingBot-specific leaked in either.
    banned = ("lingbot", "wan_va", "robotwin", "action_mode", "cache_name")
    leaks = [f"{f.relative_to(ROOT)}:{w}" for f in files for w in banned
             if w in f.read_text().lower().replace("lingbot-va's", "").replace("lingbot-va", "")]
    check(not leaks, "no LingBot-specific identifiers in the algorithm", "; ".join(leaks))


# -- 2. the grid --------------------------------------------------------------------------------

def test_grid_matches_the_sampler_convention():
    print("\n=== 2. the shifted grid reproduces FlowMatchScheduler's warp ===")
    # LingBot-VA: sigma' = shift*sigma / (1 + (shift-1)*sigma). Ours takes s = 1/shift.
    for shift in (1.0, 3.0, 5.0):
        for u in (0.0, 0.25, 0.5, 1.0):
            theirs = shift * u / (1.0 + (shift - 1.0) * u)
            ours = shift_time(u, 1.0 / shift)
            if abs(theirs - ours) > 1e-9:
                check(False, f"shift={shift} u={u}", f"{ours} != {theirs}")
                return
    check(True, "shift_time(u, 1/shift) == shift*u/(1+(shift-1)u) across shifts and u")

    g = Grid.from_shift(n_intervals=8, block=4, shift=5.0, scale=1000.0)
    check(g.nfe == 2, "NFE = N/L", f"N={g.n_intervals} L={g.block} -> {g.nfe}")
    check(abs(g.times[0] - 1000.0) < 1e-6 and abs(g.times[-1]) < 1e-6,
          "grid runs from t_start*scale to 0", f"{g.times[0]:.1f} -> {g.times[-1]:.1f}")
    check(all(g.h(k) < 0 for k in range(g.n_intervals)), "intervals are negative (noise -> data)")
    try:
        Grid.from_shift(n_intervals=9, block=4)
        check(False, "a block that does not divide N is rejected")
    except ValueError as e:
        check("does not divide" in str(e), "a block that does not divide N is rejected", str(e)[:48])


# -- 3. the sampler -----------------------------------------------------------------------------

class ConstantField:
    """velocity = c everywhere. Exact endpoint: x0 + c * (t_end - t_start)."""

    def __init__(self, c):
        self.c = c

    def velocity(self, x, t, *, cond=None):
        return self.c.expand_as(x)


class HeadsFromField:
    """A 'student' that reports the true field as every head. Perfect on a constant field."""

    def __init__(self, field, n_heads):
        self.field, self.n_heads = field, n_heads

    def heads(self, x, t, *, cond=None):
        return torch.stack([self.field.velocity(x, t) for _ in range(self.n_heads)], dim=0)


def test_block_sampling_is_exact_on_a_constant_field():
    print("\n=== 3. block-step composition (Eq 10) is exact, for every block size ===")
    torch.manual_seed(0)
    c = torch.randn(3)
    x0 = torch.randn(5, 3)
    for block in (1, 2, 4, 8):
        g = Grid.from_shift(n_intervals=8, block=block, shift=3.0)
        out = block_sample(HeadsFromField(ConstantField(c), g.n_intervals), x0, g)
        want = x0 + c * (g.times[-1] - g.times[0])
        err = float((out - want).abs().max())
        check(err < 1e-5, f"block={block} (NFE={g.nfe}) reaches the analytic endpoint",
              f"max|err| = {err:.2e}")


# -- 4. the loss --------------------------------------------------------------------------------

class LinearODE:
    """velocity = a*x. Curved trajectory, so the mean velocity is not the endpoint velocity."""

    def __init__(self, a):
        self.a = a

    def velocity(self, x, t, *, cond=None):
        return self.a * x


class TrainableHeads:
    def __init__(self, n_heads, d):
        self.n_heads = n_heads
        self.w = torch.nn.Parameter(torch.zeros(n_heads, d))
        self.calls = 0

    def heads(self, x, t, *, cond=None):
        self.calls += 1
        return self.w.unsqueeze(1).expand(self.n_heads, *x.shape) + 0.0 * x

    def parameters(self):
        return [self.w]


def test_loss_is_zero_when_the_student_is_already_right():
    print("\n=== 4. the loss vanishes exactly when the student matches the teacher ===")
    torch.manual_seed(0)
    c = torch.randn(3)
    g = Grid.from_shift(n_intervals=4, block=2, shift=1.0)
    student = HeadsFromField(ConstantField(c), g.n_intervals)
    x_n = torch.randn(6, 3)
    val, m = pdd_loss(student, ConstantField(c), x_n, g, n=0, k=1)
    check(float(val) < 1e-10, "constant field: exact zero loss", f"loss = {float(val):.3e}")
    check(m["pdd/span"] == 1.0, "metrics record the supervised span")


def test_midpoint_beats_euler_on_a_curved_trajectory():
    print("\n=== 5. the midpoint solver is second order where Euler is first ===")
    torch.manual_seed(0)
    a = 0.9
    g = Grid.from_shift(n_intervals=2, block=1, shift=1.0, t_start=0.0, t_end=1.0)
    x = torch.randn(4, 3).abs() + 1.0
    # exact mean velocity of dx/dt = a x over [t0, t1] is x0*(exp(a h) - 1)/h
    h = g.h(0)
    exact = x * (torch.tensor(a * h).exp() - 1.0) / h
    e_eul = float((mean_velocity_euler(LinearODE(a), x, g, 0) - exact).abs().max())
    e_mid = float((mean_velocity_midpoint(LinearODE(a), x, g, 0) - exact).abs().max())
    check(e_mid < e_eul, "midpoint is closer to the exact mean velocity",
          f"euler {e_eul:.3e} vs midpoint {e_mid:.3e}")


def test_gradient_reaches_only_the_supervised_head():
    print("\n=== 6. stop-gradient on X_k: only head k receives gradient ===")
    torch.manual_seed(0)
    d = 3
    g = Grid.from_shift(n_intervals=4, block=4, shift=1.0)
    n_heads = g.n_intervals
    student = TrainableHeads(g.n_intervals, d)
    x_n = torch.randn(5, d)
    val, _ = pdd_loss(student, LinearODE(0.5), x_n, g, n=0, k=2)
    val.backward()
    gn = student.w.grad.abs().sum(dim=1)
    check(float(gn[2]) > 0, "supervised head 2 has gradient", f"{float(gn[2]):.3e}")
    others = [i for i in range(n_heads) if i != 2]
    check(all(float(gn[i]) == 0.0 for i in others),
          "heads used only to build X_k get no gradient",
          " ".join(f"h{i}={float(gn[i]):.1e}" for i in others))
    check(student.calls == 1, "exactly one student forward per step", f"calls={student.calls}")


def test_out_of_block_supervision_is_refused():
    print("\n=== 7. supervising an interval outside the block is an error ===")
    g = Grid.from_shift(n_intervals=8, block=2, shift=1.0)
    student = HeadsFromField(ConstantField(torch.randn(3)), g.n_intervals)
    try:
        pdd_loss(student, ConstantField(torch.randn(3)), torch.randn(4, 3), g, n=0, k=5)
        check(False, "k outside the block rejected")
    except ValueError as e:
        check("outside the block" in str(e), "k outside the block rejected", str(e)[:52])


if __name__ == "__main__":
    test_pdd_package_does_not_import_instinctwm()
    test_grid_matches_the_sampler_convention()
    test_block_sampling_is_exact_on_a_constant_field()
    test_loss_is_zero_when_the_student_is_already_right()
    test_midpoint_beats_euler_on_a_curved_trajectory()
    test_gradient_reaches_only_the_supervised_head()
    test_out_of_block_supervision_is_refused()
    print("\n" + "=" * 64)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        raise SystemExit(1)
    print("PASS: PDD core is correct and the package boundary holds")

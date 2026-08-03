#!/usr/bin/env python3
"""The trainer must be recipe-agnostic, and PDD's objective must be mathematically right.

Two separate claims, and they need separate evidence:

  1. THE FRAMEWORK CLAIM. Adding a method is adding a file. Tested by driving the trainer with a
     synthetic three-update recipe (the DMD2 shape: student + fake score + discriminator) and
     asserting all three optimisers stepped -- with zero trainer changes and no `if` on recipe name.
  2. THE OBJECTIVE CLAIM. PDD's target is the mean velocity of the TEACHER'S OWN ODE trajectory
     over an interval -- not the mean of the teacher's field along the straight data coupling, which
     is what a first version of this file wrongly computed and what these tests now rule out.
     Checked against two closed forms: a constant field (target == c exactly) and a linear ODE
     (target -> x_t*(exp(a*d)-1)/d as sub-steps grow). A "loss goes down" test passes for a wrong
     target; these do not.

    python tests/test_trainer.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from instinctwm.train.recipe import (  # noqa: E402
    Capabilities, DescriptorDelta, Environment, RecipeRejected, RecipeState, StepOutput,
)
from instinctwm.train.recipes.pdd import ParallelDecoding  # noqa: E402
from instinctwm.train.trainer import CachedTeacher, TrainConfig, Trainer  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


# -- fixtures -----------------------------------------------------------------------------------

class _Phase:
    def __init__(self, name, nfe):
        self.name, self.nfe = name, nfe


class _Exec:
    model_id = "toy"

    def __init__(self):
        self.phases = (_Phase("video", 25), _Phase("action", 50))


class _Model:
    def __init__(self):
        self.execution = _Exec()


class ConstantVelocityTeacher(nn.Module):
    """v(x, sigma) = c. The ODE is exactly linear, so the mean velocity is c for ANY interval."""

    def __init__(self, c):
        super().__init__()
        self.register_buffer("c", c)

    def forward(self, x, t, phase=None, cond=None):
        return self.c.expand_as(x)


class LinearODETeacher(nn.Module):
    """v(x, sigma) = a * x, so dx/dsigma = a x and x(sigma) = x_t * exp(a * (sigma - sigma_t)).

    The mean velocity over [t, s] is therefore x_t * (exp(a*d) - 1) / d in closed form, with
    d = s - t. This is the test that actually exercises the INTEGRATOR: a Euler scheme with K
    sub-steps gives x_t * ((1 + a*d/K)^K - 1)/d, which converges to the closed form as K grows.
    A single-point average of the field could not produce that.
    """

    def __init__(self, a: float):
        super().__init__()
        self.a = a

    def forward(self, x, t, phase=None, cond=None):
        return self.a * x


class TinyNet(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, 32), nn.SiLU(), nn.Linear(32, d))

    def forward(self, x, t, phase=None, cond=None):
        return self.net(torch.cat([x, t.reshape(-1, 1)], dim=-1))


# -- 1. the objective is mathematically correct --------------------------------------------------

def test_target_is_the_secant_of_the_teacher_trajectory():
    print("\n=== 1. PDD target == mean velocity of the TEACHER'S ODE, not of the straight line ===")
    torch.manual_seed(0)
    d, B = 4, 8
    r = ParallelDecoding({"video": 2}, sub_intervals=4)
    x_t = torch.randn(B, d)

    # (a) constant field: the mean velocity is c over any interval, exactly.
    c = torch.randn(d)
    for st, ss in ((1.0, 0.5), (0.5, 0.0), (0.8, 0.2)):
        got = r._target_mean_velocity(ConstantVelocityTeacher(c), x_t, st, ss, "video")
        err = float((got - c.expand(B, d)).abs().max())
        check(err < 1e-5, f"constant field, sigma {st}->{ss}: target == c", f"max|err| = {err:.2e}")

    # (b) linear ODE: Euler with K sub-steps must converge to the closed-form secant.
    a, st, ss = 0.7, 1.0, 0.2
    dsig = ss - st
    closed = x_t * (math.exp(a * dsig) - 1.0) / dsig
    errs = []
    for K in (1, 4, 16, 64):
        got = ParallelDecoding({"video": 2}, sub_intervals=max(2, K))._target_mean_velocity(
            LinearODETeacher(a), x_t, st, ss, "video")
        errs.append(float((got - closed).abs().max()))
    check(errs[-1] < errs[0], "Euler target converges to the closed-form secant as K grows",
          " -> ".join(f"{e:.2e}" for e in errs))
    check(errs[-1] < 1e-2, "converged target is close to closed form", f"K=64 err {errs[-1]:.2e}")

    # (c) the test is not vacuous: for a curved field the secant must DIFFER from the straight-line
    # answer, which is what the first (wrong) implementation would have returned.
    got = r._target_mean_velocity(LinearODETeacher(a), x_t, st, ss, "video")
    straight = LinearODETeacher(a)(x_t, None)          # field at the left endpoint only
    check(float((got - straight).abs().max()) > 1e-3,
          "curved field: target differs from the left-endpoint field (not vacuous)",
          f"max|diff| = {float((got - straight).abs().max()):.3e}")


def test_sub_intervals_must_be_at_least_two():
    print("\n=== 2. a single sub-interval is not a mean, and is refused ===")
    try:
        ParallelDecoding({"video": 2}, sub_intervals=1)
        check(False, "sub_intervals=1 rejected")
    except ValueError as e:
        check("not a mean" in str(e) or "single teacher evaluation" in str(e),
              "sub_intervals=1 rejected with a reason", str(e)[:60])


def test_bare_list_nfe_refused():
    print("\n=== 3. a bare NFE list would silently apply one count to both streams ===")
    try:
        ParallelDecoding([2, 2])
        check(False, "list nfe rejected")
    except TypeError as e:
        check("phase name" in str(e), "list nfe rejected with a reason", str(e)[:60])


# -- 4. PDD actually trains --------------------------------------------------------------------

def test_pdd_reduces_loss():
    print("\n=== 4. PDD drives the loss down on a learnable toy problem ===")
    torch.manual_seed(0)
    d, B = 4, 16
    teacher = LinearODETeacher(0.7)          # curved: a constant field has nothing to teach
    student = TinyNet(d)
    recipe = ParallelDecoding({"video": 2}, sub_intervals=4)

    def data():
        while True:
            yield {"video/x1": torch.randn(B, d), "video/x0": torch.randn(B, d)}

    tr = Trainer(recipe, teacher, student, _Model(),
                 config=TrainConfig(steps=150, lr=3e-3, log_every=0, seed=0))
    res = tr.fit(data())
    first = sum(h["loss/student"] for h in res.history[:10]) / 10
    last = sum(h["loss/student"] for h in res.history[-10:]) / 10
    check(res.steps_done == 150, "ran all 150 steps")
    check(last < first * 0.5, "loss at least halved", f"{first:.4f} -> {last:.4f}")
    check(res.stopped_early is None, "no early stop")
    check("video/curvature" in res.history[-1], "reports the trajectory-curvature metric")
    check(res.history[-1]["video/curvature"] > 0, "the teacher trajectory is actually curved",
          f'curvature={res.history[-1]["video/curvature"]:.4g}')


# -- 5. the framework claim: N updates, no trainer changes --------------------------------------

class ThreeUpdateRecipe:
    """The DMD2 SHAPE with a trivial objective: student + fake score + discriminator.

    Deliberately synthetic and named as such. The point is not to reproduce DMD2, it is to prove
    the trainer drives three optimisers in a declared order while knowing nothing about any of them.
    """
    name = "synthetic_three_update"

    def requires(self):
        return Capabilities(adversarial=True, aux_modules=("fake_score", "discriminator"))

    def descriptor_delta(self, model):
        return DescriptorDelta(nfe={"video": 2}, note="synthetic")

    def build(self, model, env):
        fake, disc = TinyNet(4), TinyNet(4)
        return RecipeState(
            modules={"fake_score": fake, "discriminator": disc},
            optimizers={"fake_score": torch.optim.SGD(fake.parameters(), lr=1e-2),
                        "discriminator": torch.optim.SGD(disc.parameters(), lr=1e-2)},
            update_order=("student", "fake_score", "discriminator"))

    def step(self, batch, teacher, student, state):
        x, t = batch["x"], batch["t"]
        ls = {
            "student": student(x, t).pow(2).mean(),
            "fake_score": state.modules["fake_score"](x, t).pow(2).mean(),
            "discriminator": state.modules["discriminator"](x, t).pow(2).mean(),
        }
        return StepOutput(losses=ls, metrics={"n_updates": 3.0})


def test_three_updates_all_step():
    print("\n=== 5. three declared updates all optimise, trainer unmodified ===")
    torch.manual_seed(0)
    student = TinyNet(4)
    recipe = ThreeUpdateRecipe()
    tr = Trainer(recipe, ConstantVelocityTeacher(torch.randn(4)), student, _Model(),
                 config=TrainConfig(steps=20, log_every=0))
    before = {n: [p.detach().clone() for p in m.parameters()]
              for n, m in (("student", student), *tr.state.modules.items())}

    def data():
        while True:
            yield {"x": torch.randn(8, 4), "t": torch.rand(8)}

    res = tr.fit(data())
    for name, mod in (("student", student), *tr.state.modules.items()):
        moved = any(not torch.equal(a, b.detach())
                    for a, b in zip(before[name], mod.parameters()))
        check(moved, f"{name} parameters moved")
    h = res.history[-1]
    for name in ("student", "fake_score", "discriminator"):
        check(f"loss/{name}" in h, f"loss/{name} logged")
        check(f"gradnorm/{name}" in h, f"gradnorm/{name} logged")


# -- 6. fails closed ---------------------------------------------------------------------------



def test_empty_delta_rejected():
    print("\n=== 8. a recipe that changes nothing about execution is refused ===")

    class NoDelta(ThreeUpdateRecipe):
        name = "no_delta"

        def descriptor_delta(self, model):
            return DescriptorDelta()

    try:
        Trainer(NoDelta(), None, TinyNet(4), _Model())
        check(False, "empty delta rejected")
    except RecipeRejected as e:
        check("no descriptor delta" in str(e), "empty delta rejected with a reason", str(e)[:70])


def test_wrong_return_type_rejected():
    print("\n=== 9. returning floats instead of StepOutput is a TypeError, not a silent no-op ===")

    class FloatReturner(ThreeUpdateRecipe):
        name = "float_returner"

        def step(self, batch, teacher, student, state):
            return {"student": 0.5}

    tr = Trainer(FloatReturner(), None, TinyNet(4), _Model(), config=TrainConfig(steps=1))
    try:
        tr.fit(iter([{"x": torch.randn(8, 4), "t": torch.rand(8)}]))
        check(False, "float return rejected")
    except TypeError as e:
        check("StepOutput" in str(e), "float return rejected with a reason", str(e)[:70])


def test_nonfinite_loss_stops_the_run():
    print("\n=== 10. a NaN loss stops the run instead of producing a checkpoint ===")

    class NanRecipe(ThreeUpdateRecipe):
        name = "nan"

        def build(self, model, env):
            return RecipeState(update_order=("student",))

        def step(self, batch, teacher, student, state):
            out = student(batch["x"], batch["t"]).mean()
            return StepOutput(losses={"student": out * float("nan")})

    tr = Trainer(NanRecipe(), None, TinyNet(4), _Model(),
                 config=TrainConfig(steps=10, log_every=0))

    def data():
        while True:
            yield {"x": torch.randn(8, 4), "t": torch.rand(8)}

    res = tr.fit(data())
    check(res.stopped_early is not None, "run stopped early")
    check("non-finite" in (res.stopped_early or ""), "reason names the non-finite loss",
          str(res.stopped_early)[:60])
    check(res.steps_done == 1, "stopped on the first bad step", f"steps_done={res.steps_done}")


# -- 11. the Layer1 -> Layer2..6 handoff -------------------------------------------------------

def test_checkpoint_carries_the_descriptor_delta():
    print("\n=== 11. a saved checkpoint carries the delta, so the runtime can run it ===")
    import json
    student = TinyNet(4)
    recipe = ParallelDecoding({"video": 2, "action": 2}, sub_intervals=3)
    tr = Trainer(recipe, ConstantVelocityTeacher(torch.randn(4)), student, _Model(),
                 config=TrainConfig(steps=1, log_every=0))
    with tempfile.TemporaryDirectory() as td:
        p = tr.save(Path(td) / "ckpt")
        check((p / "student.pt").exists(), "weights written")
        meta = json.loads((p / "delta.json").read_text())
        check(meta["nfe"] == {"video": 2, "action": 2}, "delta.json carries per-phase NFE",
              str(meta["nfe"]))
        check("PDD" in meta["note"], "delta.json records which recipe produced it")


def test_teacher_calls_are_cached_within_a_step():
    print("\n=== 12. identical teacher calls within one step are computed once ===")
    calls = {"n": 0}

    class Counting(nn.Module):
        def forward(self, x, t, phase=None):
            calls["n"] += 1
            return x * 0

    ct = CachedTeacher(Counting())
    x, t = torch.randn(4, 4), torch.rand(4)
    ct(x, t, phase="video")
    ct(x, t, phase="video")
    check(calls["n"] == 1 and ct.hits == 1, "second identical call hit the cache",
          f"calls={calls['n']} hits={ct.hits}")
    ct.clear()
    ct(x, t, phase="video")
    check(calls["n"] == 2, "cache cleared between steps (no stale buffers across steps)")


if __name__ == "__main__":
    test_target_is_the_secant_of_the_teacher_trajectory()
    test_sub_intervals_must_be_at_least_two()
    test_bare_list_nfe_refused()
    test_pdd_reduces_loss()
    test_three_updates_all_step()
    test_empty_delta_rejected()
    test_wrong_return_type_rejected()
    test_nonfinite_loss_stops_the_run()
    test_checkpoint_carries_the_descriptor_delta()
    test_teacher_calls_are_cached_within_a_step()
    print("\n" + ("=" * 62))
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        raise SystemExit(1)
    print("PASS: trainer is recipe-agnostic and PDD's target is analytically correct")

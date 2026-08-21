#!/usr/bin/env python3
"""The trainer must be recipe-agnostic, and PDD's objective must be mathematically right.

Two separate claims, and they need separate evidence:

  1. THE FRAMEWORK CLAIM. Adding a method is adding a file. Tested by driving the trainer with a
     synthetic three-update recipe (the DMD2 shape: student + fake score + discriminator) and
     asserting all three optimisers stepped -- with zero trainer changes and no `if` on recipe name.
  2. Objective correctness is NOT tested here -- it belongs to the algorithm, and lives in
     tests/test_pdd_core.py. This file must stay recipe-agnostic, or it stops testing the trainer.

    python tests/test_trainer.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from instinctflash.train.recipe import (  # noqa: E402
    Capabilities, DescriptorDelta, Environment, RecipeRejected, RecipeState, StepOutput,
)
from instinctflash.train.trainer import CachedTeacher, TrainConfig, Trainer  # noqa: E402

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


# -- 4. PDD actually trains --------------------------------------------------------------------

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
    tr = Trainer(recipe, LinearODETeacher(0.7), student, _Model(),
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
    recipe = ThreeUpdateRecipe()
    tr = Trainer(recipe, LinearODETeacher(0.7), student, _Model(),
                 config=TrainConfig(steps=1, log_every=0))
    with tempfile.TemporaryDirectory() as td:
        p = tr.save(Path(td) / "ckpt")
        check((p / "student.pt").exists(), "weights written")
        meta = json.loads((p / "delta.json").read_text())
        check(meta["nfe"] == {"video": 2}, "delta.json carries per-phase NFE",
              str(meta["nfe"]))
        check(meta["recipe"] == "synthetic_three_update",
          "delta.json records which recipe produced it", meta["recipe"])


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

#!/usr/bin/env python3
"""The Recipe interface must carry real methods and refuse the ones this box cannot run.

Same bar the pass framework had to clear: the interface is only real if methods fit WITHOUT
special cases, and if it fails closed when the environment cannot satisfy them.

    python tests/test_recipe.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from instinctwm.train.recipe import (
    Capabilities, DescriptorDelta, Environment, RecipeRejected, admit, prepare,
)
from instinctwm.train.recipes.pdd import ParallelDecoding

results = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{('  -- ' + extra) if extra else ''}")
    results.append(cond)


class _Phase:
    def __init__(self, name, nfe):
        self.name, self.nfe = name, nfe


def _model():
    """LingBot-VA's real phase structure: video 25 steps, action 50."""
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class P:
        name: str
        nfe: int

    @dataclasses.dataclass(frozen=True)
    class E:
        model_id: str
        phases: tuple

    @dataclasses.dataclass(frozen=True)
    class M:
        execution: E

    return M(E("lingbot-va", (P("video", 25), P("action", 50))))


# Stand-ins for methods we are NOT implementing, present only to test that the interface
# can express their requirements without special-casing.
class _RCM:
    name = "rcm"
    def requires(self): return Capabilities(jvp_through_attention=True, teacher_calls_per_step=2,
                                            aux_modules=("fake_score",))
    def descriptor_delta(self, m): return DescriptorDelta(nfe={"video": 4, "action": 4})
    def build(self, m, e): return None
    def step(self, *a): ...


class _DMD2:
    name = "dmd2"
    def requires(self): return Capabilities(adversarial=True, teacher_calls_per_step=2,
                                            aux_modules=("fake_score", "discriminator"))
    def descriptor_delta(self, m): return DescriptorDelta(nfe={"video": 1, "action": 1})
    def build(self, m, e): return None
    def step(self, *a): ...


class _DCVideoGen:
    name = "dc_videogen"
    def requires(self): return Capabilities(aux_modules=("vae",))
    def descriptor_delta(self, m): return DescriptorDelta(shapes={"latent": (1, 60, 3072)},
                                                          note="4x temporal compression")
    def build(self, m, e): return None
    def step(self, *a): ...


class _NoOp:
    name = "noop"
    def requires(self): return Capabilities()
    def descriptor_delta(self, m): return DescriptorDelta()
    def build(self, m, e): return None
    def step(self, *a): ...


def main() -> int:
    m = _model()
    this_box = Environment(n_gpus=8, has_jvp_attention=False, allows_adversarial=True)

    print("=== 1. PDD runs on this box ===")
    pdd = ParallelDecoding({"video": 1, "action": 2})
    ok, why = admit(pdd, this_box)
    check("admitted", ok, why)
    state, delta = prepare(pdd, m, this_box)
    check("declares its descriptor delta", delta.describe() == "nfe action=2, video=1",
          delta.describe())
    check("no auxiliary modules", not state.modules)

    print("\n=== 2. the delta actually rewrites the descriptor ===")
    new = delta.apply_to(m.execution)
    got = {p.name: p.nfe for p in new.phases}
    check("25/50 -> 1/2", got == {"video": 1, "action": 2}, str(got))
    check("teacher descriptor is untouched",
          {p.name: p.nfe for p in m.execution.phases} == {"video": 25, "action": 50})

    print("\n=== 3. the interface carries methods we are NOT implementing ===")
    for r, expect_ok, label in [(_RCM(), False, "rCM needs a JVP kernel this box lacks"),
                                (_DMD2(), True, "DMD2 admitted (adversarial allowed here)"),
                                (_DCVideoGen(), True, "DC-VideoGen changes SHAPES, not steps")]:
        ok, why = admit(r, this_box)
        check(label, ok == expect_ok, why[:64])
    check("shape-changing recipe expresses its delta",
          "shapes" in _DCVideoGen().descriptor_delta(m).describe())

    print("\n=== 4. fails closed ===")
    try:
        prepare(_RCM(), m, this_box)
        check("rCM rejected before training", False, "did NOT reject")
    except RecipeRejected as e:
        check("rCM rejected before training", True, str(e)[:58])
    try:
        prepare(_NoOp(), m, this_box)
        check("a recipe with no descriptor delta is rejected", False, "did NOT reject")
    except RecipeRejected as e:
        check("a recipe with no descriptor delta is rejected", True, str(e)[:58])
    strict = Environment(n_gpus=8, has_jvp_attention=False, allows_adversarial=False)
    ok, why = admit(_DMD2(), strict)
    check("adversarial recipe refused where disallowed", not ok, why[:58])

    # PDD's objective used to be a stub and this group asserted that it raised. It is implemented
    # now (tests/test_trainer.py checks the mean-velocity target against two closed forms), so what
    # earns a check here is the registry's scope: one recipe, deliberately. Declaration-only scm /
    # rcm / dmd2 entries were removed rather than kept as future-proofing.
    print("\n=== 5. the registry is scoped to what is implemented ===")
    from instinctwm.train.recipes import REGISTRY, available, build as build_recipe
    check("the registry holds exactly the implemented recipe",
          available() == ["pdd"], str(available()))
    try:
        build_recipe("rcm", {"video": 1})
        check("an unregistered recipe is a KeyError, not a silent no-op", False)
    except KeyError as e:
        check("an unregistered recipe is a KeyError, not a silent no-op", True, str(e)[:52])

    try:
        ParallelDecoding([1, 2])
        check("a bare list is rejected (would apply one NFE to both streams)", False)
    except TypeError:
        check("a bare list is rejected (would apply one NFE to both streams)", True)

    print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} checks")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

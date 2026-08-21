#!/usr/bin/env python3
"""PDD as an InstinctFlash recipe: composition, not algorithm.

test_pdd_core.py proves the algorithm. This file proves the ORCHESTRATION around it:

  * the paper's primary configuration is what you get by default (N=256, Euler, MSE), and the
    2-NFE target yields L=128;
  * the two streams are driven on DIFFERENT grids, because LingBot-VA serves video at snr_shift 5.0
    and action at 1.0 -- a single shared grid would train the student to jump between times the
    sampler never visits, and would still look fine in the loss curve;
  * ONE student forward per stream per step. The data-free rollout must reuse the same head output
    for the loss and for advancing the trajectory; recomputing doubles the dominant cost of a step
    on a 14B backbone and nothing else would notice;
  * the rollout advances, wraps, and restarts;
  * a checkpoint carries the per-phase NFE the runtime needs.

    python tests/test_pdd_recipe.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from instinctflash.train.recipe import Environment, prepare  # noqa: E402
from instinctflash.train.recipes.pdd import ParallelDecoding  # noqa: E402
from instinctflash.train.trainer import TrainConfig, Trainer  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


class _P:
    def __init__(self, name, nfe):
        self.name, self.nfe = name, nfe


class _E:
    model_id = "lingbot-va"

    def __init__(self):
        self.phases = (_P("video", 25), _P("action", 50))


class _M:
    execution = _E()


class ToyTeacher:
    """Counts calls so teacher cost per step is observable."""

    def __init__(self, a=0.3):
        self.a, self.calls = a, 0

    def velocity(self, x, t, *, cond=None):
        self.calls += 1
        return self.a * x


class ToyStudent(nn.Module):
    """Multi-head student that can actually represent the teacher: head k predicts a_k * x.

    Made x-dependent on purpose. An earlier version returned a bare per-head vector, ignoring the
    state, and in data-free mode that produced a runaway: the heads set the trajectory, the
    trajectory sets the target, and a state-blind student chases its own tail. That is a real
    property of on-policy training rather than a quirk of the toy, and it is why the paper
    initialises every head from the teacher's own final layer -- `init_a` reproduces that here.
    """

    def __init__(self, n_heads, d, init_a=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.a = nn.Parameter(torch.full((n_heads,), float(init_a)))
        self.calls = 0

    def heads(self, x, t, *, cond=None):
        self.calls += 1
        return self.a.view(-1, *([1] * x.dim())) * x.unsqueeze(0)


def _models(n_heads, d, init_a=0.0):
    return ({"video": ToyStudent(n_heads, d, init_a), "action": ToyStudent(n_heads, d, init_a)},
            {"video": ToyTeacher(0.3), "action": ToyTeacher(0.7)})


class _Wrap(nn.Module):
    def __init__(self, mods):
        super().__init__()
        self.mods = nn.ModuleDict(mods)

    def __getitem__(self, k):
        return self.mods[k]


def test_paper_defaults():
    print("\n=== 1. the defaults are the paper's primary configuration ===")
    r = ParallelDecoding({"video": 2, "action": 2})
    check(r.n_intervals == 256, "N = 256", str(r.n_intervals))
    check(r.solver == "euler", "Euler solver", r.solver)
    check(r.loss == "mse", "MSE, unweighted", r.loss)
    check(r.data_free is True, "data-free training (the paper's video setting)")
    st = r.build(_M(), Environment())
    for ph in ("video", "action"):
        g = st.extra["grids"][ph]
        check(g.block == 128 and g.nfe == 2, f"{ph}: 2-NFE target gives L=128",
              f"N={g.n_intervals} L={g.block}")
    check(r.requires().jvp_through_attention is False, "no JVP required")
    check(r.requires().adversarial is False, "no adversary required")


def test_streams_get_different_grids():
    print("\n=== 2. the two streams are driven on different SNR-shifted grids ===")
    st = ParallelDecoding({"video": 2, "action": 2}).build(_M(), Environment())
    gv, ga = st.extra["grids"]["video"], st.extra["grids"]["action"]
    check(gv.times != ga.times, "video and action grids differ")
    # action shift = 1.0 is the identity warp, so its grid must be exactly linear in t.
    # instinct-pdd's axis ascends: with shift=1 the identity warp leaves t = i/N.
    lin = [i / 256 for i in range(257)]
    check(max(abs(a - b) for a, b in zip(ga.times, lin)) < 1e-6,
          "action (shift 1.0) is the linear grid in t", f"t1={ga.times[1]:.5f}")
    # DIRECTION MATTERS, and the sign of the comparison depends on which axis you read it in.
    # shift > 1 warps SIGMA upward, holding the trajectory at high noise for more of the grid
    # (FlowMatchScheduler at N=25, shift=5: sigma_1 = 0.99174 against a linear 0.96). Under
    # t = 1 - sigma that same fact reads as a SHORTER first interval in t -- which is what
    # "steps concentrated at the noise end" means on this axis.
    check(gv.times[1] < ga.times[1] - 1e-4,
          "video (shift 5.0) takes a shorter first step in t (steps at the noise end)",
          f"video t1={gv.times[1]:.5f} vs action {ga.times[1]:.5f}")
    check(abs(gv.cond(0) - 1000.0) < 1e-6,
          "but both still condition the backbone on sigma * 1000", f"cond0={gv.cond(0):.1f}")


def test_one_student_forward_per_stream_per_step():
    print("\n=== 3. exactly one student forward per stream per step (data-free) ===")
    torch.manual_seed(0)
    d, N = 4, 8
    students, teachers = _models(N, d)
    r = ParallelDecoding({"video": 2, "action": 2}, n_intervals=N)
    st = r.build(_M(), Environment())
    batch = {"video/noise_like": torch.randn(3, d), "action/noise_like": torch.randn(3, d)}
    out = r.step(batch, teachers, students, st)
    check(students["video"].calls == 1, "one video student forward",
          f"calls={students['video'].calls}")
    check(students["action"].calls == 1, "one action student forward",
          f"calls={students['action'].calls}")
    check(teachers["video"].calls == 1, "one video teacher call (Euler)",
          f"calls={teachers['video'].calls}")
    check("student" in out.losses, "one fused loss for the student update")
    check(out.losses["student"].requires_grad, "loss carries grad")


def test_midpoint_doubles_teacher_cost():
    print("\n=== 4. the midpoint solver costs two teacher calls, as declared ===")
    torch.manual_seed(0)
    d, N = 4, 8
    students, teachers = _models(N, d)
    r = ParallelDecoding({"video": 2}, n_intervals=N, solver="midpoint")
    st = r.build(_M(), Environment())
    r.step({"video/noise_like": torch.randn(3, d)}, teachers, students, st)
    check(teachers["video"].calls == 2, "two teacher calls per supervised interval",
          f"calls={teachers['video'].calls}")
    check(r.requires().teacher_calls_per_step == 2, "and requires() declares it",
          str(r.requires().teacher_calls_per_step))


def test_rollout_advances_and_restarts():
    print("\n=== 5. the data-free rollout walks the grid and restarts ===")
    torch.manual_seed(0)
    d, N = 4, 8                                  # NFE=2 -> L=4, so 2 blocks per trajectory
    students, teachers = _models(N, d)
    r = ParallelDecoding({"video": 2}, n_intervals=N)
    st = r.build(_M(), Environment())
    batch = {"video/noise_like": torch.randn(3, d)}
    seen_n = []
    for _ in range(5):
        out = r.step(batch, teachers, students, st)
        seen_n.append(int(out.metrics["video/n"]))     # the block actually trained on
    roll = st.extra["rollouts"]["video"]
    check(seen_n == [0, 4, 0, 4, 0], "block index cycles 0 -> L -> restart", str(seen_n))
    check(max(seen_n) < N, "the pointer never supervises past the end of the grid")
    check(roll.trajectories >= 2, "trajectory restarted after N/L blocks",
          f"trajectories={roll.trajectories}")
    check(roll.blocks == 5, "one block advanced per step", f"blocks={roll.blocks}")


def test_data_based_variant_uses_the_interpolant():
    print("\n=== 6. the data-based variant (Alg 2) draws X_n from the interpolant ===")
    torch.manual_seed(0)
    d, N = 4, 8
    students, teachers = _models(N, d)
    r = ParallelDecoding({"video": 2}, n_intervals=N, data_free=False)
    st = r.build(_M(), Environment())
    batch = {"video/x1": torch.randn(3, d), "video/x0": torch.randn(3, d)}
    out = r.step(batch, teachers, students, st)
    check(out.losses["student"].requires_grad, "produces a differentiable loss")
    check(not st.extra["rollouts"], "no rollout state is created in the data-based variant")


def test_it_trains_through_the_platform_trainer():
    print("\n=== 7. the recipe drives the real Trainer end to end (data-based) ===")
    torch.manual_seed(0)
    d, N = 4, 8
    students, teachers = _models(N, d)
    r = ParallelDecoding({"video": 2}, n_intervals=N, data_free=False)
    tr = Trainer(r, teachers, _Wrap({"video": students["video"]}), _M(),
                 config=TrainConfig(steps=60, lr=5e-2, log_every=0))

    def data():
        while True:
            yield {"video/x1": torch.randn(3, d), "video/x0": torch.randn(3, d)}

    res = tr.fit(data())
    check(res.steps_done == 60, "ran 60 steps")
    check(res.stopped_early is None, "no NaN stop")
    first = sum(h["loss/student"] for h in res.history[:10]) / 10
    last = sum(h["loss/student"] for h in res.history[-10:]) / 10
    check(last < first * 0.5, "loss at least halved", f"{first:.4f} -> {last:.4f}")
    learned = students["video"].a.detach()
    check(float(learned.abs().mean()) > 0.05,
          "heads moved toward the teacher's velocity scale", f"mean|a| = {float(learned.abs().mean()):.3f}")


def test_data_free_is_stable_from_a_teacher_initialised_student():
    print("\n=== 7b. data-free on-policy training is stable when heads start at the teacher ===")
    torch.manual_seed(0)
    d, N = 4, 8
    # The paper initialises each of the N final layers from the teacher's single final layer, so the
    # student begins by predicting very nearly the teacher's velocity and the rollout is sane from
    # step one. Reproduced here by starting every head at the teacher's coefficient.
    students, teachers = _models(N, d, init_a=0.3)
    r = ParallelDecoding({"video": 2}, n_intervals=N)
    tr = Trainer(r, teachers, _Wrap({"video": students["video"]}), _M(),
                 config=TrainConfig(steps=40, lr=1e-3, log_every=0))

    def data():
        while True:
            yield {"video/noise_like": torch.randn(3, d)}

    res = tr.fit(data())
    losses = [h["loss/student"] for h in res.history]
    check(res.stopped_early is None, "no divergence to NaN")
    check(max(losses) < 1.0, "loss stays bounded across the rollout",
          f"max = {max(losses):.4f}")
    check(losses[0] < 0.05, "teacher-initialised heads start near zero loss",
          f"first = {losses[0]:.5f}")
    check("video/traj" in res.history[-1], "rollout progress is reported")


def test_checkpoint_carries_per_phase_nfe():
    print("\n=== 8. the checkpoint tells the runtime how to run the student ===")
    torch.manual_seed(0)
    d, N = 4, 8
    students, teachers = _models(N, d)
    r = ParallelDecoding({"video": 2, "action": 2}, n_intervals=N)

    tr = Trainer(r, teachers, _Wrap(students), _M(), config=TrainConfig(steps=1, log_every=0))
    with tempfile.TemporaryDirectory() as td:
        p = tr.save(Path(td) / "ckpt")
        meta = json.loads((p / "delta.json").read_text())
        check(meta["nfe"] == {"video": 2, "action": 2}, "per-phase NFE recorded", str(meta["nfe"]))
        check("guidance distilled in" in meta["note"],
              "the note records that guidance is baked into the weights", meta["note"][:70])


def test_guardrails():
    print("\n=== 9. the configurations that would be silently wrong are refused ===")
    try:
        ParallelDecoding([2, 2])
        check(False, "bare list rejected")
    except TypeError as e:
        check("phase name" in str(e), "bare NFE list rejected", str(e)[:56])
    try:
        ParallelDecoding({"video": 3}, n_intervals=256)
        check(False, "NFE not dividing N rejected")
    except ValueError as e:
        check("does not divide" in str(e), "NFE that does not divide N rejected", str(e)[:56])
    try:
        ParallelDecoding({"depth": 2}, shifts={"depth": 1.0}).descriptor_delta(_M())
        check(False, "unknown phase rejected")
    except ValueError as e:
        check("does not" in str(e), "unknown phase name rejected", str(e)[:56])
    try:
        ParallelDecoding({"video": 2}, shifts={}).build(_M(), Environment())
        check(False, "missing shift rejected")
    except ValueError as e:
        check("SNR shift" in str(e), "a phase with no declared SNR shift is refused", str(e)[:56])
    ok, why = None, None
    try:
        st, delta = prepare(ParallelDecoding({"video": 2}), _M(), Environment())
        ok = True
    except Exception as e:
        ok, why = False, str(e)
    check(ok, "prepare() admits PDD on this box (no flash-attn needed)", why or "")


if __name__ == "__main__":
    test_paper_defaults()
    test_streams_get_different_grids()
    test_one_student_forward_per_stream_per_step()
    test_midpoint_doubles_teacher_cost()
    test_rollout_advances_and_restarts()
    test_data_based_variant_uses_the_interpolant()
    test_it_trains_through_the_platform_trainer()
    test_data_free_is_stable_from_a_teacher_initialised_student()
    test_checkpoint_carries_per_phase_nfe()
    test_guardrails()
    print("\n" + "=" * 64)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        raise SystemExit(1)
    print("PASS: the PDD recipe composes the core correctly at the paper's settings")

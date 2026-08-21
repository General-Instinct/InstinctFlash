"""The training loop the PLATFORM owns.

The contract this file has to keep is one line long:

    nothing in here may branch on which recipe is running.

No `if recipe.name == ...`, no isinstance checks, no per-method special cases. Every axis a method
could vary along is a declaration the recipe makes and this loop reads:

    Capabilities        -> checked once, before any GPU work, and it fails closed
    RecipeState.modules -> what else is trainable besides the student
    RecipeState.optimizers / update_order -> how many updates per step, and in what order
    StepOutput.losses   -> one scalar per update; the trainer does backward/clip/step/zero

That is what makes "adding a paper is adding a file" true rather than aspirational. If a future
recipe cannot be expressed here, the fix is a new declaration, not an `if`.

WHAT IS DELIBERATELY NOT HERE: any notion of diffusion, velocity, noise schedule, consistency or
guidance. The trainer never learns what a timestep is. It moves batches, drives optimizers, caches
teacher calls and writes checkpoints.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from instinctflash.train.recipe import (
    Environment, Recipe, RecipeRejected, RecipeState, StepOutput, prepare,
)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    lr: float = 1e-5
    grad_clip: float | None = 1.0
    log_every: int = 10
    ckpt_every: int = 0                  # 0 disables intermediate checkpoints
    out_dir: str | None = None
    seed: int = 0
    #: Fail the run if the loss is not finite. On by default: a NaN that is logged and trained
    #: through produces a checkpoint that certifies as a catastrophic regression hours later, and
    #: the certificate cannot tell you it was NaN rather than a bad objective.
    stop_on_nonfinite: bool = True


@dataclass
class TrainResult:
    steps_done: int
    history: list[dict[str, float]] = field(default_factory=list)
    stopped_early: str | None = None
    seconds: float = 0.0

    def final(self) -> dict[str, float]:
        return dict(self.history[-1]) if self.history else {}

    def summary(self) -> str:
        out = [f"steps {self.steps_done}, {self.seconds:.1f}s"]
        if self.stopped_early:
            out.append(f"STOPPED EARLY: {self.stopped_early}")
        for k, v in sorted(self.final().items()):
            if k != "step":
                out.append(f"  {k}: {v:.6g}")
        return "\n".join(out)


class CachedTeacher:
    """Wraps the teacher so identical calls within one step are computed once.

    `Capabilities.teacher_calls_per_step` exists so this is plannable rather than discovered. The
    cache is cleared every step on purpose -- a teacher call is keyed on tensor *identity*, not
    value, so holding entries across steps would be a correctness trap the moment a recipe reuses a
    buffer. Cheap wins only.
    """

    def __init__(self, teacher):
        self._teacher = teacher
        self._cache: dict[Any, Any] = {}
        self._children: dict[Any, "CachedTeacher"] = {}
        self.calls = 0
        self.hits = 0

    def __getitem__(self, key):
        """Index through to a per-stream teacher, keeping each one cached.

        A multi-stream recipe is handed a mapping of oracles, one per phase. `__getattr__` forwards
        attribute access but NOT subscripting -- Python looks `__getitem__` up on the type, so it
        never reaches the delegate -- which made wrapping a dict of teachers fail with "not
        subscriptable" the moment a second stream existed. Wrapping each child rather than returning
        it bare keeps per-stream caching, and `clear()` below resets the whole tree together so no
        child can outlive a step.
        """
        if key not in self._children:
            self._children[key] = CachedTeacher(self._teacher[key])
        return self._children[key]

    def __call__(self, *args, **kwargs):
        key = (tuple(id(a) for a in args), tuple(sorted((k, id(v)) for k, v in kwargs.items())))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        out = self._teacher(*args, **kwargs)
        self._cache[key] = out
        return out

    def clear(self) -> None:
        self._cache.clear()
        for child in self._children.values():
            child.clear()

    @property
    def total_calls(self) -> int:
        return self.calls + sum(c.total_calls for c in self._children.values())

    @property
    def total_hits(self) -> int:
        return self.hits + sum(c.total_hits for c in self._children.values())

    def __getattr__(self, name):           # delegate anything else to the real teacher
        return getattr(self._teacher, name)


class Trainer:
    def __init__(self, recipe: Recipe, teacher, student, model, *,
                 env: Environment | None = None, config: TrainConfig | None = None):
        self.recipe = recipe
        self.teacher = teacher
        self.student = student
        self.model = model
        self.env = env or Environment()
        self.config = config or TrainConfig()
        # prepare() runs the capability check and rejects BEFORE anything is allocated.
        self.state, self.delta = prepare(recipe, model, self.env)
        self._ensure_student_optimizer()

    def _ensure_student_optimizer(self) -> None:
        """Give "student" an optimizer if the recipe did not build one.

        A recipe that trains only the student should not have to write optimizer boilerplate, but a
        recipe that needs a specific optimizer (DMD2's two-time-scale rates) must be able to say so
        and be left alone. So: fill the gap, never overwrite.
        """
        if "student" not in self.state.update_order or "student" in self.state.optimizers:
            return
        import torch
        params = [p for p in self.student.parameters() if p.requires_grad]
        if not params:
            raise RecipeRejected(
                f"{self.recipe.name}: the student has no parameters with requires_grad=True, so "
                f"there is nothing to train. This is usually a frozen teacher passed as the student.")
        self.state.optimizers["student"] = torch.optim.AdamW(params, lr=self.config.lr)

    def _modules_for(self, update: str):
        if update == "student":
            return self.student
        if update in self.state.modules:
            return self.state.modules[update]
        raise RecipeRejected(
            f"{self.recipe.name}: update_order names {update!r}, but RecipeState.modules has "
            f"{sorted(self.state.modules)} and it is not 'student'. Every update must correspond to "
            f"a module the recipe built.")

    def _apply(self, out: StepOutput) -> dict[str, float]:
        """Backward, clip, step, zero -- for each declared update, in declared order."""
        import torch
        logged: dict[str, float] = {}
        order = self.state.update_order
        for i, name in enumerate(order):
            loss = out.loss_for(name)
            opt = self.state.optimizers.get(name)
            if opt is None:
                raise RecipeRejected(
                    f"{self.recipe.name}: no optimizer for update {name!r}. Build one in "
                    f"RecipeState.optimizers; the trainer only auto-creates 'student'.")
            opt.zero_grad(set_to_none=True)
            # retain_graph for all but the last update: several updates commonly share one forward
            # (DMD2's student and fake-score losses both flow from the same student output).
            loss.backward(retain_graph=(i < len(order) - 1))
            mod = self._modules_for(name)
            if self.config.grad_clip is not None:
                gn = torch.nn.utils.clip_grad_norm_(mod.parameters(), self.config.grad_clip)
                logged[f"gradnorm/{name}"] = float(gn)
            opt.step()
            logged[f"loss/{name}"] = float(loss.detach())
        return logged

    def fit(self, data: Iterable[Mapping[str, Any]]) -> TrainResult:
        import torch
        torch.manual_seed(self.config.seed)
        cached = CachedTeacher(self.teacher)
        result = TrainResult(steps_done=0)
        t0 = time.time()
        out_dir = Path(self.config.out_dir) if self.config.out_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "delta.json").write_text(json.dumps(
                {"recipe": self.recipe.name, "delta": self.delta.describe(),
                 "note": self.delta.note, "config": asdict(self.config)}, indent=2))

        it = iter(data)
        for step in range(self.config.steps):
            try:
                batch = next(it)
            except StopIteration:                    # restart finite datasets
                it = iter(data)
                batch = next(it)

            cached.clear()
            out = self.recipe.step(batch, cached, self.student, self.state)
            if not isinstance(out, StepOutput):
                raise TypeError(
                    f"{self.recipe.name}.step() must return StepOutput (losses as tensors so the "
                    f"trainer can own backward); got {type(out).__name__}")

            logged = self._apply(out)
            row = {"step": float(step), **logged, **{k: float(v) for k, v in out.metrics.items()},
                   "teacher_calls": float(cached.total_calls),
                   "teacher_hits": float(cached.total_hits)}
            result.history.append(row)
            result.steps_done = step + 1

            if self.config.stop_on_nonfinite:
                bad = [k for k, v in logged.items() if k.startswith("loss/") and not math.isfinite(v)]
                if bad:
                    result.stopped_early = f"non-finite loss at step {step}: {bad}"
                    break

            if self.config.log_every and step % self.config.log_every == 0:
                bits = " ".join(f"{k.split('/')[-1]}={v:.4g}"
                                for k, v in logged.items() if k.startswith("loss/"))
                print(f"  step {step:>6}  {bits}", flush=True)
            if out_dir and self.config.ckpt_every and step and step % self.config.ckpt_every == 0:
                self.save(out_dir / f"step_{step}")

        result.seconds = time.time() - t0
        if out_dir:
            self.save(out_dir / "final")
            (out_dir / "history.json").write_text(json.dumps(result.history, indent=2))
        return result

    def save(self, path: str | Path) -> Path:
        """Weights plus the descriptor delta, because a checkpoint the runtime cannot run is useless.

        This is the Layer 1 -> Layer 2..6 handoff: `delta.json` is what lets `plan.serve()` execute
        the student with no glue. Saving weights alone would recreate exactly the problem the
        DescriptorDelta abstraction exists to remove.
        """
        import torch
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.student.state_dict(), path / "student.pt")
        for name, mod in self.state.modules.items():
            if hasattr(mod, "state_dict"):
                torch.save(mod.state_dict(), path / f"{name}.pt")
        (path / "delta.json").write_text(json.dumps(
            {"recipe": self.recipe.name, "nfe": dict(self.delta.nfe or {}),
             "shapes": {k: list(v) for k, v in (self.delta.shapes or {}).items()},
             "dtypes": dict(self.delta.dtypes or {}), "note": self.delta.note}, indent=2))
        return path

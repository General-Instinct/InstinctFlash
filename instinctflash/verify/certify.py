"""Certification: paired non-inferiority on task success.

WHY THIS IS NOT `probe_bitexact`

Every gate in this project so far asks `max|delta action| == 0`. That is the right question for a
runtime pass and a meaningless one for a model-level optimization: a distilled student produces
different actions by construction. The question becomes statistical -- *is the student's task
success acceptably close to the teacher's* -- and statistical questions are easy to answer
dishonestly.

Three rules, each of which exists because the obvious alternative is wrong:

1. **PAIRED.** The same episodes, same seeds, teacher and student. Two independent 2500-episode
   runs would let ordinary between-run variance masquerade as a real difference; pairing removes
   it. The test is McNemar on the discordant pairs, which is the correct test for paired binary
   outcomes and ignores episodes where both arms agree.

2. **THE MARGIN IS DECLARED BEFORE THE RUN.** `certify()` requires `margin` as an argument and
   records it in the certificate. A threshold chosen after seeing the delta is not a gate, it is a
   narrative. This is the same discipline that made the equivalence tiers useful.

3. **IT CAN FAIL.** A certificate that only ever says yes is decoration. The verdict is
   non-inferiority at the declared margin, and `FAIL` is a normal outcome that should block a
   release.

Incomplete or unpaired inputs are refused outright rather than partially analysed, matching
`aggregate.py`'s `REPORTABLE: NO`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence


class NotCertifiable(RuntimeError):
    """The inputs cannot support a certificate. Never downgrade this to a warning."""


@dataclass(frozen=True)
class Outcome:
    """One episode's result on one arm. `success` is the scored binary outcome."""
    episode_id: str
    seed: int
    task: str
    success: bool


@dataclass
class Certificate:
    teacher_hash: str
    student_hash: str
    n_pairs: int
    teacher_success: float
    student_success: float
    delta: float
    ci95: tuple[float, float]
    margin_declared: float
    verdict: str
    p_value: float
    discordant: tuple[int, int]          # (teacher-only wins, student-only wins)
    harness: str
    recipe: str
    seeds: str
    #: task -> (n, teacher successes, student successes). A macro number can hide a task that went
    #: to zero while others improved, and for a policy that is the failure that matters.
    per_task: Mapping[str, tuple] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return self.verdict.startswith("PASS")

    def to_json(self, **kw) -> str:
        d = asdict(self)
        d["ci95"] = list(self.ci95)
        d["discordant"] = list(self.discordant)
        d["per_task"] = {t: list(v) for t, v in self.per_task.items()}
        return json.dumps(d, indent=2, **kw)

    def per_task_table(self) -> str:
        if not self.per_task:
            return "(no per-task breakdown)"
        out = [f"  {'task':<26}{'n':>4}{'teacher':>9}{'student':>9}{'delta':>8}"]
        for t, (n, tw, sw) in self.per_task.items():
            dt = (sw - tw) / max(n, 1)
            flag = "  <-- collapsed" if tw > 0 and sw == 0 else ""
            out.append(f"  {t:<26}{n:>4}{tw / n:>9.2f}{sw / n:>9.2f}{dt:>+8.2f}{flag}")
        return "\n".join(out)

    def __str__(self) -> str:
        b, c = self.discordant
        return (
            f"teacher {self.teacher_success:.4f}   student {self.student_success:.4f}\n"
            f"delta   {self.delta:+.4f}   95% CI [{self.ci95[0]:+.4f}, {self.ci95[1]:+.4f}]\n"
            f"paired  n={self.n_pairs}  discordant: teacher-only {b}, student-only {c}  "
            f"p={self.p_value:.4g}\n"
            f"margin declared BEFORE the run: {self.margin_declared:+.4f}\n"
            f"VERDICT: {self.verdict}")


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar. b, c are the discordant counts.

    Exact rather than chi-square because discordant counts on 50-task suites are routinely small,
    and the chi-square approximation is bad exactly there.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _paired_delta_ci(pairs: Sequence[tuple[bool, bool]], z: float = 1.96) -> tuple[float, float]:
    """95% CI on (student - teacher) success rate for PAIRED binary data.

    Uses the variance of the paired differences, which is driven entirely by the discordant pairs;
    an unpaired two-proportion interval would be wider and wrong here.
    """
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    d = [(1 if s else 0) - (1 if t else 0) for t, s in pairs]
    mean = sum(d) / n
    if n < 2:
        return (mean, mean)
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    half = z * math.sqrt(var / n)
    return (mean - half, mean + half)


def required_pairs(pairs: Sequence[tuple[bool, bool]], margin: float, z: float = 1.96) -> int:
    """How many paired episodes would be needed to decide non-inferiority at this margin.

    Added after the first real run: 20 paired episodes produced a CI of [-0.12, +0.22] against a
    -0.05 margin. The verdict was FAIL, which was correct, but "FAIL" alone conflates *we measured
    a regression* with *we cannot tell yet* -- and those call for opposite responses. This turns
    the second case into an actionable number.
    """
    n = len(pairs)
    if n == 0:
        return 0
    d = [(1 if s else 0) - (1 if t else 0) for t, s in pairs]
    mean = sum(d) / n
    if n < 2:
        return 0
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    slack = mean - margin                  # how much room between the estimate and the margin
    if slack <= 0 or var == 0:
        return 0                           # a point estimate already at/below the margin
    return int(math.ceil(var * (z / slack) ** 2))


def _index(outcomes: Sequence[Outcome], arm: str) -> dict[tuple[str, int], Outcome]:
    seen: dict[tuple[str, int], Outcome] = {}
    for o in outcomes:
        key = (o.episode_id, o.seed)
        if key in seen:
            raise NotCertifiable(
                f"{arm}: duplicate episode {key}. Certification needs exactly one outcome per "
                f"(episode, seed); duplicates mean the run is not what it claims to be.")
        seen[key] = o
    return seen


def certify(teacher: Sequence[Outcome], student: Sequence[Outcome], *,
            margin: float, teacher_hash: str = "?", student_hash: str = "?",
            harness: str = "?", recipe: str = "?", seeds: str = "?",
            min_pairs: int = 1) -> Certificate:
    """Paired non-inferiority certificate. `margin` MUST be supplied and is recorded.

    `margin` is the largest success-rate drop that is still acceptable, as a NEGATIVE number:
    -0.05 means "the student may be up to 5 points worse". Non-inferiority holds when the lower
    bound of the CI on (student - teacher) is above the margin.
    """
    if margin > 0:
        raise NotCertifiable(
            f"margin={margin} is positive. A non-inferiority margin is the acceptable LOSS and "
            f"must be <= 0; a positive value would certify a student that is worse than allowed.")

    ti, si = _index(teacher, "teacher"), _index(student, "student")
    common = sorted(set(ti) & set(si))
    notes: list[str] = []
    if not common:
        raise NotCertifiable("no (episode, seed) pairs are common to both arms; the runs are not "
                             "paired and cannot be compared this way.")
    only_t, only_s = set(ti) - set(si), set(si) - set(ti)
    if only_t or only_s:
        raise NotCertifiable(
            f"arms are not on the same episodes: {len(only_t)} teacher-only, {len(only_s)} "
            f"student-only. Certification refuses partial overlap rather than silently "
            f"comparing whatever happens to match.")
    if len(common) < min_pairs:
        raise NotCertifiable(f"{len(common)} pairs < min_pairs={min_pairs}")

    for k in common:
        if ti[k].task != si[k].task:
            raise NotCertifiable(f"episode {k} is task {ti[k].task!r} for the teacher and "
                                 f"{si[k].task!r} for the student")

    pairs = [(ti[k].success, si[k].success) for k in common]
    n = len(pairs)
    per_task: dict[str, list] = {}
    for k in common:
        row = per_task.setdefault(ti[k].task, [0, 0, 0])
        row[0] += 1
        row[1] += 1 if ti[k].success else 0
        row[2] += 1 if si[k].success else 0
    t_rate = sum(1 for t, _ in pairs if t) / n
    s_rate = sum(1 for _, s in pairs if s) / n
    delta = s_rate - t_rate
    b = sum(1 for t, s in pairs if t and not s)      # teacher-only wins
    c = sum(1 for t, s in pairs if s and not t)      # student-only wins
    p = _mcnemar_exact(b, c)
    lo, hi = _paired_delta_ci(pairs)

    if lo > margin:
        verdict = f"PASS non-inferiority at margin {margin:+.4f} (CI lower bound {lo:+.4f})"
    elif hi < margin:
        # the whole interval is below the margin: a regression, detected
        verdict = (f"FAIL (regression detected) at margin {margin:+.4f}: the entire 95% CI "
                   f"[{lo:+.4f}, {hi:+.4f}] lies below the margin")
    else:
        # the interval straddles the margin: we cannot tell, and saying so is the honest answer
        need = required_pairs(pairs, margin)
        verdict = (f"FAIL (insufficient evidence) at margin {margin:+.4f}: CI [{lo:+.4f}, "
                   f"{hi:+.4f}] straddles it. n={n} is too small to decide; "
                   f"~{need} paired episodes are needed at the observed discordance rate")
        notes.append(f"underpowered: {n} pairs, ~{need} needed for margin {margin:+.4f}")
    if b + c == 0:
        notes.append("zero discordant pairs: the arms agreed on every episode, so McNemar has no "
                     "information and p=1 by construction")
    return Certificate(
        teacher_hash=teacher_hash, student_hash=student_hash, n_pairs=n,
        teacher_success=t_rate, student_success=s_rate, delta=delta, ci95=(lo, hi),
        margin_declared=margin, verdict=verdict, p_value=p, discordant=(b, c),
        harness=harness, recipe=recipe, seeds=seeds,
        per_task={t: tuple(v) for t, v in sorted(per_task.items())}, notes=tuple(notes))


def load_jsonl(path: str) -> list[Outcome]:
    """Per-episode JSONL: one object per episode with episode_id, seed, task, success."""
    out = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(Outcome(str(d["episode_id"]), int(d["seed"]), str(d["task"]),
                                   bool(d["success"])))
            except (KeyError, ValueError, TypeError) as e:
                raise NotCertifiable(f"{path}:{i} is not a valid outcome record ({e}). "
                                     f"Required: episode_id, seed, task, success") from e
    if not out:
        raise NotCertifiable(f"{path} contains no outcomes")
    return out

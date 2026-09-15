"""Startup self-check gating a family's default CUDA-graph capture. Family-generic.

The pi05 pattern (examples/pi05_vla/pi05_iwm/static_capture.py::_self_check), extracted for
every family that captures by default: immediately after the FIRST capture, replay is compared
against the family's UPSTREAM eager callable (bound before install patched it) on staged inputs
the capture never saw. What each staged case IS — fresh x_t from a dedicated generator, distinct
schedule timesteps, a synthetically REFILLED prefix so a graph that baked K/V values instead of
reading the live buffers cannot pass — stays the family module's business (the staging is where
the model knowledge lives); comparing, timing, verdict-keeping and saying the verdict OUT LOUD is
this module's, so every family fails the same way in the same words.

The gate is the family's own verified numeric tier:

  * a BITEXACT family (LingBot-VLA-4B, GR00T-N1.7; pi05 keeps its in-plugin original) requires
    exact equality (``tolerance=0.0``): replay re-runs the same kernels at the same addresses,
    so ANY drift is evidence something is not being re-read;
  * a NUMERIC family (LingBot-VLA-V2, whose upstream fused-MoE kernel disagrees with ITSELF on
    identical seeds) uses an explicitly declared numeric guard. V2 retains a legacy action-domain
    threshold for velocity comparisons; that domain mismatch is reported, not presented as
    calibrated numerical or task-quality evidence. Repeated comparisons can tighten admission
    without increasing the guard.

PASS → replay serves. FAIL → the caller releases its graphs and rebinds upstream (its release
hook prints that part) — serving continues on eager arithmetic. Verdicts land on stderr
deliberately: they arrive at first capture, i.e. during serving, and ``cli_config.execute``
defers stdout until the command returns — which for a persistent ``instinctflash serve`` is
never. The server's live log stream is stderr.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Iterable

__all__ = ["run_capture_self_check", "record_self_check_on_plan"]


def compare_tensors(ref, out) -> dict:
    """Finite same-dtype/shape bytes are the exact contract; error is diagnostic."""
    import torch
    if ref.shape != out.shape or ref.dtype != out.dtype or ref.numel() == 0:
        return {"valid": False, "bitexact": False, "max_abs_delta": float("inf")}
    a, b = ref.detach().contiguous(), out.detach().to(ref.device).contiguous()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return {"valid": False, "bitexact": False, "max_abs_delta": float("inf")}
    exact = torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
    delta = float((a.to(torch.float64) - b.to(torch.float64)).abs().max().item())
    return {"valid": True, "bitexact": bool(exact), "max_abs_delta": delta}


def run_capture_self_check(*, family: str,
                           cases: Iterable[tuple[str, Callable[[], "object"],
                                                 Callable[[], "object"]]],
                           tolerance: float = 0.0,
                           tolerance_provenance: str = "",
                           comparison_domain: str | None = None,
                           tolerance_domain: str | None = None,
                           repeats: int = 1) -> dict:
    """Run staged (label, run_eager, run_replay) cases and return the verdict dict.

    Each ``run_eager`` must produce the UPSTREAM answer for a staged input and each
    ``run_replay`` the captured graph's answer for the same input; sequencing side effects
    (copying into static buffers, refilling a perturbed prefix, restoring the caller's real
    state afterwards) belong to the family's case closures and its enclosing try/finally.

    The verdict dict carries: ``n``, ``passed``, ``bitexact``, ``max_abs_delta``,
    ``tolerance``, ``tolerance_provenance``, ``seconds`` and per-case ``cases`` records.
    This function prints the verdict line; a failing caller's release hook prints the
    graphs-released consequence.
    """
    import torch

    if type(repeats) is not int or repeats < 1:
        raise ValueError("self-check repeats must be a positive integer")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    worst, recorded = 0.0, []
    exact, valid = True, True
    for i, (label, run_eager, run_replay) in enumerate(cases):
        checks = []
        for repeat in range(repeats):
            with torch.no_grad():
                # Freeze reusable eager storage before graph replay can overwrite it.
                ref = run_eager().detach().clone()
                out = run_replay()
            checks.append(compare_tensors(ref, out))
        check = {"valid": all(c["valid"] for c in checks),
                 "bitexact": all(c["bitexact"] for c in checks),
                 "max_abs_delta": max(c["max_abs_delta"] for c in checks)}
        worst = max(worst, check["max_abs_delta"])
        exact = exact and check["bitexact"]
        valid = valid and check["valid"]
        row = {"input": i + 1, "stage": label, **check}
        if repeats > 1:
            row["repeat_checks"] = checks
        recorded.append(row)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    valid = valid and bool(recorded)
    passed = valid and (exact if tolerance == 0.0 else worst <= tolerance)
    verdict = {
        "n": len(recorded),
        "passed": passed,
        "bitexact": valid and exact,
        "max_abs_delta": worst,
        "tolerance": float(tolerance),
        "tolerance_provenance": tolerance_provenance,
        "seconds": time.perf_counter() - t0,
        "cases": recorded,
    }
    if repeats > 1:
        verdict.update(repeats=repeats, comparisons=len(recorded) * repeats)
    if comparison_domain is not None or tolerance_domain is not None:
        verdict.update(comparison_domain=comparison_domain, tolerance_domain=tolerance_domain,
                       calibration_status=("legacy_cross_domain_guard"
                                           if comparison_domain != tolerance_domain else "same_domain_declared"))
    if not passed:
        # The PASS line is the plan recorder's (one line per verdict, pi05's split); the FAIL
        # is printed here TOO so a caller without a recorder can never fail silently.
        print(f"[{family} static_capture] SELF-CHECK FAILED: {_verdict_line(verdict)}.",
              file=sys.stderr, flush=True)
    return verdict


def _stage_summary(cases: list[dict]) -> str:
    counts: dict[str, int] = {}
    for case in cases:
        stage = str(case.get("stage", "staged"))
        counts[stage] = counts.get(stage, 0) + 1
    return " + ".join(f"{n} {stage}" for stage, n in counts.items())


def _verdict_line(res: dict) -> str:
    stages = _stage_summary(res.get("cases", []))
    if not res["passed"]:
        gate = ("exact equality" if res["tolerance"] == 0.0 else
                f"the {'legacy cross-domain guard' if res.get('calibration_status') == 'legacy_cross_domain_guard' else 'recorded envelope'} {res['tolerance']:.3e} "
                f"({res['tolerance_provenance']})")
        return (f"replay disagrees with eager by {res['max_abs_delta']:.3e} on a staged "
                f"input it was not captured from (gate: {gate}; staged {stages})")
    if res["tolerance"] == 0.0:
        return (f"self-check bit-exact on {res['n']} inputs (replay == eager exactly; "
                f"staged {stages}; {res['seconds']:.1f} s startup cost, once per process)")
    gate = 'legacy cross-domain guard' if res.get('calibration_status') == 'legacy_cross_domain_guard' else 'recorded envelope'
    return (f"self-check within the {gate} on {res['n']} inputs "
            f"(max |Δ| {res['max_abs_delta']:.3e} ≤ {res['tolerance']:.3e} — "
            f"{res['tolerance_provenance']}; staged {stages}; "
            f"{res['seconds']:.1f} s startup cost, once per process)")


def record_self_check_on_plan(capture, family: str):
    """The self-check verdict, put where a reader will look: the plan's graph_capture entry.

    ``plan.explain()`` / ``runtime.explain()`` render ``params['decision']`` lines, and the plan
    object is the same one the facade holds — so the line the serve log prints at first capture
    is the line every later ``explain()`` shows. The full verdict rides on
    ``params['self_check']`` for programmatic readers. Same recorder shape as pi05's
    ``_record_self_check_on_plan``; this one additionally phrases the NUMERIC-envelope tier.
    """
    def on_result(res: dict) -> None:
        if res["passed"]:
            line = _verdict_line(res)
        else:
            line = (f"self-check FAILED — {_verdict_line(res)}; graphs released, running "
                    f"eager (upstream's arithmetic exactly), serve continues")
        capture.params["decision"] = tuple(capture.params.get("decision", ())) + (
            f"graph_capture: {line}",)
        capture.params["self_check"] = res
        # stderr: the verdict lands at FIRST CAPTURE, during serving — see module docstring.
        print(f"InstinctFlash {family}: graph_capture {line}.", file=sys.stderr, flush=True)
    return on_result

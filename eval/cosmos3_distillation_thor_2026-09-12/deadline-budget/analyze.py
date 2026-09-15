"""Read frozen timing receipts; illustrate deadline arithmetic, not realtime admission."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def analyze(root):
    results = []
    for family, template in (
        ("Edge V17 SDE2 CFG1 cuDNN", "realtime-deployment-v17/thor-receipts/seed{seed}/cudnn.json"),
        ("Nano V18 SDE1 CFG1 combined", "realtime-deployment-v18/thor-results-v3/seed{seed}/combined.json"),
    ):
        for seed in (12031, 12032):
            path = root / template.format(seed=seed)
            raw = path.read_bytes()
            report = json.loads(raw)
            if report.get("ok") is not True:
                raise ValueError(f"Unsuccessful receipt: {path}")
            measured = [c["ms"] for c in report["calls"] if c["phase"] == "measured"]
            if not measured or not all(math.isfinite(x) and x > 0 for x in measured):
                raise ValueError(f"Invalid timing samples: {path}")
            peak = max(measured)
            results.append({
                "family": family, "seed": seed,
                "source": str(path.relative_to(root)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "measured_requests": len(measured),
                "p50_ms": float(np.median(measured)),
                "p95_ms": float(np.percentile(measured, 95)),
                "max_observed_ms": peak,
                "illustrative_deadlines": [
                    {"deadline_ms": deadline,
                     "observed_exceedances": sum(t > deadline for t in measured),
                     "margin_at_observed_max_ms": deadline - peak}
                    for deadline in (250, 500, 750, 1000)
                ],
                "illustrative_execution_windows": [
                    {"assumed_control_hz": hz, "assumed_extra_budget_ms": extra,
                     "required_scheduling": "pipelined",
                     "blocking_controller_deadline_ms": 1000 / hz,
                     "minimum_executed_steps_at_observed_max": math.ceil((peak + extra) * hz / 1000)}
                    for hz in (10, 15, 20) for extra in (0, 50)
                ],
            })
    return {"status": "analysis_complete", "realtime_certified": False,
            "task_quality_certified": False,
            "assumptions": [
                "Deadlines, control rates and extra budgets are illustrative, not controller settings.",
                "Execution-window arithmetic assumes pipelined scheduling: N/control_hz >= observed model latency + extra budget. A blocking controller instead has one control period, not N periods; actual overlap, action age and startup remain unverified.",
                "Conditioning FPS is not evidence of actuator/control frequency; output horizon is not executed chunk length.",
                "Measured maximum is a finite-sample observation, not a worst-case bound; zero observed exceedances does not certify a deadline.",
                "Receipts cover timed Runtime prediction under their frozen protocols, not a certified sensor-to-actuator path; do not double-count preprocessing already timed.",
                "Small frozen-fixture samples exclude warmup from this table and cannot establish sustained latency tails under deployment load.",
                "Both trained families retain their historical UniPC4 quality limitations; timing feasibility alone cannot admit them.",
            ], "results": results}


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    result = analyze(directory.parent)
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Illustrative controller deadline budget", "",
             "These are calculations from frozen Thor timing receipts, not realtime or quality admission. No controller deadline or executed action chunk has been supplied.", "",
             "| Model / seed | Measured requests | p50 ms | p95 ms | Observed max ms | Margin at 500 ms | Margin at 750 ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["results"]:
        peak = r["max_observed_ms"]
        lines.append(f'| {r["family"]} / {r["seed"]} | {r["measured_requests"]} | {r["p50_ms"]:.1f} | {r["p95_ms"]:.1f} | {peak:.1f} | {500-peak:+.1f} | {750-peak:+.1f} |')
    lines += ["", "Positive margin only means the observed timed call fits that illustrative budget. It must also accommodate any additional required work and deployment variability.", "",
              "For illustration only, with pipelined execution at15 Hz and50 ms of additional budget, the minimum executed chunk implied by the observed maximum is:", ""]
    for r in result["results"]:
        steps = next(x["minimum_executed_steps_at_observed_max"] for x in r["illustrative_execution_windows"] if x["assumed_control_hz"] == 15 and x["assumed_extra_budget_ms"] == 50)
        lines.append(f'- {r["family"]}, seed{r["seed"]}: {steps} steps.')
    lines += ["", *[f'- {a}' for a in result["assumptions"]], "",
              "The shared deployment assessor is `benchmarks/vla/realtime.py::assess`, exposed by `instinctflash eval realtime-report`. Its example50 Hz/50-action budget is not a deployment setting. These receipts do not record an actual executed chunk or controller schedule, so this analysis does not fabricate those fields to produce a passing assessment.", "",
              "Reproduce with `python analyze.py` from this directory. [Full calculations and source hashes](results.json).", ""]
    (directory / "README.md").write_text("\n".join(lines))

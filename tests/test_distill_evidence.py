#!/usr/bin/env python3
"""The distill evidence base: the campaigns' measured frontiers as data, and what the code reads off them.

Pinned (CPU, no torch, no weights):

  * both registered files load through the loader and the family adapters' `evidence()`, in the
    sweep-report row shape, every row citing its archive, no row of the UNTRAINED frontier being
    a trained arm;
  * the facts the campaign wrote down hold in the data: the same 1V/4A schedule spans 0.28..0.89
    across w; the video cliff at w=5 is absent at w=1/w=3; action 2->1 is a real cliff at w=1;
    the pi05 frontier is flat to nfe1 (every |delta| < 3.5 pp, lower bounds above the margin,
    zero-discordance noise floor, 174 -> ~51 ms);
  * the verdict code reads the evidence and returns the campaign's verdicts: wan_va 2V/2A ->
    NO DISTILLATION NEEDED (best untrained 2V/2A@w1); 1V/4A -> EXTEND (best w3/w1 tie by the
    one-episode rule, cost rule picks w1); 1V/1A -> TRAIN with 1V/1A@w1 as the control; pi05 ->
    NO DISTILLATION NEEDED at every nfe;
  * the control gate accepts the evidence grid as a swept grid and refuses the H1 confound on it
    (a student verified against 1V/4A@w5).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from instinctflash.distill import (  # noqa: E402
    ControlGateViolation, DistillPipeline, NoDistillationNeeded, get_family, screen_verdict, verify_point,
)
from instinctflash.distill.evidence import FILES, load_evidence, registered_families  # noqa: E402
from instinctflash.distill.pipeline import (  # noqa: E402
    VERDICT_EXTEND_SCREEN, VERDICT_NO_DISTILLATION_NEEDED, VERDICT_TRAIN, rank_candidates,
)
from instinctflash.verify.certify import Outcome  # noqa: E402
from tests.run_tests import run_module_tests  # noqa: E402

ROW_KEYS = {"point", "nfe", "guidance", "status", "point_success", "delta", "interval", "interval_method",
            "deciding_lower_bound", "n_pairs", "forwards_per_cycle", "forwards_batch1", "forwards_batch2",
            "cycle_p50_ms", "source"}


def _rows(doc):
    return {r["point"]: r for r in doc["rows"]}


def test_evidence_files_load_in_the_sweep_report_shape() -> None:
    assert registered_families() == ["pi05", "wan_va"]
    for family in FILES:
        doc = load_evidence(family)
        assert doc["kind"] == "fewstep_evidence_frontier" and doc["family"] == family
        assert doc["screening"] is True and doc["margin_declared_confers_nothing"] == -0.05
        assert doc["baseline"]["success"] is not None and doc["baseline"]["n"]
        assert doc["noise_floor"]["n_pairs"]
        for row in doc["rows"]:
            assert ROW_KEYS <= set(row), (family, row["point"], ROW_KEYS - set(row))
            assert row["status"] == "SCREENED" and row["source"], row["point"]
            assert row["interval"][0] <= row["delta"] <= row["interval"][1], row["point"]
            assert row["deciding_lower_bound"] == row["interval"][0]
            assert row["forwards_batch1"] + row["forwards_batch2"] == row["forwards_per_cycle"]
            # the guidance leg is canonical per stream, always present
            assert all({"mode", "scale"} <= set(g) for g in row["guidance"].values()), row["point"]
        assert get_family(family).evidence()["rows"] == doc["rows"]
    # the UNTRAINED frontier carries no trained arm as a row; trained arms are on the record apart
    va = load_evidence("wan_va")
    assert all("h1s" not in r["arm"] for r in va["rows"])
    assert [t["arm"] for t in va["trained_arms"]] == ["fewstep_h1s_a4", "fewstep_h1s_a2"]


def test_the_campaign_facts_hold_in_the_data() -> None:
    va = _rows(load_evidence("wan_va"))
    # the same schedule, 60 points apart across the guidance axis: nfe alone underspecifies quality
    at_1v4a = {w: va[f"1V/4A@w{w}"]["point_success"] for w in (1, 3, 5, 7, 9)}
    assert at_1v4a[9] < 0.30 < at_1v4a[7] < 0.55 < 0.70 < at_1v4a[5] < 0.80 < at_1v4a[3]
    assert abs(at_1v4a[1] - at_1v4a[3]) < 0.01
    # the w=5 video cliff (E4) is a guidance artifact: gone at w=1 and w=3
    assert va["1V/4A@w5"]["interval"][1] < -0.09
    assert va["1V/4A@w1"]["interval"][1] > 0 and va["1V/4A@w3"]["interval"][1] > 0
    # batch-1 at w=1, batch-2 otherwise; forwards = (V+1)+(A+1)+2
    assert (va["1V/4A@w1"]["forwards_batch1"], va["1V/4A@w1"]["forwards_batch2"]) == (9, 0)
    assert (va["1V/4A@w5"]["forwards_batch1"], va["1V/4A@w5"]["forwards_batch2"]) == (0, 9)
    assert va["2V/2A@w1"]["forwards_per_cycle"] == 8 and va["1V/1A@w1"]["forwards_per_cycle"] == 6
    # with guidance off: flat in video steps, flat in action steps down to 2, a real cliff at 1
    assert va["2V/2A@w1"]["delta"] > 0 and va["2V/2A@w1"]["interval"][0] > -0.05
    assert va["1V/2A@w1"]["delta"] > -0.03
    assert va["1V/1A@w1"]["interval"][1] < -0.10
    # guidance x NFE non-monotonicity: w=9 hurts even the certified 2V/4A schedule
    assert va["2V/4A@w9"]["delta"] < -0.05 < va["2V/4A@w3"]["delta"]
    # the trained arms: training bought nothing measurable over the free knob
    trained = {t["arm"]: t for t in load_evidence("wan_va")["trained_arms"]}
    c_b3 = trained["fewstep_h1s_a4"]["comparisons"]["C-B3 (student vs untrained 1V/4A@w3, best untrained knob)"]
    assert abs(c_b3["delta"]) < 1e-9 and c_b3["interval"][0] < 0 < c_b3["interval"][1]
    c_b2 = trained["fewstep_h1s_a4"]["comparisons"]["C-B2 (student vs untrained 1V/4A@w1, execution-matched)"]
    assert c_b2["interval"][0] < 0 < c_b2["interval"][1] and c_b2["mcnemar_exact_two_sided_p"] > 0.5

    pi = load_evidence("pi05")
    rows = _rows(pi)
    assert [r["point"] for r in pi["rows"]] == ["nfe5", "nfe4", "nfe3", "nfe2", "nfe1"]
    for r in pi["rows"]:
        assert abs(r["delta"]) < 0.035 and r["deciding_lower_bound"] > -0.05 and r["n_pairs"] == 160, r["point"]
        assert r["forwards_batch2"] == 0 and r["forwards_per_cycle"] == r["nfe"]["action"] + 1
        assert r["guidance"] == {"action": {"mode": "none", "scale": 1.0}}
    assert pi["noise_floor"]["discordance"] == 0.0 and pi["noise_floor"]["delta"] == 0.0
    assert 170 < pi["baseline"]["cycle_p50_ms"] < 180 and 49 < rows["nfe1"]["cycle_p50_ms"] < 53
    assert rows["nfe2"]["delta"] > 0, "the R1 nfe2 cliff does not replicate on the E9 harness"


def test_verdicts_over_the_evidence_reproduce_the_campaign() -> None:
    va = load_evidence("wan_va")
    v = screen_verdict(va, family="wan_va", nfe={"video": 2, "action": 2}, source="evidence")
    assert v.verdict == VERDICT_NO_DISTILLATION_NEEDED and v.best["point"] == "2V/2A@w1"
    assert [c["point"] for c in v.candidates] == ["2V/2A@w9", "2V/2A@w7", "2V/2A@w5", "2V/2A@w3", "2V/2A@w1"]
    stub = v.prereg_stub()
    assert "2V/2A@w1" in stub and "8 batch-1 + 0 batch-2" in stub and "DRAFT" in stub
    # 1V/4A: w3 and w1 tie within one paired episode (0.8904 vs 0.8865 at n~229); the cost rule
    # picks the batch-1 point, and the interval straddles the margin: extend / certify first
    v = screen_verdict(va, family="wan_va", nfe={"video": 1, "action": 4})
    assert v.verdict == VERDICT_EXTEND_SCREEN and v.best["point"] == "1V/4A@w1", (v.verdict, v.best)
    ranked = rank_candidates(v.candidates)
    assert [c["point"] for c in ranked[:2]] == ["1V/4A@w1", "1V/4A@w3"]
    # 1V/2A: extend; 1V/1A: a real cliff even at w=1 -> TRAIN, with 1V/1A@w1 as the control
    assert screen_verdict(va, family="wan_va", nfe={"video": 1, "action": 2}).verdict == VERDICT_EXTEND_SCREEN
    v = screen_verdict(va, family="wan_va", nfe={"video": 1, "action": 1})
    assert v.verdict == VERDICT_TRAIN and v.best["point"] == "1V/1A@w1"
    # the pipeline refuses to train where the screen says the point is free
    try:
        DistillPipeline("wan_va", {"video": 2, "action": 2}, guidance={"video": 1.0}).train(
            teacher=None, screen=screen_verdict(va, family="wan_va", nfe={"video": 2, "action": 2}))
    except NoDistillationNeeded as verdict:
        assert verdict.verdict.best["point"] == "2V/2A@w1"
    else:
        raise AssertionError("train() ran on a free operating point")
    # pi05: no distillation needed at every nfe; the untrained point goes to certification
    pi = load_evidence("pi05")
    for n in (5, 4, 3, 2, 1):
        v = screen_verdict(pi, family="pi05", nfe={"action": n})
        assert v.verdict == VERDICT_NO_DISTILLATION_NEEDED and v.best["point"] == f"nfe{n}", (n, v.verdict)
        assert v.best["forwards_batch2"] == 0
    assert "nfe1" in screen_verdict(pi, family="pi05", nfe={"action": 1}).prereg_stub()


def _outcomes(n_true: int, n: int = 100) -> list[Outcome]:
    return [Outcome(f"t{i % 5}/s{i}", 10_000 + i, f"t{i % 5}", i < n_true) for i in range(n)]


def test_the_evidence_grid_is_the_control_gates_swept_grid() -> None:
    va = load_evidence("wan_va")
    grid = screen_verdict(va, family="wan_va", nfe={"video": 1, "action": 4}).control_grid()
    fam = get_family("wan_va").default_guidance()
    teacher, control, student = _outcomes(90), _outcomes(80), _outcomes(82)
    # the H1 confound, refused: a student at the shipped guidance verified against 1V/4A@w5
    try:
        verify_point(point="1V/4A", teacher=teacher, control=control, student=student,
                     student_schedule={"nfe": {"video": 1, "action": 4}},
                     control_schedule={"nfe": {"video": 1, "action": 4}}, margin=-0.05,
                     control_grid=grid, family_guidance=fam)
    except ControlGateViolation as refusal:
        assert "not the best untrained configuration" in str(refusal) and "1V/4A@w5" not in str(refusal)
    else:
        raise AssertionError("the shipped-guidance control passed against the evidence grid")
    # the execution-matched control at the best untrained knob passes (w1 and w3 tie within an episode)
    for w in (1.0, 3.0):
        report = verify_point(point="1V/4A", teacher=teacher, control=control, student=student,
                              student_schedule={"nfe": {"video": 1, "action": 4}, "guidance": {"video": w}},
                              control_schedule={"nfe": {"video": 1, "action": 4}, "guidance": {"video": w}},
                              margin=-0.05, control_grid=grid, family_guidance=fam)
        assert report.control_guidance["video"]["scale"] == w
        assert len(report.guidance_grid_swept) == 5


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))

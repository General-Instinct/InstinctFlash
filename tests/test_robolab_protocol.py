"""Prospective RoboLab coverage, approval, RNG pairing and outcome rejection."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.vla import robolab_protocol as rp
from instinctflash.verify.certify import Outcome, certify

INVENTORY = Path(__file__).resolve().parents[1] / (
    "eval/cosmos3_task_quality_2026-09-14/preflight/robolab120_task_inventory.json")


@pytest.fixture
def inventory():
    return json.loads(INVENTORY.read_text())


def _bindings():
    return {family: {arm: rp.digest([family, arm, "frozen execution files"])
                     for arm in rp.ARMS} for family in rp.FAMILIES}


def _plan(inventory, **kwargs):
    return rp.build_protocol(inventory, execution_bindings=_bindings(), **kwargs)


def _formal(inventory, **kwargs):
    return _plan(inventory, stage="formal", success_margin=-.05,
                 approval_reference="synthetic CPU fixture approval only", **kwargs)


def _records(plan):
    rows = []
    protocol_hash, bindings = rp.digest(plan), _bindings()
    for episode in plan["episodes"]:
        for family in rp.FAMILIES:
            for arm in rp.ARMS:
                rows.append({
                    **{k: episode[k] for k in ("pair_id", "task_id", "episode_index", "scene_seed")},
                    "family": family, "arm": arm, "protocol_sha256": protocol_hash,
                    "robolab_revision": rp.ROBOLAB_REVISION,
                    "checkpoint_revision": rp.CHECKPOINTS[family]["revision"],
                    "status": "completed", "success": episode["episode_index"] % 2 == 0,
                    "reset_count": 2, "executed_steps": 3, "generated_chunks": 1,
                    "model_seeds": [rp.model_seed(episode, 0)],
                    "initial_state_sha256": rp.digest([episode["pair_id"], "initial state"]),
                    "initial_observation_sha256": rp.digest([episode["pair_id"], "initial observation"]),
                    "scene_config_sha256": rp.digest([episode["pair_id"], "scene configuration"]),
                    "asset_inventory_sha256": rp.digest([episode["task_id"], "asset inventory"]),
                    "simulator_fingerprint_sha256": rp.digest("fixed simulator and renderer"),
                    "action_trace_sha256": rp.digest([episode["pair_id"], family, arm, "actions"]),
                    "execution_binding_sha256": bindings[family][arm],
                })
    return rows


def test_inventory_is_complete_pinned_and_simulator_free(inventory):
    assert inventory["task_count"] == 120
    assert inventory["simulator_executed"] is False
    assert inventory["robolab_revision"] == rp.ROBOLAB_REVISION
    task = next(t for t in inventory["tasks"] if t["task_id"] == "BananaInBowlTask")
    assert task["episode_length_s"] == 50
    assert task["instruction_default"] == "Pick up the banana and place it in the bowl"
    assert task["scene"] == "banana_bowl.usda"


def test_formal_manifest_contains_every_task_episode_and_no_implicit_margin(inventory):
    plan = _plan(inventory, stage="formal")
    assert plan["expected_records"] == 4800
    assert plan["expected_pairs_per_family"] == 1200
    assert len(plan["task_ids"]) == 120
    assert plan["acceptance"]["success_margin"] is None
    assert plan["acceptance"]["mode"] == "pending"
    assert plan["acceptance"]["task_collapse_gate"] is False
    assert plan["episodes"] == _plan(inventory, stage="formal")["episodes"]
    assert rp.validate_protocol(plan) == rp.digest(plan)


def test_all_reserved_scene_and_chunk_seeds_are_unique_across_stages(inventory):
    seen = set()
    for stage in ("smoke", "formal"):
        plan = _plan(inventory, stage=stage)
        for episode in plan["episodes"]:
            seeds = [episode["scene_seed"], *[rp.model_seed(episode, i)
                     for i in range(episode["max_policy_chunks"])]]
            assert all(0 <= s < 2**32 for s in seeds)
            assert len(set(seeds)) == len(seeds)
            assert not seen.intersection(seeds)
            seen.update(seeds)


@pytest.mark.parametrize("bad", [-1, True, 10000])
def test_model_seed_rejects_indices_outside_reserved_episode(inventory, bad):
    with pytest.raises(ValueError):
        rp.model_seed(_plan(inventory)["episodes"][0], bad)


@pytest.mark.parametrize("kwargs", [
    {"success_margin": -.05}, {"success_margin": float("nan"), "approval_reference": "x"},
    {"success_margin": True, "approval_reference": "x"},
    {"success_margin": .05, "approval_reference": "x"},
    {"success_margin": -.05, "approval_reference": "x", "report_only": True},
    {"approval_reference": "x"}, {"stage": "formal", "episodes_per_task": 9},
    {"episodes_per_task": 1001}, {"episodes_per_task": True},
])
def test_invalid_or_unapproved_protocol_choices_are_rejected(inventory, kwargs):
    with pytest.raises(ValueError):
        _plan(inventory, **kwargs)


def test_source_reader_refuses_worktree_drift(tmp_path, monkeypatch):
    (tmp_path / "source.py").write_text("changed")
    monkeypatch.setattr(rp, "_git", lambda *_: b"pinned")
    with pytest.raises(ValueError, match="differs from pinned"):
        rp._source(tmp_path, "source.py")


@pytest.mark.parametrize("mutation", ["empty", "duplicate", "seed", "coverage", "approval"])
def test_altered_plan_is_rejected(inventory, mutation):
    plan = _plan(inventory)
    if mutation == "empty":
        plan["episodes"] = []
    elif mutation == "duplicate":
        plan["episodes"][1] = copy.deepcopy(plan["episodes"][0])
    elif mutation == "seed":
        plan["episodes"][0]["scene_seed"] += 1
    elif mutation == "coverage":
        plan["expected_records"] = 1
    else:
        plan["acceptance"]["mode"] = "certify"
    with pytest.raises(ValueError):
        rp.validate_protocol(plan)


def test_smoke_never_certifies_even_with_a_margin(inventory):
    plan = _plan(inventory, success_margin=-.05, approval_reference="CPU fixture")
    result = rp.build_report(plan, _records(plan))
    assert result["status"] == "NOT_CERTIFIABLE"
    assert result["task_quality_validated"] is False
    assert all(r["certificate"] is None for r in result["families"].values())


def test_full_report_requires_approval_and_prebound_execution(inventory):
    plan = _plan(inventory, stage="formal", report_only=True)
    result = rp.build_report(plan, _records(plan))
    assert result["status"] == "NOT_CERTIFIABLE"
    assert result["families"]["edge"]["lower_confidence_bound"] < 0
    assert result["families"]["edge"]["upper_confidence_bound"] > 0
    assert len(result["families"]["edge"]["per_task"]) == 120
    assert all(t["pairs"] == 10 for t in result["families"]["edge"]["per_task"].values())
    plan = rp.build_protocol(inventory, stage="formal", success_margin=-.05,
                             approval_reference="CPU fixture")
    result = rp.build_report(plan, _records(plan))
    assert result["status"] == "NOT_CERTIFIABLE"
    assert "exact deployments were not bound before capture" in result["pending"]


def test_formal_success_gate_reuses_the_existing_one_sided_score_method(inventory):
    plan = _formal(inventory)
    rows = _records(plan)
    report = rp.build_report(plan, rows)
    assert report["status"] == "PASS"
    assert report["task_quality_validated"] is True
    for family in rp.FAMILIES:
        outcomes = {arm: [Outcome(r["pair_id"], r["scene_seed"], r["task_id"], r["success"])
                          for r in rows if r["family"] == family and r["arm"] == arm]
                    for arm in rp.ARMS}
        direct = certify(outcomes["baseline"], outcomes["candidate"], margin=-.05,
                         interval="tango_one_sided95", min_pairs=1200, fail_on_task_collapse=False)
        actual = report["families"][family]["certificate"]
        assert actual["lower_confidence_bound"] == direct.lower_confidence_bound
        assert actual["verdict"] == direct.verdict
        assert actual["lower_confidence_level"] == .95


def test_large_global_regression_fails_even_if_other_family_passes(inventory):
    plan = _formal(inventory)
    rows = _records(plan)
    for r in rows:
        if r["family"] == "nano" and r["arm"] == "candidate":
            r["success"] = False
    report = rp.build_report(plan, rows)
    assert report["status"] == "FAIL"
    assert report["families"]["edge"]["task_quality_validated"] is True
    assert report["families"]["nano"]["task_quality_validated"] is False
    assert report["families"]["nano"]["collapsed_tasks_descriptive"]


def test_descriptive_task_collapse_does_not_override_the_approved_global_criterion(inventory):
    plan = _formal(inventory)
    rows = _records(plan)
    task_id = plan["task_ids"][0]
    for row in rows:
        if row["family"] == "nano" and row["task_id"] == task_id:
            row["success"] = row["arm"] == "baseline" and row["episode_index"] == 0
    report = rp.build_report(plan, rows)
    assert report["status"] == "PASS"
    nano = report["families"]["nano"]
    assert nano["task_collapse_gate"] is False
    assert nano["collapsed_tasks_descriptive"] == [task_id]
    assert nano["lower_confidence_bound"] > -.05


def test_legacy_unexecuted_drafts_cannot_certify_with_unapproved_secondary_gate(inventory):
    plan = _formal(inventory)
    plan["schema_version"] = 1
    plan["kind"] = "cosmos_robolab_paired_quality_v1"
    plan["acceptance"]["task_collapse_gate"] = True
    with pytest.raises(ValueError, match="unapproved task-collapse"):
        rp.validate_protocol(plan)


@pytest.mark.parametrize("field,value", [
    ("success", 1), ("success", None), ("status", "error"), ("reset_count", 1),
    ("reset_count", True), ("scene_seed", 0), ("episode_index", True),
    ("protocol_sha256", "0" * 64), ("checkpoint_revision", "f" * 40),
    ("robolab_revision", "f" * 40), ("execution_binding_sha256", "f" * 64),
    ("initial_state_sha256", "f" * 64), ("scene_config_sha256", "f" * 64),
    ("initial_observation_sha256", "f" * 64),
    ("asset_inventory_sha256", "f" * 64), ("simulator_fingerprint_sha256", "f" * 64),
    ("action_trace_sha256", None), ("model_seeds", []), ("generated_chunks", 2),
    ("executed_steps", 0), ("family", "dreamzero"), ("arm", "unknown"),
])
def test_malformed_or_mismatched_record_is_never_certified(inventory, field, value):
    plan = _plan(inventory)
    rows = _records(plan)
    rows[0][field] = value
    with pytest.raises(ValueError):
        rp.build_report(plan, rows)


@pytest.mark.parametrize("change", ["empty", "missing", "duplicate", "unexpected"])
def test_partial_or_extra_episode_coverage_is_rejected(inventory, change):
    plan = _plan(inventory)
    rows = _records(plan)
    if change == "empty":
        rows = []
    elif change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[0]["pair_id"] = "not a planned episode"
    with pytest.raises(ValueError):
        rp.build_report(plan, rows)


def test_different_termination_times_keep_shared_query_seed_prefix(inventory):
    plan = _plan(inventory)
    rows = _records(plan)
    row = rows[1]
    episode = plan["episodes"][0]
    row.update(executed_steps=33, generated_chunks=2,
               model_seeds=[rp.model_seed(episode, i) for i in range(2)])
    assert rp.build_report(plan, rows)["status"] == "NOT_CERTIFIABLE"


def _renderer_records(plan):
    rows = _records(plan)
    for row in rows:
        pair = row["pair_id"]
        unique = [pair, row["family"], row["arm"]]
        row.update(
            pairing_mode=rp.NATIVE_RENDERER_PAIRING,
            initial_observation_sha256=rp.digest([unique, "actual native image and observation"]),
            initial_nonvisual_observation_sha256=rp.digest([pair, "all nonvisual observation values"]),
            initial_image_schema_sha256=rp.digest([pair, "all image keys dtypes and shapes"]),
            raw_scene_config_sha256=rp.digest([unique, "complete raw session and scene"]),
            render_product_identity_sha256=rp.digest([unique, "actual native view handles"]),
            renderer_process_exit_code=0,
            renderer_process_completion_sha256=rp.digest([unique, "synthetic clean completion receipt"]),
            collector_source_sha256=rp.digest("synthetic CPU collector identity"),
        )
    return rows


def test_native_renderer_pairing_is_prospective_and_keeps_frozen_legacy_valid(inventory):
    legacy = _plan(inventory, report_only=True)
    native = _plan(inventory, report_only=True, pairing_mode=rp.NATIVE_RENDERER_PAIRING)
    assert legacy["schema_version"] == 2
    assert "pairing_mode" not in legacy["contract"]
    assert native["schema_version"] == 3
    assert native["contract"]["pairing_mode"] == rp.NATIVE_RENDERER_PAIRING
    assert native["acceptance"] == legacy["acceptance"]
    assert native["episodes"] == legacy["episodes"]
    assert rp.validate_protocol(native) == rp.digest(native)
    historical = json.loads((INVENTORY.parent.parent / "rtx_preparation" /
                             "robolab_smoke_live_bindings_v1.json").read_text())
    assert rp.validate_protocol(historical) == "1328c20ebc76c02eaf02cadedfe9a8e5b1dd8a1c97aa377481382ad6a70e610f"


def test_native_renderer_retains_different_real_images_without_pixel_certification(inventory):
    plan = _plan(inventory, report_only=True, pairing_mode=rp.NATIVE_RENDERER_PAIRING)
    rows = _renderer_records(plan)
    assert len({r["initial_observation_sha256"] for r in rows}) == len(rows)
    assert len({r["raw_scene_config_sha256"] for r in rows}) == len(rows)
    result = rp.build_report(plan, rows)
    assert result["status"] == "NOT_CERTIFIABLE"
    assert result["pairing_mode"] == rp.NATIVE_RENDERER_PAIRING
    assert result["task_quality_validated"] is False
    assert "not pixel equivalence" in result["limits"][-1]


@pytest.mark.parametrize("field", [
    "initial_state_sha256", "initial_nonvisual_observation_sha256",
    "initial_image_schema_sha256", "scene_config_sha256", "asset_inventory_sha256",
    "simulator_fingerprint_sha256", "execution_binding_sha256",
])
def test_native_renderer_still_rejects_nonvisual_scene_or_execution_drift(inventory, field):
    plan = _plan(inventory, pairing_mode=rp.NATIVE_RENDERER_PAIRING)
    rows = _renderer_records(plan)
    rows[1][field] = "f" * 64
    with pytest.raises(ValueError):
        rp.build_report(plan, rows)


@pytest.mark.parametrize("field", [
    "initial_observation_sha256", "raw_scene_config_sha256", "render_product_identity_sha256",
    "initial_nonvisual_observation_sha256", "initial_image_schema_sha256", "pairing_mode",
    "renderer_process_exit_code", "renderer_process_completion_sha256", "collector_source_sha256",
])
def test_native_renderer_requires_all_per_arm_original_evidence(inventory, field):
    plan = _plan(inventory, pairing_mode=rp.NATIVE_RENDERER_PAIRING)
    rows = _renderer_records(plan)
    del rows[0][field]
    with pytest.raises(ValueError):
        rp.build_report(plan, rows)


def test_pairing_contract_cannot_be_reinterpreted_after_capture(inventory):
    legacy = _plan(inventory)
    native = _plan(inventory, pairing_mode=rp.NATIVE_RENDERER_PAIRING)
    with pytest.raises(ValueError, match="another protocol"):
        rp.build_report(native, _records(legacy))
    altered = copy.deepcopy(legacy)
    altered["contract"]["pairing_mode"] = rp.NATIVE_RENDERER_PAIRING
    with pytest.raises(ValueError, match="explicitly bound"):
        rp.validate_protocol(altered)
    del native["contract"]["pairing_mode"]
    with pytest.raises(ValueError, match="explicitly bound"):
        rp.validate_protocol(native)
    with pytest.raises(ValueError, match="Unknown renderer pairing"):
        _plan(inventory, pairing_mode="ignore_scene_changes")


def test_native_renderer_contract_keeps_the_approved_success_margin_and_ci(inventory):
    legacy = _formal(inventory)
    native = _formal(inventory, pairing_mode=rp.NATIVE_RENDERER_PAIRING)
    legacy_rows, native_rows = _records(legacy), _renderer_records(native)
    # A material loss must fail identically despite harmless native pixel variation.
    for rows in (legacy_rows, native_rows):
        for row in rows:
            if row["family"] == "nano" and row["arm"] == "candidate":
                row["success"] = False
    old_report, new_report = rp.build_report(legacy, legacy_rows), rp.build_report(native, native_rows)
    assert old_report["status"] == new_report["status"] == "FAIL"
    assert native["acceptance"] == legacy["acceptance"]
    assert native["acceptance"]["success_margin"] == -.05
    for family in rp.FAMILIES:
        old, new = old_report["families"][family], new_report["families"][family]
        for key in ("baseline_success", "candidate_success", "delta", "lower_confidence_bound",
                    "upper_confidence_bound", "task_quality_validated", "task_collapse_gate"):
            assert old[key] == new[key]
        assert old["certificate"]["verdict"] == new["certificate"]["verdict"]


@pytest.mark.parametrize("pairing_mode", [None, rp.NATIVE_RENDERER_PAIRING])
@pytest.mark.parametrize("failure", [
    {"process_exit_code": 139}, {"renderer_process_exit_code": 139},
    {"renderer_process_exit_code": False}, {"native_app_close": "failed"},
    {"cleanup_errors": ["native environment close failed"]},
])
def test_completed_native_file_cannot_hide_explicit_process_or_cleanup_failure(inventory, pairing_mode, failure):
    plan = _plan(inventory, pairing_mode=pairing_mode)
    rows = _renderer_records(plan) if pairing_mode else _records(plan)
    rows[0].update(failure)
    assert rows[0]["status"] == "completed"
    with pytest.raises(ValueError):
        rp.build_report(plan, rows)


def test_cli_writes_new_draft_and_refuses_overwrite(inventory, tmp_path):
    source, target = tmp_path / "inventory.json", tmp_path / "plan.json"
    source.write_text(json.dumps(inventory))
    args = ["plan", "--inventory", str(source), "--stage", "formal", "--output", str(target)]
    assert rp.main(args) == 0
    assert json.loads(target.read_text())["acceptance"]["success_margin"] is None
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        rp.main(args)
    assert target.read_bytes() == before


def test_cli_report_only_exits_nonzero_without_certification(inventory, tmp_path):
    plan = _plan(inventory)
    protocol, records, output = [tmp_path / n for n in ("plan.json", "episodes.jsonl", "report.json")]
    protocol.write_text(json.dumps(plan))
    records.write_text("\n".join(json.dumps(r) for r in _records(plan)))
    assert rp.main(["report", "--protocol", str(protocol), "--records", str(records),
                    "--output", str(output)]) == 2
    assert json.loads(output.read_text())["task_quality_validated"] is False

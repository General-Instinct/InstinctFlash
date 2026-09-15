"""CPU checks of fixed coverage, native arguments, statistics and finite scheduling."""

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tested_timed_screen", HERE / "run_timed_screen_v1.py")
screen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = screen
spec.loader.exec_module(screen)


@pytest.fixture(scope="module")
def inputs():
    configuration = screen.read(HERE / "timed_screen_config_v1.json")
    return configuration, screen.load_inputs(configuration)


def test_only_output_roots_run_name_and_server_budget_change(inputs):
    configuration, values = inputs
    actual = values[0]
    expected = screen.read(configuration["original_config"]["local"])
    expected["run_id"] = configuration["run_id"]
    for host in ("thor", "renderer"):
        expected[host]["output_root"] = configuration["roots"][host]
    expected["limits"]["server_seconds"] = 14400
    assert actual == expected
    assert screen.sha(actual["protocol"]["local"]) == actual["protocol"]["sha256"]


def test_exact_24_native_jobs_and_fresh_unchanged_arguments(inputs):
    _, (config, plan, protocol, _, helper, _, _, _) = inputs
    jobs = []
    for cell in screen.formal.CELLS:
        for episode in plan["selection"]["episodes"]:
            reference = screen.formal.specifications(config, helper, 0, cell, episode)
            actual = screen.specifications(config, helper, cell, episode, remaining=3600)
            assert actual["command"] == reference["command"]
            assert actual["bound_files"] == reference["bound_files"]
            assert actual["environment"] == reference["environment"]
            assert actual["max_seconds"] == 3600
            argv = actual["command"]
            assert argv[argv.index("--pair-id") + 1] == episode["pair_id"]
            assert argv[argv.index("--protocol") + 1] == config["protocol"]["renderer"]
            assert argv[argv.index("--device") + 1] == "cuda:0"
            jobs.append(actual)
    assert len(jobs) == len({job["job_id"] for job in jobs}) == 24
    assert protocol["contract"]["steps"] == 4
    assert protocol["contract"]["guidance"] == 3.0
    assert protocol["contract"]["native_initial_resets"] == 2
    assert protocol["contract"]["executed_action_horizon"] == 32
    assert sum(row["max_episode_steps"] for row in plan["selection"]["episodes"]) == 4950


class FakeTransport:
    """No remote actions; only the controller schedule is exercised."""

    cleanup = False

    def __init__(self):
        self.spawns = []

    def remaining(self):
        return 3600

    def spawn(self, host, job):
        self.spawns.append((host, job["job_id"]))
        return {"cpu_test_only": True}


def make_controller(inputs, tmp_path, controller_type=None):
    configuration, values = inputs
    transport = FakeTransport()
    constructor = controller_type or screen.ScreenController
    return constructor(configuration, *values, transport, tmp_path)


def synthetic_records(controller, family="edge", *, successes=(True, True), candidate_count=6):
    """Synthetic count inputs, explicitly not native evidence."""
    result = []
    for arm, success in zip(("baseline", "candidate"), successes):
        count = 6 if arm == "baseline" else candidate_count
        for row in controller.episodes[:count]:
            result.append({**copy.deepcopy(row), "family": family, "arm": arm,
                "success": success, "generated_chunks": 1,
                "initial_state_sha256": "1" * 64, "scene_config_sha256": "2" * 64,
                "asset_inventory_sha256": "3" * 64, "initial_nonvisual_observation_sha256": "4" * 64,
                "initial_image_schema_sha256": "5" * 64, "simulator_fingerprint_sha256": "6" * 64,
                "execution_binding_sha256": controller.protocol["execution_bindings"][family][arm]})
    return result


@pytest.mark.parametrize("successes,expected_gate", [((True, True), False), ((False, True), True)])
def test_six_pair_numeric_bound_never_promotes_screen(inputs, tmp_path, successes, expected_gate):
    controller = make_controller(inputs, tmp_path)
    controller.records = synthetic_records(controller, successes=successes)
    controller.closed_cells = {"edge-" + route: {"status": "passed_complete_six"}
                               for route in ("eager_native", "runtime_selected")}
    controller.screen_reports()
    result = screen.read(tmp_path / "edge_screen.json")
    assert result["status"] == "SCREEN_COMPLETE"
    assert result["fixed_family_gate_evaluated"] is True
    assert result["fixed_family_gate_passed"] is expected_gate
    assert result["task_quality_validated"] is False and result["certificate"] is None
    if successes == (True, True):
        assert result["lower_confidence_bound"] == pytest.approx(-0.310783981694, abs=1e-8)
    assert screen.read(tmp_path / "nano_screen.json")["fixed_family_gate_evaluated"] is False


def test_missing_pair_or_unclean_cell_cannot_evaluate_fixed_gate(inputs, tmp_path):
    controller = make_controller(inputs, tmp_path)
    controller.records = synthetic_records(controller, candidate_count=5)
    controller.closed_cells = {"edge-" + route: {"status": "passed_complete_six"}
                               for route in ("eager_native", "runtime_selected")}
    controller.screen_reports()
    assert screen.read(tmp_path / "edge_screen.json")["fixed_family_gate_evaluated"] is False
    controller.records = synthetic_records(controller)
    controller.closed_cells["edge-runtime_selected"]["status"] = "partial_or_failed"
    controller.screen_reports()
    assert screen.read(tmp_path / "edge_screen.json")["fixed_family_gate_evaluated"] is False


class ScheduleController(screen.ScreenController):
    """Fake native boundaries, real run() order and completion conditions."""

    def prepare_hosts(self):
        self.trace = []

    def ensure_capacity(self, host, directory):
        pass

    def ready(self, server, cell):
        return {"cpu_test_only": True}

    def screen_reports(self):
        pass

    def collect_screen_episode(self, cell, episode, ordinal, offset, server):
        self.trace.append((cell, episode["pair_id"], ordinal, offset))
        row = {"generated_chunks": ordinal + 1}
        self.records.append(row)
        return row

    def close_screen_cell(self, server, cell, rows):
        self.closed_cells[cell] = {"status": "passed_complete_six" if len(rows) == 6 else "partial_or_failed"}
        self.server = None


def test_full_schedule_is_four_servers_24_jobs_resets_offsets_per_cell(inputs, tmp_path):
    controller = make_controller(inputs, tmp_path, ScheduleController)
    controller.run()
    expected = [(cell, episode["pair_id"], i, i * (i + 1) // 2)
                for cell in screen.formal.CELLS for i, episode in enumerate(controller.episodes)]
    assert controller.trace == expected
    assert len(controller.transport.spawns) == 4
    assert controller.state["status"] == "completed_fixed_SCREEN"
    assert controller.state["task_quality_validated"] is False


def test_expired_work_budget_starts_no_native_job(inputs, tmp_path, monkeypatch):
    controller = make_controller(inputs, tmp_path, ScheduleController)
    monkeypatch.setattr(screen.time, "time", lambda: controller.deadline - 299)
    with pytest.raises(screen.ScreenDeadline):
        controller.run()
    assert controller.transport.spawns == []
    assert controller.state["status"] == "stopped_partial_SCREEN"
    assert controller.state["task_quality_validated"] is False

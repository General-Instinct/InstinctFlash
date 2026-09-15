"""Public serving qualification must exercise protocol and history, not just a port."""
from copy import deepcopy
import json

import numpy as np
import pytest

from benchmarks.regression import serve_smoke as smoke
from benchmarks.regression.reproduce import ACTION_SHAPES, fixture_path, make_plan
from instinctflash.serving import msgpack_numpy


def metadata(cell):
    return {"model_id": cell["model_id"],
            "protocol": {"wire": "openpi-websocket-msgpack-numpy", "reset_extension": True},
            "precision": cell["expected_runtime_kwargs"].get("precision", "native"),
            "execution_policy": {"nfe": cell["effective_schedule"]["nfe"],
                                 "tier_ceiling": cell["expected_runtime_kwargs"].get("tier_ceiling", "bitexact")}}


class Connection:
    def __init__(self, cell, bad=False):
        self.sent = []
        self.replies = [metadata(cell)]
        for _ in range(2):
            self.replies.append({"server_timing": {"infer_ms": 1}})
            for _ in range(3):
                action = np.zeros(cell["action_shape"], np.float32)
                if bad:
                    action.flat[0] = float("nan")
                self.replies.append({"action": action, "server_timing": {"infer_ms": 2}})

    def recv(self, timeout):
        assert timeout > 0
        return msgpack_numpy.packb(self.replies.pop(0))

    def send(self, frame):
        self.sent.append(msgpack_numpy.unpackb(frame))


@pytest.mark.parametrize("family", ACTION_SHAPES)
def test_client_checks_two_complete_episodes_and_history(family):
    cell = smoke.selected_cell(make_plan(family))
    connection = Connection(cell)
    _, calls, actions = smoke.client_requests(connection, cell, fixture_path(), 2)
    assert not connection.replies
    assert actions.shape == (6, *ACTION_SHAPES[family])
    assert len(calls) == 6
    assert [m.get("reset") for m in connection.sent] == [True, None, None, None] * 2
    assert connection.sent[0]["prompt"] != connection.sent[4]["prompt"]
    if family == "va":
        assert [len(connection.sent[i]["obs"]) for i in (1, 2, 3)] == [1, 4, 8]
        assert all("executed_action" in connection.sent[i] for i in (1, 2, 3, 5, 6, 7))
    if family == "dreamzero":
        assert [len(connection.sent[i]["observation/exterior_image_0_left"]) for i in (1, 2, 3)] == [1, 4, 4]


def test_nonfinite_wire_action_fails():
    cell = smoke.selected_cell(make_plan("pi05"))
    with pytest.raises(ValueError, match="nonfinite"):
        smoke.client_requests(Connection(cell, bad=True), cell, fixture_path(), 2)


@pytest.mark.parametrize("change", ["model_id", "precision", "nfe", "tier", "wire"])
def test_server_must_advertise_the_requested_execution(change):
    cell = smoke.selected_cell(make_plan("va", "2v4a-fp8"))
    value = metadata(cell)
    if change == "model_id":
        value["model_id"] = "different/checkpoint"
    elif change == "precision":
        value["precision"] = "native"
    elif change == "nfe":
        value["execution_policy"]["nfe"] = {"video": 25, "action": 50}
    elif change == "tier":
        value["execution_policy"]["tier_ceiling"] = "bitexact"
    else:
        value["protocol"]["wire"] = "unknown"
    with pytest.raises(ValueError):
        smoke.validate_metadata(value, cell)


def test_cli_config_preserves_all_explicit_mode_options(tmp_path):
    from instinctflash.cli import ServeConfig
    from instinctflash.cli_config import parse_config
    plan = make_plan("dreamzero", "dynamic-fp8")
    cell = smoke.selected_cell(plan)
    config = smoke.serve_config(cell, tmp_path, 8000, None)
    path = tmp_path / "serve.json"
    path.write_text(json.dumps(config))
    parsed = parse_config(ServeConfig, [f"--config_path={path}"])
    # parse_config returns config and parser bookkeeping in current CLI.
    value = parsed[0] if isinstance(parsed, tuple) else parsed
    assert value.runtime.precision == "fp8"
    assert value.runtime.step_cache == "dynamic"
    assert value.runtime.tier_ceiling == "behavioral"
    assert value.serve.seed is None


def test_optional_seed_respects_actual_engine_contract(tmp_path):
    native = smoke.selected_cell(make_plan("pi05", "native"))
    fp8 = smoke.selected_cell(make_plan("pi05", "fp8"))
    assert smoke.serve_config(native, tmp_path, 8000, 9173)["serve"]["seed"] == 9173
    assert "seed" not in smoke.serve_config(fp8, tmp_path, 8000, None)["serve"]
    with pytest.raises(ValueError, match="FP8 serving engine does not support --seed"):
        smoke.serve_config(fp8, tmp_path, 8000, 9173)


def test_selected_cell_cannot_start_an_eager_reference_server():
    plan = make_plan("pi05")
    with pytest.raises(ValueError, match="Runtime"):
        smoke.selected_cell(plan, "pi05-eager_native")
    before = deepcopy(plan)
    assert smoke.selected_cell(plan)["arm"] == "runtime_selected"
    assert plan == before


def test_error_text_is_not_a_valid_msgpack_result():
    class ErrorConnection:
        def recv(self, timeout):
            return "server traceback"
    with pytest.raises(ValueError, match="error text"):
        smoke.receive(ErrorConnection(), 1)

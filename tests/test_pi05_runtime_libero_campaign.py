"""Refuse changed scene populations or incompatible action protocols before GPU work."""
import copy

import pytest

from benchmarks.vla.pi05_runtime_libero_campaign import frozen_jobs, validate_identity
from benchmarks.vla.util import ConfigurationError


def scenes():
    return {"scenes": {
        f"libero_10/{task}/{task * 10000 + repeat}": {
            "task": f"libero_10/{task}", "resolved_seed": task * 10000 + repeat,
            "requested_seed": task * 10000 + repeat, "init_state": [0.0], "prompt": "move",
        } for task in range(10) for repeat in range(2)
    }}


def test_frozen_population_preserves_task_seed_pairing():
    jobs = frozen_jobs(scenes())
    assert [(j["task"], j["seed"]) for j in jobs] == [
        (task, task * 10000 + repeat) for task in range(10) for repeat in range(2)]


@pytest.mark.parametrize("change", ["missing", "replacement_seed", "wrong_task", "extra"])
def test_refuses_changed_population(change):
    document = scenes()
    population = document["scenes"]
    if change == "missing":
        del population["libero_10/0/0"]
    elif change == "replacement_seed":
        population["libero_10/0/0"]["resolved_seed"] = 123
    elif change == "wrong_task":
        population["libero_10/0/0"]["task"] = "libero_10/1"
    else:
        population["libero_10/0/2"] = copy.deepcopy(population["libero_10/0/0"])
    with pytest.raises(ConfigurationError):
        frozen_jobs(document)


@pytest.mark.parametrize("field,value", [("n_action_steps", 10), ("action_dim", 32),
                                         ("action_nfe", 4), ("precision", "auto")])
def test_refuses_incompatible_server(field, value):
    identity = {
        "protocol": "pi05-public-libero-50-v1", "model_id": "lerobot/pi05_libero_finetuned_v044",
        "model_revision": "8e174154ef5f6c60a8da12ae99c303d8963138c1",
        "n_action_steps": 50, "action_dim": 7, "action_nfe": 10, "precision": "native",
    }
    validate_identity(identity)
    identity[field] = value
    with pytest.raises(ConfigurationError):
        validate_identity(identity)

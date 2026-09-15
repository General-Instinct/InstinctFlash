#!/usr/bin/env python3
"""The pi05 LIBERO schedule-sweep driver: request interpretation, refusals, spec compilation.

What is pinned here (CPU, no torch, no simulator):

  * schedule dispatch REFUSES rather than defaults — a sweep arm without a schedule, a
    misnamed phase, or a non-positive step count must never silently serve the baseline;
  * the closed-loop task grammar and the fixed-seed law (LIBERO init states are fixed files;
    an increment_until_stable request would break the seed -> init-state pairing);
  * the committed sweep spec `config/sweep.pi05_libero_nfe.json` compiles through the real
    `build_sweep_plan` against the committed screening profiles, every generated closed-loop
    job carries its arm's schedule, and the per-pair counterbalance holds — the plan the
    campaign runs is exactly the plan this test builds.
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmarks.vla.pi05_libero_driver as driver  # noqa: E402
from benchmarks.vla.instinctflash_driver import DriverRefusal  # noqa: E402
from benchmarks.vla.registry import load_registry  # noqa: E402
from benchmarks.vla.schedule_sweep import build_sweep_plan, load_sweep, repeat_arm_id  # noqa: E402
from tests.run_tests import run_module_tests  # noqa: E402

SPEC_PATH = ROOT / "benchmarks" / "vla" / "config" / "sweep.pi05_libero_nfe.json"
MODEL = "lerobot/pi05_libero_finetuned_v044"


def _request(**overrides) -> dict:
    request = {
        "schema_version": 1,
        "task": "libero_spatial/3",
        "requested_seed": 40103,
        "model": {
            "backbone": "pi05",
            "checkpoint": {"id": MODEL, "revision": "8e" + "0" * 38},
        },
        "suite": {"id": "libero_spatial", "kind": "closed_loop", "seed_strategy": "fixed"},
        "arm": {
            "id": "nfe2",
            "role": "treatment",
            "operating_point": {
                "name": "nfe2",
                "tier": "BEHAVIORAL",
                "schedule": {"nfe": {"action": 2}},
            },
        },
    }
    request.update(overrides)
    return request


def _refuses(fn, *args, needle: str) -> None:
    try:
        fn(*args)
    except DriverRefusal as error:
        assert needle in str(error), f"refusal does not name the law: {error}"
    else:
        raise AssertionError(f"{fn.__name__}{args!r} did not refuse")


def test_schedule_dispatch_refuses_rather_than_defaults() -> None:
    assert driver.schedule_steps(_request()) == 2
    no_schedule = _request()
    del no_schedule["arm"]["operating_point"]["schedule"]
    _refuses(driver.schedule_steps, no_schedule, needle="schedule-sweep")
    misnamed = _request()
    misnamed["arm"]["operating_point"]["schedule"] = {"nfe": {"video": 2}}
    _refuses(driver.schedule_steps, misnamed, needle="one stream")
    both = _request()
    both["arm"]["operating_point"]["schedule"] = {"nfe": {"action": 2, "video": 1}}
    _refuses(driver.schedule_steps, both, needle="one stream")
    zero = _request()
    zero["arm"]["operating_point"]["schedule"] = {"nfe": {"action": 0}}
    _refuses(driver.schedule_steps, zero, needle="positive")
    # the guidance leg of the operating point: pi05 has none, so a point may STATE that (and
    # get batch-1 forwards reported) but a point requesting a negative branch is refused, never
    # silently served at the baseline combine
    stated = _request()
    stated["arm"]["operating_point"]["schedule"] = {"nfe": {"action": 2}, "guidance": {"action": "none"}}
    assert driver.schedule_steps(stated) == 2
    guided = _request()
    guided["arm"]["operating_point"]["schedule"] = {"nfe": {"action": 2}, "guidance": {"action": 3}}
    _refuses(driver.schedule_steps, guided, needle="classifier-free")
    foreign = _request()
    foreign["arm"]["operating_point"]["schedule"] = {"nfe": {"action": 2}, "guidance": {"video": "cfg"}}
    _refuses(driver.schedule_steps, foreign, needle="one stream")


def test_closed_loop_task_grammar_and_seed_law() -> None:
    assert driver.parse_closed_loop_task(_request()) == ("libero_spatial", 3)
    _refuses(driver.parse_closed_loop_task, _request(task="robotwin/3"), needle="LIBERO")
    _refuses(driver.parse_closed_loop_task, _request(task="libero_spatial/x"), needle="LIBERO")
    mismatched = _request(task="libero_object/3")
    _refuses(driver.parse_closed_loop_task, mismatched, needle="does not belong")
    unstable = _request()
    unstable["suite"] = dict(unstable["suite"], seed_strategy="increment_until_stable")
    _refuses(driver.episode_request, unstable, Path("/tmp/x"), needle="fixed seeds")


def test_episode_order_is_fully_determined_by_the_plan_request() -> None:
    order = driver.episode_request(_request(), Path("/snapshots/v044"))
    assert order == {
        "op": "episode",
        "suite": "libero_spatial",
        "task_id": 3,
        "seed": 40103,
        "num_inference_steps": 2,
        "n_action_steps": driver.N_ACTION_STEPS,
        "snapshot": "/snapshots/v044",
        "model_revision": "8e" + "0" * 38,
    }
    assert driver.N_ACTION_STEPS == 10, "the locked eval protocol (research log 2026-08-21)"


def test_committed_sweep_spec_compiles_through_the_committed_profiles() -> None:
    spec = load_sweep(SPEC_PATH)
    assert [point["nfe"]["action"] for point in spec["points"]] == [5, 4, 3, 2, 1]
    assert spec["baseline"]["nfe"] == {"action": 10}
    for point in [spec["baseline"], *spec["points"]]:
        assert point["forwards_per_cycle"] == point["nfe"]["action"] + 1, (
            "pi05 cycle = 1 prefix prefill + N denoise forwards"
        )
    environ = dict(os.environ)
    os.environ["IFL_BENCH_DRIVER_REVISION"] = "test-revision"
    try:
        registry = load_registry()
        plans = {
            profile: build_sweep_plan(registry, copy.deepcopy(spec), profile, [MODEL])
            for profile in (
                "screen_libero_spatial", "screen_libero_object",
                "screen_libero_goal", "screen_libero_10",
            )
        }
    finally:
        os.environ.clear()
        os.environ.update(environ)
    for profile, plan in plans.items():
        suite_id = profile.removeprefix("screen_")
        assert plan["pair_count"] == 41, profile  # 10 tasks x 4 seeds + 1 latency pair
        assert plan["job_count"] == 41 * 7, profile  # 7 arms: baseline, 5 points, repeat
        arm_ids = [arm["id"] for arm in plan["arms"]]
        assert arm_ids == ["nfe10", "nfe5", "nfe4", "nfe3", "nfe2", "nfe1", repeat_arm_id(spec)]
        orders: dict[str, list[str]] = {}
        for job in plan["jobs"]:
            request = job["request"]
            if request["suite"]["kind"] == "closed_loop":
                from benchmarks.vla.adapters import validate_bound_adapter
                assert validate_bound_adapter(request)["id"] == "pi05-libero-schedule-v1"
            else:
                assert "adapter" not in request
            assert request["suite_id"] in (suite_id, "single_gpu_latency")
            schedule = request["arm"]["operating_point"]["schedule"]
            assert set(schedule["nfe"]) == {"action"}, "every arm carries its schedule"
            assert job["driver"]["revision"] == "test-revision"
            orders.setdefault(request["pair_id"], []).append(request["arm"]["id"])
        assert any(order[0] == "nfe10" for order in orders.values())
        assert any(order[-1] == "nfe10" for order in orders.values()), "counterbalanced"



def test_incompatible_or_missing_contract_refused_before_loading_checkpoint():
    from unittest.mock import patch
    from benchmarks.vla.adapters import bind_adapter
    from benchmarks.vla.util import ConfigurationError
    request = _request()
    request['model']['checkpoint']['revision'] = '8e174154ef5f6c60a8da12ae99c303d8963138c1'
    descriptor = {'adapter_id': 'pi05-libero-schedule-v1',
                  'command': [sys.executable, '-m', 'benchmarks.vla.pi05_libero_driver']}
    bind_adapter(request, descriptor)
    for change in ('checkpoint', 'binding'):
        bad = copy.deepcopy(request)
        if change == 'checkpoint': bad['model']['checkpoint']['id'] = 'lerobot/pi05_base'
        else: bad.pop('adapter')
        with patch.object(driver, 'resolve_snapshot') as load:
            try: driver.run_closed_loop({'request': bad})
            except ConfigurationError: pass
            else: raise AssertionError('incompatible request accepted')
            load.assert_not_called()


def test_closed_loop_preserves_controller_step_count():
    from unittest.mock import patch
    from benchmarks.vla.adapters import bind_adapter
    request = _request()
    request['model']['checkpoint']['revision'] = '8e174154ef5f6c60a8da12ae99c303d8963138c1'
    bind_adapter(request, {'adapter_id': 'pi05-libero-schedule-v1',
                          'command': [sys.executable, '-m', 'benchmarks.vla.pi05_libero_driver']})
    reply = {'model_revision': request['model']['checkpoint']['revision'], 'success': True,
             'finite': True, 'action_digest': 'a'*64, 'action_values': [0.]*(42*7), 'n_env_steps': 42, 'environment_fingerprint': 'b'*64}
    with patch('benchmarks.vla.pi05_scenes.load_scene', return_value=({'seed': 1}, {}, 'c'*64)), patch.object(driver, 'resolve_snapshot', return_value=Path('/pinned')), patch.object(driver, '_call_server', return_value=reply):
        result = driver.run_closed_loop({'request': request})
    assert result['metrics']['executed_steps'] == 42

def test_capture_variant_is_explicit_isolated_and_preserves_nfe() -> None:
    request = _request()
    assert driver.optimization_variant(request) == "stock"
    request["arm"]["operating_point"]["optimization"] = "unknown"
    _refuses(driver.optimization_variant, request, needle="unknown pi05 optimization")
    request["arm"]["operating_point"]["optimization"] = "instinctflash_capture"
    _refuses(driver.optimization_variant, request, needle="NFE=10")
    request["arm"]["operating_point"]["schedule"]["nfe"]["action"] = 10
    assert driver.episode_request(request, Path("/weights"))["optimization"] == "instinctflash_capture"
    assert driver.socket_path("a" * 40) != driver.socket_path("a" * 40, "instinctflash_capture")


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))

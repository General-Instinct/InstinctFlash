"""Frozen scenes, real executed-action logging, raw VA wire order and paired reporting."""
from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from benchmarks.vla import robotwin_driver as d
from benchmarks.vla.registry import load_registry
from benchmarks.vla.report import _paired
from benchmarks.vla.result import validate_result
from benchmarks.vla.util import ConfigurationError, sha256_file, sha256_json, write_json_atomic
from benchmarks.vla.wan_va_benchmark_server import SeededPolicy, snapshot_identity
sys.path.insert(0, str(ROOT / "tests"))
from run_tests import run_module_tests


MODEL = "robbyant/lingbot-va-posttrain-robotwin"


def job():
    registry = load_registry()
    suite = registry.suites["robotwin50_easy"]
    request = {"model_id": MODEL, "model": {"backbone": "wan_va", "checkpoint": {
        "id": MODEL, "revision": registry.models[MODEL]["revision"]}},
        "requested_seed": 50100, "task": "adjust_bottle", "pair_id": "pair", "suite_id": suite["id"],
        "suite": {k: copy.deepcopy(suite[k]) for k in ("id", "kind", "seed_strategy", "seed_max_attempts", "protocol", "required_metrics")},
        "dataset": registry.datasets["robotwin2_eval"],
        "arm": {"id": "a", "role": "control", "operating_point": {"name": "stock", "tier": "BITEXACT"}}}
    request["suite"]["protocol"]["bridge"] = d.PROTOCOL
    request["suite"]["protocol"]["evaluation_mode"] = "paused_simulation"
    from benchmarks.vla.adapters import bind_adapter
    bind_adapter(request, {"command": [sys.executable, "-m", "benchmarks.vla.robotwin_driver"]})
    return seal(request)


def seal(request):
    digest = sha256_json(request)
    return {"request": request, "request_sha256": digest, "job_id": digest[:24],
            "driver": {"revision": d.driver_revision()}}


def refuses(fn, needle):
    try:
        fn()
    except (ConfigurationError, RuntimeError) as error:
        assert needle in str(error), str(error)
    else:
        raise AssertionError("expected refusal: " + needle)


def observation():
    return {"observation": {name: {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}
                            for name in ("head_camera", "left_camera", "right_camera")},
            "joint_action": {"vector": np.zeros(14)}, "endpose": {
        "left_endpose": [0, 0, 0, 0, 0, 0, 1], "right_endpose": [0, 0, 0, 0, 0, 0, 1],
        "left_gripper": 0, "right_gripper": 0}}


def scene():
    return {"requested_seed": 50100, "resolved_seed": 50101, "task": "adjust_bottle",
            "suite_id": "robotwin50_easy", "prompt": "adjust the bottle",
            "initial_state_sha256": d.initial_state(observation())}


class Remote:
    def __init__(self):
        self.messages = []
    def reset_episode(self, prompt, seed):
        self.messages.append(("reset", prompt, seed))
        return {}
    def infer(self, value):
        self.messages.append(value)
        if value.get("compute_kv_cache"):
            return {}
        return {"action": np.ones((16, 2, 16))}


def test_raw_wire_requires_reset_and_commit_and_observed_history():
    remote = Remote()
    bridge = d.WanVaWireBridge(remote, scene())
    refuses(lambda: bridge.infer({"obs": {}}), "reset/commit")
    bridge.infer({"reset": True, "prompt": scene()["prompt"]})
    action = bridge.infer({"obs": {}, "video_guidance_scale": 99})["action"]
    assert "video_guidance_scale" not in remote.messages[-1]
    refuses(lambda: bridge.infer({"obs": {}}), "reset/commit")
    refuses(lambda: bridge.infer({"compute_kv_cache": True, "obs": [{}] * 8}), "4 observed")
    bridge.infer({"compute_kv_cache": True, "obs": [{}] * 4, "state": action})
    bridge.infer({"obs": {}})
    refuses(lambda: bridge.infer({"compute_kv_cache": True, "obs": [{}] * 4}), "8 observed")
    bridge.infer({"compute_kv_cache": True, "obs": [{}] * 8, "state": action})
    assert bridge.cycles == 2
    assert remote.messages[0] == ("reset", scene()["prompt"], 50101)


def test_nonfinite_remote_actions_are_not_successful_episodes():
    remote = Remote()
    remote.infer = lambda value: {"action": np.full((16, 2, 16), np.nan)}
    bridge = d.WanVaWireBridge(remote, scene())
    bridge.infer({"reset": True, "prompt": scene()["prompt"]})
    refuses(lambda: bridge.infer({"obs": {}}), "finite action")


def test_identity_and_scene_refusals_are_before_simulator_loading():
    value = job()
    assert d.validate_request(value)["task"] == "adjust_bottle"
    changed = copy.deepcopy(value)
    changed["request"]["task"] = "../other"
    refuses(lambda: d.validate_request(changed), "digest")
    changed = copy.deepcopy(value["request"])
    changed["model"]["backbone"] = "pi05"
    refuses(lambda: d.validate_request(seal(changed)), "no RoboTwin policy bridge")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scenes.json"
        manifest = {"protocol": d.PROTOCOL, "reset_policy": d.RESET_POLICY, "scenes": {d.scene_key(value["request"]): scene()}}
        write_json_atomic(path, manifest)
        ref = {"path": str(path), "sha256": sha256_file(path)}
        assert d.load_scene(value["request"], ref)[1] == scene()
        manifest["scenes"]["duplicate"] = scene()
        write_json_atomic(path, manifest)
        refuses(lambda: d.load_scene(value["request"], ref), "digest mismatch")
        ref["sha256"] = sha256_file(path)
        refuses(lambda: d.load_scene(value["request"], ref), "reuses")


class Env:
    def __init__(self):
        self.take_action_cnt = 0
        self.test_num = 0
        self.closed = 0
        self.plan_success = True
        self.render_freq = 0
    def setup_demo(self, **kwargs):
        self.current_seed = kwargs["seed"]
        self.take_action_cnt = 0
    def close_env(self, **kwargs):
        self.closed += 1
    def take_action(self, action, action_type):
        assert action_type == "ee"
        if self.take_action_cnt < 2:
            self.take_action_cnt += 1
    def get_obs(self):
        return observation()
    def play_once(self):
        return {"info": {}}
    def check_success(self):
        return self.current_seed % 2 == 1


def fake_client(env, *, fault=None):
    def evaluate(task, env, args, model, seed, **kw):
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True)
        prompt = client.generate_episode_descriptions(None)[0]["seen"][0]
        model.infer({"reset": True, "prompt": prompt})
        action = model.infer({"obs": {}})["action"]
        for _ in range(3):  # third action is ignored after terminal success
            env.take_action(np.arange(16), action_type="ee")
        if fault:
            raise RuntimeError(fault)
        model.infer({"compute_kv_cache": True, "obs": [{}] * 4, "state": action})
        env.test_num = 1
        return seed, 1
    def main(args):
        client.WebsocketClientPolicy(port=0)
        settings = {k: v for k, v in args.items() if k not in {"seed", "port", "test_num"}}
        client.eval_policy(args["task_name"], env, settings, None, 10000,
                           save_visualization=True, instruction_type="seen")
    client = SimpleNamespace(main=main, eval_policy=evaluate, WebsocketClientPolicy=lambda **kw: None,
                             generate_episode_descriptions=lambda *a: [{"seen": ["original"]}],
                             save_comparison_video=lambda **kw: None, UnStableError=type("Unstable", (Exception,), {}))
    return client


def test_rollout_hashes_controller_accepted_actions_and_restores_hooks():
    env = Env()
    client = fake_client(env)
    original = client.eval_policy
    with tempfile.TemporaryDirectory() as tmp:
        metrics = d.rollout(client, job()["request"], scene(), Remote(), tmp)
        assert metrics["success"] is True and metrics["executed_steps"] == 2
        assert metrics["cycles"] == 1
        assert len(__import__("json").loads((Path(tmp) / "executed_actions.json").read_text())) == 2
    assert client.eval_policy is original and env.closed > 0


def test_runtime_failure_never_becomes_a_completed_success():
    env = Env()
    client = fake_client(env, fault="simulator crash")
    original = client.eval_policy
    with tempfile.TemporaryDirectory() as tmp:
        refuses(lambda: d.rollout(client, job()["request"], scene(), Remote(), tmp), "simulator crash")
    assert client.eval_policy is original and env.closed > 0


def test_expert_preparation_is_bounded_and_does_not_reuse_scenes():
    env = Env()
    client = fake_client(env)
    with tempfile.TemporaryDirectory() as tmp:
        prepared = d.prepare_scene(client, job()["request"], tmp)
        assert prepared["resolved_seed"] == 50101
        prepared = d.prepare_scene(client, job()["request"], tmp, {50101})
        assert prepared["resolved_seed"] == 50103
        bad = job()["request"]
        bad["suite"]["seed_max_attempts"] = 1
        refuses(lambda: d.prepare_scene(client, bad, tmp), "no valid scene")


def test_expert_retries_only_known_unreachable_grasp_assertion():
    env = Env()
    original = env.play_once
    def play():
        if env.current_seed == 50100:
            raise AssertionError("target_pose cannot be None for move action.")
        return original()
    env.play_once = play
    with tempfile.TemporaryDirectory() as tmp:
        assert d.prepare_scene(fake_client(env), job()["request"], tmp)["resolved_seed"] == 50101
        def unexpected():
            raise AssertionError("unexpected simulator invariant")
        env.play_once = unexpected
        try:
            d.prepare_scene(fake_client(env), job()["request"], tmp)
        except AssertionError as error:
            assert str(error) == "unexpected simulator invariant"
        else:
            raise AssertionError("unexpected errors must abort preparation")


def test_server_seed_is_per_episode_and_frame_and_reset_is_acknowledged():
    draws = []
    class Model:
        def _infer(self, obs, frame_st_id=0):
            return {"frame": frame_st_id}
        def infer(self, obs):
            return {} if obs.get("reset") else self._infer(obs, frame_st_id=3)
    identity = {"model": "test"}
    policy = SeededPolicy(Model(), identity, draws.append)
    refuses(lambda: policy.infer({}), "seeded reset")
    response = policy.infer({"reset": True, "benchmark_seed": 42,
                             "benchmark_identity_sha256": sha256_json(identity)})
    assert response["benchmark_seed"] == 42
    policy.infer({})
    assert draws == [42, 45]
    refuses(lambda: policy.infer({"reset": True, "benchmark_seed": 42}), "identity mismatch")


def test_equal_seeds_with_different_instructions_are_not_pairs():
    a = job()
    b = copy.deepcopy(a)
    b["job_id"] = "other"
    b["request"]["arm"]["id"] = "b"
    results = {a["job_id"]: {"resolved_seed": 50101, "provenance": {"scene_sha256": "a"}},
               b["job_id"]: {"resolved_seed": 50101, "provenance": {"scene_sha256": "b"}}}
    pairs, issues = _paired({"jobs": [a, b], "control_arm": "a"}, results, "b")
    assert not pairs and "different frozen scenes" in issues[0]


def test_snapshot_identity_requires_real_weight_files_and_pinned_layout():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "models--org--model" / "snapshots" / ("a" * 40)
        path.mkdir(parents=True)
        refuses(lambda: snapshot_identity(path), "no transformer")
        (path / "transformer").mkdir()
        (path / "transformer/weights.safetensors").write_bytes(b"fixture, not loaded")
        identity = snapshot_identity(path)
        assert identity["model_id"] == "org/model" and identity["model_revision"] == "a" * 40
        (path / "transformer/weights.safetensors").write_bytes(b"changed")
        assert snapshot_identity(path)["checkpoint_sha256"] != identity["checkpoint_sha256"]


def test_quality_plan_budget_is_explicit_and_draft_cannot_run():
    arms = {"schema_version": 1, "control_arm": "a", "arms": []}
    for name, role in (("a", "control"), ("b", "treatment")):
        arms["arms"].append({"id": name, "role": role,
            "driver": {"command": [sys.executable, "-m", "benchmarks.vla.robotwin_driver"],
                       "environment": {}, "timeout_seconds": 600, "revision": "filled-by-builder"},
            "operating_point": {"name": name, "tier": "NUMERIC"}})
    with tempfile.TemporaryDirectory() as tmp:
        armfile = Path(tmp) / "arms.json"
        write_json_atomic(armfile, arms)
        plan = d.make_plan(armfile, Path(tmp) / "draft.json", "robotwin_quality_smoke", -0.01, 100)
        assert plan["pair_count"] == 2
        for entry in plan["jobs"]:
            d.validate_request(entry)
            assert entry["request"]["suite"]["protocol"]["margin"] == -0.01
            assert "scene_manifest" not in entry["request"]["arm"]["operating_point"]
        assert load_registry().suites["robotwin50_easy"]["protocol"]["margin"] == -0.05


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))

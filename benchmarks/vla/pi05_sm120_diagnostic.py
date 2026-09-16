"""Three-arm, source-locked LIBERO diagnosis; not a new qualification gate.

Reuse the completed campaign's scenes, calibration and rollout unchanged. The
official LeRobot arm receives the same BF16-rounded noise values (promoted to
FP32 for its action projection). This is deliberately not stock FP32 sampling.
Videos and replayable arrays are local diagnostic artifacts, not wheel assets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from benchmarks.vla import pi05_sm120_libero as matched
from benchmarks.vla.pi05_weights import verify_loaded_weights
from benchmarks.vla.util import sha256_file, write_json_atomic

ARMS = ("official", "native", "fp8")
PROTOCOL = "pi05-sm120-three-arm-diagnostic-v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def diagnostic_sources():
    return {**matched.source_hashes(), **{name: sha256_file(matched.ROOT / name) for name in (
        "benchmarks/vla/pi05_sm120_diagnostic.py", "benchmarks/vla/pi05_weights.py")}}


def validate_episode(row):
    if type(row["success"]) is not bool or type(row["steps"]) is not int or not 1 <= row["steps"] <= 280:
        raise ValueError("invalid success or step count")
    if row["scene"]["seed"] != row["seed"] or row["scene"]["init_state_index"] != row["seed"] % 50:
        raise ValueError("wrong frozen scene")
    hashes = row["noise_sha256"]
    if len(hashes) != (row["steps"] + 9) // 10 or any(
            not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
        raise ValueError("incomplete or invalid noise trace")


def validate_pair(left, right):
    for row in (left, right):
        validate_episode(row)
    common = min(len(left["noise_sha256"]), len(right["noise_sha256"]))
    if left["seed"] != right["seed"] or left["scene"] != right["scene"]:
        raise ValueError("unmatched seed or initial observation/state")
    if left["noise_sha256"][:common] != right["noise_sha256"][:common]:
        raise ValueError("unmatched policy noise")
    return common


def validate_trace(row, arrays):
    """Tie the portable receipt to actual actions, noise and simulator steps."""
    validate_episode(row)
    steps, calls = row["steps"], len(row["noise_sha256"])
    shapes = {"actions": (steps, 7), "states": (steps + 1, 8),
              "input_state": (calls, 8), "chunks": (calls, 10, 7), "noises": (calls, 50, 32)}
    for name, shape in shapes.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"invalid trace array: {name}")
    actions = arrays["actions"]
    if matched.action_digest(actions.astype(np.float64).ravel()) != row["action_digest"]:
        raise ValueError("trace actions differ from receipt")
    if not np.array_equal(actions, arrays["chunks"].reshape(-1, 7)[:steps]):
        raise ValueError("executed actions differ from predicted queue")
    if not np.array_equal(arrays["input_state"], arrays["states"][:-1:10]):
        raise ValueError("inference state differs from simulator observation")
    noises = np.asarray(arrays["noises"], dtype="<f4").view("<u4")
    if np.any(noises & 0xffff):
        raise ValueError("trace noise is not exactly BF16 representable")
    hashes = [hashlib.sha256((noise >> 16).astype("<u2").tobytes()).hexdigest() for noise in noises]
    if hashes != row["noise_sha256"]:
        raise ValueError("trace noise differs from receipt")
    goals = arrays["goal_satisfied"]
    if goals.shape != (steps + 1, len(row["goal_predicates"])) or goals.dtype != np.bool_:
        raise ValueError("incomplete success-predicate trace")
    if not goals.shape[1] or bool(goals[-1].all()) != row["success"]:
        raise ValueError("success does not match simulator predicates")


def baseline_identity(directory, task):
    names = ("plan.json", "calibration.json", "calibration.npz", "results.json",
             f"scenes-{task}.json", f"native-{task}.json", f"fp8-{task}.json")
    identity = {name: sha256_file(directory / name) for name in names}
    plan, result = read_json(directory / "plan.json"), read_json(directory / "results.json")
    if result["status"] != "PASS" or result["summary"]["pairs"] != 500:
        raise ValueError("requires the completed 500-pair baseline")
    if plan["source_sha256"] != matched.source_hashes():
        raise ValueError("qualified source changed; do not mix protocols")
    if identity["calibration.json"] != plan["calibration_sha256"]:
        raise ValueError("baseline calibration changed")
    if identity["calibration.npz"] != read_json(directory / "calibration.json")["archive_sha256"]:
        raise ValueError("baseline calibration archive changed")
    if result["plan"] != plan:
        raise ValueError("baseline result and plan differ")
    scenes = read_json(directory / f"scenes-{task}.json")
    for arm in ("native", "fp8"):
        name = f"{arm}-{task}.json"
        receipt = read_json(directory / name)
        if identity[name] != result["receipt_sha256"][name] or receipt["plan_sha256"] != identity["plan.json"]:
            raise ValueError("baseline receipt changed")
        if [r["seed"] for r in receipt["episodes"]] != list(range(40100, 40150)):
            raise ValueError("baseline requires all 50 unique seeds")
        for row in receipt["episodes"]:
            validate_episode(row)
            if row["scene"] != scenes[str(row["seed"])]:
                raise ValueError("baseline scene manifest changed")
    return identity, plan


class OfficialRuntime:
    """Unmodified LeRobot model and checkpoint processors, with explicit noise."""

    def __init__(self, checkpoint, prompt):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.compile_model, config.device, config.n_action_steps = False, "cuda", 10
        config.validate_features()
        if (config.type, config.chunk_size, config.max_action_dim, config.num_inference_steps) != ("pi05", 50, 32, 10):
            raise ValueError("unexpected official checkpoint contract")
        self.policy = PI05Policy.from_pretrained(checkpoint, config=config).eval()
        self.loaded_weights = verify_loaded_weights(self.policy, checkpoint)
        pre, self.post = make_pre_post_processors(config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
            postprocessor_overrides={"device_processor": {"device": "cuda"}})
        # Call the very same input adapter used by the numerical reference gate,
        # without constructing an unused FlashRT model in the official process.
        self.processor = SimpleNamespace(_prompt=prompt, _pre=pre,
            _preprocess_images=self.policy._preprocess_images, frontend=SimpleNamespace(num_views=2))
        self._noise_buf = torch.empty((50, 32), device="cuda", dtype=torch.bfloat16)

    def infer(self, observation):
        import torch
        from flash_rt.frontends.torch.pi05_checkpoint import Pi05CheckpointFrontend
        with torch.no_grad():
            _, batch = Pi05CheckpointFrontend.prepare(self.processor, observation)
            self._noise_buf.normal_()
            rng = torch.cuda.get_rng_state()
            actions = self.policy.predict_action_chunk(batch, noise=self._noise_buf.float()[None])
            actions = self.post(actions)[0, :10].float().cpu().numpy()
            if not torch.equal(rng, torch.cuda.get_rng_state()):
                raise ValueError("official inference consumed unexpected random numbers")
        return {"actions": actions}

    def close(self):
        self.policy = self.processor = self.post = self._noise_buf = None


class TraceRuntime:
    def __init__(self, runtime):
        self.runtime = runtime
        self.inputs, self.actions, self.noises = [], [], []

    @property
    def _noise_buf(self):
        return self.runtime._noise_buf

    def infer(self, observation):
        result = self.runtime.infer(observation)
        self.inputs.append({k: np.array(observation[k], copy=True) for k in ("image", "wrist_image", "state")})
        self.actions.append(result["actions"].copy())
        self.noises.append(self._noise_buf.float().cpu().numpy().copy())
        return result


class TraceEnvironment:
    """Observe the existing rollout without adding resets or simulator steps."""

    def __init__(self, vec, video):
        import imageio.v2 as imageio
        self.vec = vec
        self.states, self.actions, self.qpos, self.body_positions, self.goals = [], [], [], [], []
        self.writer = imageio.get_writer(str(video), fps=20, codec="libx264", quality=7,
                                         ffmpeg_params=["-threads", "1"])

    def __getattr__(self, name):
        return getattr(self.vec, name)

    def record(self, observation):
        env = self.vec.envs[0].unwrapped.inner.env
        self.states.append(observation["state"][0].copy())
        self.qpos.append(env.sim.data.qpos.copy())
        self.body_positions.append(env.sim.data.body_xpos.copy())
        self.goals.append([bool(env._eval_predicate(g)) for g in env.parsed_problem["goal_state"]])
        self.body_names = list(env.sim.model.body_names)
        self.goal_predicates = env.parsed_problem["goal_state"]
        pixels = observation["pixels"]
        frame = np.concatenate([pixels[k][0, ::-1, ::-1] for k in ("image", "image2")], axis=1)
        self.writer.append_data(np.ascontiguousarray(frame))

    def reset(self, **kwargs):
        result = self.vec.reset(**kwargs)
        self.record(result[0])
        return result

    def step(self, action):
        self.actions.append(action[0].copy())
        result = self.vec.step(action)
        self.record(result[0])
        return result

    def finish(self, path, runtime):
        self.writer.close()
        np.savez_compressed(path, actions=np.asarray(self.actions), states=np.asarray(self.states),
            qpos=np.asarray(self.qpos), body_positions=np.asarray(self.body_positions),
            goal_satisfied=np.asarray(self.goals), body_names=np.asarray(self.body_names),
            image=np.asarray([i["image"] for i in runtime.inputs]),
            wrist_image=np.asarray([i["wrist_image"] for i in runtime.inputs]),
            input_state=np.asarray([i["state"] for i in runtime.inputs]),
            chunks=np.asarray(runtime.actions), noises=np.asarray(runtime.noises))


def check_plan(output, gpu=False):
    plan = read_json(output / "plan.json")
    baseline = Path(plan["baseline"])
    identity, previous = baseline_identity(baseline, plan["task"])
    if plan["source_sha256"] != diagnostic_sources() or identity != plan["baseline_sha256"]:
        raise ValueError("diagnostic sources or baseline changed")
    if gpu:
        if matched.environment_identity() != previous["environment"]:
            raise ValueError("qualified GPU, dependencies, assets or binaries changed")
        checkpoint = Path(previous["config"]["checkpoint"])
        if sha256_file(checkpoint / "config.json") != plan["checkpoint_config_sha256"]:
            raise ValueError("checkpoint config changed")
        if sha256_file(checkpoint / "model.safetensors") != matched.MODEL_SHA256:
            raise ValueError("checkpoint weights changed")
        for name, expected in previous["processor_files"].items():
            if sha256_file(checkpoint / name) != expected:
                raise ValueError("checkpoint processor changed")
    return plan, previous


def worker(output, arm):
    plan, previous = check_plan(output, gpu=True)
    baseline, task = Path(plan["baseline"]), plan["task"]
    scenes = read_json(baseline / f"scenes-{task}.json")
    prompt = scenes[str(plan["seeds"][0])]["prompt"]
    matched._apply_eval_numeric_environment()
    matched.seed_everything(5090120)
    if arm == "official":
        runtime = OfficialRuntime(previous["config"]["checkpoint"], prompt)
    else:
        settings = previous["selected"] if arm == "fp8" else {}
        runtime = matched.make_runtime(previous["config"], arm, settings.get("bf16_layers", []))
        runtime.set_prompt(prompt)
        matched.seed_everything(5090120)
        calibration, _ = matched.calibration_frames(baseline, prompt)
        runtime.calibrate(calibration, percentile=settings.get("percentile", 99.9))
    result = {"arm": arm, "task": task, "plan_sha256": sha256_file(output / "plan.json"), "episodes": []}
    if arm == "official":
        result["loaded_weights"] = runtime.loaded_weights
    else:
        result["settings"] = settings
        result["scales"] = {k: float(v.download_new((1,), np.float32)[0])
                            for k, v in runtime.pipeline.fp8_act_scales.items()}
        old_scales = read_json(baseline / f"{arm}-{task}.json")["scales"]
        result["historical_scale_changes"] = {k: {"previous": old_scales[k], "current": v}
                                              for k, v in result["scales"].items() if v != old_scales[k]}
    directory = output / arm
    directory.mkdir(exist_ok=True)
    vec = matched.make_vec(task)
    try:
        for index, seed in enumerate(plan["seeds"]):
            video, trace = directory / f"{seed}.mp4", directory / f"{seed}.npz"
            recorded = TraceEnvironment(vec, video)
            observed = TraceRuntime(runtime)
            try:
                row = matched.run_episode(recorded, observed, seed, scenes[str(seed)])
                recorded.finish(trace, observed)
            finally:
                recorded.writer.close()
            row["goal_predicates"] = recorded.goal_predicates
            row["artifacts"] = {str(p.relative_to(output)): sha256_file(p) for p in (video, trace)}
            validate_episode(row)
            result["episodes"].append(row)
            write_json_atomic(output / f"{arm}.json", result)
            print(f"{arm} task={task} episode={index + 1}/{len(plan['seeds'])} seed={seed} "
                  f"success={row['success']} steps={row['steps']}", flush=True)
    finally:
        vec.close()
        runtime.close()


def summarize(output):
    from instinctflash.verify.certify import _tango_paired_score_bounds
    plan, previous = check_plan(output)
    receipts = {arm: read_json(output / f"{arm}.json") for arm in ARMS}
    for arm, receipt in receipts.items():
        if receipt["arm"] != arm or receipt["task"] != plan["task"] or receipt["plan_sha256"] != sha256_file(output / "plan.json"):
            raise ValueError("receipt belongs to a different arm or plan")
        if [r["seed"] for r in receipt["episodes"]] != plan["seeds"]:
            raise ValueError("missing, duplicate or unplanned episode seeds")
        for row in receipt["episodes"]:
            validate_episode(row)
            expected_artifacts = {f"{arm}/{row['seed']}.npz", f"{arm}/{row['seed']}.mp4"}
            if set(row["artifacts"]) != expected_artifacts:
                raise ValueError("missing or unexpected episode artifacts")
            for name, digest in row["artifacts"].items():
                path = (output / name).resolve()
                if not path.is_relative_to(output.resolve()) or sha256_file(path) != digest:
                    raise ValueError("trace or video changed")
            with np.load(output / f"{arm}/{row['seed']}.npz", allow_pickle=False) as arrays:
                validate_trace(row, arrays)
    comparisons, noise_calls = {}, {}
    for left, right in itertools.combinations(ARMS, 2):
        rows = list(zip(receipts[left]["episodes"], receipts[right]["episodes"]))
        noise_calls[f"{left}:{right}"] = sum(validate_pair(a, b) for a, b in rows)
        outcomes = [(a["success"], b["success"]) for a, b in rows]
        comparisons[f"{right}_minus_{left}"] = {
            "difference": sum(int(b) - int(a) for a, b in outcomes) / len(outcomes),
            "central_ci95": list(_tango_paired_score_bounds(outcomes, z=1.959963984540054)),
            "right_only": sum(b and not a for a, b in outcomes),
            "left_only": sum(a and not b for a, b in outcomes)}
    replay = {}
    for arm in ("native", "fp8"):
        old = {r["seed"]: r for r in read_json(Path(plan["baseline"]) / f"{arm}-{plan['task']}.json")["episodes"]}
        rows = receipts[arm]["episodes"]
        for row in rows:
            validate_pair(row, old[row["seed"]])
        replay[arm] = {key: [r["seed"] for r in rows if r[key] != old[r["seed"]][key]]
                       for key in ("success", "steps", "action_digest", "noise_sha256")}
    groups = Counter("/".join("success" if receipts[a]["episodes"][i]["success"] else "failure" for a in ARMS)
                     for i in range(len(plan["seeds"])))
    result = {"protocol": PROTOCOL, "status": "DIAGNOSTIC", "task": plan["task"],
        "prompt": receipts["official"]["episodes"][0]["scene"]["prompt"],
        "episodes_per_arm": len(plan["seeds"]), "arm_order": list(ARMS),
        "successes": {a: sum(r["success"] for r in v["episodes"]) for a, v in receipts.items()},
        "outcome_groups": dict(groups), "comparisons": comparisons, "historical_replay_mismatches": replay,
        "historical_scale_changes": {a: receipts[a]["historical_scale_changes"] for a in ("native", "fp8")},
        "matched_noise_calls": noise_calls, "official_loaded_weights": receipts["official"]["loaded_weights"],
        "source_sha256": plan["source_sha256"], "environment": previous["environment"],
        "plan": plan, "receipt_sha256": {a: sha256_file(output / f"{a}.json") for a in ARMS},
        "episodes": {a: v["episodes"] for a, v in receipts.items()},
        "limitations": ["Post-hoc diagnosis of a selected task, not a suite-wide non-inferiority gate.",
                        "Official LeRobot uses shared BF16-rounded explicit noise, not its stock FP32 sampler.",
                        "All arms share the qualified simulator and input adapter; common-mode adapter errors remain possible.",
                        "Trace and video collection adds overhead; these timings are not a speed qualification."]}
    write_json_atomic(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--worker", choices=ARMS)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        worker(output, args.worker)
        return
    if not args.report_only:
        if args.baseline is None or not 0 <= args.task < 10 or not 1 <= args.episodes <= 50:
            parser.error("requires --baseline, task in [0, 10), and episodes in [1, 50]")
        baseline = args.baseline.resolve()
        if output == baseline or output.is_relative_to(baseline):
            parser.error("diagnostics must not write inside the qualified baseline")
        identity, previous = baseline_identity(baseline, args.task)
        plan = {"protocol": PROTOCOL, "baseline": str(baseline), "baseline_sha256": identity,
                "source_sha256": diagnostic_sources(), "task": args.task,
                "checkpoint_config_sha256": sha256_file(Path(previous["config"]["checkpoint"]) / "config.json"),
                "seeds": list(range(40100, 40100 + args.episodes))}
        if (output / "plan.json").exists():
            if read_json(output / "plan.json") != plan:
                raise ValueError("resume requires identical plan and source")
        else:
            if output.exists() and any(output.iterdir()):
                raise ValueError("refusing to reuse a nonempty output directory")
            output.mkdir(parents=True, exist_ok=True)
            write_json_atomic(output / "plan.json", plan)
        for arm in ARMS:
            receipt = output / f"{arm}.json"
            if receipt.exists() and [r["seed"] for r in read_json(receipt)["episodes"]] == plan["seeds"]:
                continue
            with (output / f"{arm}.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "benchmarks.vla.pi05_sm120_diagnostic",
                    "--output", str(output), "--worker", arm], stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=7200)
    result = summarize(output)
    print(json.dumps({k: result[k] for k in ("status", "successes", "comparisons", "historical_replay_mismatches")}, indent=2))


if __name__ == "__main__":
    main()

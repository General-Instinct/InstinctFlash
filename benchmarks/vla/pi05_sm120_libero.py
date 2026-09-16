"""Source-locked FlashRT extension of pi05_libero_driver's matched episode protocol.

The Native arm is FlashRT BF16, not LeRobot Runtime. Both arms use the same
10-action queue, NFE=10, single synchronous environment, fixed initial states,
checked reset observations and Python/NumPy/Torch seeds. Calibration selection
uses demonstration frames only; no closed-loop result selects a candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from benchmarks.vla.pi05_libero_driver import N_ACTION_STEPS, _apply_eval_numeric_environment
from benchmarks.vla.instinctflash_driver import action_digest, seed_everything
from benchmarks.vla.pi05_scenes import checked_reset, observation_digest, state_record
from benchmarks.vla.util import sha256_file, sha256_json, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
MODEL_REVISION = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
MODEL_SHA256 = "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"
DATA_REVISION = "d86c0b94922572b3b657e1d1a3d01f0952ddeb46"
ASSET_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
PROTOCOL = "pi05-checkpoint-sm120-matched-v2"


def source_hashes():
    paths = [ROOT / f"benchmarks/vla/{n}.py" for n in (
        "pi05_sm120_libero", "pi05_libero_driver", "pi05_scenes", "instinctflash_driver", "util")]
    paths += sorted((ROOT / "serving/flash_rt").rglob("*.py"))
    paths.append(ROOT / "instinctflash/verify/certify.py")
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def environment_identity():
    import libero
    import lerobot
    import transformers
    from huggingface_hub import try_to_load_from_cache
    from libero.libero import get_libero_path
    from flash_rt import flash_rt_kernels, flash_rt_fa2
    import torch
    package = Path(libero.__file__).resolve().parent
    assets = Path(get_libero_path("assets")).resolve()
    if (package / "libero/assets").resolve() != assets:
        raise ValueError("LIBERO assets link differs from configured assets")
    def tree(folder):
        return sha256_json({str(p.relative_to(folder)): sha256_file(p)
            for p in sorted(folder.rglob("*")) if p.is_file()
            and not {"__pycache__", ".cache"}.intersection(p.parts) and p.suffix != ".pyc"})
    token_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        path = try_to_load_from_cache("google/paligemma-3b-pt-224", name)
        if not isinstance(path, str):
            raise ValueError(f"missing checkpoint tokenizer file: {name}")
        token_files[name] = sha256_file(Path(path))
    return {"packages": {k: importlib.metadata.version(k) for k in
            ("hf-libero", "robosuite", "mujoco", "bddl", "gymnasium", "torch", "numpy", "pyarrow",
             "lerobot", "transformers", "tokenizers", "torchcodec")},
        "processor_source_sha256": {"lerobot": tree(Path(lerobot.__file__).parent),
                                    "transformers": tree(Path(transformers.__file__).parent)},
        "tokenizer_sha256": token_files,
        "libero_tree_sha256": tree(package), "assets_tree_sha256": tree(assets),
        "binary_sha256": {"flash_rt_kernels": sha256_file(Path(flash_rt_kernels.__file__)),
                          "flash_rt_fa2": sha256_file(Path(flash_rt_fa2.__file__))},
        "device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "render_environment": {k: os.environ.get(k) for k in
            ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID", "PYTHONHASHSEED")}}


def resize(image):
    import torch
    from torch.nn import functional as F
    t = torch.from_numpy(np.array(image, copy=True, order="C")).permute(2, 0, 1)[None]
    return F.interpolate(t, (224, 224), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()


def prepare_calibration(config, output):
    """Freeze task-matched demonstration frames and disjoint numeric holdout rows."""
    import pyarrow.parquet as pq
    from PIL import Image
    dataset = Path(config["dataset"])
    tasks = pq.read_table(dataset / "meta/tasks.parquet").to_pandas()
    prompts = {int(row.task_index): str(index) for index, row in tasks.iterrows()}
    selected, files, arrays = {}, {}, {}
    for path in sorted((dataset / "data").rglob("*.parquet")):
        files[str(path.relative_to(dataset))] = sha256_file(path)
        table = pq.read_table(path)
        meta = table.select(["task_index", "episode_index", "frame_index"]).to_pandas()
        for task_id in sorted(set(meta.task_index)):
            prompt = prompts[int(task_id)]
            if prompt in selected:
                continue
            rows = meta.index[meta.task_index == task_id].to_numpy()
            if len(rows) < 16:
                continue
            picks = rows[np.linspace(0, len(rows) - 1, 16, dtype=int)]
            records = table.take(picks.tolist()).to_pylist()
            key = str(len(selected))
            images, states = [], []
            for row in records:
                images.append([np.asarray(Image.open(io.BytesIO(row[name]["bytes"])).convert("RGB"))
                               for name in ("observation.images.image", "observation.images.wrist_image")])
                states.append(row["observation.state"])
            arrays[key] = np.asarray(images, dtype=np.uint8)
            arrays[key + "_states"] = np.asarray(states, dtype=np.float32)
            selected[prompt] = {"key": key, "file": str(path.relative_to(dataset)),
                "rows": picks.tolist(), "task_index": int(task_id),
                "calibration_positions": list(range(0, 16, 2)),
                "holdout_positions": list(range(1, 16, 2))}
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    missing = [suite.get_task(t).language for t in range(10) if suite.get_task(t).language not in selected]
    if missing:
        raise ValueError(f"calibration shards lack task-matched real frames: {missing}")
    np.savez_compressed(output / "calibration.npz", **arrays)
    manifest = {"dataset_revision": DATA_REVISION, "files": files,
        "tasks_sha256": sha256_file(dataset / "meta/tasks.parquet"), "tasks": selected,
        "archive_sha256": sha256_file(output / "calibration.npz")}
    write_json_atomic(output / "calibration.json", manifest)
    return manifest


def calibration_frames(output, prompt):
    manifest = json.loads((output / "calibration.json").read_text())
    if sha256_file(output / "calibration.npz") != manifest["archive_sha256"]:
        raise ValueError("calibration archive changed")
    task = manifest["tasks"][prompt]
    with np.load(output / "calibration.npz", allow_pickle=False) as archive:
        images = archive[task["key"]]
        states = archive[task["key"] + "_states"]
    convert = lambda i: {"image": images[i, 0], "wrist_image": images[i, 1], "state": states[i]}
    return ([convert(i) for i in task["calibration_positions"]],
            [convert(i) for i in task["holdout_positions"]])


def make_vec(task_id):
    """hf-libero backend; exposes the frozen-scene interface shared with LeRobot."""
    import gymnasium as gym
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from lerobot.processor.env_processor import LiberoProcessorStep
    state_processor = LiberoProcessorStep()

    class SceneEnv(gym.Env):
        def __init__(self):
            suite = benchmark.get_benchmark_dict()["libero_spatial"]()
            task = suite.get_task(task_id)
            self.task_description = task.language
            self._init_states = torch.load(Path(get_libero_path("init_states")) /
                task.problem_folder / task.init_states_file, weights_only=False)
            if len(self._init_states) != 50:
                raise ValueError("qualification requires the pinned 50 initial states")
            self.init_state_id = 0
            self.observation_space = gym.spaces.Dict({"pixels": gym.spaces.Dict({
                k: gym.spaces.Box(0, 255, (256, 256, 3), np.uint8) for k in ("image", "image2")}),
                "state": gym.spaces.Box(-np.inf, np.inf, (8,), np.float32)})
            self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
            self.inner = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
                task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)

        @staticmethod
        def observation(raw):
            angle = state_processor._quat2axisangle(torch.as_tensor(raw["robot0_eef_quat"])[None])[0].numpy()
            state = np.concatenate([raw["robot0_eef_pos"], angle,
                                    raw["robot0_gripper_qpos"]]).astype(np.float32)
            return {"pixels": {"image": raw["agentview_image"], "image2": raw["robot0_eye_in_hand_image"]}, "state": state}

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.inner.seed(seed)
            self.inner.reset()
            raw = self.inner.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
            for _ in range(10):
                raw, _, _, _ = self.inner.step([0.] * 6 + [-1.])
            return self.observation(raw), {}

        def step(self, action):
            raw, reward, done, info = self.inner.step(action)
            success = bool(self.inner.check_success())
            return self.observation(raw), reward, bool(done or success), False, {"is_success": success}

        def close(self):
            self.inner.close()

    return gym.vector.SyncVectorEnv([SceneEnv], autoreset_mode=gym.vector.AutoresetMode.DISABLED)


def run_episode(vec, runtime, seed, expected=None):
    """Identical initial state and fresh policy RNG per episode, independent of calibration."""
    import torch
    seed_everything(seed)
    vec.envs[0].unwrapped.init_state_id = seed
    record = state_record(vec, seed)
    if expected is None:
        obs, _ = vec.reset(seed=[seed])
        scene = {"seed": seed, **record, "initial_observation_sha256": observation_digest(obs)}
    else:
        with checked_reset(vec, expected):
            obs, _ = vec.reset(seed=[seed])
        scene = expected
    # Reset may consume NumPy/Python randomness in the simulator; reset policy noise explicitly.
    seed_everything(seed)
    actions, noises, latency = [], [], []
    queue = []
    started = time.perf_counter()
    success = False
    for step in range(280):
        if not queue:
            pixels = obs["pixels"]
            observation_digest(obs)  # fail closed on broken/blank EGL, not an honest-looking zero
            if any(float(image.mean()) <= 3 or float(image.std()) <= 1 for image in pixels.values()):
                raise ValueError("blank EGL camera")
            inputs = {"image": np.ascontiguousarray(pixels["image"][0, ::-1, ::-1]),
                      "wrist_image": np.ascontiguousarray(pixels["image2"][0, ::-1, ::-1]),
                      "state": obs["state"][0]}
            t0 = time.perf_counter()
            predicted = runtime.infer(inputs)["actions"]
            latency.append((time.perf_counter() - t0) * 1000)
            if predicted.shape != (N_ACTION_STEPS, 7) or not np.isfinite(predicted).all():
                raise ValueError("invalid action chunk")
            noises.append(hashlib.sha256(runtime._noise_buf.view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
            queue = list(predicted)
        action = np.asarray(queue.pop(0), dtype=np.float32)
        actions.append(action.copy())
        obs, _, done, truncated, info = vec.step(action[None])
        success = bool(info.get("is_success", [False])[0])
        if success or bool(done[0]) or bool(truncated[0]):
            break
    trace = np.asarray(actions, dtype=np.float64)
    return {"seed": seed, "success": success, "scene": scene, "steps": len(trace),
        "action_digest": action_digest(trace.ravel()), "noise_sha256": noises,
        "latency_ms": latency, "wall_s": time.perf_counter() - started}


def make_runtime(config, mode, layers=()):
    import torch
    from flash_rt.frontends.torch.pi05_checkpoint import Pi05CheckpointFrontend
    if torch.cuda.get_device_capability(0) != (12, 0) or "5090" not in torch.cuda.get_device_name(0):
        raise ValueError("qualification requires RTX 5090 / SM120")
    _apply_eval_numeric_environment()
    seed_everything(5090120)
    torch.cuda.reset_peak_memory_stats()
    return Pi05CheckpointFrontend(config["checkpoint"], hardware="rtx_sm120", action_horizon=N_ACTION_STEPS,
                               use_fp8=mode != "native", bf16_encoder_down_layers=tuple(layers))


def worker(config, output, task_id, arm, study=False):
    import torch
    from libero.libero import benchmark
    plan = json.loads((output / "plan.json").read_text())
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("campaign source changed; create a new campaign directory")
    if environment_identity() != plan["environment"]:
        raise ValueError("campaign simulator, assets, GPU, or binary changed")
    if sha256_file(output / "calibration.json") != plan["calibration_sha256"]:
        raise ValueError("frozen calibration manifest changed")
    for name, expected in plan["processor_files"].items():
        if sha256_file(Path(config["checkpoint"]) / name) != expected:
            raise ValueError(f"checkpoint processor changed: {name}")
    settings = plan["candidates"][arm] if study else plan["selected"] if arm == "fp8" else {}
    runtime = make_runtime(config, arm, settings.get("bf16_layers", []))
    prompt = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(task_id).language
    calibration, holdout = calibration_frames(output, prompt)
    runtime.set_prompt(prompt)
    seed_everything(5090120)
    runtime.calibrate(calibration, percentile=settings.get("percentile", 99.9))
    scales = {k: float(v.download_new((1,), np.float32)[0]) for k, v in runtime.pipeline.fp8_act_scales.items()}
    result = {"task_id": task_id, "arm": arm, "settings": settings,
              "plan_sha256": sha256_file(output / "plan.json"), "scales": scales,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "resident_allocated_bytes": torch.cuda.memory_allocated()}
    if study:
        values, noises = [], []
        for i, obs in enumerate(holdout):
            seed_everything(123456 + i)
            values.append(runtime.infer(obs)["actions"])
            noises.append(runtime._noise_buf.float().cpu().numpy().copy())
        latencies = []
        for i in range(33):
            t0 = time.perf_counter()
            runtime.infer(holdout[i % len(holdout)])
            if i >= 12:
                latencies.append((time.perf_counter() - t0) * 1000)
        np.savez(output / f"study-{arm}.npz", actions=np.stack(values), noises=np.stack(noises))
        result["latency_ms"] = latencies
        write_json_atomic(output / f"study-{arm}.json", result)
        return
    vec = make_vec(task_id)
    try:
        scenes_path = output / f"scenes-{task_id}.json"
        scenes = json.loads(scenes_path.read_text()) if scenes_path.exists() else {}
        result["episodes"] = []
        for index in range(config.get("episodes", 50)):
            seed = 40100 + index
            expected = scenes.get(str(seed))
            episode = run_episode(vec, runtime, seed, expected)
            scenes[str(seed)] = episode["scene"]
            result["episodes"].append(episode)
            write_json_atomic(scenes_path, scenes)
            write_json_atomic(output / f"{arm}-{task_id}.json", result)
            print(f"{arm} task={task_id} episode={index+1} success={episode['success']} steps={episode['steps']}", flush=True)
    finally:
        vec.close()


def paired_statistics(pairs, margin=0.05):
    """Use the repository's tested Tango matched-pair score bounds."""
    from instinctflash.verify.certify import _tango_paired_score_bounds
    if not pairs:
        raise ValueError("no paired outcomes")
    native = np.array([int(p["native"]) for p in pairs])
    fp8 = np.array([int(p["fp8"]) for p in pairs])
    delta = fp8 - native
    binary_pairs = [(bool(p["native"]), bool(p["fp8"])) for p in pairs]
    lower, _ = _tango_paired_score_bounds(binary_pairs)
    interval = list(_tango_paired_score_bounds(binary_pairs, z=1.959963984540054))
    collapsed = [t for t in sorted({p["task_id"] for p in pairs})
                 if any(p["native"] for p in pairs if p["task_id"] == t)
                 and not any(p["fp8"] for p in pairs if p["task_id"] == t)]
    return {"pairs": len(pairs), "native_successes": int(native.sum()), "fp8_successes": int(fp8.sum()),
        "native_rate": float(native.mean()), "fp8_rate": float(fp8.mean()), "difference": float(delta.mean()),
        "paired_difference_ci95": interval,
        "one_sided_lower95": lower, "noninferiority_margin": margin,
        "noninferior": lower > -margin and not collapsed, "collapsed_tasks": collapsed,
        "fp8_only_success": int((delta == 1).sum()),
        "native_only_success": int((delta == -1).sum()),
        "method": "Tango paired score: central 95% CI, one-sided 95% lower decision bound; task-collapse guard"}


def report(output, tasks, episodes):
    plan = json.loads((output / "plan.json").read_text())
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("campaign source changed before reporting")
    pairs, per_task = [], []
    for task in tasks:
        left = json.loads((output / f"native-{task}.json").read_text())
        right = json.loads((output / f"fp8-{task}.json").read_text())
        if left["plan_sha256"] != sha256_file(output / "plan.json") or right["plan_sha256"] != left["plan_sha256"]:
            raise ValueError("receipt belongs to a different plan")
        if len(left["episodes"]) != episodes or len(right["episodes"]) != episodes:
            raise ValueError("incomplete matched campaign")
        for arm in (left, right):
            if [e["seed"] for e in arm["episodes"]] != list(range(40100, 40100 + episodes)):
                raise ValueError("missing, duplicate, or unexpected episode seeds")
        task_pairs = []
        for a, b in zip(left["episodes"], right["episodes"]):
            count = min(len(a["noise_sha256"]), len(b["noise_sha256"]))
            if a["seed"] != b["seed"] or a["scene"] != b["scene"] or a["noise_sha256"][:count] != b["noise_sha256"][:count]:
                raise ValueError("unmatched seed, initial state/observation, or policy noise")
            task_pairs.append({"task_id": task, "seed": a["seed"], "native": a["success"], "fp8": b["success"]})
        pairs.extend(task_pairs)
        per_task.append({"task_id": task, **paired_statistics(task_pairs)})
    stats = paired_statistics(pairs)
    result = {"protocol": PROTOCOL, "status": ("PASS" if stats["noninferior"] else "FAIL")
              if list(tasks) == list(range(10)) and episodes == 50 else "SCREEN",
              "summary": stats, "tasks": per_task, "pairs": pairs,
              "plan": json.loads((output / "plan.json").read_text()),
              "calibration": json.loads((output / "calibration.json").read_text()),
              "receipt_sha256": {p.name: sha256_file(p) for p in sorted(output.glob("*.json")) if p.name != "results.json"}}
    write_json_atomic(output / "results.json", result)
    return result


def run_child(config_path, output, task, arm, study=False):
    command = [sys.executable, "-m", "benchmarks.vla.pi05_sm120_libero", "--worker", str(config_path),
               "--task", str(task), "--arm", arm]
    if study:
        command.append("--study")
    with (output / f"{'study-' if study else ''}{arm}-{task}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)


def campaign(config_path):
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    if set(config) - {"checkpoint", "dataset", "output", "tasks", "episodes"}:
        raise ValueError("unknown campaign configuration key")
    if type(config.get("episodes", 50)) is not int or not 1 <= config.get("episodes", 50) <= 50:
        raise ValueError("episodes must be in [1, 50]")
    tasks = config.get("tasks", list(range(10)))
    if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks) or any(type(t) is not int or not 0 <= t < 10 for t in tasks):
        raise ValueError("tasks must be unique integer IDs in [0, 10)")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if sha256_file(Path(config["checkpoint"]) / "model.safetensors") != MODEL_SHA256:
        raise ValueError("checkpoint does not match pinned model")
    if (output / "plan.json").exists():
        plan = json.loads((output / "plan.json").read_text())
        if plan["source_sha256"] != source_hashes() or plan["config"] != config:
            raise ValueError("resume requires identical config and sources")
    else:
        prepare_calibration(config, output)
        plan = {"protocol": PROTOCOL, "config": config, "source_sha256": source_hashes(),
            "calibration_sha256": sha256_file(output / "calibration.json"),
            "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "assets_revision": ASSET_REVISION, "action_horizon": 10, "computed_action_chunk": 50, "nfe": 10,
            "processor": "checkpoint LeRobot state-token/image and MEAN_STD action processors",
            "processor_files": {p.name: sha256_file(p) for p in sorted(Path(config["checkpoint"]).glob("policy_*")) if p.is_file()},
            "replan_steps": 10, "max_steps": 280, "wait_steps": 10,
            "seeds": list(range(40100, 40150)), "noninferiority_margin": .05,
            "analysis": "Tango paired score one-sided 95% lower bound; task-collapse guard",
            "environment": environment_identity(),
            "selection": "highest minimum holdout action cosine among speedup >=1.5; fastest qualified candidate otherwise",
            "candidates": {"native": {}, "p99": {"percentile": 99.0},
                "p999": {"percentile": 99.9}, "p100": {"percentile": 100.0}}}
        write_json_atomic(output / "plan.json", plan)
    if "selected" not in plan:
        for arm in ("native", "p99", "p999", "p100"):
            run_child(config_path, output, 7, arm, study=True)
        scales = json.loads((output / "study-p999.json").read_text())["scales"]
        median = float(np.median(list(scales.values())))
        layers = sorted(int(k.rsplit("_", 1)[1]) for k, v in scales.items()
                        if k.startswith("encoder_ffn_down_w_") and v > 20 * median)
        plan["candidates"]["mixed"] = {"percentile": 99.9, "bf16_layers": layers}
        write_json_atomic(output / "plan.json", plan)
        run_child(config_path, output, 7, "mixed", study=True)
        with np.load(output / "study-native.npz") as data:
            baseline, baseline_noise = data["actions"].copy(), data["noises"].copy()
        native_ms = float(np.median(json.loads((output / "study-native.json").read_text())["latency_ms"]))
        scores = {}
        for arm in ("p99", "p999", "p100", "mixed"):
            with np.load(output / f"study-{arm}.npz") as data:
                if not np.array_equal(baseline_noise, data["noises"]):
                    raise ValueError("study noise is not matched")
                a, b = baseline.reshape(len(baseline), -1).astype(float), data["actions"].reshape(len(baseline), -1).astype(float)
                cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            timing = json.loads((output / f"study-{arm}.json").read_text())
            scores[arm] = {"min_cosine": float(cosine.min()), "mean_abs_error": float(np.abs(a - b).mean()),
                           "speedup": native_ms / float(np.median(timing["latency_ms"]))}
        qualified = [a for a in scores if scores[a]["min_cosine"] >= .98]
        if not qualified:
            raise ValueError(f"no numerically qualified calibration: {scores}")
        fast = [a for a in qualified if scores[a]["speedup"] >= 1.5]
        chosen = max(fast, key=lambda a: scores[a]["min_cosine"]) if fast else max(qualified, key=lambda a: scores[a]["speedup"])
        plan["selected"] = plan["candidates"][chosen]
        plan["selected_name"], plan["study_scores"] = chosen, scores
        plan["performance_target_met"] = scores[chosen]["speedup"] >= 1.5
        write_json_atomic(output / "plan.json", plan)
        print(f"Calibration selected: {chosen}; {scores}", flush=True)
    tasks, episodes = config.get("tasks", list(range(10))), config.get("episodes", 50)
    for task in tasks:
        for arm in (("native", "fp8") if task % 2 == 0 else ("fp8", "native")):
            receipt = output / f"{arm}-{task}.json"
            if receipt.exists() and len(json.loads(receipt.read_text()).get("episodes", [])) == episodes:
                continue
            run_child(config_path, output, task, arm)
            print(f"Completed {arm}, task {task}, {episodes} episodes", flush=True)
    result = report(output, tasks, episodes)
    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--study", action="store_true")
    args = p.parse_args()
    cfg = json.loads(args.worker.read_text())
    worker(cfg, Path(cfg["output"]), args.task, args.arm, args.study)

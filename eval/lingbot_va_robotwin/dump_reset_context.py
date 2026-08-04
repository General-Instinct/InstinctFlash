#!/usr/bin/env python3
"""Capture real conditioning contexts from RoboTwin resets. CLIENT-SIDE ($IWM_CLIENT_PY).

Stage one of the PDD conditioning pipeline. A chunk-0 video training context is exactly
(observation, prompt) -- measured, see probe_chunk0_cache.py -- and both come from a sim reset with
no policy in the loop. So collecting them needs no rollout at all: reset, read the cameras, write.

WHY THIS IS A SEPARATE PROCESS FROM THE ENCODER. The client needs sapien 3.0.0b1 on torch 2.4 and
the server needs torch 2.9 / diffusers 0.36; they are dependency-incompatible, which is why upstream
put a websocket between them. Rather than smuggle tensors across that boundary, this dumps plain
numpy and `encode_reset_context.py` picks it up in the server env.

The observation format is `format_obs` from the real eval client, copied deliberately rather than
imported: importing would drag in the whole evaluation module, and the point is to produce bytes the
server already knows how to eat.

LAUNCH IT EXACTLY LIKE THIS. Two environment requirements, and each fails in a way that looks
like the other's problem:

    cd "$ROBOTWIN_ROOT" && env ROBOTWIN_ROOT="$ROBOTWIN_ROOT" PYTHONPATH="$ROBOTWIN_ROOT" \
      PYTHONWARNINGS=ignore::UserWarning CUDA_VISIBLE_DEVICES=0 \
      "$IWM_CLIENT_PY" -u <abs path>/dump_reset_context.py --tasks adjust_bottle --episodes 1 --out DIR

  * CUDA_VISIBLE_DEVICES must be set. With all 8 GPUs visible, sapien/Vulkan initialisation HANGS
    indefinitely -- no output, no error, no traceback. Thirty minutes of silence looks exactly like a
    slow sim. run_eval.sh always sets it per worker, which is why the harness never hit this.
  * PYTHONPATH must include ROBOTWIN_ROOT even when cwd is already ROBOTWIN_ROOT. Running a script by
    ABSOLUTE path puts that script's directory on sys.path[0], not the working directory, so
    RoboTwin's `description` package never resolves.

With only the first fix it fails fast on the import; with only the second it hangs silently.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from envs.utils.create_actor import UnStableError  # noqa: E402
from description.utils.generate_episode_instructions import (  # noqa: E402
    generate_episode_descriptions,
)

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/home/ubuntu/RoboTwin")
sys.path.insert(0, ROBOTWIN_ROOT)


def class_decorator(task_name):
    """Same import dance the eval client does (eval_polict_client_openpi.py:263)."""
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except AttributeError as e:
        raise SystemExit(f"No {task_name} class in envs.{task_name}") from e


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def build_args(task_name: str, task_config: str = "demo_clean"):
    with open(f"{ROBOTWIN_ROOT}/task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config
    with open(f"{ROBOTWIN_ROOT}/task_config/_camera_config.yml", "r", encoding="utf-8") as f:
        cams = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["head_camera_h"] = cams[args["camera"]["head_camera_type"]]["h"]
    args["head_camera_w"] = cams[args["camera"]["head_camera_type"]]["w"]

    # Mirrors eval_polict_client_openpi.py:320-352 exactly: the table lives in task_config/ under
    # a LEADING-UNDERSCORE name, and file_path is used as-is (relative to ROBOTWIN_ROOT, which is
    # why this must run with that as cwd).
    with open(f"{ROBOTWIN_ROOT}/task_config/_embodiment_config.yml", "r", encoding="utf-8") as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    emb = args["embodiment"]
    def _path(name):
        fp = emb_types[name]["file_path"]
        if fp is None:
            raise SystemExit(f"no embodiment file for {name}")
        return fp
    if len(emb) == 1:
        args["left_robot_file"] = args["right_robot_file"] = _path(emb[0])
        args["dual_arm_embodied"] = True
    else:
        args["left_robot_file"], args["right_robot_file"] = _path(emb[0]), _path(emb[1])
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def format_obs(observation, prompt):
    """Verbatim from eval_polict_client_openpi.py:423 -- the exact dict the server is sent."""
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": observation["joint_action"]["vector"],
        "task": prompt,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=1, help="resets per task")
    ap.add_argument("--seed", type=int, default=0, help="matches run_eval.sh's --seed 0")
    ap.add_argument("--task-config", default="demo_clean")
    ap.add_argument("--instruction-type", default="seen",
                    help="matches the eval client's instruction_type (:308)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seed-skips", type=int, default=20,
                    help="how many unstable seeds to skip before giving up on a task")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st_seed = 10000 * (1 + a.seed)          # the official RoboTwin seed sequence
    n = 0

    for task in a.tasks:
        args = build_args(task, a.task_config)
        env = class_decorator(task)
        # SOME SEEDS ARE PHYSICALLY UNSTABLE and RoboTwin refuses them: `_base_task._init_task_env_`
        # raises UnStableError when an object has not settled. That is not an error to propagate --
        # the real eval client advances the seed and retries (eval_policy_client.py:396-403), and a
        # collector that instead died would leave a task short and bias the context pool toward
        # whatever tasks happened to have stable early seeds.
        ep, seed, attempts = 0, st_seed, 0
        max_attempts = a.episodes + a.max_seed_skips
        while ep < a.episodes and attempts < max_attempts:
            attempts += 1
            try:
                env.setup_demo(now_ep_num=ep, seed=seed, is_test=True, **args)
            except UnStableError as e:
                print(f"  {task} seed{seed}: unstable, skipping ({e})", flush=True)
                try:
                    env.close_env()
                except Exception:
                    pass
                seed += 1
                continue
            # The instruction is NOT available straight after setup_demo -- get_instruction()
            # returns None until one is set. The real client derives it from the episode info that
            # play_once() produces, then set_instruction()s it (client :518, :554-557). Reproduced
            # here, because text_emb is required conditioning for the video stream and a 'None'
            # prompt would silently train against the wrong embedding.
            # play_once() RUNS THE WHOLE DEMO, so the scene must be reset again afterwards or
            # get_obs() would return the end-of-episode state instead of the reset state. The client
            # does exactly this: play_once at :518, then a second setup_demo at :552 before any
            # observation is read at :598.
            episode_info = env.play_once()
            results = generate_episode_descriptions(task, [episode_info["info"]], 1)
            instruction = np.random.choice(results[0][a.instruction_type])
            env.setup_demo(now_ep_num=ep, seed=seed, is_test=True, **args)   # back to the reset state
            env.set_instruction(instruction=instruction)
            prompt = env.get_instruction()
            if isinstance(prompt, (list, tuple)):
                prompt = prompt[0]
            if prompt in (None, "None", ""):
                raise SystemExit(
                    f"{task} ep{ep}: empty instruction after set_instruction. Refusing to write a "
                    f"context with no prompt -- it would train against the wrong text embedding.")
            obs = env.get_obs()
            f = format_obs(obs, prompt)
            path = out / f"{task}__ep{ep}__seed{seed}.npz"
            np.savez_compressed(
                path,
                cam_high=f["observation.images.cam_high"],
                cam_left_wrist=f["observation.images.cam_left_wrist"],
                cam_right_wrist=f["observation.images.cam_right_wrist"],
                state=np.asarray(f["observation.state"], dtype=np.float32),
                prompt=np.array(str(prompt)),
                task=np.array(task),
                seed=np.array(seed),
            )
            n += 1
            print(f"  {path.name}  cam_high={f['observation.images.cam_high'].shape} "
                  f"state={np.asarray(f['observation.state']).shape}  prompt={str(prompt)[:52]!r}",
                  flush=True)
            try:
                env.close_env()
            except Exception:
                pass
            ep += 1
            seed += 1

        if ep < a.episodes:
            # Say so rather than exit 0 with a short pool. A context set that quietly covered 40 of
            # 50 tasks would bias training toward whatever survived, and nothing downstream would
            # report it.
            print(f"  {task}: only {ep}/{a.episodes} after {attempts} attempts "
                  f"({a.max_seed_skips} unstable-seed skips allowed)", flush=True)
    print(f"\nwrote {n} conditioning contexts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the Thor-side assets for the T3 REAL certificate (runs on the H100 sim box).

    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl /home/ubuntu/tools/pi05env313/bin/python \
        make_calib_and_synth.py <out_dir>

Produces:
    <out_dir>/calib_obs.npz     N real LIBERO frames in ENGINE WIRE FORM (fp16 HWC [-1,1],
                                rot180 like LiberoProcessorStep, resize_with_pad 224) — the
                                per-length fp8 calibration set. Keys: image[N], wrist_image[N].
    <out_dir>/thor_assets.json  {"synth_ids": {L: [ids]}, "replay": {...}} — for every prompt
                                token length L in the PROVEN range, one well-formed
                                'Task: ..., State: ...;\nAction: ' prompt whose lerobot
                                tokenization (AutoTokenizer paligemma-3b-pt-224, the exact
                                TokenizerProcessorStep call) is exactly L ids. Content is a
                                real libero_spatial task text + real-format state bins so the
                                calibration activations are representative.

Why the range [41, 62] is closed for libero_spatial: state is 8 dims discretized to bins in
{-1, 0..255}; measured with the eval tokenizer, "-1" contributes the same token count as a
1-digit bin, so all-1-digit is the floor and all-3-digit the ceiling over the 10 task texts.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIN_LEN, MAX_LEN = 41, 62
STATE_DIM = 8
CALIB_TASKS = [0, 2, 4, 6, 8]          # spread over the suite for representative pixels
SEED = 1000


def lerobot_tokenize(tok, text: str) -> list[int]:
    """The exact TokenizerProcessorStep call (max_length padding stripped by the mask)."""
    out = tok(text, max_length=200, truncation=True, padding="max_length",
              padding_side="right", return_tensors="np")
    ids = out["input_ids"][0]
    mask = out["attention_mask"][0].astype(bool)
    return [int(x) for x in ids[mask]]


def build_prompt(task_text: str, bins) -> str:
    cleaned = task_text.strip().replace("_", " ").replace("\n", " ")
    state_str = " ".join(str(int(b)) for b in bins)
    return f"Task: {cleaned}, State: {state_str};\nAction: "


def synth_ids(tok, tasks: list[str]):
    """For each target length, a real prompt tokenizing to exactly that many ids."""
    got: dict[int, dict] = {}
    for task_text in tasks:
        for n3 in range(STATE_DIM + 1):
            for n2 in range(STATE_DIM + 1 - n3):
                n1 = STATE_DIM - n3 - n2
                bins = [200] * n3 + [50] * n2 + [7] * n1
                prompt = build_prompt(task_text, bins)
                ids = lerobot_tokenize(tok, prompt)
                L = len(ids)
                if MIN_LEN <= L <= MAX_LEN and L not in got:
                    got[L] = {"ids": ids, "prompt": prompt}
        if len(got) == MAX_LEN - MIN_LEN + 1:
            break
    missing = [L for L in range(MIN_LEN, MAX_LEN + 1) if L not in got]
    if missing:
        raise SystemExit(f"could not synthesise prompts for lengths {missing}")
    return got


def engine_frames(task: int):
    """Reset the env, return (image, wrist) in engine wire form — the proxy's exact transform."""
    import torch
    from lerobot.envs.factory import make_env, make_env_config
    from remote_policy import to_engine_frame
    cfg = make_env_config("libero", task="libero_spatial", task_ids=[task])
    envs = make_env(cfg, n_envs=1, use_async_envs=False)
    env = next(iter(envs["libero_spatial"].values()))  # keyed by task id
    obs, _ = env.reset(seed=[SEED], options={})
    frames = []
    for key in ("image", "image2"):
        img = torch.from_numpy(np.asarray(obs["pixels"][key])).to(torch.float32) / 255.0
        img = img.permute(0, 3, 1, 2)          # BHWC -> BCHW
        img = torch.flip(img, dims=[2, 3])     # LiberoProcessorStep rot180
        frames.append(to_engine_frame(img))
    env.close()
    return frames


def main() -> int:
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    from libero.libero import benchmark
    bm = benchmark.get_benchmark_dict()["libero_spatial"]()
    tasks = [bm.get_task(i).language for i in range(bm.n_tasks)]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    got = synth_ids(tok, tasks)

    # Null-control prompts: task 0 text, two DIFFERENT real-format State suffixes with
    # DIFFERENT token lengths — the state-liveness + no-recapture probe pair.
    bins_a = [128, 137, 99, 202, 115, 143, 27, 241]     # mixed 2/3-digit
    bins_b = [7, 3, 99, 202, 115, 143, 27, 8]           # more 1-digit -> shorter
    prompt_a = build_prompt(tasks[0], bins_a)
    prompt_b = build_prompt(tasks[0], bins_b)
    ids_a = lerobot_tokenize(tok, prompt_a)
    ids_b = lerobot_tokenize(tok, prompt_b)
    assert len(ids_a) != len(ids_b), "null pair must exercise two different buckets"
    for ids in (ids_a, ids_b):
        assert MIN_LEN <= len(ids) <= MAX_LEN

    assets = {
        "min_len": MIN_LEN, "max_len": MAX_LEN, "state_dim": STATE_DIM,
        "tokenizer": "google/paligemma-3b-pt-224 via TokenizerProcessorStep call",
        "tasks": tasks,
        "synth_ids": {str(L): got[L]["ids"] for L in sorted(got)},
        "synth_prompts": {str(L): got[L]["prompt"] for L in sorted(got)},
        "replay": {
            "prompt_a": prompt_a, "ids_a": ids_a, "len_a": len(ids_a),
            "prompt_b": prompt_b, "ids_b": ids_b, "len_b": len(ids_b),
        },
    }
    with open(os.path.join(out_dir, "thor_assets.json"), "w") as f:
        json.dump(assets, f)
    print(f"synth ids for lengths {sorted(got)} -> thor_assets.json "
          f"(null pair lens {len(ids_a)}/{len(ids_b)})", flush=True)

    imgs, wrists = [], []
    for t in CALIB_TASKS:
        fr = engine_frames(t)
        imgs.append(fr[0])
        wrists.append(fr[1])
        print(f"calib frame from task {t}: {fr[0].shape} {fr[0].dtype}", flush=True)
    np.savez(os.path.join(out_dir, "calib_obs.npz"),
             image=np.stack(imgs), wrist_image=np.stack(wrists))
    print(f"calib_obs.npz: {len(imgs)} frames from tasks {CALIB_TASKS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

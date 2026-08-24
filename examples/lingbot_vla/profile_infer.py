#!/usr/bin/env python3
"""Where does a LingBot-VLA infer spend its time? Decides whether static-KV capture is the right
optimization before anything is built.

Decomposition of one `LingbotVLAServer.infer(obs)` call:
    obs pipeline     resize + tensorize + feature_transform
    prefill          embed_prefix + the fill_kv_cache forward (VLM over ~prefix tokens)
    denoise loop     10 x predict_velocity (suffix forward over 36 layers with per-layer KV cat)
    post             action slice + unnormalize + cpu

Run from /home/ubuntu/lingbot-vla-repo with its venv:
    CUDA_VISIBLE_DEVICES=6 .venv/bin/python /home/ubuntu/InstinctFlash/examples/lingbot_vla/profile_infer.py
"""
import glob
import json
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/lingbot-vla-repo")

SNAP = glob.glob(
    "/home/ubuntu/.cache/huggingface/hub/models--robbyant--lingbot-vla-4b-posttrain-robotwin/snapshots/*/"
)[0]
NORM = "/home/ubuntu/lingbot-vla-repo/assets/norm_stats/robotwin_50.json"

CAMS = ["observation.images.cam_high", "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"]


def make_obs(rng):
    obs = {k: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for k in CAMS}
    obs["observation.state"] = np.zeros(14, dtype=np.float32)
    obs["task"] = obs["prompt"] = "Use the left arm to pick up the block and place it in the tray"
    return obs


def main():
    from deploy.lingbot_vla_policy import LingbotVLAServer

    server = LingbotVLAServer(SNAP, use_length=25, robot_norm_path=NORM, num_denoising_step=10)
    fm = server.vla.model                       # FlowMatching

    acc = {"prefill_ms": 0.0, "denoise_ms": 0.0, "denoise_calls": 0}

    orig_expert_fwd = type(fm.qwenvl_with_expert).forward
    orig_velocity = type(fm).predict_velocity

    def timed_expert_fwd(self, *a, **k):
        if k.get("fill_kv_cache"):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = orig_expert_fwd(self, *a, **k)
            torch.cuda.synchronize(); acc["prefill_ms"] += (time.perf_counter() - t0) * 1000
            return out
        return orig_expert_fwd(self, *a, **k)

    def timed_velocity(self, *a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig_velocity(self, *a, **k)
        torch.cuda.synchronize(); acc["denoise_ms"] += (time.perf_counter() - t0) * 1000
        acc["denoise_calls"] += 1
        return out

    type(fm.qwenvl_with_expert).forward = timed_expert_fwd
    type(fm).predict_velocity = timed_velocity

    rng = np.random.default_rng(0)
    server.infer(dict(reset=True, robo_name="robotwin"))
    for _ in range(3):
        server.infer(make_obs(rng))

    for k in acc:
        acc[k] = 0.0 if k != "denoise_calls" else 0
    total = []
    N = 10
    for _ in range(N):
        obs = make_obs(rng)
        t0 = time.perf_counter()
        server.infer(obs)
        total.append((time.perf_counter() - t0) * 1000)

    res = {
        "infer_ms_p50": round(statistics.median(total), 1),
        "prefill_ms_per_infer": round(acc["prefill_ms"] / N, 1),
        "denoise_loop_ms_per_infer": round(acc["denoise_ms"] / N, 1),
        "denoise_ms_per_step": round(acc["denoise_ms"] / acc["denoise_calls"], 2),
        "steps_per_infer": acc["denoise_calls"] / N,
        "other_ms": round(statistics.median(total) - (acc["prefill_ms"] + acc["denoise_ms"]) / N, 1),
    }
    print(json.dumps(res, indent=1))
    open("/home/ubuntu/iwm_distill/bench_vla_h100/profile.json", "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

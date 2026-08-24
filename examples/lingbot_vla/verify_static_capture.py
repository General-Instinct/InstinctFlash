#!/usr/bin/env python3
"""Gates for LingBot-VLA static-KV capture, in the pi05 verify discipline.

Determinism lever: `sample_actions` draws its noise from the global torch RNG, so seeding before
each infer makes stock and captured runs consume identical noise. References are computed on the
STOCK model first; the static path is installed afterwards on the same instance, warmed for one
full chunk (eager on static buffers), captured on the second, and every gate case then replays
the graph on inputs the capture never saw.

    CUDA_VISIBLE_DEVICES=6 .venv/bin/python examples/lingbot_vla/verify_static_capture.py
"""
import glob
import json
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/lingbot-vla-repo")
sys.path.insert(0, "/home/ubuntu/InstinctFlash/examples/lingbot_vla")

SNAP = glob.glob(
    "/home/ubuntu/.cache/huggingface/hub/models--robbyant--lingbot-vla-4b-posttrain-robotwin/snapshots/*/"
)[0]
NORM = "/home/ubuntu/lingbot-vla-repo/assets/norm_stats/robotwin_50.json"
CAMS = ["observation.images.cam_high", "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"]
PROMPT_A = "Use the left arm to pick up the block and place it in the tray"
PROMPT_B = "Stack the red bowl on top of the blue plate with the right arm"


def make_obs(seed, prompt):
    rng = np.random.default_rng(seed)
    obs = {k: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for k in CAMS}
    obs["observation.state"] = rng.normal(0, 0.1, size=14).astype(np.float32)
    obs["task"] = obs["prompt"] = prompt
    return obs


#: (obs seed, torch seed, prompt) — cases 3-7 are the unseen set: fresh noise, fresh observations,
#: and two on a DIFFERENT prompt, which forces a prefix refill before replay
CASES = [(0, 100, PROMPT_A), (1, 101, PROMPT_A), (2, 102, PROMPT_A),
         (3, 103, PROMPT_A), (4, 104, PROMPT_B), (5, 105, PROMPT_B)]


def run_case(server, case):
    obs_seed, torch_seed, prompt = case
    torch.manual_seed(torch_seed)
    ret = server.infer(make_obs(obs_seed, prompt))
    return np.asarray(ret["action"], dtype=np.float64)


def main():
    from deploy.lingbot_vla_policy import LingbotVLAServer
    from static_capture import install_static_capture

    server = LingbotVLAServer(SNAP, use_length=25, robot_norm_path=NORM, num_denoising_step=10)
    server.infer(dict(reset=True, robo_name="robotwin"))

    # -- stock references (and the stock timing while we are here) -----------------------------
    for _ in range(3):
        server.infer(make_obs(99, PROMPT_A))
    stock_lat = []
    for _ in range(12):
        obs = make_obs(98, PROMPT_A)
        t0 = time.perf_counter()
        server.infer(obs)
        stock_lat.append((time.perf_counter() - t0) * 1000)
    refs = [run_case(server, c) for c in CASES]

    # -- install, warm one chunk eagerly on static buffers, capture on the next ----------------
    d = install_static_capture(server.vla.model)
    warm = run_case(server, CASES[0])            # chunk 1: eager static path
    d_static_eager = float(np.abs(warm - refs[0]).max())

    outs, replays_before = [], d.replays
    for c in CASES:
        outs.append(run_case(server, c))
    assert d.replays > replays_before, "graph never replayed — capture did not engage"

    # -- gates ----------------------------------------------------------------------------------
    print(f"static path, eager (pre-capture) vs stock:   max |d| {d_static_eager:.3e}")
    gate = {"static_eager_vs_stock": d_static_eager, "cases": []}
    for c, ref, out in zip(CASES, refs, outs):
        dmax = float(np.abs(out - ref).max())
        gate["cases"].append({"case": list(c[:2]) + [c[2][:24]], "max_abs_d": dmax})
        print(f"replay vs stock  seed={c[1]} prompt={'A' if c[2] == PROMPT_A else 'B'}:"
              f"   max |d| {dmax:.3e}")

    # -- timing ----------------------------------------------------------------------------------
    ours_lat = []
    for _ in range(12):
        obs = make_obs(98, PROMPT_A)
        t0 = time.perf_counter()
        server.infer(obs)
        ours_lat.append((time.perf_counter() - t0) * 1000)

    res = {
        "gates": gate,
        "stock_ms_p50_inprocess": round(statistics.median(stock_lat), 1),
        "ours_ms_p50_inprocess": round(statistics.median(ours_lat), 1),
        "speedup_inprocess": round(statistics.median(stock_lat) / statistics.median(ours_lat), 2),
        "replays": d.replays,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "gates"}, indent=1))
    open("/home/ubuntu/InstinctFlash/examples/lingbot_vla/static_capture_results.json",
         "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

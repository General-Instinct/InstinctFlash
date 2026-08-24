#!/usr/bin/env python3
"""Gates for the GR00T DiT static capture: bitexact vs stock, then the measured win.

Determinism control: get_action samples its flow-matching noise from torch's global RNG, so both
arms run under identical torch.manual_seed per case — the same technique that made the
LingBot-VLA gates meaningful. Cases include new observations AND a different prompt (a new
vl-sequence shape, which must capture its own graph rather than replay a wrong one).
"""
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/iwm_distill/bench_groot_h100")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_groot_eager import synth_obs  # noqa: E402
from static_capture import install_static_capture  # noqa: E402


def build_policy():
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    path = "/home/ubuntu/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots"
    snap = os.path.join(path, os.listdir(path)[0])
    return Gr00tPolicy(embodiment_tag=EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT,
                       model_path=snap, device="cuda:0")


def make_cases(policy):
    cases = []
    for seed in (0, 1, 2, 3):
        obs = synth_obs(policy)
        rng = np.random.default_rng(100 + seed)
        for k in obs["video"]:
            obs["video"][k] = rng.integers(0, 256, size=(1, 2, 256, 256, 3), dtype=np.uint8)
        cases.append((f"obs{seed}", obs, 1234 + seed))
    long_obs = synth_obs(policy)
    for k in long_obs["language"]:
        long_obs["language"][k] = [["carefully pick up the leftmost small red block and place "
                                    "it inside the open drawer on the right side of the table"]]
    cases.append(("new-prompt", long_obs, 4321))
    cases.append(("new-prompt-2", long_obs, 4322))
    return cases


def run_arm(policy, cases):
    outs = []
    for name, obs, seed in cases:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ret = policy.get_action(obs)
        a = ret[0] if isinstance(ret, tuple) else ret
        flat = {k: np.asarray(v.cpu() if torch.is_tensor(v) else v)
                for k, v in (a.items() if isinstance(a, dict) else [("action", a)])}
        outs.append((name, flat))
    return outs


def main():
    policy = build_policy()
    cases = make_cases(policy)

    stock = run_arm(policy, cases)                     # arm 1: stock
    handle = install_static_capture(policy.model)     # arm 2: captured
    _ = run_arm(policy, cases)                         # warm: captures graphs per signature
    cap = run_arm(policy, cases)

    print(f"graphs captured: {handle.captures}   replays: {handle.replays}")
    worst = 0.0
    for (name, s), (_, c) in zip(stock, cap):
        d = max(float(np.abs(s[k] - c[k]).max()) for k in s)
        worst = max(worst, d)
        print(f"  {name:14} max |d| = {d:.3e}")
    print(f"GATE {'PASS (bitexact)' if worst == 0.0 else f'delta {worst:.3e}'}")

    obs = cases[0][1]
    for _ in range(3):
        policy.get_action(obs)
    lat = []
    for _ in range(15):
        t0 = time.perf_counter()
        policy.get_action(obs)
        lat.append((time.perf_counter() - t0) * 1000)
    p50 = statistics.median(lat)
    print(f"ours p50: {p50:.1f} ms")
    json.dump({"label": "groot_instinctflash_static_capture", "wall_ms_p50": round(p50, 1),
               "gate_worst_abs_delta": worst,
               "cases": [n for n, *_ in cases],
               "graphs_captured": handle.captures, "replays": handle.replays},
              open("/home/ubuntu/iwm_distill/bench_groot_h100/ours.json", "w"), indent=1)


if __name__ == "__main__":
    main()

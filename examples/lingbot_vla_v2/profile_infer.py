#!/usr/bin/env python3
"""LingBot-VLA-V2 in-process profile: where does one infer() spend its time?

Loads through the official deploy server class so the measured pipeline is exactly what ships.
Decomposition: prefill (fill_kv_cache forward) vs the 10-step Euler denoise loop vs everything
else — measured by CUDA-event bracketing installed on the model's own methods, no code edits.
Also records stock in-process p50 for compile ON (as-shipped default) and OFF.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/lingbot-vla-v2-repo")

SNAP = sorted(Path("/home/ubuntu/.cache/huggingface/hub/"
                   "models--robbyant--lingbot-vla-v2-6b-robotwin/snapshots").iterdir())[0]
MODEL_PATH = str(SNAP / "checkpoints/global_step_50000/hf_ckpt")

CAMS = ["observation.images.cam_high", "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"]


def make_obs(rng):
    obs = {k: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for k in CAMS}
    obs["observation.state"] = np.zeros(14, dtype=np.float32)
    obs["task"] = obs["prompt"] = "Use the left arm to pick up the block and place it in the tray"
    return obs


def bench(server, n=12, warmup=3):
    rng = np.random.default_rng(0)
    lat = []
    for i in range(warmup + n):
        obs = make_obs(rng)
        t0 = time.perf_counter()
        server.infer(obs)
        torch.cuda.synchronize()
        if i >= warmup:
            lat.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(lat), 1)


def main():
    use_compile = "--compile" in sys.argv
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    server = LingbotVLAv2Server(MODEL_PATH, use_length=50, chunk_ret=True,
                                use_bf16=True, use_fp32=False, use_compile=use_compile)
    server.reset("robotwin")   # installs the robot feature transform, as the ws flow does

    # decomposition via method bracketing on the un-compiled path only (compile hides boundaries)
    decomp = {}
    if not use_compile:
        model = server.vla.model
        orig_fwd = model.qwenvl_with_expert.forward
        stats = {"prefill_ms": [], "step_ms": []}

        def timed_fwd(*a, **k):
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()
            out = orig_fwd(*a, **k)
            ev1.record()
            ev1.synchronize()
            key = "prefill_ms" if k.get("fill_kv_cache") else "step_ms"
            stats[key].append(ev0.elapsed_time(ev1))
            return out

        model.qwenvl_with_expert.forward = timed_fwd
        rng = np.random.default_rng(1)
        for _ in range(3):
            stats["prefill_ms"].clear(); stats["step_ms"].clear()
            server.infer(make_obs(rng))
        decomp = {"prefill_ms": round(sum(stats["prefill_ms"]), 1),
                  "denoise_loop_ms": round(sum(stats["step_ms"]), 1),
                  "n_steps": len(stats["step_ms"]),
                  "per_step_ms": round(statistics.mean(stats["step_ms"]), 2)}
        model.qwenvl_with_expert.forward = orig_fwd

    p50 = bench(server)
    label = "compile_on" if use_compile else "eager"
    res = {"label": f"vla2_inprocess_{label}", "wall_ms_p50": p50, "decomposition": decomp}
    print(json.dumps(res, indent=1))
    out = Path("/home/ubuntu/iwm_distill/bench_vla2_h100")
    (out / f"profile_{label}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

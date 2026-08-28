#!/usr/bin/env python3
"""Gate for the LingBot-VLA-V2 Triton kernels, under the SAME 6-case protocol as the published row.

What is being judged: the deterministic (expert-sorted, atomics-free) sparse-MoE reduction and the
fused RMSNorm/AdaRMSNorm kernels, against vendor robby_moe on H100. The prior evidence was a
4-case A100 gate against a null-derived threshold — weaker than the protocol behind the
published H100 row — so the kernels ship DEFAULT OFF until this gate passes.

Gate standard (same as verify_static_capture.py): this model is nondeterministic against ITSELF —
the vendor fused-MoE kernel reduces with atomics — so the null control is three stock runs on
identical seeds, and the kernel arms' deltas vs stock must sit inside that stock-vs-stock
envelope. The kernel arms are ALSO run twice each: an atomics-free reduction should be
self-consistent (max self-delta 0), which is the property the deterministic rewrite buys.

    CUDA_VISIBLE_DEVICES=7 QWEN3VL_PATH=Qwen/Qwen3-VL-4B-Instruct \
        /home/ubuntu/lingbot-vla-v2-repo/.venv/bin/python examples/lingbot_vla_v2/verify_moe_kernel.py
"""
import glob
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("LINGBOT_VLA_V2_ROOT", "/home/ubuntu/lingbot-vla-v2-repo"))
sys.path.insert(0, HERE)

SNAP = glob.glob(
    "/home/ubuntu/.cache/huggingface/hub/models--robbyant--lingbot-vla-v2-6b-robotwin/snapshots/*/"
)[0] + "checkpoints/global_step_50000/hf_ckpt"
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


#: identical to verify_static_capture.py — fresh noise, fresh observations, two prompt switches
CASES = [(0, 100, PROMPT_A), (1, 101, PROMPT_A), (2, 102, PROMPT_A),
         (3, 103, PROMPT_A), (4, 104, PROMPT_B), (5, 105, PROMPT_B)]


def run_cases(server):
    outs = []
    for obs_seed, torch_seed, prompt in CASES:
        torch.manual_seed(torch_seed)
        ret = server.infer(make_obs(obs_seed, prompt))
        outs.append(np.asarray(ret["action"], dtype=np.float64))
    return outs


def max_delta(a, b):
    return max(float(np.abs(x - y).max()) for x, y in zip(a, b))


def time_p50(server, n=12):
    lat = []
    for _ in range(n):
        obs = make_obs(98, PROMPT_A)
        t0 = time.perf_counter()
        server.infer(obs)
        lat.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(lat), 1)


def main():
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from lingbot_vla_v2_iwm.moe_kernel import install_lingbot_moe_kernel
    from lingbot_vla_v2_iwm.rmsnorm_kernel import install_lingbot_rmsnorm_kernel

    server = LingbotVLAv2Server(SNAP, use_length=50, chunk_ret=True, use_bf16=True,
                                use_fp32=False, use_compile=False)
    # Upstream resolves robot configs relative to its project root (same scoped-cwd dance the
    # adapter does in _project_cwd); inference itself uses no relative paths.
    prev_cwd = os.getcwd()
    os.chdir(sys.path[0] if os.path.isdir(os.path.join(sys.path[0], "configs")) else
             os.environ.get("LINGBOT_VLA_V2_ROOT", "/home/ubuntu/lingbot-vla-v2-repo"))
    try:
        server.reset("robotwin")
    finally:
        os.chdir(prev_cwd)
    for _ in range(3):
        server.infer(make_obs(99, PROMPT_A))

    # -- null control: three stock runs on identical seeds bound the model's own nondeterminism --
    stock_runs = [run_cases(server) for _ in range(3)]
    null = [max_delta(stock_runs[i], stock_runs[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    envelope = max(null)
    stock_p50 = time_p50(server)
    print(f"null control (stock vs stock, 3 pairs): {['%.3e' % d for d in null]}  "
          f"envelope = {envelope:.3e}")
    refs = stock_runs[0]

    # -- arm B: the deterministic sparse-MoE kernel alone ----------------------------------------
    moe = install_lingbot_moe_kernel(server.vla.model)
    print(f"moe kernel installed: {moe.layers} layers, {moe.converted_layers} converted")
    b1, b2 = run_cases(server), run_cases(server)
    b_vs_stock = max(max_delta([o], [r]) for o, r in zip(b1, refs))
    b_self = max_delta(b1, b2)
    b_p50 = time_p50(server)
    print(f"B moe-kernel vs stock: max |d| {b_vs_stock:.3e}   self-consistency: {b_self:.3e}")

    # -- arm C: + fused RMSNorm/AdaRMSNorm --------------------------------------------------------
    norm = install_lingbot_rmsnorm_kernel(server.vla.model)
    print(f"rmsnorm kernel installed: {norm.modules} modules")
    c1, c2 = run_cases(server), run_cases(server)
    c_vs_stock = max(max_delta([o], [r]) for o, r in zip(c1, refs))
    c_self = max_delta(c1, c2)
    c_p50 = time_p50(server)
    print(f"C moe+rmsnorm vs stock: max |d| {c_vs_stock:.3e}   self-consistency: {c_self:.3e}")

    b_pass = b_vs_stock <= envelope
    c_pass = c_vs_stock <= envelope
    print(f"GATE B (moe kernel):     {'PASS' if b_pass else 'FAIL'} "
          f"({b_vs_stock:.3e} vs envelope {envelope:.3e})")
    print(f"GATE C (moe + rmsnorm):  {'PASS' if c_pass else 'FAIL'} "
          f"({c_vs_stock:.3e} vs envelope {envelope:.3e})")
    print(f"p50 eager: stock {stock_p50} ms   moe {b_p50} ms   moe+rmsnorm {c_p50} ms")

    res = {
        "protocol": "6-case H100, null-control envelope (same as verify_static_capture.py)",
        "null_control_deltas": null,
        "envelope": envelope,
        "moe_kernel": {"vs_stock_max_abs_d": b_vs_stock, "self_consistency_max_abs_d": b_self,
                       "p50_ms_eager": b_p50, "pass": b_pass},
        "moe_plus_rmsnorm": {"vs_stock_max_abs_d": c_vs_stock,
                             "self_consistency_max_abs_d": c_self,
                             "p50_ms_eager": c_p50, "pass": c_pass},
        "stock_p50_ms_eager": stock_p50,
        "device": torch.cuda.get_device_name(0),
        "cases": [[c[0], c[1], c[2][:24]] for c in CASES],
    }
    with open(os.path.join(HERE, "moe_kernel_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    return 0 if (b_pass and c_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())

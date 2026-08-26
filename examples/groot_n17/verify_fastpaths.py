#!/usr/bin/env python3
"""Full-stack gate for the GR00T N1.7 fastpaths: fixed-seed exactness, then the measured win.

Arms:
  A  stock             upstream policy, process-default threads, no patches
  A2 stock + 16 thread torch/cv2 pinned to 16 -- quantifies the HOST-SPECIFIC pinning effect
                       (measured ~1.5 ms here on a 208-CPU H100 vs ~165 ms reported on a
                       240-CPU box; the pin is restored before the next arm)
  B  full stack        fast_decode + backbone_fastpath + DiT static capture, default threads
  C  DiT capture only  the previously published arm, for decomposition

Protocol mirrors verify_static_capture.py (the measured-artifact protocol reference): same
6 cases including two prompt-shape switches, same per-case seeds. The gate is max|delta| == 0.0
(np.array_equal-grade) for B vs A across all cases. Results land in fastpath_results.json
next to this script.
"""
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/ubuntu/iwm_distill/bench_groot_h100")
sys.path.insert(0, HERE)
from bench_groot_eager import synth_obs  # noqa: E402


def build_policy():
    # transformers 4.57.3 _patch_mistral_regex calls the Hub API even in offline
    # mode; the checkpoint's Qwen3 tokenizer is not a mistral model, skip it.
    import transformers.tokenization_utils_base as tub

    def _no_mistral_patch(cls, tokenizer, *args, **kwargs):
        return tokenizer

    tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(_no_mistral_patch)
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    path = os.environ.get(
        "GR00T_N17_SNAPSHOTS",
        "/home/ubuntu/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots")
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


def time_p50(policy, obs, warm=3, iters=15):
    for _ in range(warm):
        policy.get_action(obs)
    torch.cuda.synchronize()
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        policy.get_action(obs)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return statistics.median(lat)


def main():
    import cv2
    default_torch_threads = torch.get_num_threads()
    default_cv2_threads = cv2.getNumThreads()
    print(f"default threads: torch={default_torch_threads} cv2={default_cv2_threads} "
          f"nproc={os.cpu_count()}")

    policy = build_policy()
    cases = make_cases(policy)

    # ---- Arm A: stock, default threads
    stock = run_arm(policy, cases)
    stock_p50 = time_p50(policy, cases[0][1])
    print(f"A  stock (default threads)              p50 = {stock_p50:.1f} ms")

    # ---- Arm A2: thread-pin-only measurement on the stock path; restored below
    torch.set_num_threads(16)
    cv2.setNumThreads(16)
    stock16 = run_arm(policy, cases)
    stock16_p50 = time_p50(policy, cases[0][1])
    print(f"A2 stock + 16 threads                   p50 = {stock16_p50:.1f} ms")
    worst_t = max(max(float(np.abs(s[k] - c[k]).max()) for k in s)
                  for (_, s), (_, c) in zip(stock, stock16))
    print(f"   thread-pin exactness vs A: max|d| = {worst_t:.3e}")
    torch.set_num_threads(default_torch_threads)
    cv2.setNumThreads(default_cv2_threads)

    # ---- Arm B: full stack at default threads
    from groot_n17_iwm.fast_decode import install_fast_decode
    from groot_n17_iwm.backbone_fastpath import install_backbone_fastpath
    from static_capture import install_static_capture

    fd = install_fast_decode(policy)
    print(f"fast_decode installed: {fd is not None} (supported={getattr(fd, 'supported', None)})")
    bf = install_backbone_fastpath(policy.model)
    drv = install_static_capture(policy.model)
    _ = run_arm(policy, cases)  # warm: capture graphs + populate caches
    full = run_arm(policy, cases)

    worst = 0.0
    for (name, s), (_, c) in zip(stock, full):
        d = max(float(np.abs(s[k] - c[k]).max()) for k in s)
        worst = max(worst, d)
        print(f"  {name:14} max |d| = {d:.3e}")
    print(f"GATE full-stack vs stock: {'PASS (bitexact)' if worst == 0.0 else f'delta {worst:.3e}'}")
    print(f"backbone cache hits={bf.hits} misses={bf.misses}  "
          f"graphs={drv.captures} replays={drv.replays}")

    full_p50 = time_p50(policy, cases[0][1])
    print(f"B  full stack (default threads)         p50 = {full_p50:.1f} ms")

    # ---- Arm C: DiT capture only (the previously published arm) for decomposition
    fd_obj = getattr(policy.processor, "_instinctflash_fast_decoder", None)
    if fd_obj is not None:
        policy.processor.decode_action = fd_obj._fallback
        delattr(policy.processor, "_instinctflash_fast_decoder")
    bf.close()
    ours_p50 = time_p50(policy, cases[0][1])
    print(f"C  DiT capture only, default threads    p50 = {ours_p50:.1f} ms")

    out = {
        "stock_default_threads_p50": round(stock_p50, 1),
        "stock_16_threads_p50": round(stock16_p50, 1),
        "thread_pin_max_abs_delta": worst_t,
        "full_stack_p50": round(full_p50, 1),
        "full_stack_max_abs_delta": worst,
        "capture_only_default_threads_p50": round(ours_p50, 1),
        "graphs": drv.captures, "replays": drv.replays,
        "backbone_hits": bf.hits, "backbone_misses": bf.misses,
        "cases": [n for n, *_ in cases],
        "device": torch.cuda.get_device_name(0),
    }
    with open(os.path.join(HERE, "fastpath_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0 if worst == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

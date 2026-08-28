#!/usr/bin/env python3
"""Rerun the README LingBot-VLA-V2 H100 pair: upstream eager vs the Runtime DEFAULT arm.

Protocol, exactly as the published row: in-process ``infer``, p50 of 12 timed calls after
3 warmup calls, one idle H100. The stock arm is the upstream server with ``use_compile=False``
(their eager reference — their own compile default is a separate arm in the eval archive).
The ours arm is ``Runtime.from_pretrained`` with no flags: what the family DEFAULT serves —
the static-KV denoise CUDA graph, the vision/prefill graphs, and GPU preprocessing, gated at
startup by the capture self-check.

Each arm runs in its OWN subprocess. That is protocol, not convenience: the upstream deploy
stack keeps process-global state, and a second model built in the same process was measured to
fail the capture self-check (which falls back loudly, exactly as designed — but then this
script would be timing the fallback, not the default arm). A fresh process per arm is also
what any real deployment of either arm looks like.

Quality gate, compared across the two arms: six fixed-seed cases (two on a second prompt,
forcing a prefix refill). BITEXACT is unattainable for ANY serving of this model — upstream's
fused-MoE kernel disagrees with itself on identical seeds — so the gate is the family's
recorded stock-vs-stock envelope (``lingbot_vla_v2_iwm.static_capture.NULL_ENVELOPE``, from
moe_kernel_results.json), the same standard the row was verified under. Tier: NUMERIC.

    examples/lingbot_vla_v2/reproduce_h100.sh          # wraps this with the venv/GPU knobs

Env:
    VLA2_SNAP        upstream hf_ckpt dir for the stock arm (default: HF cache snapshot)
    IFL_VLA2_CKPT    checkpoint dir or Hub id for the Runtime arm
                     (default robbyant/lingbot-vla-v2-6b-robotwin)
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("LINGBOT_VLA_V2_ROOT",
                                  str(Path.home() / "lingbot-vla-v2-repo")))
sys.path.insert(0, str(HERE))              # this checkout's lingbot_vla_v2_iwm
sys.path.insert(0, str(HERE.parents[1]))   # repo root: the package imports instinctflash

ARM_ENV = "IFL_VLA2_REPRO_ARM"

CAMS = ["observation.images.cam_high", "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"]
PROMPT_A = "Use the left arm to pick up the block and place it in the tray"
PROMPT_B = "Stack the red bowl on top of the blue plate with the right arm"
#: (obs seed, torch seed, prompt) — the verify_static_capture.py gate seeds; the last two force
#: a prefix refill on a second prompt.
CASES = [(0, 100, PROMPT_A), (1, 101, PROMPT_A), (2, 102, PROMPT_A),
         (3, 103, PROMPT_A), (4, 104, PROMPT_B), (5, 105, PROMPT_B)]


def make_obs(seed, prompt):
    rng = np.random.default_rng(seed)
    obs = {k: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for k in CAMS}
    obs["observation.state"] = rng.normal(0, 0.1, size=14).astype(np.float32)
    obs["task"] = obs["prompt"] = prompt
    return obs


def measure(predict):
    import torch

    for _ in range(3):
        predict(make_obs(99, PROMPT_A))
    lat = []
    for _ in range(12):
        obs = make_obs(98, PROMPT_A)
        t0 = time.perf_counter()
        predict(obs)
        lat.append((time.perf_counter() - t0) * 1000)
    outs = []
    for obs_seed, torch_seed, prompt in CASES:
        torch.manual_seed(torch_seed)
        outs.append(np.asarray(predict(make_obs(obs_seed, prompt)), dtype=np.float64))
    return statistics.median(lat), outs


def _skip_transformers_mistral_hub_probe():
    import transformers.tokenization_utils_base as tub

    def _no_mistral_patch(cls, tokenizer, *args, **kwargs):
        return tokenizer

    tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(_no_mistral_patch)


def run_stock_arm():
    _skip_transformers_mistral_hub_probe()
    if os.environ.get("VLA2_SNAP"):
        snap = os.environ["VLA2_SNAP"]
    else:
        from huggingface_hub import snapshot_download

        snap = os.path.join(snapshot_download("robbyant/lingbot-vla-v2-6b-robotwin"),
                            "checkpoints", "global_step_50000", "hf_ckpt")
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    server = LingbotVLAv2Server(snap, use_length=50, chunk_ret=True,
                                use_bf16=True, use_fp32=False, use_compile=False)
    server.reset("robotwin")
    return measure(lambda obs: server.infer(obs)["action"])


def run_ours_arm():
    _skip_transformers_mistral_hub_probe()
    from instinctflash import Runtime

    ckpt = os.environ.get("IFL_VLA2_CKPT", "robbyant/lingbot-vla-v2-6b-robotwin")
    runtime = Runtime.from_pretrained(ckpt)
    with runtime, runtime.episode(prompt=PROMPT_A) as episode:
        return measure(lambda obs: episode.predict(obs)["action"])


def arm_main(arm: str) -> int:
    p50, outs = run_stock_arm() if arm == "stock" else run_ours_arm()
    np.savez(HERE / f".reproduce_{arm}.npz", **{f"case{i}": a for i, a in enumerate(outs)})
    (HERE / f".reproduce_{arm}.json").write_text(json.dumps({"p50_ms": p50}))
    print(f"[{arm}] p50 = {p50:.1f} ms")
    return 0


def main() -> int:
    arm = os.environ.get(ARM_ENV)
    if arm:
        return arm_main(arm)

    for arm in ("stock", "ours"):
        subprocess.run([sys.executable, str(Path(__file__).resolve())],
                       env={**os.environ, ARM_ENV: arm}, check=True)

    stock_p50 = json.loads((HERE / ".reproduce_stock.json").read_text())["p50_ms"]
    ours_p50 = json.loads((HERE / ".reproduce_ours.json").read_text())["p50_ms"]
    stock = np.load(HERE / ".reproduce_stock.npz")
    ours = np.load(HERE / ".reproduce_ours.npz")
    print(f"stock (upstream eager, in-process)      p50 = {stock_p50:.1f} ms")
    print(f"ours (Runtime default arm)              p50 = {ours_p50:.1f} ms")

    from lingbot_vla_v2_iwm.static_capture import NULL_ENVELOPE

    worst, gates = 0.0, []
    for i, (obs_seed, torch_seed, prompt) in enumerate(CASES):
        dmax = float(np.abs(ours[f"case{i}"] - stock[f"case{i}"]).max())
        worst = max(worst, dmax)
        gates.append({"case": [obs_seed, torch_seed, prompt[:24]], "max_abs_d": dmax})
        print(f"ours vs stock  seed={torch_seed} prompt={'A' if prompt == PROMPT_A else 'B'}:"
              f"   max |d| {dmax:.3e}")
    ok = worst <= NULL_ENVELOPE
    print(f"GATE ours vs stock: max |d| {worst:.3e} "
          f"{'INSIDE' if ok else 'OUTSIDE'} the recorded stock-vs-stock envelope "
          f"{NULL_ENVELOPE:.3e} (moe_kernel_results.json null_control_deltas)")

    import torch

    res = {
        "protocol": "in-process infer, p50 of 12 calls x 3 warmup, one fresh process per arm; "
                    "6 gate cases vs the recorded stock-vs-stock envelope",
        "stock_ms_p50": round(stock_p50, 1),
        "ours_ms_p50": round(ours_p50, 1),
        "speedup": round(stock_p50 / ours_p50, 2),
        "gate_max_abs_d": worst,
        "null_envelope": NULL_ENVELOPE,
        "gates": gates,
        "device": torch.cuda.get_device_name(0),
    }
    print(json.dumps({k: v for k, v in res.items() if k != "gates"}, indent=1))
    (HERE / "reproduce_h100_results.json").write_text(json.dumps(res, indent=1))
    for leftover in HERE.glob(".reproduce_*"):
        leftover.unlink()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

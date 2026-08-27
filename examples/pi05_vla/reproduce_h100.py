#!/usr/bin/env python3
"""Rerun the README pi05 H100 pair: lerobot eager vs the InstinctFlash static-capture arm.

Published row (H100, 2026-08-24 remeasured sweep):

    pi05    207 -> 73 ms, 2.84x, BITEXACT
    checkpoint lerobot/pi05_libero_finetuned_v044 (bf16-stored, the realistic serving artifact)

Protocol, exactly as measured: one control chunk = prefill + 10 flow-matching denoise steps,
median of 15 timed chunks after 2 warm chunks, one H100, no other GPU load. The T1 arm is
`pi05_iwm.static_capture` (replay-safe static max-extent KV buffers + per-timestep step tables,
`IFL_PI05_STEP_TABLES` default on) over the bit-exact loop-constant hoists in `pi05_iwm.surface`.
The bitexactness claim itself is gated by verify_static_capture.py, not here; this script is the
latency pair only and asserts the arms agree bitwise on the input it times as a sanity floor.

    examples/pi05_vla/reproduce_h100.sh          # wraps this with the venv/GPU knobs

Env:
    IFL_PI05_CKPT    checkpoint repo id or local path (default lerobot/pi05_libero_finetuned_v044)
    IFL_PI05_STEP_TABLES=0   disables the step-table hoist inside the captured region
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))    # repo root, so pi05_iwm can import instinctflash

from lerobot.policies.pi05 import modeling_pi05 as M  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402

from pi05_iwm.static_capture import WARMUP_STEPS, install_static_capture  # noqa: E402
from pi05_iwm.surface import Pi05Surface  # noqa: E402

CKPT = os.environ.get("IFL_PI05_CKPT", "lerobot/pi05_libero_finetuned_v044")
DEV = "cuda:0"
README_PAIR = (206.7, 72.8)


def prefill(model, prompt_seed: int):
    """Prefix K/V for a seeded synthetic observation — same construction as the verify gates,
    so this script needs the weights and nothing else (no gated tokenizer)."""
    g = torch.Generator(device="cpu").manual_seed(prompt_seed)
    im = [torch.rand(1, 3, 224, 224, generator=g).to(DEV) * 2 - 1 for _ in range(3)]
    mk = [torch.ones(1, dtype=torch.bool, device=DEV) for _ in range(3)]
    tk = torch.randint(2, 1000, (1, 48), generator=g).to(DEV)
    ms = torch.ones(1, 48, dtype=torch.bool, device=DEV)
    with torch.no_grad():
        pe, ppm, pam = model.embed_prefix(im, mk, tk, ms)
        a2d = M.make_att_2d_masks(ppm, pam)
        pos = torch.cumsum(ppm, dim=1) - 1
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, kv = model.paligemma_with_expert.forward(
            attention_mask=M.prepare_attention_masks_4d(a2d), position_ids=pos,
            past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
    return ppm, kv


def timed_chunks(model, denoise, n=15, warm=2):
    """Median over n full chunks; each chunk is one prefill + 10 denoise steps."""
    def chunk():
        ppm, kv = prefill(model, prompt_seed=9)
        with torch.no_grad():
            g = torch.Generator(device="cpu").manual_seed(500)
            xt = torch.randn(1, model.config.chunk_size, model.config.max_action_dim,
                             generator=g).to(DEV)
            for k in range(10):
                v = denoise(ppm, kv, xt, torch.full((1,), 1.0 - k / 10, device=DEV))
                xt = xt - 0.1 * v
        return xt

    for _ in range(warm):
        chunk()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = chunk()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), out


def main() -> int:
    if not torch.cuda.is_available():
        print("this reproduction needs a CUDA GPU (the README pair is an H100 measurement)")
        return 2
    print(f"loading {CKPT} ...")
    policy = PI05Policy.from_pretrained(CKPT).to(DEV).eval()
    model = policy.model
    orig_denoise = type(model).denoise_step

    def eager(ppm, kv, x, t):
        with torch.no_grad():
            return orig_denoise(model, prefix_pad_masks=ppm, past_key_values=kv, x_t=x, timestep=t)

    ms_eager, out_eager = timed_chunks(model, eager)

    # T1 arm: bit-exact hoists, then the replay-safe static-KV graph (+ step tables).
    Pi05Surface(model).hoist_loop_constants()
    den = install_static_capture(model)
    ppm_w, kv_w = prefill(model, prompt_seed=9)
    with torch.no_grad():                                     # drive past warmup into capture
        g = torch.Generator(device="cpu").manual_seed(500)
        x_w = torch.randn(1, model.config.chunk_size, model.config.max_action_dim,
                          generator=g).to(DEV)
        for _ in range(WARMUP_STEPS + 1):
            den(ppm_w, kv_w, x_w, torch.full((1,), 0.5, device=DEV))
    assert den.replays >= 1, "graph was never captured"
    ms_ours, out_ours = timed_chunks(model, den)

    d = (out_eager - out_ours).abs().max().item()
    speedup = ms_eager / ms_ours
    result = {
        "checkpoint": CKPT,
        "protocol": "full chunk = prefill + 10 denoise steps, median of 15 (2 warm)",
        "torch_eager_chunk_ms": round(ms_eager, 1),
        "instinctflash_static_capture_chunk_ms": round(ms_ours, 1),
        "speedup": round(speedup, 2),
        "arms_bitexact_on_timed_input": d == 0.0,
        "readme_pair_ms": list(README_PAIR),
        "step_tables": os.environ.get("IFL_PI05_STEP_TABLES", "1") != "0",
    }
    print(json.dumps(result, indent=1))
    print(f"\npi05  {ms_eager:.1f} -> {ms_ours:.1f} ms   {speedup:.2f}x   "
          f"(README: {README_PAIR[0]:.0f} -> {README_PAIR[1]:.0f} ms, 2.84x)")
    if d != 0.0:
        print(f"WARNING: arms disagree on the timed input (max |d| {d:.3e}); "
              f"run verify_static_capture.py before trusting either number")
        return 1
    type(model).denoise_step = orig_denoise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

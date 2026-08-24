#!/usr/bin/env python3
"""The gates for static-KV graph capture, in the order they can fail.

    CUDA_VISIBLE_DEVICES=4 HF_TOKEN=... python examples/pi05_vla/verify_static_capture.py

Gate 0  hoisted embed_suffix is bitexact against upstream's (re-establishes the old measurement
        in this process, so gate 1's reference is upstream's numerics, not a hoisted cousin's).
Gate 1  replay vs eager on the CAPTURED input: must be bitexact.
Gate 2  replay vs eager on inputs the capture never saw — new x_t, new timestep, and a NEW PROMPT
        whose prefix is refilled into the static buffers. This is the exact failure class that
        rejected the DynamicCache capture (measured max |d| 2.116e-01 on a new x_t); every case
        here must read 0.
Gate 3  timing: denoise step replay vs eager, and full chunk (prefill + 10 steps) vs eager,
        medians over >= 15.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from lerobot.policies.pi05 import modeling_pi05 as M  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402

from pi05_iwm.static_capture import WARMUP_STEPS, install_static_capture  # noqa: E402
from pi05_iwm.surface import Pi05Surface  # noqa: E402

DEV = "cuda:0"
PROMPTS = ["Put the exhaust fans back to the slots.",
           "pick up the black bowl and place it on the plate"]


def prefill(model, cfg, prompt_seed: int):
    """Prefix K/V for a synthetic observation. Seeded so arms and reruns see the same bytes."""
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


def sample_input(model, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, model.config.chunk_size, model.config.max_action_dim, generator=g).to(DEV)
    t = torch.rand(1, generator=g).to(DEV).clamp(0.02, 0.98)
    return x, t


def timed(fn, n=15, warm=3):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def main() -> int:
    policy = PI05Policy.from_pretrained("lerobot/pi05_base").to(DEV).eval()
    model = policy.model
    orig_denoise = type(model).denoise_step
    orig_embed_suffix = type(model).embed_suffix
    results: dict = {}

    ppm_a, kv_a = prefill(model, policy.config, prompt_seed=7)
    x0, t0 = sample_input(model, seed=100)

    # -- gate 0: the hoist is bitexact ---------------------------------------------------------
    with torch.no_grad():
        ref_pre = orig_denoise(model, prefix_pad_masks=ppm_a, past_key_values=kv_a, x_t=x0, timestep=t0)
    Pi05Surface(model).hoist_loop_constants()
    with torch.no_grad():
        ref_post = orig_denoise(model, prefix_pad_masks=ppm_a, past_key_values=kv_a, x_t=x0, timestep=t0)
    d0 = (ref_pre - ref_post).abs().max().item()
    results["gate0_hoist_bitexact"] = d0
    print(f"gate 0  hoist vs upstream          max |d| {d0:.3e}   {'PASS' if d0 == 0 else 'FAIL'}")
    if d0 != 0:
        return 1

    def eager(ppm, kv, x, t):
        with torch.no_grad():
            return orig_denoise(model, prefix_pad_masks=ppm, past_key_values=kv, x_t=x, timestep=t)

    # -- install the static path; drive it past warmup into capture on the captured input -------
    den = install_static_capture(model)
    with torch.no_grad():
        for _ in range(WARMUP_STEPS + 1):                     # last call captures + replays
            out_cap = den(ppm_a, kv_a, x0, t0)
    assert den.replays >= 1, "graph was never captured"

    d1 = (eager(ppm_a, kv_a, x0, t0) - out_cap).abs().max().item()
    results["gate1_captured_input"] = d1
    print(f"gate 1  replay vs eager, captured  max |d| {d1:.3e}   {'PASS' if d1 == 0 else 'FAIL'}")

    # -- gate 2: inputs the capture never saw ----------------------------------------------------
    cases = []
    for i, seed in enumerate((201, 202, 203)):                 # new x_t and new timestep
        x, t = sample_input(model, seed)
        with torch.no_grad():
            d = (eager(ppm_a, kv_a, x, t) - den(ppm_a, kv_a, x, t)).abs().max().item()
        cases.append((f"new x_t/t #{i + 1}", d))
    ppm_b, kv_b = prefill(model, policy.config, prompt_seed=8)  # NEW PROMPT: refill path
    for i, seed in enumerate((301, 302)):
        x, t = sample_input(model, seed)
        with torch.no_grad():
            d = (eager(ppm_b, kv_b, x, t) - den(ppm_b, kv_b, x, t)).abs().max().item()
        cases.append((f"new prompt, input #{i + 1}", d))
    worst = max(d for _, d in cases)
    results["gate2_new_inputs"] = {name: d for name, d in cases}
    for name, d in cases:
        print(f"gate 2  {name:24}  max |d| {d:.3e}   {'PASS' if d == 0 else 'FAIL'}")
    print(f"gate 2  worst                       max |d| {worst:.3e}  "
          f"(old DynamicCache failure was 2.116e-01)")

    # -- gate 3: what the capture buys -----------------------------------------------------------
    x, t = sample_input(model, 400)
    ms_eager = timed(lambda: eager(ppm_a, kv_a, x, t))
    ms_replay = timed(lambda: den(ppm_a, kv_a, x, t))

    def chunk(fn):
        ppm, kv = prefill(model, policy.config, prompt_seed=9)
        with torch.no_grad():
            xt, _ = sample_input(model, 500)
            for k in range(10):
                v = fn(ppm, kv, xt, torch.full((1,), 1.0 - k / 10, device=DEV))
                xt = xt - 0.1 * v
        return xt

    ms_chunk_eager = timed(lambda: chunk(eager), n=15, warm=2)
    ms_chunk_replay = timed(lambda: chunk(den), n=15, warm=2)
    results["gate3_timing_ms"] = {
        "denoise_eager": round(ms_eager, 2), "denoise_replay": round(ms_replay, 2),
        "step_speedup": round(ms_eager / ms_replay, 2),
        "chunk_eager": round(ms_chunk_eager, 1), "chunk_replay": round(ms_chunk_replay, 1),
        "chunk_speedup": round(ms_chunk_eager / ms_chunk_replay, 2),
        "note": "chunk includes prefill (uncaptured) + 10 steps; replays counted="
                + str(den.replays)}
    print(f"gate 3  denoise step   {ms_eager:6.2f} -> {ms_replay:6.2f} ms   "
          f"{ms_eager / ms_replay:.2f}x")
    print(f"gate 3  chunk          {ms_chunk_eager:6.1f} -> {ms_chunk_replay:6.1f} ms   "
          f"{ms_chunk_eager / ms_chunk_replay:.2f}x   (prefill + 10 steps)")

    ok = d1 == 0 and worst == 0
    out = HERE / "static_capture_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\n{'ALL GATES PASS' if ok else 'GATES FAILED'}   -> {out}")

    # restore the class, so nothing leaks into a caller that imports this module
    type(model).denoise_step = orig_denoise
    type(model).embed_suffix = orig_embed_suffix
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

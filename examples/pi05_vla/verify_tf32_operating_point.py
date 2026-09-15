#!/usr/bin/env python3
"""Real-weight H100 gate for the declared pi0.5 TF32 NUMERIC operating point."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lerobot.policies.pi05 import modeling_pi05 as M  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
from pi05_iwm.precision import Pi05PrecisionLease  # noqa: E402
from pi05_iwm.static_capture import install_static_capture  # noqa: E402
from pi05_iwm.surface import Pi05Surface  # noqa: E402

DEV = "cuda:0"
MAX_ALLOWED_DELTA = 0.002482


def make_case(seed: int, valid_tokens: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = [torch.rand(1, 3, 224, 224, generator=generator).to(DEV).mul_(2).sub_(1)
              for _ in range(3)]
    image_masks = [torch.ones(1, dtype=torch.bool, device=DEV) for _ in range(3)]
    tokens = torch.randint(2, 1000, (1, 48), generator=generator).to(DEV)
    masks = torch.zeros(1, 48, dtype=torch.bool, device=DEV)
    masks[:, :valid_tokens] = True
    noise = torch.randn(1, 50, 32, generator=generator).to(DEV)
    return images, image_masks, tokens, masks, noise


def prefill(model, case):
    images, image_masks, tokens, masks, _noise = case
    embeds, pad, att = model.embed_prefix(images, image_masks, tokens, masks)
    att2d = M.make_att_2d_masks(pad, att)
    positions = torch.cumsum(pad, dim=1) - 1
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=M.prepare_attention_masks_4d(att2d), position_ids=positions,
        past_key_values=None, inputs_embeds=[embeds, None], use_cache=True,
    )
    return pad, cache


def integrate(model, case, denoise):
    pad, cache = prefill(model, case)
    x = case[-1].clone()
    for step in range(10):
        timestep = torch.full((1,), 1.0 - step / 10, device=DEV)
        velocity = denoise(pad, cache, x, timestep)
        x = x - 0.1 * velocity
    return x


def timed(call, n: int):
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(n):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> int:
    policy = PI05Policy.from_pretrained("lerobot/pi05_base", local_files_only=True).to(DEV).eval()
    model = policy.model
    original = model.denoise_step

    cases = (
        ("A_mask48", make_case(101, 48)),
        ("B_new_image_tokens_mask31", make_case(102, 31)),
        ("A_after_B", make_case(101, 48)),
        ("C_new_image_tokens_mask19", make_case(103, 19)),
    )

    with Pi05PrecisionLease(torch, "fp32"), torch.no_grad():
        fp32 = {name: integrate(model, case, original) for name, case in cases}
        fp32_ms = timed(lambda: integrate(model, cases[1][1], original), 11)

    with Pi05PrecisionLease(torch, "tf32"), torch.no_grad():
        tf32_eager = {name: integrate(model, case, original) for name, case in cases}
        tf32_repeat = {name: integrate(model, case, original) for name, case in cases}
        tf32_eager_ms = timed(lambda: integrate(model, cases[1][1], original), 11)

        Pi05Surface(model).hoist_loop_constants()
        static = install_static_capture(model, step_tables=True)
        while static.replays == 0:
            integrate(model, cases[0][1], static)
        captured = {name: integrate(model, case, static) for name, case in cases}
        captured_repeat = {name: integrate(model, case, static) for name, case in cases}
        static_ms = timed(lambda: integrate(model, cases[1][1], static), 15)
        prefill_ms = timed(lambda: prefill(model, cases[1][1]), 21)

    gates = {}
    for name, _case in cases:
        gates[name] = {
            "max_abs_vs_fp32": (captured[name] - fp32[name]).abs().max().item(),
            "static_vs_tf32_eager": (captured[name] - tf32_eager[name]).abs().max().item(),
            "tf32_eager_repeat": (tf32_eager[name] - tf32_repeat[name]).abs().max().item(),
            "static_repeat": (captured[name] - captured_repeat[name]).abs().max().item(),
        }
        print(name, gates[name], flush=True)
    worst = max(row["max_abs_vs_fp32"] for row in gates.values())
    result = {
        "status": "PASS" if worst <= MAX_ALLOWED_DELTA else "FAIL",
        "tier": "NUMERIC",
        "device": torch.cuda.get_device_name(0),
        "checkpoint": "lerobot/pi05_base@b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba",
        "closed_loop": "NOT RUN - see tf32_closed_loop_preregistration.json",
        "max_allowed_delta": MAX_ALLOWED_DELTA,
        "max_observed_delta": worst,
        "cases": gates,
        "timing_ms": {
            "fp32_eager_chunk": fp32_ms,
            "tf32_eager_chunk": tf32_eager_ms,
            "tf32_static_chunk": static_ms,
            "tf32_prefill": prefill_ms,
            "speedup_vs_fp32_eager": fp32_ms / static_ms,
        },
    }
    ok = (
        worst <= MAX_ALLOWED_DELTA
        and all(row["static_vs_tf32_eager"] == 0 for row in gates.values())
        and all(row["tf32_eager_repeat"] == 0 for row in gates.values())
        and all(row["static_repeat"] == 0 for row in gates.values())
        and static_ms < fp32_ms
    )
    result["status"] = "PASS" if ok else "FAIL"
    output = HERE / "tf32_static_h100_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["timing_ms"], "->", output)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare stock and InstinctFlash pi0.5 core latency on one fixed LIBERO observation.

Reuses one processed observation for both arms. Times prefix embedding/prefill,
seeded noise generation, and 10 denoising steps; excludes simulator setup and
observation preprocessing. Reports the median of 15 chunks after two warmups.
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
sys.path.insert(0, str(HERE.parents[1]))

from lerobot.policies.pi05 import modeling_pi05 as M  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.utils import preprocess_observation

from pi05_iwm.static_capture import WARMUP_STEPS, install_static_capture  # noqa: E402
from pi05_iwm.surface import Pi05Surface  # noqa: E402

CKPT = os.environ.get("IFL_PI05_CKPT", "lerobot/pi05_libero_finetuned_v044")
DEV = "cuda:0"
README_PAIR = (206.7, 72.8)


def prepare_model_inputs(policy, batch: dict[str, torch.Tensor]):
    """Prepare model images and retrieve tokenized task inputs.

    Returns:
        Images, camera-validity masks, language token IDs, and language-validity masks.
    """
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]

    return images, img_masks, tokens, masks


def create_env(suite, task_id):
    """Create one synchronous environment for the selected LIBERO task.

    Returns:
        The environment configuration and suite/task-indexed environment mapping.
    """
    env_cfg = make_env_config("libero", task=suite, task_ids=[task_id])
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    return env_cfg, envs


def prefill(model, images, image_masks, tokens, masks):
    """Compute observation-prefix keys and values from prepared model inputs.

    Returns:
        Prefix padding masks and the observation-prefix KV cache.
    """
    with torch.no_grad():
        pe, ppm, pam = model.embed_prefix(images, image_masks, tokens, masks)
        a2d = M.make_att_2d_masks(ppm, pam)
        pos = torch.cumsum(ppm, dim=1) - 1
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, kv = model.paligemma_with_expert.forward(
            attention_mask=M.prepare_attention_masks_4d(a2d),
            position_ids=pos,
            past_key_values=None,
            inputs_embeds=[pe, None],
            use_cache=True,
        )
    return ppm, kv


def timed_chunks(model, denoise, images, image_masks, tokens, masks, n=15, warm=2):
    """Measure repeated full chunks using a fixed observation and seeded noise.

    Each chunk includes prefix embedding/prefill and 10 denoising steps. Warmup
    chunks are excluded, and CUDA is synchronized around each timed chunk.

    Returns:
        Median wall-clock milliseconds and the final predicted padded action chunk.
    """

    def chunk():
        """Predict one padded action chunk with the supplied denoising implementation."""
        ppm, kv = prefill(model, images, image_masks, tokens, masks)
        with torch.no_grad():
            g = torch.Generator(device="cpu").manual_seed(500)
            xt = torch.randn(
                1, model.config.chunk_size, model.config.max_action_dim, generator=g
            ).to(DEV)
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


def main():
    """Load one LIBERO observation, benchmark both arms, and report latency and equality."""
    if not torch.cuda.is_available():
        print("this reproduction needs a CUDA GPU")
        return 2
    print(f"loading {CKPT} ...")
    config = PI05Config.from_pretrained(CKPT)
    config.compile_model = False
    config.device = DEV
    policy = PI05Policy.from_pretrained(CKPT, config=config).to(DEV).eval()
    model = policy.model
    env_cfg, envs = create_env("libero_object", 0)
    env = envs["libero_object"][0]
    orig_denoise = type(model).denoise_step

    observation, info = env.reset()

    obs = preprocess_observation(observation)
    task_desc = list(env.call("task_description"))
    obs["task"] = task_desc
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy.config
    )
    obs = env_preprocessor(obs)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=CKPT
    )
    batch = preprocessor(obs)
    im, im_mk, tk, ms = prepare_model_inputs(policy, batch)
    env.close()

    def eager(ppm, kv, x, t):
        """Run the checkpoint's original denoising step without IF capture."""
        with torch.no_grad():
            return orig_denoise(
                model, prefix_pad_masks=ppm, past_key_values=kv, x_t=x, timestep=t
            )

    ms_stock, out_stock = timed_chunks(model, eager, im, im_mk, tk, ms)

    Pi05Surface(model).hoist_loop_constants()
    den = install_static_capture(model)
    ppm_w, kv_w = prefill(model, im, im_mk, tk, ms)
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(500)
        x_w = torch.randn(
            1, model.config.chunk_size, model.config.max_action_dim, generator=g
        ).to(DEV)
        for _ in range(WARMUP_STEPS + 1):
            den(ppm_w, kv_w, x_w, torch.full((1,), 0.5, device=DEV))
    assert den.replays >= 1, "graph was never captured"
    ms_if, out_if = timed_chunks(model, den, im, im_mk, tk, ms)

    d = (out_stock - out_if).abs().max().item()
    speedup = ms_stock / ms_if
    result = {
        "checkpoint": CKPT,
        "protocol": "full chunk = prefill + 10 denoise steps, median of 15 (2 warm)",
        "torch_eager_chunk_ms": round(ms_stock, 1),
        "instinctflash_static_capture_chunk_ms": round(ms_if, 1),
        "speedup": round(speedup, 2),
        "arms_bitexact_on_timed_input": d == 0.0,
        "readme_pair_ms": list(README_PAIR),
        "step_tables": os.environ.get("IFL_PI05_STEP_TABLES", "1") != "0",
    }
    print(json.dumps(result, indent=1))
    print(
        f"\npi05  {ms_stock:.1f} -> {ms_if:.1f} ms   {speedup:.2f}x   "
        f"(README: {README_PAIR[0]:.0f} -> {README_PAIR[1]:.0f} ms, 2.84x)"
    )
    if d != 0.0:
        print(
            f"WARNING: arms disagree on the timed input (max |d| {d:.3e}); "
            f"run verify_static_capture.py before trusting either number"
        )
        return 1
    type(model).denoise_step = orig_denoise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

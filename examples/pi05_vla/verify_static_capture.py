#!/usr/bin/env python3
"""The gates for static-KV graph capture, in the order they can fail.

    # Hub checkpoint (historical default):
    CUDA_VISIBLE_DEVICES=0 python examples/pi05_vla/verify_static_capture.py

    # Local checkpoint following current LeRobot/Transformers main:
    CUDA_VISIBLE_DEVICES=0 python examples/pi05_vla/verify_static_capture.py \
        --checkpoint /workspace/models/pi05_libero_finetuned_v044 \
        --device cuda:0 --output /tmp/pi05-static-kv-5090.json

Gate 0  hoisted embed_suffix is bitexact against upstream's (re-establishes the old measurement
        in this process, so gate 1's reference is upstream's numerics, not a hoisted cousin's).
Gate 1  replay vs eager on the CAPTURED input: must be bitexact.
Gate 2  replay vs eager on inputs the capture never saw — new x_t, new timestep, and a NEW PROMPT
        whose prefix and valid-token mask are refilled into the static buffers — plus the complete
        Euler-integrated action chunk. This is the exact failure class that rejected the DynamicCache
        capture (measured max |d| 2.116e-01 on a new x_t); every case here must read 0.
Gate 3  timing: denoise step replay vs eager, and full chunk (prefill + 10 steps) vs eager,
        medians over >= 15. With --prefix-graph, sample_actions is the two-graph end-to-end path.
"""
from __future__ import annotations

import argparse
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


def policy_inputs(cfg, prompt_seed: int, valid_tokens: int):
    """Return deterministic images/masks/tokens at the checkpoint's static extents."""
    g = torch.Generator(device="cpu").manual_seed(prompt_seed)
    num_views = len(getattr(cfg, "image_features", ())) or 3
    prompt_extent = int(getattr(cfg, "tokenizer_max_length", 200))
    valid_tokens = max(1, min(int(valid_tokens), prompt_extent))
    images = [
        torch.rand(1, 3, 224, 224, generator=g).to(DEV) * 2 - 1
        for _ in range(num_views)
    ]
    image_masks = [
        torch.ones(1, dtype=torch.bool, device=DEV) for _ in range(num_views)
    ]
    tokens = torch.randint(2, 1000, (1, prompt_extent), generator=g).to(DEV)
    token_mask = torch.arange(prompt_extent, device=DEV)[None, :] < valid_tokens
    return images, image_masks, tokens, token_mask


def prefill(model, cfg, prompt_seed: int, valid_tokens: int):
    """Build deterministic prefix K/V at the checkpoint's published static prompt extent.

    Values and valid-token masks both vary between prompt seeds. The tensor extent stays fixed,
    matching the real processor pipeline's padded prompt contract and exercising the per-chunk
    mask/position refresh without forcing a graph recapture.
    """
    im, mk, tk, ms = policy_inputs(cfg, prompt_seed, valid_tokens)
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


def _local_hf_revision(checkpoint: str):
    """Read the local-dir tree manifest written by ``hf download``, when unambiguous."""
    tree_dir = Path(checkpoint) / ".cache" / "huggingface" / "trees"
    manifests = sorted(tree_dir.glob("*.json")) if tree_dir.is_dir() else []
    return manifests[0].stem if len(manifests) == 1 else None


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Verify replay-safe, bit-exact Static-KV CUDA Graph capture for Pi0.5")
    ap.add_argument("--checkpoint", default="lerobot/pi05_base",
                    help="Hub id or local LeRobot Pi0.5 checkpoint")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=str(HERE / "static_capture_results.json"))
    ap.add_argument("--prompt-valid-a", type=int, default=13)
    ap.add_argument("--prompt-valid-b", type=int, default=31)
    ap.add_argument("--timing-iters", type=int, default=15)
    ap.add_argument("--timing-warm", type=int, default=3)
    ap.add_argument("--no-step-tables", action="store_true",
                    help="Capture the original timestep MLP/AdaRMS projections instead of tables")
    ap.add_argument("--full-chunk-graph", action="store_true",
                    help="Capture the complete fixed Euler loop; defaults to original kernels")
    ap.add_argument("--full-step-tables", action="store_true",
                    help="Bake exact time-MLP/AdaRMS outputs into the fixed full-loop graph")
    ap.add_argument("--prefix-graph", action="store_true",
                    help="Also capture original vision/language prefix; implies full chunk")
    args = ap.parse_args()
    if args.full_step_tables:
        args.full_chunk_graph = True
    if args.prefix_graph:
        args.full_chunk_graph = True
    if args.full_step_tables and args.no_step_tables:
        ap.error("--full-step-tables conflicts with --no-step-tables")
    return args


def main() -> int:
    args = _parse_args()
    global DEV
    DEV = args.device

    import lerobot
    import transformers
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    local_checkpoint = Path(args.checkpoint).exists()
    config = PI05Config.from_pretrained(
        args.checkpoint, local_files_only=local_checkpoint)
    # torch.compile changes the kernel path before this verifier can establish its eager oracle.
    # Static-KV captures the checkpoint's original PyTorch modules instead.
    config.compile_model = False
    config.device = DEV
    policy = PI05Policy.from_pretrained(
        args.checkpoint, config=config, local_files_only=local_checkpoint,
        strict=True).to(DEV).eval()
    model = policy.model
    model_allocated_bytes = torch.cuda.memory_allocated(torch.device(DEV))
    orig_denoise = type(model).denoise_step
    orig_embed_suffix = type(model).embed_suffix
    orig_sample_actions = model.sample_actions
    results: dict = {
        "checkpoint": str(Path(args.checkpoint).resolve()) if local_checkpoint else args.checkpoint,
        "checkpoint_revision": _local_hf_revision(args.checkpoint) if local_checkpoint else None,
        "device": DEV,
        "python": sys.version.split()[0],
        "lerobot": getattr(lerobot, "__version__", None),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.device(DEV)),
        "compute_capability": list(torch.cuda.get_device_capability(torch.device(DEV))),
        "config": {
            "dtype": policy.config.dtype,
            "chunk_size": policy.config.chunk_size,
            "num_inference_steps": policy.config.num_inference_steps,
            "num_views": len(policy.config.image_features),
            "prompt_extent": policy.config.tokenizer_max_length,
            "step_tables": args.full_step_tables or (
                not args.no_step_tables and not args.full_chunk_graph),
            "full_step_tables": args.full_step_tables,
            "full_chunk_graph": args.full_chunk_graph,
            "prefix_graph": args.prefix_graph,
        },
    }

    try:
        ppm_a, kv_a = prefill(
            model, policy.config, prompt_seed=7, valid_tokens=args.prompt_valid_a)
        x0, t0 = sample_input(model, seed=100)
        policy_args = policy_inputs(
            policy.config, prompt_seed=17, valid_tokens=args.prompt_valid_a)
        policy_args_new = policy_inputs(
            policy.config, prompt_seed=18, valid_tokens=args.prompt_valid_b)
        policy_noise, _ = sample_input(model, seed=700)
        policy_noise_new, _ = sample_input(model, seed=701)
        policy_noise_dynamic, _ = sample_input(model, seed=702)
        dynamic_steps = max(1, int(policy.config.num_inference_steps) // 2)
        policy_ref = policy_ref_new = policy_ref_dynamic = None
        if args.full_chunk_graph:
            with torch.no_grad():
                policy_ref = orig_sample_actions(
                    *policy_args, noise=policy_noise.clone(),
                    num_steps=policy.config.num_inference_steps)
                policy_ref_new = orig_sample_actions(
                    *policy_args_new, noise=policy_noise_new.clone(),
                    num_steps=policy.config.num_inference_steps)
                policy_ref_dynamic = orig_sample_actions(
                    *policy_args_new, noise=policy_noise_dynamic.clone(),
                    num_steps=dynamic_steps)

        # -- gate 0: the hoist is bitexact -----------------------------------------------------
        with torch.no_grad():
            ref_pre = orig_denoise(
                model, prefix_pad_masks=ppm_a, past_key_values=kv_a,
                x_t=x0, timestep=t0)
        Pi05Surface(model).hoist_loop_constants()
        with torch.no_grad():
            ref_post = orig_denoise(
                model, prefix_pad_masks=ppm_a, past_key_values=kv_a,
                x_t=x0, timestep=t0)
        d0 = (ref_pre - ref_post).abs().max().item()
        results["gate0_hoist_bitexact"] = d0
        print(f"gate 0  hoist vs upstream          max |d| {d0:.3e}   "
              f"{'PASS' if d0 == 0 else 'FAIL'}")
        if d0 != 0:
            results["status"] = "failed_gate0"
            Path(args.output).write_text(json.dumps(results, indent=1) + "\n")
            return 1

        def eager(ppm, kv, x, t):
            with torch.no_grad():
                return orig_denoise(
                    model, prefix_pad_masks=ppm, past_key_values=kv,
                    x_t=x, timestep=t)

        ms_sample_eager = None
        if args.full_chunk_graph:
            ms_sample_eager = timed(
                lambda: orig_sample_actions(
                    *policy_args, noise=policy_noise.clone(),
                    num_steps=policy.config.num_inference_steps),
                n=args.timing_iters, warm=args.timing_warm)

        # -- install; drive past warmup into capture on the captured input ----------------------
        den = install_static_capture(
            model,
            step_tables=args.full_step_tables or (
                not args.no_step_tables and not args.full_chunk_graph),
            full_chunk=args.full_chunk_graph,
            prefix_graph=args.prefix_graph,
        )
        with torch.no_grad():
            if args.full_chunk_graph:
                # Keep this process representative of production's TWO graphs. The per-step path
                # is exercised eagerly below; it has its own standalone capture gate otherwise.
                out_cap = den(ppm_a, kv_a, x0, t0)
            else:
                for _ in range(WARMUP_STEPS + 1):
                    out_cap = den(ppm_a, kv_a, x0, t0)
                assert den.replays >= 1, "per-step graph was never captured"

        d1 = (eager(ppm_a, kv_a, x0, t0) - out_cap).abs().max().item()
        results["gate1_captured_input"] = d1
        gate1_label = (
            "static vs eager, pre-capture" if args.full_chunk_graph
            else "replay vs eager, captured")
        print(f"gate 1  {gate1_label:27} max |d| {d1:.3e}   "
              f"{'PASS' if d1 == 0 else 'FAIL'}")

        # -- gate 2: inputs and prompt mask/content the graph never saw --------------------------
        cases = []
        for i, seed in enumerate((201, 202, 203)):
            x, t = sample_input(model, seed)
            with torch.no_grad():
                d = (eager(ppm_a, kv_a, x, t) - den(ppm_a, kv_a, x, t)).abs().max().item()
            cases.append((f"new x_t/t #{i + 1}", d))
        ppm_b, kv_b = prefill(
            model, policy.config, prompt_seed=8, valid_tokens=args.prompt_valid_b)
        for i, seed in enumerate((301, 302)):
            x, t = sample_input(model, seed)
            with torch.no_grad():
                d = (eager(ppm_b, kv_b, x, t) - den(ppm_b, kv_b, x, t)).abs().max().item()
            cases.append((f"new prompt/mask #{i + 1}", d))
        worst = max(d for _, d in cases)
        results["gate2_new_inputs"] = {name: d for name, d in cases}
        for name, d in cases:
            print(f"gate 2  {name:24}  max |d| {d:.3e}   "
                  f"{'PASS' if d == 0 else 'FAIL'}")
        print(f"gate 2  worst                       max |d| {worst:.3e}  "
              f"(old DynamicCache failure was 2.116e-01)")

        # -- gate 3: timing ---------------------------------------------------------------------
        x, t = sample_input(model, 400)
        ms_eager = timed(
            lambda: eager(ppm_a, kv_a, x, t),
            n=args.timing_iters, warm=args.timing_warm)
        ms_replay = None
        if not args.full_chunk_graph:
            ms_replay = timed(
                lambda: den(ppm_a, kv_a, x, t),
                n=args.timing_iters, warm=args.timing_warm)

        num_steps = int(policy.config.num_inference_steps)

        def chunk(fn):
            ppm, kv = prefill(
                model, policy.config, prompt_seed=9,
                valid_tokens=args.prompt_valid_a)
            with torch.no_grad():
                xt, _ = sample_input(model, 500)
                if args.full_chunk_graph and hasattr(fn, "run_full_chunk"):
                    return fn.run_full_chunk(ppm, kv, xt, num_steps)
                for k in range(num_steps):
                    timestep = torch.full(
                        (1,), 1.0 - k / num_steps, device=DEV)
                    v = fn(ppm, kv, xt, timestep)
                    xt = xt - (1.0 / num_steps) * v
            return xt

        with torch.no_grad():
            chunk_ref = chunk(eager)
            if args.full_chunk_graph:
                chunk(den)  # one eager, static-KV warmup schedule before whole-loop capture
            chunk_replay = chunk(den)
        d_chunk = (chunk_ref - chunk_replay).abs().max().item()
        results["gate2_full_chunk"] = d_chunk
        print(f"gate 2  full {num_steps}-step chunk          max |d| {d_chunk:.3e}   "
              f"{'PASS' if d_chunk == 0 else 'FAIL'}")

        d_policy = 0.0
        if args.full_chunk_graph:
            with torch.no_grad():
                if args.prefix_graph:
                    model.sample_actions(
                        *policy_args, noise=policy_noise.clone(),
                        num_steps=policy.config.num_inference_steps)
                policy_replay = model.sample_actions(
                    *policy_args, noise=policy_noise.clone(),
                    num_steps=policy.config.num_inference_steps)
                policy_replay_new = model.sample_actions(
                    *policy_args_new, noise=policy_noise_new.clone(),
                    num_steps=policy.config.num_inference_steps)
            policy_cases = {
                "captured prompt/noise":
                    (policy_ref - policy_replay).abs().max().item(),
                "new prompt/mask/noise":
                    (policy_ref_new - policy_replay_new).abs().max().item(),
            }
            d_policy = max(policy_cases.values())
            results["gate2_sample_actions"] = d_policy
            results["gate2_sample_actions_cases"] = policy_cases
            for name, delta in policy_cases.items():
                print(f"gate 2  sample_actions {name:21} max |d| {delta:.3e}   "
                      f"{'PASS' if delta == 0 else 'FAIL'}")

        chunk_warm = min(args.timing_warm, 2)
        ms_chunk_eager = timed(
            lambda: chunk(eager), n=args.timing_iters, warm=chunk_warm)
        ms_chunk_replay = timed(
            lambda: chunk(den), n=args.timing_iters, warm=chunk_warm)
        ms_sample_replay = None
        if args.full_chunk_graph:
            ms_sample_replay = timed(
                lambda: model.sample_actions(
                    *policy_args, noise=policy_noise.clone(),
                    num_steps=policy.config.num_inference_steps),
                n=args.timing_iters, warm=args.timing_warm)

        results["gate3_timing_ms"] = {
            "denoise_eager": round(ms_eager, 2),
            "denoise_replay": round(ms_replay, 2) if ms_replay is not None else None,
            "step_speedup": round(ms_eager / ms_replay, 2)
            if ms_replay is not None else None,
            "chunk_eager": round(ms_chunk_eager, 1),
            "chunk_replay": round(ms_chunk_replay, 1),
            "chunk_speedup": round(ms_chunk_eager / ms_chunk_replay, 2),
            "timing_iters": args.timing_iters,
            "sample_actions_eager": round(ms_sample_eager, 1)
            if ms_sample_eager is not None else None,
            "sample_actions_replay": round(ms_sample_replay, 1)
            if ms_sample_replay is not None else None,
            "sample_actions_speedup": round(ms_sample_eager / ms_sample_replay, 2)
            if ms_sample_eager is not None and ms_sample_replay is not None else None,
            "chunk_graph_replays": den.chunk_replays,
            "prefix_graph_replays": den.prefix_replays,
            "note": "chunk includes prefill + denoise steps; replays counted=" + str(den.replays),
        }
        fixed_path_self_check = den.self_check
        results["runtime_self_check"] = fixed_path_self_check
        results["memory_gib"] = {
            "allocated_after_model": round(model_allocated_bytes / 2**30, 3),
            "allocated_after_graphs": round(torch.cuda.memory_allocated(DEV) / 2**30, 3),
            "reserved_after_graphs": round(torch.cuda.memory_reserved(DEV) / 2**30, 3),
        }
        d_dynamic = 0.0
        if args.full_chunk_graph:
            with torch.no_grad():
                policy_dynamic = model.sample_actions(
                    *policy_args_new, noise=policy_noise_dynamic.clone(),
                    num_steps=dynamic_steps)
            d_dynamic = (policy_ref_dynamic - policy_dynamic).abs().max().item()
            results["gate2_dynamic_nfe_fallback"] = d_dynamic
            print(f"gate 2  dynamic NFE={dynamic_steps} fallback       max |d| "
                  f"{d_dynamic:.3e}   {'PASS' if d_dynamic == 0 else 'FAIL'}")
        if ms_replay is not None:
            print(f"gate 3  denoise step   {ms_eager:6.2f} -> {ms_replay:6.2f} ms   "
                  f"{ms_eager / ms_replay:.2f}x")
        else:
            print(f"gate 3  denoise eager  {ms_eager:6.2f} ms   "
                  "(per-step graph intentionally not captured in two-graph run)")
        print(f"gate 3  chunk          {ms_chunk_eager:6.1f} -> {ms_chunk_replay:6.1f} ms   "
              f"{ms_chunk_eager / ms_chunk_replay:.2f}x   "
              f"(prefill + {num_steps} steps)")
        if ms_sample_replay is not None:
            print(f"gate 3  sample_actions {ms_sample_eager:6.1f} -> "
                  f"{ms_sample_replay:6.1f} ms   "
                  f"{ms_sample_eager / ms_sample_replay:.2f}x   "
                  f"(prefix graph {'on' if args.prefix_graph else 'off'})")

        self_check_ok = (
            fixed_path_self_check is None or (
                fixed_path_self_check.get("bitexact") is True and not den.rejected))
        ok = (d1 == 0 and worst == 0 and d_chunk == 0 and d_policy == 0
              and d_dynamic == 0 and self_check_ok)
        results["status"] = "pass" if ok else "failed_bitexact"
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1) + "\n")
        print(f"\n{'ALL GATES PASS' if ok else 'GATES FAILED'}   -> {out}")
        return 0 if ok else 1
    finally:
        # The verifier runs one model per process, but remove instance wrappers and table modules
        # as well so calling ``main`` from a test process cannot leak treatment into a control arm.
        driver = locals().get("den")
        if driver is not None and driver._denses is not None:
            for norm, real in driver._denses:
                norm.dense = real
        model.__dict__.pop("_ifl_static_denoiser", None)
        model.__dict__.pop("denoise_step", None)
        model.__dict__.pop("embed_suffix", None)
        model.__dict__.pop("sample_actions", None)
        type(model).denoise_step = orig_denoise
        type(model).embed_suffix = orig_embed_suffix


if __name__ == "__main__":
    raise SystemExit(main())

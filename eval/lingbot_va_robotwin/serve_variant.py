#!/usr/bin/env python3
"""Launch the LingBot-VA server with InstinctWM optimization variants toggled on.

This is an A/B harness, not a fork. It imports the upstream server unmodified and patches
named behaviours at runtime, so `git status` in lingbot-va stays clean and every variant is
one flag away from the stock baseline. Each variant exists because a specific cost was
measured or read out of the code.

Variants (all default OFF; stock behaviour is the baseline):

  --no-fsdp
      Skip `shard_model`. `distributed/util.py:15-19` applies FSDP `fully_shard` whenever
      `dist.is_initialized()`, which is ALWAYS -- `init_distributed` is called
      unconditionally, even for a single-GPU server. `fsdp.py:28-34` wraps 4 units per
      block (attn1, attn2, ffn, block) across 30 blocks plus the root = 121 units, with
      `reshard_after_forward=True`. At world_size=1 the all-gather is a no-op collective
      but PyTorch still pays the flat-param copy and stream sync on every unit on every
      forward: ~9,300 shard/unshard round trips per 77-forward cycle. Expected to be
      numerically identical at world_size=1 (MixedPrecisionPolicy param_dtype=bf16 on an
      already-bf16 model), but that is a claim to VERIFY, not assume.

  --no-empty-cache
      Neuter `torch.cuda.empty_cache()`, which the server calls on every chunk
      (`wan_va_server.py:569`) and every KV update (`:603`). It releases the caching
      allocator back to the driver, forcing cudaMalloc again next cycle.

  --no-debug-dump
      Neuter `save_async`. It is async only for the disk write: `utils.py:63-64` does a
      BLOCKING `.cpu()` of the full latent/action tensors on the critical path, three times
      per cycle, unconditionally, with no upstream flag to disable it.

Usage mirrors the upstream launcher:
    python serve_variant.py --config-name robotwin --port 29061 --save_root /tmp/x --no-fsdp
"""
from __future__ import annotations

import argparse
import os
import sys

LINGBOT_ROOT = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
sys.path.insert(0, os.path.join(LINGBOT_ROOT, "wan_va"))
sys.path.insert(0, LINGBOT_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="robotwin")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--save_root", default=None)
    ap.add_argument("--no-fsdp", action="store_true")
    ap.add_argument("--no-empty-cache", action="store_true")
    ap.add_argument("--no-debug-dump", action="store_true")
    ap.add_argument(
        "--conditioning-prefill", action="store_true",
        help="Cache the episode-constant cross-attention K/V for all 30 layers. model.py:331 "
             "withholds attn_caches from cross-attention, so the text K/V is re-projected on "
             "all 77 forwards; that is ~39%% of a control cycle's arithmetic, and 67.6%% of an "
             "action forward's layer FLOPs (32-token query vs 512-token text).")
    ap.add_argument(
        "--deterministic-seed", type=int, default=None,
        help="Seed torch before each chunk's noise draw. REQUIRED to compare two variants: "
             "_infer draws torch.randn for the initial video latents and action tokens "
             "(wan_va_server.py:449-462) with no seeding, so two stock servers already "
             "disagree and any A/B on output values is meaningless without this.")
    args = ap.parse_args()

    import torch

    import wan_va_server as S

    applied = []

    if args.no_fsdp:
        # wan_va_server binds _configure_model at import time, so patch the BOUND name.
        def _configure_model_nofsdp(model, shard_fn, param_dtype, device, eval_mode=True):
            if eval_mode:
                model.eval().requires_grad_(False)
            model.to(param_dtype)
            model.to(device)
            return model

        S._configure_model = _configure_model_nofsdp
        applied.append("no-fsdp")

    if args.no_empty_cache:
        # Patch on the torch module the server actually calls through.
        torch.cuda.empty_cache = lambda *a, **k: None
        applied.append("no-empty-cache")

    if args.no_debug_dump:
        S.save_async = lambda obj, path: None
        applied.append("no-debug-dump")

    if getattr(args, "conditioning_prefill", False):
        sys.path.insert(0, "/home/ubuntu/InstinctWM")
        from instinctwm.runtime.lingbot_install import install_conditioning_prefill

        install_conditioning_prefill(S, S.VA_Server)
        applied.append("conditioning-prefill")

    if args.deterministic_seed is not None:
        # Seed as a function of frame_st_id, not a constant: a constant would make every
        # chunk in an episode start from the SAME noise, which is not the stock
        # distribution and would itself change behaviour.
        _orig_infer = S.VA_Server._infer

        def _seeded_infer(self, obs, frame_st_id=0):
            torch.manual_seed(args.deterministic_seed + frame_st_id)
            torch.cuda.manual_seed_all(args.deterministic_seed + frame_st_id)
            return _orig_infer(self, obs, frame_st_id=frame_st_id)

        S.VA_Server._infer = _seeded_infer
        applied.append(f"deterministic-seed={args.deterministic_seed}")

    print("=" * 72, flush=True)
    print(f"InstinctWM serve_variant: {applied if applied else ['STOCK BASELINE']}", flush=True)
    print(f"  ckpt   : {os.environ.get('LINGBOT_CKPT')}", flush=True)
    print(f"  config : {args.config_name}   port: {args.port}", flush=True)
    print("=" * 72, flush=True)

    class _A:
        pass

    a = _A()
    a.config_name = args.config_name
    a.port = args.port
    a.save_root = args.save_root

    S.init_logger()
    S.run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    ap.add_argument("--hoist-casts", action="store_true",
        help="Cast loop-invariant constants once per episode instead of once per forward: "
             "FP32LayerNorm's weight/bias (4,740 casts of a constant per cycle) and the block's "
             "scale_shift_table, which lets the modulation add promote bf16->fp32 for free and "
             "deletes a 35.4 MB materialization per block.")
    ap.add_argument("--ring-kv", action="store_true",
        help="Address the KV pool by ring interval instead of boolean mask. Removes the "
             "per-layer-per-forward mask.nonzero() host sync and the full-pool advanced-index "
             "gather (model.py:451-453); valid becomes a slice. Falls back to stock when the "
             "interval wraps, so key order stays ascending and the pass stays bit-exact.")
    ap.add_argument("--graph-blocks", action="store_true",
        help="[NOT SHIPPABLE -- 2.17x but NOT bit-exact, max|d action| 1.398 = 136%% of real "
             "movement. The captured region mutates host-side ring bookkeeping that replay never "
             "re-executes. See graph_capture.py. Kept for measurement only.] "
             "Run the 30-block transformer stack from a captured CUDA graph (E3). Requires "
             "--ring-kv: the stock mask.nonzero() is a data-dependent shape and capture of a "
             "stock block fails with cudaErrorStreamCaptureInvalidated. Measured per-op cost is "
             "6.2 us of which 83.6%% is cudaLaunchKernel; replay is ~1.17 us.")
    ap.add_argument("--stable-pools", action="store_true",
        help="E1: reset clears logical KV state in place instead of reallocating the pools, so "
             "captured graphs stay valid across episodes. Only has an effect with --graph-blocks, "
             "which still verifies every pool pointer survived before keeping its graphs.")
    ap.add_argument("--no-keep-graphs", action="store_true",
        help="Drop captured graphs at every reset even when --stable-pools certifies pointer "
             "stability. Preservation is ON by default and gated by that certificate; this is the "
             "escape hatch if a new pass introduces episode-scoped device state the certificate "
             "does not yet cover.")
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

    if getattr(args, "ring_kv", False):
        sys.path.insert(0, "/home/ubuntu/InstinctWM")
        from instinctwm.optimizer.passes.ring_kv import RingKVAddressing
        RingKVAddressing().install(S, S.VA_Server)
        applied.append("ring-kv")

    if getattr(args, "hoist_casts", False):
        sys.path.insert(0, "/home/ubuntu/InstinctWM")
        from instinctwm.optimizer.passes.hoist_invariant_casts import HoistInvariantCasts
        HoistInvariantCasts().install(S, S.VA_Server)
        applied.append("hoist-casts")

    _pools_pass = None
    if getattr(args, "stable_pools", False):
        sys.path.insert(0, "/home/ubuntu/InstinctWM")
        from instinctwm.optimizer.passes.stable_pools import StableStatePools
        _pools_pass = StableStatePools()
        _pools_pass.install(S, S.VA_Server)
        applied.append("stable-pools")

    if getattr(args, "graph_blocks", False):
        if not getattr(args, "ring_kv", False):
            print("REFUSING: --graph-blocks requires --ring-kv. The stock KV path calls "
                  "mask.nonzero() per layer per forward, which is a data-dependent shape; "
                  "capture fails with cudaErrorStreamCaptureInvalidated.", flush=True)
            return 2
        sys.path.insert(0, "/home/ubuntu/InstinctWM")
        from instinctwm.optimizer.passes.graph_capture import GraphBlockStack
        _graph_pass = GraphBlockStack()
        _graph_pass.install(S, S.VA_Server)
        applied.append("graph-blocks")

        if _pools_pass is not None:
            # Bind the pools on first use, then let the graph pass consult them at every reset.
            _orig_reset_bind = S.VA_Server._reset

            def _reset_bind(self, prompt=None, _p=_pools_pass, _o=_orig_reset_bind):
                out = _o(self, prompt=prompt)
                if hasattr(self, "transformer"):
                    _p.bind(self.transformer)
                return out

            S.VA_Server._reset = _reset_bind
            _graph_pass.bind_hook = _pools_pass.bind
            if not getattr(args, "no_keep_graphs", False):
                _graph_pass.stability_check = lambda: _pools_pass.pointers_stable()

        # Report capture/replay counts at the end of each chunk so the recapture rate is visible
        # rather than inferred -- if the key churns, the win evaporates and we need to know.
        _orig_infer_g = S.VA_Server._infer

        def _infer_reporting(self, obs, frame_st_id=0):
            out = _orig_infer_g(self, obs, frame_st_id=frame_st_id)
            print(f"[graph_block_stack] {_graph_pass.stats()}"
                  + (f" | {_pools_pass.stats()}" if _pools_pass else ""), flush=True)
            return out

        S.VA_Server._infer = _infer_reporting

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

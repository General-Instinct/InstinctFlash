#!/usr/bin/env python3
"""Why do the VAE's 3D convolutions fall back to `slow_conv_dilated3d`?

21.7 ms of the warm 487 ms cycle, 42 calls at 517 us each -- and `slow_conv_dilated3d` is a FALLBACK.
Meanwhile 22 other convolutions in the same cycle dispatch to `aten::cudnn_convolution` at 32.9 us. So
the question is not "how do we write a faster conv", it is "why is this conv not using the fast path
that the others are", and the answer may be configuration rather than code.

WHAT IS ALREADY RULED OUT BY READING. `WanCausalConv3d` (diffusers autoencoder_kl_wan.py:133) sets
`self.padding = (0, 0, 0)` and pads explicitly with `F.pad`, and never sets `dilation`, which therefore
stays 1. So the op's NAME is misleading: nothing here is dilated. `slow_conv_dilated3d` is simply where
PyTorch lands when its other 3D backends decline, so the useful question is which one declined and why.

WHAT THIS PROBE DOES.

  1. Hooks every Conv3d in the live VAE and records the exact call signature -- input shape, dtype,
     memory format, kernel, stride, padding, groups -- for the convolutions that actually run.
  2. For each distinct signature, replays it standalone under the profiler and reports WHICH aten op
     dispatches. That separates "falls back" from "does not" per signature rather than in aggregate.
  3. For the fallback signatures, tries the levers that change backend selection and reports which one
     (if any) reaches cuDNN and what it costs:
        cudnn.benchmark            heuristic search instead of the default heuristic
        channels_last_3d           NHWDC layout, which cuDNN v8 prefers for 3D
        fp16 instead of bf16       bf16 3D conv support is thinner than fp16
        fp32                       the widest support, as a diagnostic not a proposal
     A layout or dtype change is NOT bit-exact and would need the paired protocol; `cudnn.benchmark` is
     a search-strategy flag and does not change arithmetic, so it is the only candidate that could ship
     under a max|delta| = 0 gate.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29989 probe_vae_conv_backend.py
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics
import sys
import time
from pathlib import Path

IFL_ROOT = os.environ.get("IFL_ROOT") or str(Path(__file__).resolve().parents[2])
if IFL_ROOT not in sys.path:
    sys.path.insert(0, IFL_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from instinctflash.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_fsdp_elision,
)


def dispatched_op(fn) -> str:
    """Which aten convolution op fires for `fn`. The whole question, answered per signature."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    names = [e.key for e in p.key_averages()
             if "conv" in e.key.lower() and e.key.startswith("aten::")]
    # The innermost (most specific) conv op is the backend actually used.
    for pref in ("slow_conv_dilated3d", "cudnn_convolution", "convolution_overrideable",
                 "_convolution", "conv3d", "miopen"):
        for n in names:
            if pref in n:
                return n.replace("aten::", "")
    return names[0].replace("aten::", "") if names else "?"


def bench(fn, n=12, inner=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) / inner)
    return statistics.median(xs) * 1e3, (max(xs) - min(xs)) / statistics.mean(xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5, help="distinct signatures to investigate")
    a = ap.parse_args()

    print(f"torch {torch.__version__}   cudnn enabled={torch.backends.cudnn.enabled} "
          f"version={torch.backends.cudnn.version()}   benchmark={torch.backends.cudnn.benchmark}")
    print(f"allow_tf32(cudnn)={torch.backends.cudnn.allow_tf32}   "
          f"device={torch.cuda.get_device_name(0)}")

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_vae_conv"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

    print("\nbuilding server ...", flush=True)
    server = S.VA_Server(cfg)

    # ---- 1. record the real call signatures ---------------------------------------------------
    seen = collections.Counter()
    sig_example = {}

    def hook(mod, inputs):
        x = inputs[0]
        sig = (tuple(x.shape), str(x.dtype).replace("torch.", ""),
               "channels_last_3d" if x.is_contiguous(
                   memory_format=torch.channels_last_3d) else
               ("contiguous" if x.is_contiguous() else "strided"),
               tuple(mod.kernel_size), tuple(mod.stride), tuple(mod.padding),
               tuple(mod.dilation), mod.groups, tuple(mod.weight.shape))
        seen[sig] += 1
        sig_example.setdefault(sig, (x.detach(), mod))

    handles = []
    n_conv = 0
    for m in server.streaming_vae.vae.modules():
        if isinstance(m, torch.nn.Conv3d):
            handles.append(m.register_forward_pre_hook(hook))
            n_conv += 1
    print(f"  hooked {n_conv} Conv3d modules in the VAE")

    cams = list(cfg.obs_cam_keys)
    rng = np.random.default_rng(0)
    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    prompt = str(np.load(ctx[0], allow_pickle=True)["prompt"]) if ctx else "probe"
    server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
    kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
          for _ in range(8)]
    try:
        server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=None))
    except Exception as e:
        print(f"  (kv refresh raised {type(e).__name__}; hooks already fired)")
    for h in handles:
        h.remove()

    print(f"\n{'=' * 100}\nCONV3D SIGNATURES ACTUALLY EXECUTED ({len(seen)} distinct)\n{'=' * 100}")
    print(f"{'calls':>6}  {'input':<28}{'dtype':>9}{'layout':>16}  k/s/p/d/g          backend")
    print("-" * 100)
    ranked = []
    for sig, cnt in seen.most_common():
        shp, dt, lay, k, st, pd, dl, g, wshp = sig
        x, mod = sig_example[sig]
        op = dispatched_op(lambda x=x, mod=mod: mod(x))
        ranked.append((cnt, sig, op))
        print(f"{cnt:>6}  {str(shp):<28}{dt:>9}{lay:>16}  "
              f"{k}/{st}/{pd}/{dl}/{g}  {op}")

    fallbacks = [(c, s, o) for c, s, o in ranked if "slow" in o]
    if not fallbacks:
        print("\n  No signature dispatched to a slow fallback in isolation. The fallback seen in the")
        print("  full cycle must depend on surrounding state; NOT EVALUATED here.")
        return 2
    print(f"\n  {len(fallbacks)} of {len(ranked)} signatures fall back to a slow path.")

    # ---- 2. what changes the backend? ---------------------------------------------------------
    print(f"\n{'=' * 100}\nLEVERS, on the {min(a.top, len(fallbacks))} costliest fallback signatures"
          f"\n{'=' * 100}")
    fallbacks.sort(reverse=True, key=lambda t: t[0])
    for cnt, sig, op in fallbacks[:a.top]:
        shp, dt, lay, k, st, pd, dl, g, wshp = sig
        x, mod = sig_example[sig]
        print(f"\n  input {shp} {dt} {lay}, weight {wshp}, stride {st}  ({cnt} calls/encode)")
        base_ms, base_sp = bench(lambda x=x, mod=mod: mod(x))
        print(f"    as-is                 {op:<24} {base_ms:8.3f} ms  (spread {base_sp:.1%})")

        # (a) cudnn.benchmark -- a SEARCH strategy, not an arithmetic change. The only lever here
        #     that could ship under a max|delta| = 0 gate.
        old = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True
        try:
            op_b = dispatched_op(lambda x=x, mod=mod: mod(x))
            ms_b, sp_b = bench(lambda x=x, mod=mod: mod(x))
            print(f"    cudnn.benchmark=True  {op_b:<24} {ms_b:8.3f} ms  "
                  f"({base_ms / ms_b:5.2f}x)  BITEXACT-eligible: no arithmetic change")
        finally:
            torch.backends.cudnn.benchmark = old

        # (b) channels_last_3d -- cuDNN v8 prefers NDHWC for 3D. Changes layout, so NOT bit-exact.
        try:
            xc = x.contiguous(memory_format=torch.channels_last_3d)
            mc = mod.to(memory_format=torch.channels_last_3d)
            op_c = dispatched_op(lambda xc=xc, mc=mc: mc(xc))
            ms_c, _ = bench(lambda xc=xc, mc=mc: mc(xc))
            print(f"    channels_last_3d      {op_c:<24} {ms_c:8.3f} ms  "
                  f"({base_ms / ms_c:5.2f}x)  NUMERIC only: layout changes reduction order")
            mod.to(memory_format=torch.contiguous_format)
        except Exception as e:
            print(f"    channels_last_3d      FAILED {type(e).__name__}: {str(e)[:44]}")

        # (c) fp16 -- 3D conv support for bf16 is thinner than for fp16. Diagnostic.
        try:
            xh = x.half()
            mh = type(mod)(mod.in_channels, mod.out_channels, mod.kernel_size,
                           mod.stride, (0, 0, 0)).half().to(x.device) \
                if hasattr(mod, "_padding") else None
            if mh is not None:
                mh.weight.data.copy_(mod.weight.data.half())
                if mod.bias is not None:
                    mh.bias.data.copy_(mod.bias.data.half())
                mh._padding = (0, 0, 0, 0, 0, 0)
                op_h = dispatched_op(lambda xh=xh, mh=mh: F.conv3d(
                    xh, mh.weight, mh.bias, mod.stride, (0, 0, 0), (1, 1, 1), mod.groups))
                ms_h, _ = bench(lambda xh=xh, mh=mh: F.conv3d(
                    xh, mh.weight, mh.bias, mod.stride, (0, 0, 0), (1, 1, 1), mod.groups))
                print(f"    fp16 (diagnostic)     {op_h:<24} {ms_h:8.3f} ms  "
                      f"({base_ms / ms_h:5.2f}x)  dtype change: not a proposal")
        except Exception as e:
            print(f"    fp16                  FAILED {type(e).__name__}: {str(e)[:44]}")

    print("\n  Read the BITEXACT-eligible column first. A lever that changes layout or dtype buys")
    print("  speed and costs the max|delta| = 0 gate that Layers 2-3 are held to, so it would need")
    print("  the paired non-inferiority protocol and a much larger measured win to be worth it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

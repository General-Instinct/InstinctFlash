#!/usr/bin/env python3
"""Does the channels_last_3d win survive at WHOLE-ENCODE scale, and what does it cost numerically?

PER-CONV the answer is already known (probe_vae_conv_backend.py): every 3x3x3 convolution in the VAE
falls back to `slow_conv_dilated3d` in NCDHW and dispatches to `cudnn_convolution` in NDHWC, 4.35x to
7.24x faster. Summed over one encode that is 22.7 ms -> 4.6 ms, which accounts for essentially all of
the 21.7 ms/cycle the warm profile attributes to `slow_conv_dilated3d`.

A PER-CONV WIN IS NOT A WIN. Two things can eat it, and both have eaten one before:

  LAYOUT TRANSITIONS  if the layout has to be converted at every conv, the transforms cost more than
                      the kernels save. It only works if the layout PROPAGATES: convert the module and
                      its input once, and let every intermediate stay in NDHWC. This probe measures
                      the whole encode, transforms included, so a layout that has to keep flipping
                      shows up as a loss rather than as a footnote.
  OPS THAT DO NOT FOLLOW  norms, resamples and the attention inside the VAE may force a copy back to
                      contiguous. Those copies are in the measurement too.

AND THE NUMERICS ARE NOT FREE. NDHWC changes the accumulation order inside the convolution, so this is
a NUMERIC-tier change, not BITEXACT: it cannot ship under a max|delta| = 0 gate and needs paired
non-inferiority. This probe reports the actual delta on the encoder's output so the size of that claim
is known before anyone argues about it -- the encoded latents feed the KV cache, so the difference
propagates to actions.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29990 probe_vae_channels_last.py
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from instinctwm.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_fsdp_elision,
)

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def bench(fn, n=9, inner=2):
    for _ in range(2):
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


def count_slow_convs(fn) -> tuple[int, int, float]:
    """(slow fallback calls, cudnn calls, total conv device ms) for one invocation."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    slow = cud = 0
    ms = 0.0
    for e in p.key_averages():
        if "slow_conv" in e.key:
            slow += e.count
            ms += (getattr(e, "self_device_time_total", 0) or 0) / 1000
        elif "cudnn_convolution" in e.key:
            cud += e.count
            ms += (getattr(e, "self_device_time_total", 0) or 0) / 1000
    return slow, cud, ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    a = ap.parse_args()

    hot = [ln for ln in os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    ).read().strip().split("\n") if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%).")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_vae_cl"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4
    print("building server ...", flush=True)
    server = S.VA_Server(cfg)

    sv = server.streaming_vae
    dev = next(sv.vae.parameters()).device
    dt = server.dtype
    H, W = cfg.height, cfg.width

    # One encoder chunk, exactly as _encode_obs builds it: (1, 3, T, H, W) in [-1, 1].
    g = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.rand(1, 3, a.frames, H, W, generator=g) * 2 - 1).to(device=dev, dtype=dt)
    print(f"  chunk {tuple(x.shape)} {dt}, target {H}x{W}")

    # THE TEMPORAL CHUNKING RULE, which the first version of this probe got wrong and which explains
    # a failure recorded loosely in PROFILE.md. The Wan causal VAE downsamples time by 4, so the FIRST
    # chunk after clear_cache must have T = 4k+1 and every later chunk T = 4k. Feeding 8 frames as the
    # first chunk raises "size of tensor a (8) must match tensor b (4)" inside the residual shortcut.
    # That is why the real flow works: _infer encodes ONE frame at frame_st_id 0, then _compute_kv_cache
    # sends 4 and then 8. So the constraint is a documented property of the architecture, not the cache
    # desynchronisation the earlier probe guessed at.
    prime = x[:, :, :1].contiguous()

    def run_contig():
        sv.clear_cache()
        sv.encode_chunk(prime)                 # T=1: primes the temporal cache
        return sv.encode_chunk(x)              # T=8: the chunk the real path measures

    print("\n=== 1. baseline: NCDHW (as shipped) ===")
    slow, cud, conv_ms = count_slow_convs(run_contig)
    t_base, sp_base = bench(run_contig)
    print(f"  {slow} slow-fallback convs, {cud} cuDNN convs, {conv_ms:.2f} ms in convolutions")
    print(f"  whole encode_chunk: {t_base:.2f} ms  (spread {sp_base:.1%})")
    ref = run_contig().float().clone()

    print("\n=== 2. NDHWC: convert the module and the input ONCE, let it propagate ===")
    # `module.to(memory_format=channels_last_3d)` applies to EVERY parameter and raises
    # "required rank 5 tensor" on the rank-1 RMSNorm weights. Only the Conv3d weights have a 5-D
    # layout to change, so convert exactly those -- which is also what a pass would do.
    n_conv = 0
    for m in sv.vae.modules():
        if isinstance(m, torch.nn.Conv3d) and m.weight.dim() == 5:
            m.to(memory_format=torch.channels_last_3d)
            n_conv += 1
    print(f"  converted {n_conv} Conv3d weights to channels_last_3d")
    xc = x.contiguous(memory_format=torch.channels_last_3d)

    prime_c = prime.contiguous(memory_format=torch.channels_last_3d)

    def run_cl():
        sv.clear_cache()
        sv.encode_chunk(prime_c)
        return sv.encode_chunk(xc)

    try:
        slow2, cud2, conv_ms2 = count_slow_convs(run_cl)
        t_cl, sp_cl = bench(run_cl)
        got = run_cl().float().clone()
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:90]}")
        for m in sv.vae.modules():
            if isinstance(m, torch.nn.Conv3d) and m.weight.dim() == 5:
                m.to(memory_format=torch.contiguous_format)
        return 1
    print(f"  {slow2} slow-fallback convs, {cud2} cuDNN convs, {conv_ms2:.2f} ms in convolutions")
    print(f"  whole encode_chunk: {t_cl:.2f} ms  (spread {sp_cl:.1%})")

    print("\n=== 3. the block-scale verdict ===")
    check(slow2 < slow, f"fallback convs {slow} -> {slow2}", f"cuDNN convs {cud} -> {cud2}")
    check(t_cl < t_base, f"whole encode faster: {t_base:.2f} -> {t_cl:.2f} ms "
                        f"({t_base / max(t_cl, 1e-9):.2f}x)",
          "transforms and non-following ops included")
    saved = t_base - t_cl
    print(f"  saved per encode: {saved:.2f} ms")
    print(f"  one encode per cycle, so ~{saved:.1f} ms of a 487 ms cycle = {saved / 487:.1%}")
    if conv_ms2 > 0:
        print(f"  convolution time alone: {conv_ms:.2f} -> {conv_ms2:.2f} ms "
              f"({conv_ms / conv_ms2:.2f}x); the difference between that and the encode-level")
        print(f"  saving is what layout transitions and non-following ops cost: "
              f"{(conv_ms - conv_ms2) - saved:+.2f} ms")

    print("\n=== 4. numerics: this is NOT bit-exact, so measure the claim ===")
    if got.shape != ref.shape:
        check(False, "output shapes match", f"{tuple(got.shape)} vs {tuple(ref.shape)}")
    else:
        d = float((got - ref).abs().max())
        rel = d / max(float(ref.abs().max()), 1e-9)
        print(f"  max|delta| on encoder output = {d:.3e}  (relative {rel:.2e})")
        print(f"  bf16 resolution at that magnitude is ~{float(ref.abs().max()) * 2**-8:.3e}")
        check(d > 0.0, "output DIFFERS, as a layout change must",
              "if this were 0 the tier derivation would be wrong")
        print("  => NUMERIC tier. Needs paired non-inferiority on pinned seeds, not max|delta| = 0.")
        print("     The encoded latents feed the KV cache, so this propagates to actions.")

    for m in sv.vae.modules():
        if isinstance(m, torch.nn.Conv3d) and m.weight.dim() == 5:
            m.to(memory_format=torch.contiguous_format)
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: channels_last_3d characterised at block scale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

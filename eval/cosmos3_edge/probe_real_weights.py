#!/usr/bin/env python3
"""Profile the SHIPPED Cosmos3-Edge transformer, with its real weights.

This probe loads a real checkpoint and measures its MoT trunk. Random-weight structural
probes live in probe_mot_stack.py and profile_stack.py; neither measures task accuracy.

Use a pinned diffusers build compatible with the checkpoint's declared architecture.
Record that build's commit with the measurement. A clean load is required; this probe
refuses missing, unexpected, or mismatched weights rather than timing a partial load.

Drives the decoder stack directly rather than Cosmos3OmniTransformer.forward, which needs ~20
structured arguments (vision/sound/action tokens, indexes, timesteps). The layer contract is
small and is the whole trunk cost:

    decoder_layer(und_seq, gen_seq, rotary_emb) -> (und_seq, gen_seq)
    rotary_emb = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])   # transformer_cosmos3.py:792

NOT MEASURED HERE: the vision encoder, the VAE, and the action heads. This is the MoT trunk only,
not an end-to-end policy latency or accuracy measurement.

Run: python eval/cosmos3_edge/probe_real_weights.py --ckpt /path/to/transformer
"""
from __future__ import annotations

import argparse
import collections
import math
import time

import torch

UND, GEN = 111, 456          # the served pack, fixed synthetic action-policy pack
NFE = 16                     # forwards per control step

# Same buckets as profile_stack.py. First match wins.
BUCKETS = (
    ("attention", ("sdpa", "flash", "fmha", "attention")),
    ("GEMM", ("gemm", "cutlass", "sm80_xmma", "s16816", "matmul", "addmm")),
    ("reduce", ("reduce", "norm_kernel", "softmax")),
    ("scatter/idx", ("scatter", "gather", "index", "nonzero")),
    ("copy/cast", ("copy", "cast", "convert", "cat", "contiguous", "clone")),
    ("elementwise", ("elementwise", "vectorized", "unrolled", "mul", "add")),
)


def bucket_of(name: str) -> str:
    low = name.lower()
    for label, keys in BUCKETS:
        if any(k in low for k in keys):
            return label
    return "other"


def validate_loading_info(info):
    """Refuse performance claims on partially initialized checkpoints."""
    problems = {key: info[key] for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if info.get(key)}
    if problems:
        raise RuntimeError(f"Checkpoint load is not clean; refusing to benchmark: {problems}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Local transformer checkpoint or HF repository")
    ap.add_argument("--revision", help="Pinned HF checkpoint revision")
    ap.add_argument("--bandwidth-gbps", type=float,
                    help="Optional device datasheet bandwidth for a theoretical floor")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    if args.iters < 1 or args.warmup < 0 or args.top < 1:
        ap.error("iters and top must be positive; warmup must be nonnegative")
    if args.bandwidth_gbps is not None and (
            not math.isfinite(args.bandwidth_gbps) or args.bandwidth_gbps <= 0):
        ap.error("bandwidth-gbps must be finite and positive")

    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 2

    from diffusers import Cosmos3OmniTransformer

    dev, dt = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(0)

    model, info = Cosmos3OmniTransformer.from_pretrained(
        args.ckpt, revision=args.revision, torch_dtype=dt, low_cpu_mem_usage=True,
        output_loading_info=True)
    validate_loading_info(info)
    missing, unexpected = info.get("missing_keys", []), info.get("unexpected_keys", [])
    model = model.to(dev).eval().requires_grad_(False)

    n_par = sum(p.numel() for p in model.parameters())
    bytes_par = sum(p.numel() * p.element_size() for p in model.parameters())
    layers = model.layers
    cfg = model.config
    hidden = cfg.hidden_size

    print("=" * 92)
    print("Cosmos3-Edge transformer -- SHIPPED WEIGHTS")
    print("=" * 92)
    import diffusers
    print(f"  checkpoint {args.ckpt} revision={args.revision or 'local/unpinned'}")
    print(f"  torch {torch.__version__}  diffusers {diffusers.__version__}  "
          f"{torch.cuda.get_device_name(0)}")
    print(f"  load: missing={len(missing)} unexpected={len(unexpected)}"
          f"   {'CLEAN' if not missing and not unexpected else 'MISMATCH -- do not quote'}")
    print(f"  {len(layers)} layers, hidden {hidden}, hidden_act {cfg.hidden_act}, "
          f"pack {UND} und + {GEN} gen = {UND + GEN} tokens, NFE {NFE}")
    print(f"  params {n_par / 1e9:.3f} B   weights {bytes_par / 2**30:.2f} GiB bf16")

    # Inputs. rotary_emb is the 4-tuple the transformer builds at transformer_cosmos3.py:792.
    und_seq = torch.randn(UND, hidden, device=dev, dtype=dt)
    gen_seq = torch.randn(GEN, hidden, device=dev, dtype=dt)
    pos = torch.arange(UND + GEN, device=dev)[None, :].expand(3, 1, -1)
    cos, sin = model.rotary_emb(pos, dev, dt)
    if cos.ndim == 3:
        cos, sin = cos[0], sin[0]
    rot = (cos[:UND], sin[:UND], cos[UND:], sin[UND:])

    def one_forward():
        u, g = und_seq, gen_seq
        for layer in layers:
            u, g = layer(u, g, rot)
        return u, g

    with torch.no_grad():
        for _ in range(args.warmup):
            one_forward()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.iters):
            one_forward()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / args.iters * 1000

    print(f"\n--- floors (datasheet, not achieved) ----------------------------------------")
    if args.bandwidth_gbps is not None:
        mem_floor = bytes_par / (args.bandwidth_gbps * 1e9) * 1000
        print(f"  memory floor : {mem_floor:6.2f} ms @ {args.bandwidth_gbps:g} GB/s")
    print(f"  measured     : {wall:6.2f} ms   eager, no capture")

    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            one_forward()
        torch.cuda.synchronize()

    by_ms, by_n = collections.Counter(), collections.Counter()
    per_kernel = collections.Counter()
    per_kernel_n = collections.Counter()
    for e in prof.key_averages():
        if not e.self_device_time_total:
            continue
        by_ms[bucket_of(e.key)] += e.self_device_time_total
        by_n[bucket_of(e.key)] += e.count
        per_kernel[e.key] += e.self_device_time_total
        per_kernel_n[e.key] += e.count

    total_us = sum(by_ms.values())
    if total_us <= 0:
        raise RuntimeError("Profiler returned no CUDA kernel timings")
    print(f"\n--- GPU kernel time, eager ({args.iters} forwards) -------------------")
    print(f"  summed kernel self-time {total_us / args.iters / 1000:.2f} ms/forward "
          f"vs {wall:.2f} ms wall")
    print(f"\n  {'bucket':14s}{'ms/fwd':>9s}{'% GPU':>8s}{'kernels/fwd':>13s}{'x NFE 16':>11s}")
    for b, us in by_ms.most_common():
        ms = us / args.iters / 1000
        print(f"  {b:14s}{ms:9.2f}{100 * us / total_us:7.1f}%"
              f"{by_n[b] / args.iters:13.0f}{ms * NFE:10.1f} ms")

    print(f"\n  top {args.top} kernels")
    print(f"  {'ms/fwd':>8s}{'% GPU':>8s}{'calls/fwd':>11s}  kernel")
    for k, us in per_kernel.most_common(args.top):
        print(f"  {us / args.iters / 1000:8.3f}{100 * us / total_us:7.1f}%"
              f"{per_kernel_n[k] / args.iters:11.1f}  {k[:70]}")

    n_k = sum(by_n.values()) / args.iters
    print(f"\n  {n_k:.0f} kernels per forward over {len(layers)} layers = "
          f"{n_k / len(layers):.1f} per layer; {n_k * NFE:.0f} per control step")
    print(f"  mean kernel duration {total_us / sum(by_n.values()):.1f} us")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

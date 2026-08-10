#!/usr/bin/env python3
"""Profile the SHIPPED Cosmos3-Edge transformer, with its real weights.

Sections 3, 6 and 7 of RESULTS.md all profile a stack this repo BUILDS. This one loads the
published checkpoint instead, so three things stop being assumptions:

  1. the weights load with zero missing and zero unexpected keys, i.e. the structure under test
     is the structure that shipped;
  2. the floors are computed from the real parameter count rather than a proxy's;
  3. the standing claim that "weights do not change latency, only shapes do" gets a number
     instead of an assertion -- compare the buckets here against RESULTS.md section 7.

WHY IT NEEDS ITS OWN ENVIRONMENT
The checkpoint declares _diffusers_version 0.40.0.dev0 and PyPI's newest release is 0.39.0, whose
Cosmos3OmniTransformer is a different network: SwiGLU with a gate_proj, and norm_q/norm_k instead
of k_norm_und_for_gen. Loading the real weights into 0.39.0 newly-initialises 112 tensors and
leaves 28 unused, and diffusers itself prints "You should probably TRAIN this model". So this runs
against diffusers built from git main in $IWM_ROOT/.venv-cosmos-real, which is an UNRELEASED
version -- pin the commit before quoting anything from here as reproducible.

Drives the decoder stack directly rather than Cosmos3OmniTransformer.forward, which needs ~20
structured arguments (vision/sound/action tokens, indexes, timesteps). The layer contract is
small and is the whole trunk cost:

    decoder_layer(und_seq, gen_seq, rotary_emb) -> (und_seq, gen_seq)
    rotary_emb = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])   # transformer_cosmos3.py:792

NOT MEASURED HERE: the vision encoder, the VAE, and the action heads. This is the MoT trunk only,
which is what sections 3/6/7 measure too, so the columns are comparable.

Run:  $IWM_ROOT/.venv-cosmos-real/bin/python eval/cosmos3_edge/probe_real_weights.py
"""
from __future__ import annotations

import argparse
import collections
import time

import torch

BASE = "/home/ubuntu/Cosmos3-Edge/transformer"
DROID = ("/home/ubuntu/.cache/huggingface/hub/models--nvidia--Cosmos3-Edge-Policy-DROID"
         "/snapshots/3ea407af3e156c0af3b4bb6edd85842cc9a58777/transformer")

UND, GEN = 111, 456          # the served pack, same as RESULTS.md sections 3/6/7
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=BASE, choices=(BASE, DROID))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0

    from diffusers import Cosmos3OmniTransformer

    dev, dt = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(0)

    model, info = Cosmos3OmniTransformer.from_pretrained(
        args.ckpt, torch_dtype=dt, low_cpu_mem_usage=True, output_loading_info=True)
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
    print(f"  {args.ckpt}")
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

    mem_floor = bytes_par / 2039e9 * 1000
    print(f"\n--- floors (datasheet, not achieved) ----------------------------------------")
    print(f"  memory floor : {mem_floor:6.2f} ms   ({bytes_par / 2**30:.2f} GiB @ 2039 GB/s)")
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

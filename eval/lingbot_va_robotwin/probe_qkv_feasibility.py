#!/usr/bin/env python3
"""Feasibility of a fused QKV projection. FIVE QUESTIONS, NO IMPLEMENTATION.

Candidate 3 (LAYER5_NEXT.md): `to_q(q), to_k(k), to_v(v)` at ring_kv.py:146 is 900 addmm calls/cycle
and 12.01 ms. If q, k and v are the same tensor, three GEMMs against a shared input can become one
against a concatenated weight -- 900 -> 300 calls. In a cycle that is 42% GPU-idle, the launch
reduction may matter more than the GEMM time.

Everything about that is conditional, so this answers the conditions before anything is built:

  1. WHICH SITES SHARE AN INPUT. Self-attention has q is k is v; cross-attention does not. Fusing the
     latter is not an optimization, it is a bug. Measured per call, by object identity.
  2. ARE THE WEIGHT SHAPES COMPATIBLE. Concatenation needs matching in_features and dtype, and the
     bias structure has to agree.
  3. DOES IT ACTUALLY REDUCE LAUNCHES 900 -> 300. Predicted from the measured share of fusible calls,
     not assumed.
  4. NUMERICAL DELTA from the changed GEMM shape. K is unchanged at 3072 but N goes 3072 -> 9216, which
     can change cuBLAS tile and split-K selection and therefore the accumulation order. Measured
     against the three separate GEMMs on real weights.
  5. CYCLE-LEVEL GAIN, estimated from the measured region delta before committing to a NUMERIC-tier
     implementation with a 555-episode certification attached.

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29995 probe_qkv_feasibility.py
"""
from __future__ import annotations

import argparse
import collections
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
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def bench(fn, n=30, inner=20):
    for _ in range(5):
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
    return statistics.median(xs) * 1e6, (max(xs) - min(xs)) / statistics.mean(xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=40)
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_qkv"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4
    print("building server ...", flush=True)
    server = S.VA_Server(cfg)
    from instinctwm.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    for n in install_conditioning_prefill(S, type(server)):
        pass
    for n in install_debug_dump_elision(S):
        pass
    from instinctwm.backends.conv.apply import install_conv_layout
    for line in install_conv_layout(server):
        pass

    # ---- 1. which attention calls share their Q/K/V input? -----------------------------------
    shared = collections.Counter()
    shapes = collections.Counter()
    probe = {"on": False}
    Attn = type(server.transformer.blocks[0].attn1)
    orig = Attn.forward

    def counting(self, q, k, v, *args, **kw):
        if probe["on"]:
            # k or v can be None: the conditioning_prefill pass (P002) serves cross-attention from a
            # cached K/V, so it passes None rather than a tensor. Those calls have NOTHING to fuse --
            # there is no second or third GEMM to merge -- which is a distinct category from
            # cross-attention with real distinct inputs, and the first version of this probe crashed
            # on them rather than counting them.
            if k is None or v is None:
                shared["cross with cached KV (no projections to fuse)"] += 1
            elif (q is k) and (k is v):
                shared["self (q is k is v) -- FUSIBLE"] += 1
                shapes[(tuple(q.shape),)] += 1
            elif (q is k) or (k is v) or (q is v):
                shared["partial alias"] += 1
            else:
                shared["cross, all distinct -- must stay split"] += 1
        return orig(self, q, k, v, *args, **kw)
    Attn.forward = counting

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    if not ctx:
        raise SystemExit("no contexts")
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = {"obs": [{full: z[s] for s, full in short.items()}], "state": z["state"]}
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)
    rng = np.random.default_rng(0)
    first = {"v": True}

    def cycle():
        if first["v"]:
            server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        act = server.infer(dict(obs=obs["obs"], prompt=prompt,
                                save_visualization=False))["action"]
        kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
              for _ in range(4 if first["v"] else 8)]
        server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=act))
        first["v"] = False

    print(f"warming {a.warm} cycles ...", flush=True)
    for _ in range(a.warm):
        cycle()
    probe["on"] = True
    cycle()
    probe["on"] = False
    Attn.forward = orig

    total = sum(shared.values())
    print(f"\n=== 1. input sharing across {total} attention calls in one warm cycle ===")
    for kind, n in shared.most_common():
        print(f"  {n:>5} ({n/total:5.1%})  {kind}")
    fusible = shared.get("self (q is k is v) -- FUSIBLE", 0)
    check(total > 0, "attention calls observed", f"{total}")
    check(fusible > 0, "some calls are self-attention and therefore fusible",
          f"{fusible} of {total} = {fusible/max(total,1):.0%}")
    for k_, msg in (("cross, all distinct -- must stay split",
                     "MUST keep the split path; fusing them would be a correctness bug"),
                    ("cross with cached KV (no projections to fuse)",
                     "have no k/v projection to fuse at all -- P002 already removed it")):
        if shared.get(k_, 0):
            print(f"  => {shared[k_]} calls {msg}")

    # ---- 2. are the weights concatenable? ----------------------------------------------------
    print("\n=== 2. weight compatibility ===")
    at = server.transformer.blocks[0].attn1
    wq, wk, wv = at.to_q.weight, at.to_k.weight, at.to_v.weight
    print(f"  to_q {tuple(wq.shape)} {wq.dtype}   to_k {tuple(wk.shape)}   to_v {tuple(wv.shape)}")
    check(wq.shape[1] == wk.shape[1] == wv.shape[1],
          "in_features match, so the weights concatenate along out_features",
          f"K = {wq.shape[1]}")
    check(wq.dtype == wk.dtype == wv.dtype, "dtypes match", str(wq.dtype))
    biases = [getattr(at, n).bias is not None for n in ("to_q", "to_k", "to_v")]
    check(len(set(biases)) == 1, "bias structure agrees across the three", f"bias present: {biases}")
    n_out = wq.shape[0] + wk.shape[0] + wv.shape[0]
    print(f"  fused GEMM would be N = {n_out} (from {wq.shape[0]} x3), K = {wq.shape[1]}")

    # ---- 3 + 4. launches and numerics, on the real weights ------------------------------------
    print("\n=== 3. launch reduction, and 4. numerical delta ===")
    dev, dt = wq.device, wq.dtype
    fused_w = torch.cat([wq, wk, wv], dim=0).contiguous()
    fused_b = (torch.cat([at.to_q.bias, at.to_k.bias, at.to_v.bias], dim=0).contiguous()
               if biases[0] else None)
    for tokens in (32, 240):
        x = torch.randn(2, tokens, wq.shape[1], device=dev, dtype=dt)

        def split():
            return (torch.nn.functional.linear(x, wq, at.to_q.bias),
                    torch.nn.functional.linear(x, wk, at.to_k.bias),
                    torch.nn.functional.linear(x, wv, at.to_v.bias))

        def fused():
            y = torch.nn.functional.linear(x, fused_w, fused_b)
            return y.split([wq.shape[0], wk.shape[0], wv.shape[0]], dim=-1)

        t_s, sp_s = bench(split)
        t_f, sp_f = bench(fused)
        qs, ks, vs = split()
        qf, kf_, vf = fused()
        d = max(float((qs - qf).abs().max()), float((ks - kf_).abs().max()),
                float((vs - vf).abs().max()))
        nbits = sum(int((a_.view(torch.int16) != b_.view(torch.int16)).sum())
                    for a_, b_ in ((qs, qf), (ks, kf_), (vs, vf)))
        ntot = qs.numel() * 3
        print(f"  tokens={tokens:>4}  split {t_s:7.1f} us   fused {t_f:7.1f} us "
              f"({t_s/max(t_f,1e-9):5.2f}x)   max|delta| {d:.3e}   "
              f"{nbits}/{ntot} words differ ({nbits/ntot:.2e})")
        if tokens == 240:
            check(t_f < t_s, f"fused is faster at {tokens} tokens", f"{t_s/max(t_f,1e-9):.2f}x")
            if nbits == 0:
                print("       => BIT-EXACT at this shape. Not claimable in general (cuBLAS may pick a")
                print("          different algorithm under other shapes), but it means the NUMERIC")
                print("          tier may be avoidable -- worth checking across every served shape.")
            else:
                print(f"       => NOT bit-exact: {nbits/ntot:.2e} of words differ. NUMERIC tier, so a")
                print("          paired certification is required before shipping.")

    # ---- 5. cycle-level estimate --------------------------------------------------------------
    print("\n=== 5. cycle-level estimate ===")
    x240 = torch.randn(2, 240, wq.shape[1], device=dev, dtype=dt)
    x32 = torch.randn(2, 32, wq.shape[1], device=dev, dtype=dt)

    def mk(x, f):
        return lambda: f(x)

    def split_of(x):
        return (torch.nn.functional.linear(x, wq, at.to_q.bias),
                torch.nn.functional.linear(x, wk, at.to_k.bias),
                torch.nn.functional.linear(x, wv, at.to_v.bias))

    def fused_of(x):
        return torch.nn.functional.linear(x, fused_w, fused_b)

    saved_us = 0.0
    # 30 blocks; video forwards see 240 tokens, action forwards 32. 3 video + 5 action + 2 kv = 10.
    for x, n_fwd, label in ((x240, 3, "video 240-token"), (x32, 7, "action/kv 32-token")):
        ts, _ = bench(mk(x, split_of))
        tf, _ = bench(mk(x, fused_of))
        per_cycle = (ts - tf) * 30 * n_fwd
        saved_us += per_cycle
        print(f"  {label:<20} split {ts:7.1f} us  fused {tf:7.1f} us  x30 blocks x{n_fwd} forwards "
              f"-> {per_cycle/1000:+6.2f} ms/cycle")
    print(f"\n  estimated saving: {saved_us/1000:.2f} ms of a 330.2 ms cycle "
          f"= {saved_us/1000/330.2:.1%}")
    print(f"  launches: {fusible} fusible attention calls x 2 fewer GEMMs = "
          f"{fusible*2} fewer launches/cycle")
    print("\n  ESTIMATE, not a result. It excludes the split() view on the fused output and any")
    print("  change in how the KV cache is written, and it assumes every fusible call is converted.")
    print("  The cycle gate is what decides, and at this magnitude it must be ABBA-ordered.")

    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: fused-QKV feasibility characterised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

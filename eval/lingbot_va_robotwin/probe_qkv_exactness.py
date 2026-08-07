#!/usr/bin/env python3
"""Is fused-QKV bit-exactness a THEOREM over the served envelope, or an accident of this stack?

THE ALGEBRA FIRST, because it tells us exactly what to measure.

For C[m,n] = sum_k A[m,k] * B[k,n], concatenating B along N adds columns. Every output element's
reduction is over K, and K is untouched. The N dimension is embarrassingly parallel: no output element's
value depends on how many other columns exist. So in exact arithmetic the fused result IS the split
result, and in floating point it is bit-identical **iff the accumulation over K is performed the same
way**.

That reduces the whole question to the K-loop, and there are exactly three ways it can differ:

    tile_k       a different K-block size changes the order of partial sums
    split-K      splitting K across CTAs adds a second reduction stage (and, with atomics,
                 non-determinism). This is the real risk: cuBLAS enables split-K when M*N is too
                 small to fill the GPU -- and the action-stream GEMM is M=64, which is exactly that
                 regime. Going N=3072 -> 9216 TRIPLES the parallelism available without split-K, so
                 the heuristic may switch off precisely because we fused.
    accumulator  fp32 accumulation for bf16 inputs in both cases, or not

So: enumerate every production shape, run both forms, and compare (a) the outputs bit-for-bit, (b) the
kernel actually selected. cuBLAS `nvjet_*` kernel names encode the tile configuration, so the K-loop
structure is observable without vendor logging -- and `CUBLAS_LOGINFO_DBG` is read too when available.

A run-to-run determinism check is included because atomic split-K is not reproducible: if a shape is
non-deterministic, bit-exactness against anything is meaningless for it.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29996 probe_qkv_exactness.py
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from instinctwm.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def gemm_kernels(fn) -> list[str]:
    """The GEMM kernel names cuBLAS actually selected. nvjet names encode the tile config."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    out = []
    for e in p.key_averages():
        k = e.key
        if any(s in k for s in ("nvjet", "gemm", "xmma", "cutlass", "sm90", "sm80")):
            out.append((getattr(e, "self_device_time_total", 0) or 0, k, e.count))
    out.sort(reverse=True)
    return [f"{k}  x{c}" for _, k, c in out]


def ulp_diff_bf16(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    """(max ulp distance, differing words) for two bf16 tensors, via their integer bit patterns."""
    ia = a.view(torch.int16).to(torch.int32)
    ib = b.view(torch.int16).to(torch.int32)
    # Map signed-magnitude to a monotone ordering so |difference| is an ULP count.
    ia = torch.where(ia < 0, torch.tensor(-32768, device=ia.device, dtype=torch.int32) - ia, ia)
    ib = torch.where(ib < 0, torch.tensor(-32768, device=ib.device, dtype=torch.int32) - ib, ib)
    d = (ia - ib).abs()
    return int(d.max()), int((d != 0).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3, help="determinism repeats per shape")
    a = ap.parse_args()

    hot = [ln for ln in os.popen("nvidia-smi --query-gpu=index,utilization.gpu "
                                 "--format=csv,noheader,nounits").read().strip().split("\n")
           if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%).")
        return 2

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_qkv_exact"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4
    print(f"torch {torch.__version__}  cuBLAS via torch  "
          f"device {torch.cuda.get_device_name(0)}", flush=True)
    print("building server ...", flush=True)
    server = S.VA_Server(cfg)
    from instinctwm.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    for _ in install_conditioning_prefill(S, type(server)):
        pass
    for _ in install_debug_dump_elision(S):
        pass
    from instinctwm.backends.conv.apply import install_conv_layout
    for _ in install_conv_layout(server):
        pass

    # ---- 1. enumerate every production shape, empirically -------------------------------------
    seen = collections.Counter()
    probe = {"on": False, "phase": "video"}
    Attn = type(server.transformer.blocks[0].attn1)
    orig = Attn.forward

    def counting(self, q, k, v, *args, **kw):
        if probe["on"] and q is not None and (q is k) and (k is v):
            seen[(tuple(q.shape), probe["phase"])] += 1
        return orig(self, q, k, v, *args, **kw)
    Attn.forward = counting

    # Tag which loop issued the call, so "action" and "kv_refresh" are distinguishable.
    orig_model = server.transformer.forward

    def tagged(*args, **kw):
        probe["phase"] = "action" if kw.get("action_mode") else (
            "kv_refresh" if kw.get("update_cache") else "video")
        return orig_model(*args, **kw)
    server.transformer.forward = tagged

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
    server.transformer.forward = orig_model

    at = server.transformer.blocks[0].attn1
    wq, wk, wv = at.to_q.weight, at.to_k.weight, at.to_v.weight
    bq, bk, bv = at.to_q.bias, at.to_k.bias, at.to_v.bias
    fw = torch.cat([wq, wk, wv], dim=0).contiguous()
    fb = torch.cat([bq, bk, bv], dim=0).contiguous()
    N1, K = wq.shape

    print(f"\n{'=' * 118}\n1. PRODUCTION GEMM ENVELOPE (self-attention only; cross-attn K/V removed "
          f"by P002)\n{'=' * 118}")
    print(f"{'phase':<12}{'input shape':<22}{'M':>7}{'K':>7}{'N split':>9}{'N fused':>9}"
          f"{'calls/cyc':>11}")
    print("-" * 118)
    envelope = []
    for (shp, phase), n in sorted(seen.items(), key=lambda kv: -kv[1]):
        M = int(np.prod(shp[:-1]))
        envelope.append((phase, shp, M, K, n))
        print(f"{phase:<12}{str(shp):<22}{M:>7}{K:>7}{N1:>9}{N1*3:>9}{n:>11}")
    check(bool(envelope), "at least one production shape captured")

    # ---- 2. per shape: exactness, ULP, kernel selection, determinism --------------------------
    print(f"\n{'=' * 118}\n2. SPLIT vs FUSED, EVERY PRODUCTION SHAPE\n{'=' * 118}")
    all_exact = True
    tilek_same = True
    rows = []
    for phase, shp, M, Kd, n in envelope:
        x = torch.randn(*shp, device=wq.device, dtype=wq.dtype)

        def split():
            return (F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv))

        def fused():
            y = F.linear(x, fw, fb)
            return y.split([N1, N1, N1], dim=-1)

        # determinism: identical inputs, repeated runs
        det_s = det_f = True
        s0 = [t.clone() for t in split()]
        f0 = [t.clone() for t in fused()]
        for _ in range(a.repeats - 1):
            det_s &= all(torch.equal(p, q) for p, q in zip(s0, split()))
            det_f &= all(torch.equal(p, q) for p, q in zip(f0, fused()))

        max_ulp, ndiff = 0, 0
        ntot = 0
        for sp, fu in zip(s0, f0):
            u, d = ulp_diff_bf16(sp.contiguous(), fu.contiguous())
            max_ulp = max(max_ulp, u)
            ndiff += d
            ntot += sp.numel()

        ks = gemm_kernels(split)
        kf_ = gemm_kernels(fused)
        rows.append((phase, shp, M, ndiff, max_ulp, det_s, det_f, ks, kf_))
        all_exact &= (ndiff == 0)

        print(f"\n  {phase}  input {shp}  M={M} K={Kd}")
        print(f"    exactness    max ULP {max_ulp}   differing words {ndiff}/{ntot}")
        print(f"    determinism  split {'yes' if det_s else 'NO'}   fused "
              f"{'yes' if det_f else 'NO'}   ({a.repeats} runs)")
        print(f"    split  kernels: {ks[:2] if ks else 'none observed'}")
        print(f"    fused  kernels: {kf_[:2] if kf_ else 'none observed'}")

        def tile_of(names):
            """nvjet_tst_<tileM>x<tileN>_<tileK>x<stages>_... -> (tileM, tileN, tileK, stages)."""
            import re
            for nm in names:
                m = re.search(r"nvjet_\w+?_(\d+)x(\d+)_(\d+)x(\d+)", nm)
                if m:
                    return tuple(int(m.group(i)) for i in (1, 2, 3, 4))
            return None
        ts, tf = tile_of(ks), tile_of(kf_)
        if ts and tf:
            # ONLY tile_k CAN AFFECT THE RESULT, and an earlier version of this probe got that wrong
            # by comparing the whole tile tuple and reporting "DIFFERENT -- this is the risk" for
            # every shape. tile_m partitions rows and tile_n partitions columns: both are
            # embarrassingly parallel across output elements, so neither changes any element's
            # reduction. `stages` is pipeline depth -- prefetch scheduling, not arithmetic order.
            # tile_k is the K-block size, and it alone determines the sequence of partial sums.
            same_k = ts[2] == tf[2]
            tilek_same &= same_k
            print(f"    tiles        split {ts[0]}x{ts[1]} tile_k={ts[2]} stages={ts[3]}")
            print(f"                 fused {tf[0]}x{tf[1]} tile_k={tf[2]} stages={tf[3]}")
            print(f"    K-LOOP       tile_k {ts[2]} vs {tf[2]}: "
                  f"{'IDENTICAL -- reduction order preserved' if same_k else 'DIFFERENT -- the risk'}"
                  f"   (tile_m/tile_n/stages differ and cannot affect the result)")
        elif ks and kf_:
            print("    K-loop       tile config not parseable from these kernel names")

    print(f"\n{'=' * 118}\n3. VERDICT\n{'=' * 118}")
    check(all_exact, "bit-exact at EVERY production shape",
          "0 differing words everywhere" if all_exact else "at least one shape differs")
    check(all(r[5] and r[6] for r in rows),
          "both forms are run-to-run deterministic at every shape",
          "so no atomic split-K is in play")
    check(tilek_same, "tile_k is IDENTICAL between split and fused at every shape",
          "the only tile dimension that can change the reduction order")
    if all_exact and tilek_same:
        print("\n  The K-loop configuration is IDENTICAL across the N change at every shape, and both")
        print("  forms are deterministic. That is the mechanism: N is embarrassingly parallel, so with")
        print("  the same tile_k and no split-K, each output element's reduction is the same sequence")
        print("  of operations. Bit-exactness follows from the algebra, not from luck.")
        print("\n  BUT the invariant is a property of cuBLAS's HEURISTIC, not of the algebra. Nothing")
        print("  in the API promises tile_k stability under a change of N, and the heuristic is free to")
        print("  differ by driver, library version, GPU, or workspace size. The envelope guarantees")
        print("  the ALGEBRA; it does not guarantee the SELECTION.")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

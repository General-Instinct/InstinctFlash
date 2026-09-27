"""Triton BF16 GEMM + rounded ReLU^2 tile sweep (tiled and persistent) vs eager, back-to-back timing."""
import itertools
import os

import torch

from benchmarks.native.roofline import bench
from instinctflash.kernels.gemm_bf16 import bf16_gemm_act, bf16_gemm_act_persistent


def main():
    dev = "cuda"
    M, N, K = 3093, 9216, 2048
    torch.manual_seed(0)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    ref = torch.relu(torch.nn.functional.linear(x, w)).square()
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    t_eager = bench(lambda: torch.relu(torch.nn.functional.linear(x, w)).square(), 10, warm_ms=500)
    print(f"eager linear+relu+square: {t_eager * 1e3:8.1f} us", flush=True)
    rows = []
    only = os.environ.get("IFL_BF16_SWEEP", "all")
    for bm, bn, bk, gm, st, wp, ctas, ws in itertools.product((128, 256), (128, 256), (64, 128), (8,), (2, 3, 4),
                                                              (4, 8), (1, 2), (False, True)):
        if (bm == 256 and ctas == 1) or only == "persistent":
            continue
        cfg = (bm, bn, bk, gm, st, wp, ctas, ws)
        try:
            bf16_gemm_act(x, w, out, config=cfg)
            torch.cuda.synchronize()
        except Exception:
            continue
        exact = torch.equal(out, ref)
        diff = (out.float() - ref.float()).abs().max().item()
        t = bench(lambda: bf16_gemm_act(x, w, out, config=cfg), 10, warm_ms=100)
        rows.append((t * 1e3, cfg, exact, diff))
    for bm, bn, bk, st, wp, split, ws in itertools.product((128,), (128, 256), (64,), (2, 3, 4, 5), (4, 8),
                                                           (1, 2), (False, True)):
        cfg = (bm, bn, bk, 8, st, wp, split, ws)
        try:
            bf16_gemm_act_persistent(x, w, out, config=cfg)
            torch.cuda.synchronize()
        except Exception as e:
            print("persistent", cfg, "failed:", type(e).__name__, flush=True)
            continue
        t = bench(lambda: bf16_gemm_act_persistent(x, w, out, config=cfg), 10, warm_ms=100)
        rows.append((t * 1e3, ("persistent",) + cfg, torch.equal(out, ref), (out.float() - ref.float()).abs().max().item()))
    rows.sort()
    for r in rows[:6]:
        print(f"triton {r[0]:8.1f} us cfg={r[1]} exact={r[2]} maxdiff={r[3]:.3g}")


if __name__ == "__main__":
    main()

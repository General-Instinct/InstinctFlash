"""torch._scaled_mm vs direct cuBLASLt (tuned TN / NN) at engine shapes, rotating weights."""
import time

import torch

from benchmarks.native.bench_fp8_layout import ROT, SHAPES
from instinctflash.kernels.cublaslt import Fp8Gemm


def warm(seconds=1.0):
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        x @ x
    torch.cuda.synchronize()


def timed(fn, reps=10):
    from benchmarks.native.roofline import bench
    return bench(fn, reps, warm_ms=50) * 1e3


def main():
    dev = "cuda"
    one = torch.ones((), device=dev)
    warm()
    for m, n, k in SHAPES:
        rot = ROT if m < 100 else 2
        a = (torch.randn(m, k, device=dev) * 20).to(torch.float8_e4m3fn)
        ws = [(torch.randn(n, k, device=dev) * 20).to(torch.float8_e4m3fn) for _ in range(rot)]
        wkn = [w.t().contiguous() for w in ws]
        out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)

        def torch_mm():
            for w in ws:
                torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16, out=out)
        t_torch = timed(torch_mm) / rot
        res = [f"{m}x{n}x{k} torch {t_torch:7.2f}"]
        for layout, wl in (("tn", ws), ("nn", wkn)):
            try:
                g = Fp8Gemm(m, n, k, layout=layout)
                first = g.candidates[0]
                g.choice = first
                t_first = timed(lambda: [g(a, w, out) for w in wl]) / rot
                t_best = g.tune(a, wl, out) * 1e3
                res.append(f"{layout}: heur0 {t_first:7.2f} best {t_best:7.2f} ({len(g.candidates)} cands)")
            except Exception as e:
                res.append(f"{layout}: {str(e)[:60]}")
        print(" | ".join(res), flush=True)


if __name__ == "__main__":
    main()

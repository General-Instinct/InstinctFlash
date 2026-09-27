"""Decoder-shape attention: single-pass + row quant vs split-KV with fused FP8 combine."""
import itertools

import torch

from benchmarks.native.roofline import bench
from instinctflash.kernels.attention import SplitWorkspace, attention
from instinctflash.kernels.norm import row_quant


def main():
    dev = "cuda"
    A, G, HD, P = 50, 8, 256, 552
    q = torch.randn(1, A, G, HD, device=dev, dtype=torch.float16)
    k = torch.randn(1, P + A + 16, 1, HD, device=dev, dtype=torch.float16)
    v = torch.randn_like(k)
    n = torch.tensor([P + A], device=dev, dtype=torch.int32)
    o = torch.empty_like(q)
    q8 = torch.empty(A, G * HD, device=dev, dtype=torch.float8_e4m3fn)
    qs = torch.empty(A, device=dev)
    bench(lambda: attention(q, k, v, out=o, kv_len=n, config=(32, 128, 8, 2)), 20, warm_ms=1000)
    base = bench(lambda: (attention(q, k, v, out=o, kv_len=n, config=(32, 128, 8, 2)),
                          row_quant(o.view(A, -1), q8, qs, num_warps=1)), 50, warm_ms=100)
    print(f"single-pass + row_quant: {base * 1e3:.2f} us", flush=True)
    rows = []
    for tc5, splits, bm, bn, w, st in itertools.product((False, True), (2, 3, 4, 6), (16, 32, 64, 128), (32, 64, 128), (4, 8), (1, 2)):
        if (tc5 and bm < 64) or (not tc5 and bm > 64):
            continue
        ws = SplitWorkspace(1, A, G, 1, HD, splits, dev)
        f = lambda: attention(q, k, v, kv_len=n, split=ws, q8=q8, q8_scale=qs, config=(bm, bn, w, st), tc5=tc5)
        try:
            f()
            torch.cuda.synchronize()
            t = bench(f, 50, warm_ms=50)
        except Exception:
            continue
        rows.append((t * 1e3, (tc5, splits, bm, bn, w, st)))
    rows.sort()
    for r in rows[:8]:
        print(f"split fused: {r[0]:.2f} us cfg={r[1]}")


if __name__ == "__main__":
    main()

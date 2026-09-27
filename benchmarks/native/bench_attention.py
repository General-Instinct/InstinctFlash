"""Correctness + config sweep for instinctflash.kernels.attention at engine shapes."""
import argparse
import itertools
import json

import torch

from benchmarks.native.roofline import bench
from instinctflash.kernels.attention import attention

SHAPES = {
    # name: (B, Tq, Hq, Tk, Hkv, D, kv_len)
    "pi05_decoder": (1, 50, 8, 610, 1, 256, 602),
    "pi05_prefix": (1, 560, 8, 560, 1, 256, 552),
    "pi05_vision": (2, 256, 16, 256, 16, 72, 256),
}


def reference(q, k, v, kv_len, scale):
    g = q.shape[2] // k.shape[2]
    kk = k[:, :kv_len].float().repeat_interleave(g, 2)
    vv = v[:, :kv_len].float().repeat_interleave(g, 2)
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kk) * scale
    return torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), vv)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="*", default=list(SHAPES))
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()
    dt = getattr(torch, args.dtype)
    results = {}
    for name in args.shapes:
        B, Tq, Hq, Tk, Hkv, D, kv_len = SHAPES[name]
        torch.manual_seed(0)
        q = torch.randn(B, Tq, Hq, D, device="cuda", dtype=dt)
        k = torch.randn(B, Tk, Hkv, D, device="cuda", dtype=dt)
        v = torch.randn(B, Tk, Hkv, D, device="cuda", dtype=dt)
        scale = D ** -0.5
        ref = reference(q, k, v, kv_len, scale)
        n = torch.tensor([kv_len], device="cuda", dtype=torch.int32)
        best = None
        rows = []
        for tc5, bm, bn, w, s in itertools.product((False, True), (16, 32, 64, 128), (32, 64, 128), (4, 8), (1, 2, 3)):
            if tc5 and bm < 64:
                continue
            try:
                out = attention(q, k, v, kv_len=n, scale=scale, config=(bm, bn, w, s), tc5=tc5)
                torch.cuda.synchronize()
                err = (out.float() - ref).abs().max().item()
                if err > 2e-2:
                    rows.append((tc5, bm, bn, w, s, None, err))
                    continue
                ms = bench(lambda: attention(q, k, v, out=out, kv_len=n, scale=scale, config=(bm, bn, w, s), tc5=tc5), iters=50, warm_ms=60)
            except Exception as e:  # out of resources etc.
                rows.append((tc5, bm, bn, w, s, None, str(e)[:60]))
                continue
            rows.append((tc5, bm, bn, w, s, ms * 1e3, err))
            if best is None or ms < best[0]:
                best = (ms, (tc5, bm, bn, w, s), err)
        # torch SDPA baseline (expands KV heads)
        g = Hq // Hkv
        qt = q.transpose(1, 2)
        kt = k[:, :kv_len].repeat_interleave(g, 2).transpose(1, 2).contiguous()
        vt = v[:, :kv_len].repeat_interleave(g, 2).transpose(1, 2).contiguous()
        sdpa = bench(lambda: torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, scale=scale), iters=100)
        print(f"{name}: best {best[0]*1e3:.2f} us cfg={best[1]} err={best[2]:.2e} | torch sdpa {sdpa*1e3:.2f} us", flush=True)
        for r in sorted([r for r in rows if r[5] is not None], key=lambda r: r[5])[:5]:
            print("   ", r)
        bad = [r for r in rows if r[5] is None and isinstance(r[6], float)]
        if bad:
            print("    WRONG RESULTS:", bad[:3])
        results[name] = {"best_us": best[0] * 1e3, "config": best[1], "sdpa_us": sdpa * 1e3}
    print(json.dumps(results))


if __name__ == "__main__":
    main()

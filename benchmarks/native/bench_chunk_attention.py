"""Check and time the fused chunk attention against the unfused kernel sequence."""
import argparse
import itertools

import torch

from benchmarks.native.roofline import bench
from instinctflash.kernels.attention import ChunkAttention, attention
from instinctflash.kernels.norm import row_quant
from instinctflash.kernels.rope import qkv_split, rope_inv_freq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", type=int, default=552)
    p.add_argument("--chunk", type=int, default=50)
    args = p.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    A, G, HD, P = args.chunk, 8, 256, args.prefix
    S = P + A + 16
    raw = (torch.randn(A, (G + 2) * HD, device=dev) * 30).to(torch.bfloat16)
    rs = torch.rand(A, device=dev) * 0.02 + 0.01
    cs = torch.rand((G + 2) * HD, device=dev) * 0.02 + 0.01
    kc = torch.randn(S, 1, HD, device=dev, dtype=torch.float16)
    vc = torch.randn(S, 1, HD, device=dev, dtype=torch.float16)
    inv = rope_inv_freq(HD, 10000.0, dev)
    plen = torch.tensor([P], device=dev, dtype=torch.int32)
    klen = torch.tensor([P + A], device=dev, dtype=torch.int32)
    q16 = torch.empty(A, G, HD, device=dev, dtype=torch.float16)
    o_ref = torch.empty(A, G * HD, device=dev, dtype=torch.float16)
    kc2, vc2 = kc.clone(), vc.clone()

    def unfused():
        qkv_split(raw, q16, kc2, vc2, heads_q=G, heads_kv=1, head_dim=HD, inv_freq=inv, row=rs, col=cs, pos=plen)
        attention(q16[None], kc2[None], vc2[None], out=o_ref.view(1, A, G, HD), kv_len=klen)

    unfused()
    q8r = torch.empty(A, G * HD, device=dev, dtype=torch.float8_e4m3fn)
    qsr = torch.empty(A, device=dev)
    row_quant(o_ref, q8r, qsr)
    t_unfused = bench(unfused, iters=100)
    best = None
    for tc5, splits, bm, bn, w, st in itertools.product((False, True), (2, 3, 4, 6, 8), (16, 32, 64, 128), (32, 64), (4, 8), (1, 2)):
        if (tc5 and bm < 64) or (not tc5 and bm > 64):
            continue
        ca = ChunkAttention(A, G, HD, dev, splits=splits, block_m=bm, block_n=bn, num_warps=w, num_stages=st, tc5=tc5)
        o = torch.empty(A, G * HD, device=dev, dtype=torch.float16)
        q8 = torch.empty(A, G * HD, device=dev, dtype=torch.float8_e4m3fn)
        qs = torch.empty(A, device=dev)
        try:
            ca(raw, rs, cs, kc[:, 0], vc[:, 0], plen, inv, q=q8, q_scale=qs, out=o)
            torch.cuda.synchronize()
        except Exception as e:
            print("fail", tc5, splits, bm, bn, w, st, str(e)[:80])
            continue
        err = (o.float() - o_ref.float()).abs().max().item()
        qerr = (q8.float() * qs[:, None] - q8r.float() * qsr[:, None]).abs().max().item()
        if err > 2e-2:
            print("WRONG", tc5, splits, bm, bn, w, st, err)
            continue
        t = bench(lambda: ca(raw, rs, cs, kc[:, 0], vc[:, 0], plen, inv, q=q8, q_scale=qs), iters=50, warm_ms=100)
        if best is None or t < best[0]:
            best = (t, (tc5, splits, bm, bn, w, st), err, qerr)
    print(f"unfused split+attn: {t_unfused * 1e3:.2f} us (+ row quant)")
    print(f"fused best: {best[0] * 1e3:.2f} us cfg={best[1]} max|dO|={best[2]:.2e} max|dq|={best[3]:.2e}")


if __name__ == "__main__":
    main()

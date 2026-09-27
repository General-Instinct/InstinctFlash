"""Warp-count sweep for GI glue kernels at pi05 shapes, plus fp16 patch conv and row dequant."""
import torch

from benchmarks.native.roofline import bench
from instinctflash.kernels.activation import gated_act_quant
from instinctflash.kernels.norm import dequant_rows, residual_norm_quant, row_quant


def sweep(name, make):
    res = []
    for w in (1, 2, 4, 8, 16):
        f = make(w)
        try:
            f()
            torch.cuda.synchronize()
            res.append((bench(f, 50, warm_ms=60) * 1e3, w))
        except Exception:
            res.append((float("inf"), w))
    res.sort()
    print(f"{name:26s} " + " ".join(f"w{w}:{t:6.1f}" for t, w in res), flush=True)


def main():
    dev = "cuda"
    q8 = torch.empty(560 * 16384, device=dev, dtype=torch.float8_e4m3fn)
    qs = torch.empty(560, device=dev)
    for D, M in ((1152, 512), (2048, 560), (1024, 50)):
        x = torch.randn(M, D, device=dev)
        d = torch.randn(M, D, device=dev).to(torch.bfloat16)
        wv = torch.rand(D, device=dev)
        rs = torch.rand(M, device=dev)
        ln = D == 1152
        sweep(f"res_norm_quant D{D} M{M}", lambda w, x=x, d=d, wv=wv, rs=rs, ln=ln, D=D, M=M: (
            lambda: residual_norm_quant(x, delta=d, delta_row=rs, delta_col=wv, layernorm=ln,
                                        weight=wv if ln else None, beta=wv if ln else None,
                                        mod_scale=None if ln else wv, mod_shift=None if ln else wv,
                                        q=q8[:M * D].view(M, D), q_scale=qs, num_warps=w)))
    raw = torch.randn(512, 4304, device=dev).to(torch.bfloat16)
    cs = torch.rand(4304, device=dev)
    rs = torch.rand(512, device=dev)
    sweep("gelu_row_quant 512x4304", lambda w: (lambda: row_quant(raw, q8[:512 * 4304].view(512, 4304), qs, row=rs,
                                                                  col=cs, bias=cs, gelu=True, num_warps=w)))
    o = torch.randn(560, 2048, device=dev).half()
    sweep("row_quant 560x2048", lambda w: (lambda: row_quant(o, q8[:560 * 2048].view(560, 2048), qs, num_warps=w)))
    for M, F in ((50, 4096), (560, 16384)):
        gu = torch.randn(M, 2 * F, device=dev).to(torch.bfloat16)
        c = torch.rand(2 * F, device=dev)
        r = torch.rand(M, device=dev)
        sweep(f"gated_act {M}x{F}", lambda w, gu=gu, c=c, r=r, M=M, F=F: (
            lambda: gated_act_quant(gu, q8[:M * F].view(M, F), qs, row=r, col=c, num_warps=w)))
    qkv = torch.randn(512, 3456, device=dev).to(torch.bfloat16)
    out16 = torch.empty(512, 3456, device=dev, dtype=torch.float16)
    c3 = torch.rand(3456, device=dev)
    for blk in (256, 512, 1024):
        f = lambda blk=blk, w=4: dequant_rows(qkv, out16, row=rs, col=c3, bias=c3, block=blk, num_warps=w)
        print(f"dequant_rows 512x3456 blk{blk}: {bench(f, 50, warm_ms=60) * 1e3:6.1f} us")
    img = torch.randn(2, 3, 224, 224, device=dev)
    pw = torch.randn(1152, 3, 14, 14, device=dev)
    pb = torch.randn(1152, device=dev)
    ih, pwh, pbh = img.half(), pw.half(), pb.half()
    print(f"patch conv fp16: {bench(lambda: torch.nn.functional.conv2d(ih, pwh, pbh, stride=14), 20) * 1e3:6.1f} us")
    pwf = pw.reshape(1152, -1).half()
    def unfold_mm():
        cols = torch.nn.functional.unfold(ih, 14, stride=14)  # [2, 588, 256]
        return torch.matmul(cols.transpose(1, 2), pwf.t())
    print(f"patch unfold+mm fp16: {bench(unfold_mm, 20) * 1e3:6.1f} us")


if __name__ == "__main__":
    main()

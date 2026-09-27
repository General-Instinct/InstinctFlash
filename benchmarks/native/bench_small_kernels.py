"""Time individual small GI kernels at pi05 decoder/prefix shapes (graph, 100 reps)."""
import torch

from benchmarks.native.roofline import bench
from instinctflash.kernels.activation import gated_act_quant
from instinctflash.kernels.attention import attention
from instinctflash.kernels.norm import residual_norm_quant, row_quant
from instinctflash.kernels.rope import qkv_split, rope_inv_freq


def main():
    dev = "cuda"
    A, G, HD, P = 50, 8, 256, 552
    raw = (torch.randn(A, 2560, device=dev) * 30).to(torch.bfloat16)
    rs = torch.rand(A, device=dev)
    cs = torch.rand(2560, device=dev)
    kc = torch.randn(640, 1, HD, device=dev, dtype=torch.float16)
    vc = torch.randn_like(kc)
    q16 = torch.empty(A, G, HD, device=dev, dtype=torch.float16)
    inv = rope_inv_freq(HD, 10000.0, dev)
    plen = torch.tensor([P], device=dev, dtype=torch.int32)
    klen = torch.tensor([P + A], device=dev, dtype=torch.int32)
    o = torch.empty(A, G * HD, device=dev, dtype=torch.float16)
    q8 = torch.empty(560 * 16384, device=dev, dtype=torch.float8_e4m3fn)
    qs = torch.empty(560, device=dev)
    x = torch.randn(A, 1024, device=dev)
    d = (torch.randn(A, 1024, device=dev)).to(torch.bfloat16)
    gu = (torch.randn(A, 8192, device=dev)).to(torch.bfloat16)
    gu_p = (torch.randn(560, 32768, device=dev)).to(torch.bfloat16)
    mod = torch.randn(3, 1024, device=dev)
    xp = torch.randn(560, 2048, device=dev)
    dp = torch.randn(560, 2048, device=dev).to(torch.bfloat16)
    cases = {
        "qkv_split_dec": lambda: qkv_split(raw, q16, kc, vc, heads_q=G, heads_kv=1, head_dim=HD, inv_freq=inv,
                                           row=rs, col=cs, pos=plen),
        "attn_dec": lambda: attention(q16[None], kc[None], vc[None], out=o.view(1, A, G, HD), kv_len=klen),
        "row_quant_dec": lambda: row_quant(o, q8[:A * 2048].view(A, 2048), qs),
        "res_ada_quant_dec": lambda: residual_norm_quant(x, delta=d, delta_row=rs, delta_col=cs[:1024], gate=mod[2],
                                                         mod_scale=mod[0], mod_shift=mod[1],
                                                         q=q8[:A * 1024].view(A, 1024), q_scale=qs),
        "gelu_quant_dec": lambda: gated_act_quant(gu, q8[:A * 4096].view(A, 4096), qs, row=rs, col=cs[:1].expand(8192).contiguous()),
        "gelu_quant_prefix": lambda: gated_act_quant(gu_p, q8.view(560, 16384), qs),
        "res_rms_quant_prefix": lambda: residual_norm_quant(xp, delta=dp, q=q8[:560 * 2048].view(560, 2048), q_scale=qs),
    }
    for name, f in cases.items():
        f()
        torch.cuda.synchronize()
        print(f"{name:22s} {bench(f, 100) * 1e3:8.2f} us", flush=True)


if __name__ == "__main__":
    main()

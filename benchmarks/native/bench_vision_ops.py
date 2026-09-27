"""Time each GI SigLIP op at the 2-view pi05 shape (warmed, graph-replayed)."""
import torch

from benchmarks.native.roofline import bench
from instinctflash.kernels.attention import attention
from instinctflash.kernels.fp8 import gemm, quantize_weight
from instinctflash.kernels.norm import residual_norm_quant, row_quant
from instinctflash.kernels.rope import qkv_split


def main():
    dev = "cuda"
    T, D, H, HD, FF = 512, 1152, 16, 72, 4304
    x = torch.randn(T, D, device=dev)
    raw = (torch.randn(T, FF, device=dev) * 30).to(torch.bfloat16)
    rs = torch.rand(T, device=dev)
    cs = torch.rand(3 * D, device=dev)
    b = torch.rand(3 * D, device=dev)
    q = torch.empty(T, H, HD, device=dev, dtype=torch.float16)
    k, v = torch.empty_like(q), torch.empty_like(q)
    o = torch.empty(T, D, device=dev, dtype=torch.float16)
    q8 = torch.empty(T * FF, device=dev, dtype=torch.float8_e4m3fn)
    qs = torch.empty(T, device=dev)
    img = torch.randn(2, 3, 224, 224, device=dev)
    pw = torch.randn(D, 3, 14, 14, device=dev)
    pb = torch.randn(D, device=dev)
    wq = quantize_weight(torch.randn(3 * D, D, device=dev))
    view = (2, 256, H, HD)
    cases = {
        "patch_conv_fp32": lambda: torch.nn.functional.conv2d(img, pw, pb, stride=14),
        "ln_quant": lambda: residual_norm_quant(x, layernorm=True, weight=pb, beta=pb, q=q8[:T * D].view(T, D), q_scale=qs),
        "qkv_gemm": lambda: gemm(q8[:T * D].view(T, D), wq, raw.view(-1)[:T * 3 * D].view(T, 3 * D)),
        "qkv_split": lambda: qkv_split(raw.view(-1)[:T * 3 * D].view(T, 3 * D), q, k, v, heads_q=H, heads_kv=H,
                                       head_dim=HD, row=rs, col=cs, bias=b),
        "attention": lambda: attention(q.view(view), k.view(view), v.view(view), out=o.view(view),
                                       config=(128, 64, 4, 3), tc5=True),
        "row_quant_o": lambda: row_quant(o, q8[:T * D].view(T, D), qs),
        "res_ln_quant": lambda: residual_norm_quant(x, delta=raw.view(-1)[:T * D].view(T, D), delta_row=rs,
                                                    delta_col=cs[:D], bias=b[:D], layernorm=True, weight=pb, beta=pb,
                                                    q=q8[:T * D].view(T, D), q_scale=qs),
        "gelu_quant_fc1": lambda: row_quant(raw, q8.view(T, FF), qs, row=rs, col=cs[:1].expand(FF).contiguous(),
                                            bias=b[:1].expand(FF).contiguous(), gelu=True),
    }
    for name, f in cases.items():
        f()
        torch.cuda.synchronize()
        print(f"{name:16s} {bench(f, 50, warm_ms=100) * 1e3:8.2f} us", flush=True)


if __name__ == "__main__":
    main()

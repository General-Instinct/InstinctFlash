"""Does FP8 GEMM speed depend on operand value distribution (power/clock limits)?"""
import torch

from benchmarks.native.roofline import bench


def codes(shape, mode, dev):
    x = torch.randn(shape, device=dev)
    if mode == "zeros":
        return torch.zeros(shape, device=dev).to(torch.float8_e4m3fn)
    if mode == "unit":         # |x| ~ 1: small exponents
        return x.to(torch.float8_e4m3fn)
    if mode == "per_row_448":  # each row's amax mapped to 448
        return (x * (448 / x.abs().amax(-1, keepdim=True))).to(torch.float8_e4m3fn)
    if mode == "per_tensor_heavy":  # one outlier row sets the scale: most values tiny
        y = x.clone()
        y[0] *= 50
        return (y * (448 / y.abs().amax())).to(torch.float8_e4m3fn)
    if mode == "half_sparse":
        y = x * (448 / x.abs().amax(-1, keepdim=True))
        y[torch.rand_like(y) < 0.5] = 0
        return y.to(torch.float8_e4m3fn)
    raise ValueError(mode)


def main():
    dev = "cuda"
    one = torch.ones((), device=dev)
    for m, n, k in [(560, 32768, 2048), (560, 2048, 16384), (50, 8192, 1024)]:
        out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        rot = 1 if m > 100 else 18
        for amode in ("unit", "per_row_448", "per_tensor_heavy", "half_sparse", "zeros"):
            for wmode in ("unit", "per_row_448"):
                a = codes((m, k), amode, dev)
                ws = [codes((n, k), wmode, dev) for _ in range(rot)]

                def f():
                    for w in ws:
                        torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16, out=out)
                t = min(bench(f, 20) for _ in range(3)) * 1e3 / rot
                print(f"{m}x{n}x{k} A={amode:17s} W={wmode:12s} {t:8.1f} us", flush=True)


if __name__ == "__main__":
    main()

"""Probe FP8 GEMM options on the current device: tensorwise vs rowwise _scaled_mm, and Triton tl.dot."""
import torch
import triton
import triton.language as tl
from benchmarks.native.roofline import bench


@triton.jit
def _fp8_gemm(a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, M, N, K,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :], mask=rm[:, None] < M, other=0.0)
        b = tl.load(b_ptr + rn[:, None] * K + (k + rk)[None, :], mask=rn[:, None] < N, other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
    sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
    sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
    out = acc * sa[:, None] * sb[None, :]
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], out.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def main():
    dev = "cuda"
    torch.manual_seed(0)
    for (m, n, k) in [(50, 2560, 1024), (50, 1024, 2048), (50, 8192, 1024), (50, 1024, 4096), (560, 2560, 2048), (560, 32768, 2048), (560, 2048, 16384), (512, 3456, 1152)]:
        a = torch.randn(m, k, device=dev)
        w = torch.randn(n, k, device=dev)
        sa = a.abs().amax(1, keepdim=True) / 448
        sw = w.abs().amax(1, keepdim=True) / 448
        a8 = (a / sa).to(torch.float8_e4m3fn)
        w8 = (w / sw).to(torch.float8_e4m3fn)
        ref = (a8.float() * sa) @ (w8.float() * sw).t()
        one = torch.ones((), device=dev)
        t_tensor = bench(lambda: torch._scaled_mm(a8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.float16))
        try:
            out = torch._scaled_mm(a8, w8.t(), scale_a=sa.float(), scale_b=sw.float().t(), out_dtype=torch.float16)
            err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
            t_row = bench(lambda: torch._scaled_mm(a8, w8.t(), scale_a=sa.float(), scale_b=sw.float().t(), out_dtype=torch.float16))
            row = f"rowwise {t_row*1e3:7.1f} us err {err:.1e}"
        except Exception as e:
            row = f"rowwise FAILED {type(e).__name__}: {str(e)[:80]}"
        try:
            c = torch.empty(m, n, device=dev, dtype=torch.float16)
            BM = 64 if m <= 64 else 128
            BN, BK = 128, 128
            grid = (triton.cdiv(m, BM), triton.cdiv(n, BN))
            f = lambda: _fp8_gemm[grid](a8, w8, c, sa, sw, m, n, k, BM=BM, BN=BN, BK=BK)
            f(); torch.cuda.synchronize()
            err = ((c.float() - ref).abs().max() / ref.abs().max()).item()
            t_tr = bench(f)
            tri = f"triton {t_tr*1e3:7.1f} us err {err:.1e}"
        except Exception as e:
            tri = f"triton FAILED {type(e).__name__}: {str(e)[:120]}"
        print(f"{m}x{n}x{k}: tensorwise {t_tensor*1e3:7.1f} us | {row} | {tri}", flush=True)


if __name__ == "__main__":
    main()

"""Per-kernel cost of tiny kernels inside a CUDA graph (launch/dependency gap)."""
import torch
import triton
import triton.language as tl

from benchmarks.native.roofline import bench


@triton.jit
def _tiny(X, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(X + i, tl.load(X + i) + 1.0)


def main():
    x = torch.zeros(1024, device="cuda")
    for n in (100, 1000):
        t = bench(lambda: [_tiny[(1,)](x, N=128) for _ in range(n)], 1, warm_ms=300)
        print(f"{n} tiny triton kernels: {t * 1e3 / n:.2f} us/kernel")
    t = bench(lambda: [x.add_(1.0) for _ in range(1000)], 1, warm_ms=300)
    print(f"1000 torch add_: {t * 1e3 / 1000:.2f} us/kernel")


if __name__ == "__main__":
    main()

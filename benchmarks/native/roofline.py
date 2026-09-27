"""Measure device copy bandwidth and FP8/FP16 GEMM throughput (roofline anchors)."""
import torch


def bench(fn, iters=50, warm_ms=300.0, samples=5):
    """Median per-iteration time (ms) of ``fn`` replayed in a CUDA graph.

    Jetson clocks (GPU and memory controller) scale with load, so the graph
    is replayed for ``warm_ms`` before the timed samples.
    """
    import time
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < warm_ms:
        g.replay()
        torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / iters)
    times.sort()
    return times[len(times) // 2]


def main():
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    for mb in (8, 64, 512):
        n = mb * 2**20 // 2
        a = torch.empty(n, dtype=torch.float16, device=dev); b = torch.empty_like(a)
        ms = bench(lambda: b.copy_(a))
        print(f"copy {mb} MiB: {2 * a.numel() * 2 / ms / 1e6:.1f} GB/s (read+write)")
    # read-only bandwidth via sum
    a = torch.randn(256 * 2**20 // 2, dtype=torch.float16, device=dev)
    out = torch.empty((), dtype=torch.float32, device=dev)
    ms = bench(lambda: torch.sum(a, dim=0, dtype=torch.float32, out=out))
    print(f"read 256 MiB: {a.numel() * 2 / ms / 1e6:.1f} GB/s")
    for (m, n, k) in [(8192, 8192, 8192), (560, 32768, 2048), (560, 2048, 16384), (50, 8192, 1024), (50, 1024, 4096), (50, 2560, 1024), (50, 1024, 2048)]:
        x = torch.randn(m, k, device=dev).to(torch.float8_e4m3fn)
        w = torch.randn(n, k, device=dev).to(torch.float8_e4m3fn)
        one = torch.ones((), device=dev)
        ms = bench(lambda: torch._scaled_mm(x, w.t(), scale_a=one, scale_b=one, out_dtype=torch.float16))
        xh, wh = x.half(), w.half()
        ms16 = bench(lambda: xh @ wh.t())
        print(f"gemm {m}x{n}x{k}: fp8 {ms*1e3:8.1f} us {2*m*n*k/ms/1e9:7.1f} TFLOPS {n*k/ms/1e6:6.1f} GB/s(w) | fp16 {ms16*1e3:8.1f} us {2*m*n*k/ms16/1e9:7.1f} TFLOPS")


if __name__ == "__main__":
    main()

"""Which cuBLASLt kernel does torch._scaled_mm pick per shape (eager vs graph)?"""
import torch
from torch.profiler import ProfilerActivity, profile

SHAPES = [(560, 32768, 2048), (560, 2048, 16384), (560, 2560, 2048), (560, 2048, 2048),
          (50, 8192, 1024), (50, 1024, 4096), (50, 2560, 1024), (50, 1024, 2048)]


def main():
    dev = "cuda"
    one = torch.ones((), device=dev)
    ops = []
    for m, n, k in SHAPES:
        a = (torch.randn(m, k, device=dev) * 20).to(torch.float8_e4m3fn)
        w = (torch.randn(n, k, device=dev) * 20).to(torch.float8_e4m3fn)
        o = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        ops.append(((m, n, k), lambda a=a, w=w, o=o: torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one,
                                                                          out_dtype=torch.bfloat16, out=o)))
    for _, f in ops:
        f()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _, f in ops:
            f()
    for mode in ("eager", "graph"):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            if mode == "eager":
                for _, f in ops:
                    f()
            else:
                g.replay()
            torch.cuda.synchronize()
        names = [(e.name, e.device_time_total) for e in prof.events()
                 if e.device_type == torch.autograd.DeviceType.CUDA]
        print(mode)
        for (shape, _), (name, t) in zip(ops, names):
            print(f"  {shape} {t:8.1f} us {name[:70]}")


if __name__ == "__main__":
    main()

"""torch._scaled_mm (TN) BF16 / FP16 / fast-accum output comparison at engine shapes, rotating distinct weights to defeat L2."""
import torch

from benchmarks.native.roofline import bench

SHAPES = [(50, 2560, 1024), (50, 1024, 2048), (50, 8192, 1024), (50, 1024, 4096),
          (560, 2560, 2048), (560, 2048, 2048), (560, 32768, 2048), (560, 2048, 16384),
          (512, 3456, 1152), (512, 1152, 1152), (512, 4304, 1152), (512, 1152, 4304)]
ROT = 18


def main():
    dev = "cuda"
    one = torch.ones((), device=dev)
    for m, n, k in SHAPES:
        a = (torch.randn(m, k, device=dev) * 20).to(torch.float8_e4m3fn)
        ws = [(torch.randn(n, k, device=dev) * 20).to(torch.float8_e4m3fn) for _ in range(ROT)]
        ob = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        oh = torch.empty(m, n, device=dev, dtype=torch.float16)

        def tn(out, fast=False):
            def f():
                for w in ws:
                    torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one, out_dtype=out.dtype, out=out,
                                     use_fast_accum=fast)
            return f
        row = [f"{m}x{n}x{k}"]
        row.append(f"TN-bf16 {bench(tn(ob), 20) * 1e3 / ROT:7.2f}")
        row.append(f"TN-fp16 {bench(tn(oh), 20) * 1e3 / ROT:7.2f}")
        row.append(f"TN-bf16-fast {bench(tn(ob, True), 20) * 1e3 / ROT:7.2f}")
        print(" | ".join(row), flush=True)


if __name__ == "__main__":
    main()

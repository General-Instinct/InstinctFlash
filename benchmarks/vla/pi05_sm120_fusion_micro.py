"""CUDA-graph/event microbenchmarks, including equal input resets in both arms."""
from __future__ import annotations
import argparse
from pathlib import Path

from benchmarks.vla.util import sha256_file, write_json_atomic


def benchmark():
    import numpy as np
    import torch
    from flash_rt import flash_rt_kernels as old
    from flash_rt import flash_rt_pi05_sm120_fusion as fusion
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("requires SM120")
    torch.manual_seed(5090)
    scale = torch.tensor([.031], device="cuda", dtype=torch.float32)
    rows = []
    for name, dimension in (("qkv", 3456), ("ffn4", 4304), ("ffn8", 4304)):
        source = torch.randn((512, dimension), device="cuda", dtype=torch.bfloat16)
        working = torch.empty_like(source)
        bias = torch.randn(dimension, device="cuda", dtype=torch.bfloat16)
        zero = torch.zeros_like(source)
        outputs = {arm: ([torch.empty((512, 1152), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
                        if name == "qkv" else [torch.empty(source.shape, device="cuda", dtype=torch.float8_e4m3fn)])
                   for arm in ("baseline", "fusion")}

        def operation(arm):
            stream = torch.cuda.current_stream().cuda_stream
            working.copy_(source)  # Identical reset in BOTH arms, inside every repetition.
            pointers = [value.data_ptr() for value in outputs[arm]]
            if name == "qkv":
                if arm == "baseline":
                    old.bias_residual(working.data_ptr(), zero.data_ptr(), bias.data_ptr(), 512, dimension, stream=stream)
                    old.qkv_split(working.data_ptr(), *pointers, 512, 1152, 1152, 1152, stream=stream)
                else:
                    fusion.bias_qkv(working.data_ptr(), bias.data_ptr(), *pointers, 512, 1152, stream)
            elif arm == "baseline":
                old.bias_residual(working.data_ptr(), zero.data_ptr(), bias.data_ptr(), 512, dimension, stream=stream)
                old.gelu_inplace(working.data_ptr(), source.numel(), stream=stream)
                old.quantize_fp8_static(working.data_ptr(), pointers[0], scale.data_ptr(), source.numel(), stream=stream)
            else:
                fusion.bias_gelu_fp8(working.data_ptr(), bias.data_ptr(), pointers[0], scale.data_ptr(), 512, dimension, int(name[-1]), stream)

        for arm in outputs:
            operation(arm)
        torch.cuda.synchronize()
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                   for a, b in zip(outputs["baseline"], outputs["fusion"]))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = {}
        for arm in outputs:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(128):
                    operation(arm)
            graphs[arm] = graph
        for _ in range(10):
            for graph in graphs.values():
                graph.replay()
        torch.cuda.synchronize()
        timings = {arm: [] for arm in outputs}
        for repeat in range(20):
            for arm in (("baseline", "fusion") if repeat % 2 == 0 else ("fusion", "baseline")):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[arm].replay()
                end.record()
                end.synchronize()
                timings[arm].append(start.elapsed_time(end) * 1000 / 128)
        medians = {arm: float(np.median(values)) for arm, values in timings.items()}
        rows.append({"name": name, "shape": [512, dimension], "bit_exact": True,
                     "median_us_including_reset": medians, "timings_us": timings,
                     "saved_us": medians["baseline"] - medians["fusion"],
                     "speedup": medians["baseline"] / medians["fusion"]})
    return {"protocol": "alternating CUDA-event timing of 128-repeat CUDA graphs; identical reset in both arms",
            "device": torch.cuda.get_device_name(), "rows": rows,
            "source_sha256": sha256_file(Path(__file__)),
            "binary_sha256": {"baseline": sha256_file(Path(old.__file__)), "fusion": sha256_file(Path(fusion.__file__))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = benchmark()
    write_json_atomic(args.output, result)
    for row in result["rows"]:
        print(row["name"], row["median_us_including_reset"], row["speedup"], flush=True)


if __name__ == "__main__":
    main()

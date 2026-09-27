"""Eager (non-graph) GI pi05 forward under the profiler, grouped by op input shape."""
import torch
from torch.profiler import ProfilerActivity, profile


def main():
    from benchmarks.native.pi05_parity import _active, _checkpoint
    from instinctflash.engines.pi05.engine import Pi05Engine
    from instinctflash.engines.pi05.weights import load_pi05_weights
    import sys
    data = torch.load(sys.argv[1])
    dev = torch.device("cuda")
    eng = Pi05Engine(load_pi05_weights(_checkpoint(), dev), 2, dev)
    x, n = data["inputs"][0], data["noise"][0]
    eng.set_prompt(x["tokens"][x["masks"]].numpy())
    eng.img.copy_(torch.stack(_active(x)))
    eng.noise.copy_(n)
    for _ in range(2):
        eng._forward(eng._rows)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
        eng._forward(eng._rows)
        torch.cuda.synchronize()
    rows = [e for e in prof.key_averages(group_by_input_shape=True) if e.key == "aten::_scaled_mm"]
    rows.sort(key=lambda e: -e.device_time_total)
    for e in rows:
        print(f"{e.device_time_total / 1e3:8.3f} ms {e.count:5d}x {e.device_time_total / e.count:8.2f} us  {e.input_shapes[:2]}")


if __name__ == "__main__":
    main()

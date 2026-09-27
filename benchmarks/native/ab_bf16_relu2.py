"""Interleaved A/B of the BF16 GEMM+ReLU^2 candidates (Thor clocks drift; alternate rounds)."""
import os

import numpy as np
import torch

from benchmarks.native.roofline import bench
from instinctflash.kernels.gemm_bf16 import bf16_gemm_act, bf16_gemm_act_persistent


def main():
    dev = "cuda"
    M, N, K = (int(v) for v in os.environ.get("IFL_BF16_SHAPE", "3093,9216,2048").split(","))
    torch.manual_seed(0)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    if os.environ.get("IFL_BF16_X"):          # a real activation captured in situ
        x = torch.load(os.environ["IFL_BF16_X"]).to(dev).contiguous()
    # Rotate through as many distinct weights as the Edge GEN stack has layers, so each call
    # streams a cold weight like the real model does (one warm weight fits in L2 partially).
    layers = int(os.environ.get("IFL_BF16_WEIGHTS", "28"))
    ws = [(torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16) for _ in range(layers)]
    if os.environ.get("IFL_BF16_REAL"):       # the checkpoint's own GEN up_proj weights
        import json
        from pathlib import Path
        from safetensors import safe_open
        snap = Path(os.environ["IFL_BF16_REAL"])
        index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
        ws = []
        for i in range(layers):
            key = f"layers.{i}.mlp_moe_gen.up_proj.weight"
            with safe_open(str(snap / index[key]), "pt") as f:
                ws.append(f.get_tensor(key).to(dev, torch.bfloat16).contiguous())
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    ref = torch.relu(torch.nn.functional.linear(x, ws[0])).square()
    cyc = [0]

    def rot(fn):
        def call():
            cyc[0] = (cyc[0] + 1) % layers
            return fn(ws[cyc[0]])
        return call

    cands = {
        "tiled(128,256,64,8,4,8)": rot(lambda w: bf16_gemm_act(x, w, out, config=(128, 256, 64, 8, 4, 8))),
        "persist(3,8,1,ws)": rot(lambda w: bf16_gemm_act_persistent(x, w, out, config=(128, 256, 64, 8, 3, 8, 1, True))),
        "persist(4,8,2,ws)": rot(lambda w: bf16_gemm_act_persistent(x, w, out, config=(128, 256, 64, 8, 4, 8, 2, True))),
        "persist(4,8,2,ws,g16)": rot(lambda w: bf16_gemm_act_persistent(x, w, out, config=(128, 256, 64, 16, 4, 8, 2, True))),
        "persist(4,8,2,ws,g4)": rot(lambda w: bf16_gemm_act_persistent(x, w, out, config=(128, 256, 64, 4, 4, 8, 2, True))),
        "eager": rot(lambda w: torch.relu(torch.nn.functional.linear(x, w)).square()),
    }
    for name, fn in cands.items():
        cyc[0] = layers - 1
        y = fn()
        y = out if y is None else y
        print(name, "exact", torch.equal(y, ref), flush=True)
    times = {k: [] for k in cands}
    gap = int(os.environ.get("IFL_BF16_GAP_CYCLES", "0"))   # idle GPU cycles between calls (serving duty cycle)
    for _ in range(int(os.environ.get("IFL_ROUNDS", "6"))):
        for name, fn in cands.items():
            if not gap:
                times[name].append(bench(fn, 10, warm_ms=300) * 1e3)
                continue
            per = []
            for i in range(60):
                torch.cuda._sleep(gap)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); fn(); e.record()
                per.append((s, e))
            torch.cuda.synchronize()
            per = sorted(a.elapsed_time(b) * 1e3 for a, b in per[20:])
            times[name].append(per[len(per) // 2])
    for name, t in times.items():
        print(f"{name:28s} median {np.median(t):8.1f} us  min {min(t):8.1f}  max {max(t):8.1f}")


if __name__ == "__main__":
    main()

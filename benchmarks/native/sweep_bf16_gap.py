"""Sweep the persistent BF16 GEMM+ReLU^2 under a serving duty cycle (idle gaps between calls).

Back-to-back GEMM loops on Thor are power-throttled; inside a real request the GEMM runs at
higher clocks and becomes L2/DRAM sensitive, so tiles are ranked with gaps and real operands.
"""
import itertools
import json
import os
from pathlib import Path

import torch

from instinctflash.kernels.gemm_bf16 import bf16_gemm_act_persistent


def per_call_us(fn, gap, n=40):
    per = []
    for _ in range(n):
        torch.cuda._sleep(gap)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        per.append((s, e))
    torch.cuda.synchronize()
    t = sorted(a.elapsed_time(b) * 1e3 for a, b in per[10:])
    return t[len(t) // 2]


def main():
    dev = "cuda"
    snap = Path(os.environ["IFL_BF16_REAL"])
    x = torch.load(os.environ["IFL_BF16_X"]).to(dev).contiguous()
    from safetensors import safe_open
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    ws = []
    for i in range(int(os.environ.get("IFL_BF16_WEIGHTS", "8"))):
        key = f"layers.{i}.mlp_moe_gen.up_proj.weight"
        with safe_open(str(snap / index[key]), "pt") as f:
            ws.append(f.get_tensor(key).to(dev, torch.bfloat16).contiguous())
    M, N = x.shape[0], ws[0].shape[0]
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    ref = torch.relu(torch.nn.functional.linear(x, ws[0])).square()
    gap = int(os.environ.get("IFL_BF16_GAP_CYCLES", "1000000"))
    cyc = [0]

    def rot(cfg):
        def call():
            cyc[0] = (cyc[0] + 1) % len(ws)
            bf16_gemm_act_persistent(x, ws[cyc[0]], out, config=cfg)
        return call

    rows = []
    for bn, g, st, wp, split, ws_ in itertools.product((256, 128), (8, 12, 16, 20, 25, 32), (3, 4), (4, 8), (1, 2), (True,)):
        cfg = (128, bn, 64, g, st, wp, split, ws_)
        try:
            bf16_gemm_act_persistent(x, ws[0], out, config=cfg)
            torch.cuda.synchronize()
        except Exception as e:
            print("fail", cfg, type(e).__name__, flush=True)
            continue
        if not torch.equal(out, ref):
            print("inexact", cfg, flush=True)
            continue
        t = per_call_us(rot(cfg), gap)
        rows.append((t, cfg))
        print(f"{t:8.1f} us {cfg}", flush=True)
    rows.sort()
    print("best:")
    for t, cfg in rows[:8]:
        print(f"  {t:8.1f} us {cfg}")


if __name__ == "__main__":
    main()

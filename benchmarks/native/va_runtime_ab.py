"""LingBot-VA public Runtime A/B on the recorded fixture (whichever instinctflash is on PYTHONPATH).

One warmup episode, then N measured episodes of three control cycles; continuation
cycles (1, 2) are the primary timing, as in the public protocol.

    python -m benchmarks.native.va_runtime_ab --nfe 2v4a --episodes 4 --output va.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL = "robbyant/lingbot-va-posttrain-robotwin"
REVISION = "8c9dea8abbc5c91cc9e18bc3264b8915083bbe70"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default=str(Path.home() / "gi_native/recorded_inputs_v1.npz"))
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--nfe", choices=("2v4a", "full"), default="2v4a")
    p.add_argument("--output")
    args = p.parse_args()
    import torch
    from benchmarks.regression.user_e2e import RecordedInputs, prompt
    from instinctflash import Runtime
    kw = dict(device="cuda:0", precision="fp8", exclude_passes=["cfg_branch_elision"])
    if args.nfe == "2v4a":
        kw["nfe"] = {"video": 2, "action": 4}
    t0 = time.perf_counter()
    rt = Runtime.from_pretrained(MODEL, revision=REVISION, **kw)
    build = time.perf_counter() - t0
    inputs = RecordedInputs(args.inputs)
    rows, actions = [], []
    for ep in range(args.episodes + 1):
        rt.reset(prompt=prompt(ep))
        for cycle in range(3):
            obs = inputs.observation("va", ep, cycle)
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = rt.predict(obs)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1e3
            if ep > 0:
                rows.append({"episode": ep, "cycle": cycle, "ms": ms})
                actions.append(np.asarray(out["action"], np.float32))
    cont = [r["ms"] for r in rows if r["cycle"] > 0]
    stats = getattr(rt._backend._loop, "backend_stats", None) if hasattr(rt._backend, "_loop") else None
    summary = {"build_s": build, "continuation_p50_ms": float(np.median(cont)),
               "cycle0_p50_ms": float(np.median([r["ms"] for r in rows if r["cycle"] == 0])),
               "per_cycle": rows, "backend": str((stats or {}).get("backend")),
               "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    print(json.dumps({k: v for k, v in summary.items() if k != "per_cycle"}, indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=1))
        np.save(args.output.replace(".json", "_actions.npy"), np.stack(actions))
    rt.close()


if __name__ == "__main__":
    main()

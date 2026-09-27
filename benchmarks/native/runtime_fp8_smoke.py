"""Public Runtime FP8 smoke: which backend serves pi05 and at what latency (recorded inputs)."""
import json
import sys
import time

import numpy as np
import torch

from benchmarks.native.pi05_engine_ab import CHECKPOINT, REVISION, _observations


def main():
    from instinctflash import Runtime
    rt = Runtime.from_pretrained(CHECKPOINT, revision=REVISION, device="cuda:0", precision="fp8")
    loop = rt._backend._loop
    print("backend loop:", type(loop).__module__, type(loop).__name__, getattr(loop, "backend_stats", None))
    obs = _observations(sys.argv[1], 30)
    rt.reset(prompt=obs[0]["prompt"])
    lat = []
    for i, o in enumerate(obs):
        loop._queue.clear()
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = rt.predict(o)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1e3)
        assert np.isfinite(out["action"]).all() and out["action"].shape == (7,)
    lat = np.array(lat[5:])
    print(json.dumps({"p50_ms": float(np.median(lat)), "p90_ms": float(np.percentile(lat, 90)), "n": len(lat)}))
    rt.close()


if __name__ == "__main__":
    main()

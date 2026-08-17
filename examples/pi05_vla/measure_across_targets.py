"""Same checkpoint, same public API, two hardware targets. Latency and agreement.

    CUDA_VISIBLE_DEVICES=0  python examples/pi05_vla/measure_across_targets.py
    CUDA_VISIBLE_DEVICES=   python examples/pi05_vla/measure_across_targets.py

Measured on one box, ACT (51.6M params), 12 cycles, warm median of the last ten:

    H100 sm90    first cycle 374.53 ms   warm median 0.68 ms
    CPU  x86_64  first cycle  68.35 ms   warm median 0.59 ms
    agreement    max|delta| 7.40e-04, cosine 0.99999875

Two things worth keeping. The CPU is not slower warm, so for a policy this size the GPU wins
nothing and a runtime that assumed otherwise would be wrong -- exactly the decision hardware
awareness exists to make, and it is now measurable per device instead of assumed. And the first
cycle INVERTS, 374 ms against 68 ms, because CUDA context creation and kernel autotune dominate a
model this small; a benchmark that reported only cold latency would rank the targets backwards.

The agreement is NUMERIC, not bit-exact. A cross-target claim needs a certificate like any other
non-bit-exact change; cosine on a 14-dim action vector is not evidence about task success.
"""
import json, os, statistics, sys, tempfile, time
from pathlib import Path
ROOT = Path("/home/ubuntu/InstinctWM")
for p in (str(ROOT), str(ROOT / "examples" / "pi05_vla")):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np
import instinctwm
from instinctwm import Runtime
from instinctwm.passes.contract import DeviceProfile
import act_iwm
if "act" not in instinctwm.available_models():
    instinctwm.register("act", act_iwm.ACTAdapter)

d = DeviceProfile.probe()
print(f"TARGET  {d.name}  sm{d.capability[0]}{d.capability[1]}  features {sorted(d.features)}")

with tempfile.TemporaryDirectory() as td:
    pkg = Path(td) / "act"; pkg.mkdir()
    (pkg / "config.json").write_text("{}")
    (pkg / "instinctwm.json").write_text(json.dumps({"instinctwm_schema": 1, "execution": {
        "model_id": "example-org/act", "backbone": "act", "servable": True,
        "nfe": {"action": 1}, "base_weights": "lerobot/act_aloha_sim_transfer_cube_human"}}))
    rt = Runtime.from_pretrained(pkg)
    obs = {"observation.images.top": np.zeros((1, 3, 480, 640), dtype=np.float32),
           "observation.state": np.zeros((1, 14), dtype=np.float32)}
    with rt, rt.episode() as ep:
        acts, times = [], []
        for i in range(12):
            t0 = time.perf_counter()
            out = ep.predict(obs)
            times.append((time.perf_counter() - t0) * 1000)
            acts.append(np.asarray(out["action"]))
    warm = times[2:]
    print(f"  first cycle {times[0]:8.2f} ms   warm median {statistics.median(warm):8.2f} ms "
          f"(n={len(warm)}, spread {(max(warm)-min(warm))/statistics.median(warm):.0%})")
    np.save(os.environ.get("SAVE", "/tmp/act_out.npy"), acts[0])
    print(f"  action[:4] {[round(float(v),6) for v in acts[0].ravel()[:4]]}")

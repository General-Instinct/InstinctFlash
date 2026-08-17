"""Runtime overhead: InstinctWM's public API vs calling the reference implementation directly.

    CUDA_VISIBLE_DEVICES=0 python examples/pi05_vla/measure_overhead_vs_reference.py
    CUDA_VISIBLE_DEVICES=  python examples/pi05_vla/measure_overhead_vs_reference.py

Measured on ACT (51.6M params), 60 cycles, median. BOTH arms read one shared cache-hot host buffer:

    CPU     reference 0.418 ms   InstinctWM 0.428 ms   +4.5%
    H100    reference 0.677 ms   InstinctWM 0.710 ms   +4.6 / +4.8 / +6.6%  (three runs)
    H100    the host->device copy alone  +0.242 ms, ~56% of a 0.414 ms forward

THE ABSTRACTION COSTS ABOUT 5% on both targets, replicated. That is the number to quote for "how close
to hand-written does the public API get on the same model and the same device": a Runtime, an Episode, a
declared observation contract and a plan cost roughly one twentieth of a control cycle.

THE BOTTLENECK IS THE UPLOAD, NOT THE NETWORK. Moving a 3.5 MB observation to the GPU costs ~58% of the
forward pass, which is also why a CPU is not slower than an H100 warm on this model -- the CPU never
pays it.

PINNED STAGING WAS TRIED AND IS SLOWER: 0.264 ms against 0.236 ms for a plain `.to(device)`. Pinned
memory pays when the copy OVERLAPS compute on another stream, and a cycle that uploads then immediately
synchronises has no overlap to exploit, so the extra host->host copy is pure cost. Removed. Do not
re-add it without a measurement that shows overlap.

TWO HARNESS ERRORS THAT BOTH FLATTERED US, recorded because they are the whole reason to write this
down. First the reference arm did `.to(dev).float()` unconditionally, copying an already-float32 tensor
that our path skips. Then, with that fixed, the reference allocated a FRESH 3.5 MB array every cycle
while our arm reused one, so it read cold memory and we read cache-hot. Each error alone produced
"InstinctWM is 10-12% FASTER", replicated three times, and both were wrong. A favourable number that
replicates is still an artifact if the baseline is doing more work.
"""
import json, statistics, sys, tempfile, time
from pathlib import Path
ROOT = Path("/home/ubuntu/InstinctWM")
for p in (str(ROOT), str(ROOT / "examples" / "pi05_vla")):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, torch
from lerobot.policies.act.modeling_act import ACTPolicy

REPO = "lerobot/act_aloha_sim_transfer_cube_human"
N = 60
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {dev}   cycles {N}")

def obs_torch(on_device=True):
    d = dev if on_device else "cpu"
    return {"observation.images.top": torch.zeros(1, 3, 480, 640, device=d),
            "observation.state": torch.zeros(1, 14, device=d)}

# ONE host buffer, reused by BOTH arms. Allocating a fresh 3.5 MB array per cycle left the reference
# reading cold memory while our arm reused a cache-hot one -- an asymmetry worth about 11% on CPU, and
# ours to keep for the wrong reason. A real camera writes into the same buffer too.
_HOST = {"observation.images.top": np.zeros((1, 3, 480, 640), np.float32),
         "observation.state": np.zeros((1, 14), np.float32)}


def obs_numpy():
    return _HOST

# --- ARM A: the reference implementation, called directly -----------------------------------------
p = ACTPolicy.from_pretrained(REPO); p.eval(); p.to(dev)
def _to_dev(v):
    """The reference arm's conversion, written to be no LESS careful than ours.

    It first did `.to(dev).float()` unconditionally, which copies an already-float32 CPU tensor and
    made the reference look 12% slower on CPU -- where our stager does nothing at all. A comparison
    whose baseline is sloppier than the thing being measured is not a comparison, and that number
    would have been ours to keep for the wrong reason.
    """
    t = v if torch.is_tensor(v) else torch.as_tensor(v)
    if t.dtype != torch.float32:
        t = t.float()
    return t if str(t.device).startswith(str(dev)) else t.to(dev)


def time_direct(make):
    p.reset()
    with torch.no_grad():
        for _ in range(5): p.select_action({k: _to_dev(v) for k, v in make().items()})
    if dev == "cuda": torch.cuda.synchronize()
    out = []
    with torch.no_grad():
        for _ in range(N):
            src = make()
            t0 = time.perf_counter()
            b = {k: _to_dev(v) for k, v in src.items()}
            p.select_action(b)
            if dev == "cuda": torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1000)
    return out

direct_dev  = time_direct(lambda: obs_torch(True))    # already resident: pure compute
direct_host = time_direct(obs_numpy)                  # host arrays: what a camera gives you

# --- ARM B: the same checkpoint through InstinctWM's public API -----------------------------------
import act_iwm, instinctwm
if "act" not in instinctwm.available_models():
    instinctwm.register("act", act_iwm.ACTAdapter)
from instinctwm import Runtime
with tempfile.TemporaryDirectory() as td:
    pkg = Path(td) / "act"; pkg.mkdir()
    (pkg / "config.json").write_text("{}")
    (pkg / "instinctwm.json").write_text(json.dumps({"instinctwm_schema": 1, "execution": {
        "model_id": "example-org/act", "backbone": "act", "servable": True,
        "nfe": {"action": 1}, "base_weights": REPO}}))
    rt = Runtime.from_pretrained(pkg, device=dev)
    obs = obs_numpy()                                   # the SAME host buffer arm A used
    with rt, rt.episode() as ep:
        for _ in range(5): ep.predict(obs)
        if dev == "cuda": torch.cuda.synchronize()
        through = []
        for _ in range(N):
            t0 = time.perf_counter(); ep.predict(obs)
            if dev == "cuda": torch.cuda.synchronize()
            through.append((time.perf_counter() - t0) * 1000)

med = statistics.median
print(f"  reference, tensors already resident   {med(direct_dev):7.3f} ms   <- pure compute")
print(f"  reference, from host arrays           {med(direct_host):7.3f} ms   <- realistic input path")
print(f"  through InstinctWM, from host arrays  {med(through):7.3f} ms")
fair = med(through) - med(direct_host)
print(f"  ABSTRACTION OVERHEAD (like for like)  {fair:+.3f} ms  ({fair/med(direct_host)*100:+.1f}%)")
xfer = med(direct_host) - med(direct_dev)
print(f"  cost of the host->device copy itself  {xfer:+.3f} ms  ({xfer/med(direct_dev)*100:+.1f}% of compute)")

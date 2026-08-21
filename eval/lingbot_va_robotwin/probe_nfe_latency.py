"""ABBA latency of 2V/2A against shipped 2V/4A, in-process, on the shipped stack.

The 1.21x for 2V/2A is a model estimate (transformer share x forward ratio). The repo's only direct
measurement of a reduced-forward arm gave 1.10x, so model and measurement disagree by ~1.1x on the
entire justification for shipping it. This measures it: ABBA-ordered (base, treat, treat, base) so
drift is visible and cancels, no profiler, warm past ring saturation.
"""
import os, sys, time, statistics
sys.path.insert(0, "/home/ubuntu/InstinctFlash")
import numpy as np, torch
from instinctflash.runtime.lingbot_install import (
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision)

hot = [l for l in os.popen("nvidia-smi --query-gpu=index,utilization.gpu "
       "--format=csv,noheader,nounits").read().strip().split("\n")
       if l.strip() and int(l.split(",")[1]) >= 15]
if hot:
    print(f"NOT EVALUATED: fleet busy ({hot})"); raise SystemExit(2)

S = import_lingbot_server()
cfg = S.VA_CONFIGS["robotwin"]
cfg.save_root = "/tmp/iwm_nfe"; os.makedirs(cfg.save_root, exist_ok=True)
S.init_distributed(1, 0, 0); cfg.rank = cfg.local_rank = 0; cfg.world_size = 1
install_fsdp_elision(S); torch.cuda.empty_cache = lambda *a, **k: None
cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4
server = S.VA_Server(cfg)
from instinctflash.passes.lingbot.ring_kv import RingKVAddressing
RingKVAddressing().install(S, type(server))
for _ in install_conditioning_prefill(S, type(server)): pass
for _ in install_debug_dump_elision(S): pass
from instinctflash.backends.conv.apply import install_conv_layout
for _ in install_conv_layout(server): pass

from pathlib import Path
ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
z = np.load(ctx[0], allow_pickle=True)
short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
obs = [{f: z[s] for s, f in short.items()}]
prompt = str(z["prompt"]); cams = list(cfg.obs_cam_keys)
rng = np.random.default_rng(0)

def cycle(first=False):
    if first:
        server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
    t0 = time.perf_counter()
    act = server.infer(dict(obs=obs, prompt=prompt, save_visualization=False))["action"]
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
          for _ in range(4 if first else 8)]
    server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                      save_visualization=False, state=act))
    return dt * 1000

def arm(V, A, warm=40, n=14):
    cfg.num_inference_steps, cfg.action_num_inference_steps = V, A
    for c in S.VA_CONFIGS.values():
        if hasattr(c, "num_inference_steps"):
            c.num_inference_steps, c.action_num_inference_steps = V, A
    cycle(True)
    for _ in range(warm): cycle()
    return statistics.median(cycle() for _ in range(n))

print("ABBA: base(2V/4A), treat(2V/2A), treat, base", flush=True)
b1 = arm(2, 4); print(f"  base  2V/4A  {b1:7.1f} ms", flush=True)
t1 = arm(2, 2); print(f"  treat 2V/2A  {t1:7.1f} ms", flush=True)
t2 = arm(2, 2); print(f"  treat 2V/2A  {t2:7.1f} ms", flush=True)
b2 = arm(2, 4); print(f"  base  2V/4A  {b2:7.1f} ms", flush=True)
base, treat = (b1+b2)/2, (t1+t2)/2
drift = abs(b2-b1)/base
tspread = abs(t2-t1)/treat
print(f"\nbase mean {base:.1f} ms   treat mean {treat:.1f} ms")
print(f"drift on repeated base  arms {drift:.1%}")
# GATE BOTH ARMS. This checked only the base arms, and that is how a 1.340x reading survived: the
# base arms agreed to 0.9% while the two IDENTICAL treat arms differed by up to 69% on other devices
# (182.1 vs 307.6 ms), one of them slower than base despite running fewer forwards. An ABBA design
# cancels drift, not variance -- if the repeated TREAT arms disagree, the treatment mean is noise and
# the ratio is uninterpretable. The P007 certificate recorded the same asymmetry at 6.4% vs 0.7% and
# called the treated path the noisier of the two; this is that failure an order of magnitude larger.
print(f"spread on repeated treat arms {tspread:.1%}")
worst = max(drift, tspread)
if worst >= 0.05:
    which = "treat" if tspread > drift else "base"
    print(f"NOT EVALUATED: repeated {which} arms differ by {worst:.1%} (>= 5%). The ratio "
          f"{base/treat:.3f}x is not interpretable; re-run on a verified-idle device.")
    raise SystemExit(2)
print(f"SPEEDUP 2V/2A vs shipped 2V/4A = {base/treat:.3f}x   ({base-treat:+.1f} ms/cycle)")
print(f"  both arms repeatable to <5%, so the ratio is admissible")

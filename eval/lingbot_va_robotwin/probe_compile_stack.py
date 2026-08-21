"""Layer 6 measurement: does compiling the transformer block stack remove host dispatch?

Two questions, one process.

(1) DIAGNOSTIC. A previous run reported 9,960 aten events and 2,143 kernels for a single compiled
    block call, ~30x eager. That is implausible for a 1-graph/0-break compile and must be explained
    before any number from it is quoted. This script snapshots dynamo's compile counters around the
    profiled region and prints the compiled profile's top events by count, so the inflation is
    attributed rather than guessed.

(2) THE UNTESTED QUESTION. One block compiled measured 0.86x -- slower. The stated reason is that
    only 55 of ~123 dispatcher ops were captured while dynamo guard evaluation was added on top, so
    the overhead was paid 300 times per cycle without the fusion to cover it. Compiling all 30 blocks
    as ONE region pays guards 10 times per cycle instead of 300 and lets inductor fuse across block
    boundaries. That is the measurement that decides whether "one persistent execution object per
    forward" ranks first in Layer 6 or is blocked.

No CUDA graphs anywhere (mode="default"), no Triton authored by us, no kernel tuning: this measures
host dispatch only. Numerics are checked against eager on every arm.
"""
import copy
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("IFL_ROOT", "/home/ubuntu/InstinctFlash"))
import numpy as np
import torch

from instinctflash.runtime.lingbot_install import (
    import_lingbot_server,
    install_conditioning_prefill,
    install_debug_dump_elision,
    install_fsdp_elision,
)

S = import_lingbot_server()
cfg = S.VA_CONFIGS["robotwin"]
cfg.save_root = "/tmp/iwm_cstack"
os.makedirs(cfg.save_root, exist_ok=True)
S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), int(os.getenv("RANK", 0)))
cfg.rank = cfg.local_rank = 0
cfg.world_size = 1
install_fsdp_elision(S)
torch.cuda.empty_cache = lambda *a, **k: None
cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4
server = S.VA_Server(cfg)

from instinctflash.passes.lingbot.ring_kv import RingKVAddressing

RingKVAddressing().install(S, type(server))
for _ in install_conditioning_prefill(S, type(server)):
    pass
for _ in install_debug_dump_elision(S):
    pass
from instinctflash.backends.conv.apply import install_conv_layout

for _ in install_conv_layout(server):
    pass

ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
z = np.load(ctx[0], allow_pickle=True)
short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
obs = [{f: z[s] for s, f in short.items()}]
prompt = str(z["prompt"])
cams = list(cfg.obs_cam_keys)
rng = np.random.default_rng(0)


def cycle(first=False):
    if first:
        server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
    act = server.infer(dict(obs=obs, prompt=prompt, save_visualization=False))["action"]
    kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
          for _ in range(4 if first else 8)]
    server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False, save_visualization=False, state=act))


cycle(True)
for _ in range(40):
    cycle()   # drive the ring past saturation so we measure the steady state

# ---- capture the exact arguments block 0 receives inside the loop ------------------------------
blocks = server.transformer.blocks
blk0 = blocks[0]
orig0 = blk0.forward
grab = {"v": None}


def snoop(*a_, **k_):
    if grab["v"] is None:
        grab["v"] = (a_, k_)
    return orig0(*a_, **k_)


blk0.forward = snoop
cycle()
blk0.forward = orig0
ar, kw = grab["v"]
print(f"captured block args: {[tuple(a.shape) if torch.is_tensor(a) else type(a).__name__ for a in ar]} "
      f"kwargs={ {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in kw.items()} }")


def one_block(*a_, **k_):
    return orig0(*a_, **k_)


def stack(*a_, **k_):
    """The whole 30-block region as a single callable -- what a persistent execution object replaces."""
    h = a_[0]
    rest = a_[1:]
    for b in blocks:
        h = b(h, *rest, **k_)
    return h


# ---- instruments -------------------------------------------------------------------------------
def counters_snapshot():
    from torch._dynamo.utils import counters
    return copy.deepcopy(dict(counters))


def counters_delta(before, after):
    out = {}
    for k in set(before) | set(after):
        b, a = before.get(k, {}), after.get(k, {})
        for kk in set(b) | set(a):
            d = a.get(kk, 0) - b.get(kk, 0)
            if isinstance(d, (int, float)) and d:
                out[f"{k}/{kk}"] = d
    return out


def profile_once(fn, label, show_top=0):
    """Profile exactly one call, after the call has already been made once. Reports dynamo
    compile activity that happened DURING the profiled window -- the contamination check."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    before = counters_snapshot()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    delta = counters_delta(before, counters_snapshot())
    ka = p.key_averages()
    aten = [e for e in ka if e.key.startswith("aten::")]
    ev = sum(e.count for e in aten)
    kern = sum(e.count for e in ka
               if not e.key.startswith("aten::") and (getattr(e, "self_device_time_total", 0) or 0) > 0)
    dev_us = sum(max(0.0, getattr(e, "self_device_time_total", 0) or 0) for e in ka)
    print(f"  {label:22s} aten_events={ev:6d}  kernels={kern:5d}  device={dev_us/1000:7.3f} ms"
          f"   compile_activity_in_window={delta if delta else 'NONE'}")
    if show_top:
        print(f"      top {show_top} aten events by count:")
        for e in sorted(aten, key=lambda e: -e.count)[:show_top]:
            print(f"        {e.count:6d}  {e.key}")
    return ev, kern


def wall(fn, n=40, inner=10):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) / inner)
    return statistics.median(xs) * 1e3, (max(xs) - min(xs)) / statistics.mean(xs)


def numerics(ref, got):
    if not (torch.is_tensor(ref) and torch.is_tensor(got)):
        return "not-a-tensor"
    d = float((ref.float() - got.float()).abs().max())
    if ref.dtype == torch.bfloat16:
        nd = int((ref.contiguous().view(torch.int16) != got.contiguous().view(torch.int16)).sum())
        return f"max|d|={d:.3e} differing={nd}/{ref.numel()} -> {'BITEXACT' if nd == 0 else 'NUMERIC'}"
    return f"max|d|={d:.3e} -> {'BITEXACT' if d == 0.0 else 'NUMERIC'}"


import torch._dynamo as dynamo

print("\n" + "=" * 100)
print("A. EAGER REFERENCE")
print("=" * 100)
with torch.no_grad():
    ref_blk = one_block(*ar, **kw)
    ref_stk = stack(*ar, **kw)
b_ev, b_k = profile_once(lambda: one_block(*ar, **kw), "eager 1 block")
s_ev, s_k = profile_once(lambda: stack(*ar, **kw), "eager 30 blocks")
b_w, b_s = wall(lambda: one_block(*ar, **kw))
s_w, s_s = wall(lambda: stack(*ar, **kw))
print(f"  eager 1 block   wall={b_w:.3f} ms (spread {b_s:.1%})")
print(f"  eager 30 blocks wall={s_w:.3f} ms (spread {s_s:.1%})   per-block {s_w/30:.3f} ms")
print(f"  x10 forwards/cycle: stack accounts for {s_w*10:.1f} ms of the 351 ms cycle")

print("\n" + "=" * 100)
print("B. DIAGNOSTIC -- where did 9,960 aten events come from?")
print("=" * 100)
dynamo.reset()
comp1 = torch.compile(one_block, mode="default", dynamic=False)
n_warm = 8
for _ in range(n_warm):
    comp1(*ar, **kw)
torch.cuda.synchronize()
print(f"  after {n_warm} warmup calls, dynamo stats: {counters_snapshot().get('stats', {})}")
print("  profiling the compiled block TWICE; if pass 1 >> pass 2 the first window captured compilation")
c1_ev, c1_k = profile_once(lambda: comp1(*ar, **kw), "compiled pass 1", show_top=8)
c2_ev, c2_k = profile_once(lambda: comp1(*ar, **kw), "compiled pass 2", show_top=8)

print("\n" + "=" * 100
      )
print("C. THE UNTESTED QUESTION -- compile the whole 30-block stack as one region")
print("=" * 100)
dynamo.reset()
comp_stack = torch.compile(stack, mode="default", dynamic=False)
t0 = time.perf_counter()
for _ in range(n_warm):
    comp_stack(*ar, **kw)
torch.cuda.synchronize()
print(f"  compile+warmup took {time.perf_counter()-t0:.1f} s; dynamo stats: {counters_snapshot().get('stats', {})}")
cs_ev, cs_k = profile_once(lambda: comp_stack(*ar, **kw), "compiled 30 blocks", show_top=8)
cs_w, cs_ss = wall(lambda: comp_stack(*ar, **kw))
c1_w, c1_s = wall(lambda: comp1(*ar, **kw))

print("\n" + "=" * 100)
print("D. RESULT")
print("=" * 100)
print(f"  {'arm':22s} {'aten ev':>9s} {'kernels':>8s} {'wall ms':>9s} {'spread':>7s} {'vs eager':>9s}")
for label, ev, k, w, sp, base in (
    ("eager 1 block", b_ev, b_k, b_w, b_s, b_w),
    ("compiled 1 block", c2_ev, c2_k, c1_w, c1_s, b_w),
    ("eager 30 blocks", s_ev, s_k, s_w, s_s, s_w),
    ("compiled 30 blocks", cs_ev, cs_k, cs_w, cs_ss, s_w),
):
    print(f"  {label:22s} {ev:9d} {k:8d} {w:9.3f} {sp:6.1%} {base/max(w,1e-9):8.2f}x")

print(f"\n  host ops removed per cycle by compiling the stack:")
print(f"    aten events  {s_ev*10:6d} -> {cs_ev*10:6d}   ({(s_ev-cs_ev)*10:+d} per cycle, "
      f"{1-cs_ev/max(s_ev,1):.0%} of the block region)")
print(f"    predicted at 3.2 us/op: {(s_ev-cs_ev)*10*3.2/1000:+.1f} ms of a 351 ms cycle")
print(f"    MEASURED wall/cycle for the block region: {s_w*10:.1f} -> {cs_w*10:.1f} ms "
      f"({(s_w-cs_w)*10:+.1f} ms)")

with torch.no_grad():
    g_blk = comp1(*ar, **kw)
    g_stk = comp_stack(*ar, **kw)
print(f"\n  numerics compiled 1 block : {numerics(ref_blk, g_blk)}")
print(f"  numerics compiled 30 blocks: {numerics(ref_stk, g_stk)}")

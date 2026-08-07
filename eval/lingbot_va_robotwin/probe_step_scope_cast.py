#!/usr/bin/env python3
"""BITEXACT gate for the timestep-cast hoist, plus proof that the casts actually stopped happening.

TWO THINGS MUST BOTH HOLD, and only one of them is about speed:

  1. max|delta action| = 0 over paired seeded cycles. Hoisting a cast to a scope where the value is
     invariant cannot change a bit -- `.float()` on an fp32 tensor returns `self`. A nonzero delta
     therefore does not mean "slightly lossy", it means a LEGALITY CONDITION IS FALSE (most likely
     condition 4: a consumer mutating the shared tensor), and the pass must be withdrawn rather than
     re-tiered as NUMERIC.
  2. The casts are actually gone. A pass that is bit-exact because it did nothing is the easy way to
     pass check 1, so the `_to_copy` count at model.py:524 is measured before and after.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29994 probe_step_scope_cast.py
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import traceback
from pathlib import Path

IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from instinctwm.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


class CountCasts(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = collections.Counter()
        self.on = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        nm = str(func).split(".")[-2] if "." in str(func) else str(func)
        if self.on and nm == "_to_copy":
            for f in reversed(traceback.extract_stack()):
                if "/torch/" in f.filename or "probe_step_scope" in f.filename \
                        or "_python_dispatch" in f.filename:
                    continue
                self.n[f"{Path(f.filename).name}:{f.lineno}"] += 1
                break
        return func(*args, **kwargs)


def build(S, cfg, hoist: bool):
    server = S.VA_Server(cfg)
    from instinctwm.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    for _ in install_conditioning_prefill(S, type(server)):
        pass
    for _ in install_debug_dump_elision(S):
        pass
    from instinctwm.backends.conv.apply import install_conv_layout
    for _ in install_conv_layout(server):
        pass
    p = None
    if hoist:
        from instinctwm.passes.lingbot.step_scope_cast import StepScopeCastHoist
        p = StepScopeCastHoist()
        names = p.install(S, type(server))
        print(f"  installed {names}", flush=True)
    return server, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=6, help="paired seeded cycles for the delta gate")
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_stepcast"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    if not ctx:
        raise SystemExit("no contexts; run collect_contexts.sh")
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = {"obs": [{full: z[s] for s, full in short.items()}], "state": z["state"]}
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)

    def run(server, n, counter=None, seed=0):
        """Deterministic replay: same prompt, same observations, same keyframe stream."""
        rng = np.random.default_rng(seed)
        acts = []
        server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        for i in range(n):
            torch.manual_seed(1234 + i)
            act = server.infer(dict(obs=obs["obs"], prompt=prompt,
                                    save_visualization=False))["action"]
            acts.append(np.asarray(act, dtype=np.float64).copy())
            kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
                  for _ in range(4 if i == 0 else 8)]
            server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                              save_visualization=False, state=act))
        return acts

    print("=== baseline server (no hoist) ===", flush=True)
    base_srv, _ = build(S, cfg, hoist=False)
    cnt_base = CountCasts()
    with cnt_base:
        cnt_base.on = True
        base_acts = run(base_srv, a.cycles)
        cnt_base.on = False
    del base_srv
    torch.cuda.synchronize()

    print("\n=== hoisted server ===", flush=True)
    hoist_srv, hoist_pass = build(S, cfg, hoist=True)
    cnt_h = CountCasts()
    with cnt_h:
        cnt_h.on = True
        hoist_acts = run(hoist_srv, a.cycles)
        cnt_h.on = False

    print(f"\n=== 1. BITEXACT gate: max|delta action| over {a.cycles} paired seeded cycles ===")
    worst = 0.0
    for i, (b, h) in enumerate(zip(base_acts, hoist_acts)):
        if b.shape != h.shape:
            check(False, f"cycle {i}: action shapes match", f"{b.shape} vs {h.shape}")
            continue
        worst = max(worst, float(np.abs(b - h).max()))
    check(worst == 0.0, f"max|delta action| = 0 over {a.cycles} cycles",
          f"max|delta| = {worst:.3e}")
    if worst != 0.0:
        print("       A nonzero delta here is NOT a small numerical difference to be re-tiered. A")
        print("       no-op cast cannot perturb a bit, so a legality condition is false -- most")
        print("       likely condition 4, a consumer mutating the shared tensor. Withdraw the pass.")

    print("\n=== 2. did the casts actually stop? ===")
    b524 = sum(v for k, v in cnt_base.n.items() if k.endswith(":524"))
    h524 = sum(v for k, v in cnt_h.n.items() if k.endswith(":524"))
    print(f"  model.py:524 _to_copy calls over {a.cycles} cycles: {b524} -> {h524}")
    check(h524 < b524 * 0.2, "the hoist removed >=80% of the casts at that callsite",
          f"{b524 - h524} of {b524} removed"
          if b524 else "baseline recorded none -- gate cannot evaluate")
    if hoist_pass is not None:
        st = hoist_pass.stats()
        print(f"  pass stats: {st['casts_avoided']} casts avoided over {st['forwards']} forwards, "
              f"{st['already_fp32']} calls already fp32")
        check(st["casts_avoided"] > 0, "the pass reports work avoided")

    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: step-scope cast hoist is bit-exact and the casts are gone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

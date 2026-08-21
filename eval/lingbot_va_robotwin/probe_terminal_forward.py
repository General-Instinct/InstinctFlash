#!/usr/bin/env python3
"""Is the terminal action forward dead? A liveness test, not an implementation.

THE CLAIM. Both denoise loops pad a terminal timestep t=0 (wan_va_server.py:473, :478) and run one more
transformer forward at it, with update_cache=1. On that last iteration the OUTPUT is provably discarded --
`if not last_step` guards every use of it (:548 action, :508 video) -- so the forward's only possible
effect is its write to the shared ring KV pool.

For the ACTION loop that write may also be dead: `_compute_kv_cache` calls `clear_pred_cache` as its FIRST
statement (:574), and between the terminal action forward and that call nothing runs a transformer.
If so the entire forward is dead: 1 of 10 forwards per cycle, ~1,860 launches, ~10,500 aten events.

For the VIDEO loop the same write is NOT dead -- the action loop runs afterwards and reads the pool -- so
this probe tests the action terminal only, and separately tests the video terminal to confirm the
asymmetry is real rather than assumed.

WHY THIS IS WORTH A PROBE. It is not a Layer 5 finding: it removes a whole host-bound segment rather than
selecting a better backend. But the transformer is host-bound at 0.145 ms of cycle per ms of device time,
and deleting a forward removes BOTH its device time and its ~10,500 host dispatches -- which is why its
value is ~30 ms of cycle where a device-only change of the same size would be worth ~2 ms.

THE TEST. Skip the forward entirely and return None. Returning None rather than zeros is deliberate: if
anything downstream does touch the value, this raises instead of silently substituting a wrong one.
Then compare actions bit-exactly over seeded cycles SPANNING ring saturation, because the pool's
behaviour changes at the wrap and a gate that ran only before it would test nothing.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY -u \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29976 probe_terminal_forward.py
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

IFL_ROOT = os.environ.get("IFL_ROOT") or str(Path(__file__).resolve().parents[2])
if IFL_ROOT not in sys.path:
    sys.path.insert(0, IFL_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from instinctflash.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)
    return cond


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=40)
    ap.add_argument("--cycles", type=int, default=45, help="spans ring saturation at ~36")
    ap.add_argument("--arm-cycles", type=int, default=20)
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_terminal"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

    print("building server at 2V/4A, shipped stack ...", flush=True)
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
    obs = [{full: z[s] for s, full in short.items()}]
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)

    tf = server.transformer
    orig_fwd = tf.forward
    mode = {"skip": None, "n_calls": 0, "n_skipped": 0}

    def forward(*args, **kwargs):
        mode["n_calls"] += 1
        uc = kwargs.get("update_cache", 0)
        am = kwargs.get("action_mode", False)
        terminal = (uc == 1)
        want = mode["skip"]
        if want == "action" and terminal and am:
            mode["n_skipped"] += 1
            return None                      # deliberately None: raises if anything reads it
        if want == "video" and terminal and not am:
            mode["n_skipped"] += 1
            return None
        return orig_fwd(*args, **kwargs)

    tf.forward = forward

    def run(n, seed=0):
        rng = np.random.default_rng(seed)
        acts = []
        for i in range(n):
            torch.manual_seed(1234 + i)
            if i == 0:
                server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
            act = server.infer(dict(obs=obs, prompt=prompt, save_visualization=False))["action"]
            acts.append(np.asarray(act, dtype=np.float64).copy())
            kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
                  for _ in range(4 if i == 0 else 8)]
            server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                              save_visualization=False, state=act))
        return acts

    print(f"warming {a.warm} cycles ...", flush=True)
    mode["skip"] = None
    run(a.warm, seed=99)

    # ---- how many forwards, and how many are terminal? -------------------------------------------
    mode["n_calls"], mode["n_skipped"] = 0, 0
    run(1, seed=5)
    per_cycle = mode["n_calls"]
    print(f"\n{'=' * 100}\n0. STRUCTURE\n{'=' * 100}")
    print(f"  transformer forwards per cycle: {per_cycle}")

    # ---- 1. the action terminal forward ----------------------------------------------------------
    print(f"\n{'=' * 100}\n1. ACTION TERMINAL FORWARD -- is it dead?  ({a.cycles} seeded cycles, "
          f"spanning saturation at ~36)\n{'=' * 100}")
    mode["skip"] = None
    base = run(a.cycles, seed=3)
    mode["skip"], mode["n_skipped"] = "action", 0
    try:
        treat = run(a.cycles, seed=3)
        skipped = mode["n_skipped"]
        worst = max(float(np.abs(x - y).max()) for x, y in zip(base, treat))
        per = [float(np.abs(x - y).max()) for x, y in zip(base, treat)]
        print(f"  forwards skipped: {skipped} over {a.cycles} cycles "
              f"({skipped / a.cycles:.2f}/cycle of {per_cycle})")
        print(f"  per-cycle max|delta|, first 12: {['%.3g' % v for v in per[:12]]}")
        print(f"  last 6 (post-saturation):       {['%.3g' % v for v in per[-6:]]}")
        check(worst == 0.0, "action terminal forward is DEAD: max|delta action| = 0",
              f"worst {worst:.3e}")
    except Exception as e:
        print(f"  RAISED: {type(e).__name__}: {e}")
        print(f"  => the returned value IS consumed somewhere; the forward is not dead.")
        FAILED.append("action terminal raises")
    mode["skip"] = None

    # ---- 2. the video terminal, to confirm the asymmetry is real ---------------------------------
    print(f"\n{'=' * 100}\n2. VIDEO TERMINAL FORWARD -- expected LIVE (the action loop reads its KV "
          f"writes)\n{'=' * 100}")
    mode["skip"], mode["n_skipped"] = "video", 0
    try:
        treat_v = run(a.cycles, seed=3)
        worst_v = max(float(np.abs(x - y).max()) for x, y in zip(base, treat_v))
        print(f"  forwards skipped: {mode['n_skipped']}")
        print(f"  max|delta action| = {worst_v:.4f}")
        if worst_v > 0:
            print(f"  => LIVE, as predicted. The asymmetry is real: the video terminal's KV writes are "
                  f"read by\n     the action loop; the action terminal's are erased by clear_pred_cache "
                  f"before any read.")
        else:
            print(f"  => ALSO DEAD. Then the finding is twice as large and the analysis above is "
                  f"incomplete.")
    except Exception as e:
        print(f"  RAISED: {type(e).__name__}: {e}  => video terminal output is consumed.")
    mode["skip"] = None

    # ---- 3. what is it worth? --------------------------------------------------------------------
    if not FAILED:
        print(f"\n{'=' * 100}\n3. CYCLE VALUE (ABBA: base, treat, treat, base)\n{'=' * 100}")
        rng = np.random.default_rng(7)
        kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
              for _ in range(8)]

        def cyc():
            act = server.infer(dict(obs=obs, prompt=prompt, save_visualization=False))["action"]
            server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                              save_visualization=False, state=act))

        def timed(n):
            xs = []
            for _ in range(n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                cyc()
                torch.cuda.synchronize()
                xs.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(xs)

        for _ in range(10):
            cyc()
        arms = {"base": [], "treat": []}
        for name in ("base", "treat", "treat", "base"):
            mode["skip"] = "action" if name == "treat" else None
            m = timed(a.arm_cycles)
            arms[name].append(m)
            print(f"  arm {name:5}  {m:7.1f} ms")
        mode["skip"] = None
        b = sum(arms["base"]) / 2
        t = sum(arms["treat"]) / 2
        drift = abs(arms["base"][0] - arms["base"][1]) / b
        print(f"\n  base  {arms['base'][0]:.1f} / {arms['base'][1]:.1f} -> {b:.1f} ms  drift {drift:.1%}")
        print(f"  treat {arms['treat'][0]:.1f} / {arms['treat'][1]:.1f} -> {t:.1f} ms")
        if drift > 0.05:
            print(f"  LATENCY NOT EVALUATED: drift {drift:.1%} > 5%")
        else:
            print(f"  {b - t:+.1f} ms/cycle   {b / t:.3f}x   "
                  f"({(b - t) / b:.1%} of the cycle, from 1 of {per_cycle} forwards)")

    print("\n" + "=" * 100)
    if FAILED:
        print("CANDIDATE REFUTED -- the terminal action forward is NOT dead. A non-zero exit here means")
        print("the liveness test did its job, not that the probe broke.")
        return 1
    print("The action terminal forward is dead on this message pattern. NOT a Layer 5 finding -- it is a")
    print("redundancy elision like P001/P002, and shipping it needs a fail-closed gate on the pattern,")
    print("because generate() (wan_va_server.py:648) drives a different one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

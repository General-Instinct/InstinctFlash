#!/usr/bin/env python3
"""Price `action_terminal_forward_elision` directly: ABBA cycle timing + device-busy delta.

The candidate: the 5th action denoise forward (t=0, update_cache=1) writes 32 `is_pred` action
K/V slots per layer that `clear_pred_cache` drops and the update_cache=2 action forward physically
overwrites before any read window contains them. If that is right the whole forward is dead and
can be replaced by its ring bookkeeping.

This probe does NOT re-derive the liveness proof. It measures the two numbers the proposal claims:

  device_ms  : device-busy interval union, CUDA-activities-only profiler, ON vs OFF.
               (LAYER6_GAPS established device busy is undistorted by the instrument: 190.9-191.8
               across three instrument levels.)
  cycle_ms   : unprofiled wall, ABBA counterbalanced, with the same k=0 gate probe_slope_clean uses.

The elision is applied at the transformer boundary -- `action_mode=True and update_cache==1` is
uniquely the terminal action forward (the other four action forwards are update_cache=0 and the KV
refresh is update_cache=2). `_prepare_latent_input` and the scheduler bookkeeping still run in the
ON arm, so this UNDERSTATES the full elision by ~0.4 ms.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY -u \
        -m torch.distributed.run --nproc_per_node 1 --master_port 29971 probe_action_terminal.py
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from instinctwm.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

REFERENCE_CYCLE_MS = 330.7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70)
    ap.add_argument("--arm-cycles", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3, help="ABBA blocks")
    ap.add_argument("--device-only", action="store_true")
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_actterm"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

    print("building server at 2V/4A, shipped stack ...", flush=True)
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

    # ---- the elision -----------------------------------------------------------------------
    MODE = {"on": False, "hits": 0}
    tf = server.transformer
    _orig_tf_forward = tf.forward
    CACHE = server.cache_name

    def ring_only_terminal_action():
        """Everything the terminal action forward leaves behind, minus the compute.

        Per layer: the metadata writes `forward` does inline, then the real `_commit`. The 32 K/V
        slots themselves are NOT written -- that is the claim under test.
        """
        for blk in tf.blocks:
            at = blk.attn1
            c = at.attn_caches[CACHE]
            r = c["_ring"]
            head = (r["start"] + r["count"]) % r["total"]
            sl = slice(head, head + 32)
            c["mask"][sl] = True
            c["id"][sl] = r["next_id"]
            c["is_pred"][sl] = True
            r["next_id"] += 1
            at._iwm_commit(CACHE, 32, 1)

    def tf_forward(input_dict, update_cache=0, cache_name="pos", action_mode=False,
                   train_mode=False):
        if MODE["on"] and action_mode and update_cache == 1:
            MODE["hits"] += 1
            ring_only_terminal_action()
            return None
        return _orig_tf_forward(input_dict, update_cache, cache_name, action_mode, train_mode)

    tf.forward = tf_forward

    # ---- workload (identical to probe_slope_clean) ------------------------------------------
    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = [{full: z[s] for s, full in short.items()}]
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)
    rng = np.random.default_rng(0)

    def cycle(keyframes=None, first=False):
        if first:
            server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        act = server.infer(dict(obs=obs, prompt=prompt, save_visualization=False))["action"]
        kf = keyframes if keyframes is not None else [
            {k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
            for _ in range(4 if first else 8)]
        server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=act))
        MODE["amax"] = float(np.abs(np.asarray(act, dtype=np.float64)).max())
        return act

    print(f"warming {a.warm} cycles ...", flush=True)
    cycle(first=True)
    for _ in range(a.warm):
        cycle()
    kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
          for _ in range(8)]

    def timed(n):
        xs = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cycle(keyframes=kf)
            torch.cuda.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(xs), (max(xs) - min(xs)) / statistics.mean(xs)

    # ---- gate --------------------------------------------------------------------------------
    MODE["on"] = False
    if a.device_only:
        base0, spread0, err = REFERENCE_CYCLE_MS, 0.0, 0.0
    else:
        base0, spread0 = timed(a.arm_cycles)
    err = abs(base0 - REFERENCE_CYCLE_MS) / REFERENCE_CYCLE_MS
    print(f"\nGATE: k=0 baseline {base0:.1f} ms (spread {spread0:.1%}) vs {REFERENCE_CYCLE_MS} ms"
          f" -> {err:+.1%}")
    if err > 0.03 and not a.device_only:
        print("NOT EVALUATED: harness baseline off the served cycle.")
        return 2

    # confirm the hook fires exactly once per cycle
    MODE["on"], MODE["hits"] = True, 0
    cycle(keyframes=kf)
    print(f"elision sites hit per cycle: {MODE['hits']}  (expect 1)")
    MODE["on"] = False

    # ---- ABBA --------------------------------------------------------------------------------
    offs, ons = [], []
    for rep in range(0 if a.device_only else a.reps):
        order = [False, True] if rep % 2 == 0 else [True, False]
        for m in order:
            MODE["on"] = m
            v, sp = timed(a.arm_cycles)
            (ons if m else offs).append(v)
            print(f"  rep {rep} {'ON ' if m else 'OFF'}  {v:8.2f} ms  (spread {sp:.1%})"
                  f"  max|action| {MODE.get('amax', float('nan')):.6f}")
    MODE["on"] = False

    if offs:
        moff, mon = statistics.median(offs), statistics.median(ons)
        drift = (max(offs) - min(offs)) / statistics.mean(offs)
        print(f"\n  OFF arms {['%.1f' % v for v in offs]}   drift {drift:.1%}")
        print(f"  ON  arms {['%.1f' % v for v in ons]}")
        print(f"\n  cycle OFF {moff:.2f} ms   ON {mon:.2f} ms   delta {moff - mon:+.2f} ms"
              f"   speedup {moff / mon:.4f}x")
        if drift > 0.03:
            print("  WARNING: control drift > 3%; treat the delta as not resolvable.")

    # ---- device busy, CUDA activities only ---------------------------------------------------
    import json

    from torch.profiler import ProfilerActivity, profile

    DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "Kernel"}

    def union_len(iv):
        if not iv:
            return 0.0
        xs = sorted(iv)
        m = [list(xs[0])]
        for s, e in xs[1:]:
            if s <= m[-1][1]:
                m[-1][1] = max(m[-1][1], e)
            else:
                m.append([s, e])
        return sum(e - s for s, e in m)

    def device_busy(mode, ncyc=8, tag="x"):
        MODE["on"] = mode
        cycle(keyframes=kf)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as p:
            for _ in range(ncyc):
                cycle(keyframes=kf)
            torch.cuda.synchronize()
        path = f"/tmp/iwm_actterm/trace_{tag}.json"
        p.export_chrome_trace(path)
        with open(path) as f:
            tr = json.load(f)
        evs = tr["traceEvents"] if isinstance(tr, dict) else tr
        dev = [e for e in evs if e.get("cat") in DEVICE_CATS and e.get("ph") == "X"
               and e.get("dur") is not None]
        busy = union_len([(e["ts"], e["ts"] + e["dur"]) for e in dev])
        return busy / 1e3 / ncyc, len(dev) / ncyc

    for rep, order in enumerate(([False, True], [True, False], [False, True])):
        res = {}
        for m in order:
            res[m] = device_busy(m, tag=f"{int(m)}_{rep}")
        b_off, n_off = res[False]
        b_on, n_on = res[True]
        print(f"\n  [rep {rep}] device busy OFF {b_off:7.2f} ms  events {n_off:8.1f}")
        print(f"  [rep {rep}] device busy ON  {b_on:7.2f} ms  events {n_on:8.1f}")
        print(f"  [rep {rep}] device removed  {b_off - b_on:7.2f} ms  events removed {n_off - n_on:8.1f}")
    MODE["on"] = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

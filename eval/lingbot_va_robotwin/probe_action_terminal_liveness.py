#!/usr/bin/env python3
"""Is the terminal ACTION forward's KV write dead? NaN-poison it and see if anything reads it.

Self-controlled: NaN propagates through softmax attention unconditionally, so "a poisoned slot was
read" needs no baseline run -- the action output becomes NaN. Three arms:

  none    control. actions finite.
  action  poison the 32 slots the terminal ACTION forward (action_mode=True, update_cache=1) just
          wrote, in every layer, immediately after the forward returns. The candidate says these
          are dropped by clear_pred_cache and physically overwritten by the update_cache=2 action
          write before entering any read window => actions must stay finite.
  video   POSITIVE CONTROL. Same treatment for the 240 slots the terminal VIDEO forward wrote.
          Those are read by all five action forwards -- that is the architecture -- so this arm
          MUST go NaN. If it does not, the poison is not landing and the `action` arm proves
          nothing.

REFUTED 2026-08-09, see LAYER5_COMPLETE.md section 4b and probe_terminal_forward.py.
The terminal action forward is NOT dead. It is dead for the first ~38 cycles and LIVE thereafter:
max|delta action| = 0 pre-saturation, then 0.0297 / 0.0234 / 0.266 / 0.406 / 0.266 / 0.102 once the
ring wraps. clear_pred_cache rolls the COUNT back (ring_kv.py:132) but the write has already EVICTED
the oldest slot and advanced start (ring_kv.py:258-259), and that is not rolled back. Any run of this
probe that stops before cycle ~38 will report the candidate alive and be wrong.
"""
from __future__ import annotations

import argparse
import os
import sys
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=45)
    ap.add_argument("--cycles", type=int, default=6)
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_actlive"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

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

    tf = server.transformer
    CACHE = server.cache_name
    _orig = tf.forward
    MODE = {"m": "none", "poisoned": 0}

    def tf_forward(input_dict, update_cache=0, cache_name="pos", action_mode=False,
                   train_mode=False):
        want = ((MODE["m"] == "action" and action_mode and update_cache == 1)
                or (MODE["m"] == "video" and (not action_mode) and update_cache == 1))
        heads = []
        if want:
            for blk in tf.blocks:
                r = blk.attn1.attn_caches[CACHE]["_ring"]
                heads.append((r["start"] + r["count"]) % r["total"])
        out = _orig(input_dict, update_cache, cache_name, action_mode, train_mode)
        if want:
            n = 32 if action_mode else 240
            for blk, h in zip(tf.blocks, heads):
                c = blk.attn1.attn_caches[CACHE]
                c["k"][:, h:h + n] = float("nan")
                c["v"][:, h:h + n] = float("nan")
            MODE["poisoned"] += 1
        return out

    tf.forward = tf_forward

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = [{full: z[s] for s, full in short.items()}]
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)
    rng = np.random.default_rng(0)

    def cycle(first=False):
        if first:
            server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        act = server.infer(dict(obs=obs, prompt=prompt, save_visualization=False))["action"]
        kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
              for _ in range(4 if first else 8)]
        server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=act))
        return np.asarray(act, dtype=np.float64)

    print(f"warming {a.warm} cycles (ring saturates at 36) ...", flush=True)
    cycle(first=True)
    for _ in range(a.warm):
        cycle()

    for mode in ("none", "action", "video"):
        MODE["m"], MODE["poisoned"] = mode, 0
        bad = 0
        mags = []
        for c in range(a.cycles):
            act = cycle()
            if not np.isfinite(act).all():
                bad += 1
            mags.append(float(np.nanmax(np.abs(act))) if np.isfinite(act).any() else float("nan"))
        MODE["m"] = "none"
        for _ in range(3):          # flush the poison out of the pool before the next arm
            cycle()
        print(f"  arm {mode:7s}  poisoned_forwards={MODE['poisoned']:3d}  "
              f"non-finite action chunks: {bad}/{a.cycles}   max|action| {mags}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

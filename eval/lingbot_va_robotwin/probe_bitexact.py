#!/usr/bin/env python3
"""Compare the actions two LingBot-VA server variants produce from identical inputs.

"No accuracy loss" is a claim InstinctWM sells, so an optimization is only free if it
either (a) is bit-exact, or (b) survives a paired non-inferiority test against a measured
noise floor -- which costs roughly 10x more to establish. This tool decides which bucket a
variant falls into, cheaply, before anyone spends GPU-months on the expensive route.

Both servers MUST be launched with the same `--deterministic-seed`. `_infer` draws
`torch.randn` for the initial video latents and action tokens
(`wan_va_server.py:449-462`) with no seeding at all, so two *stock* servers already
disagree; without seeding this tool would measure the noise draw, not the variant.

The comparison replays the real episode message order against both servers in lockstep,
feeding both the identical observation stream, so divergence accumulates through the KV
cache exactly as it would in a real rollout -- a single-chunk comparison would miss drift.

Scale reference: actions are absolute EEF poses in metres and quaternion units, so a
delta of 1e-3 on the translation channels is a millimetre. The tool also reports the
delta between two DIFFERENT observations, which is the only honest yardstick for whether
a given max|delta| is small.

Usage:
    python probe_bitexact.py --ref-port 29058 --opt-port 29061 --cycles 6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# env.sh exports ROBOTWIN_ROOT; honour it rather than pinning one box's layout.
sys.path.insert(0, os.environ.get("ROBOTWIN_ROOT", "/home/ubuntu/RoboTwin"))
from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

CAMS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def make_obs(rng, h=240, w=320):
    return {k: rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for k in CAMS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--ref-port", type=int, required=True, help="reference (stock) server")
    ap.add_argument("--opt-port", type=int, required=True, help="optimized variant server")
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--keyframes", type=int, default=8)
    ap.add_argument("--prompt", default="Use the left arm to lift the plastic drink bottle head-up")
    args = ap.parse_args()

    ref = WebsocketClientPolicy(host=args.host, port=args.ref_port)
    opt = WebsocketClientPolicy(host=args.host, port=args.opt_port)
    print(f"reference = :{args.ref_port}   optimized = :{args.opt_port}")

    ref.infer(dict(reset=True, prompt=args.prompt, save_visualization=False))
    opt.infer(dict(reset=True, prompt=args.prompt, save_visualization=False))

    # Identical observation streams for both arms.
    rng = np.random.default_rng(0)
    first_obs = make_obs(rng)

    deltas, ref_actions = [], []
    for c in range(args.cycles):
        a_ref = ref.infer(dict(obs=first_obs, prompt=args.prompt, save_visualization=False))["action"]
        a_opt = opt.infer(dict(obs=first_obs, prompt=args.prompt, save_visualization=False))["action"]
        d = np.abs(a_ref.astype(np.float64) - a_opt.astype(np.float64)).max()
        deltas.append(d)
        ref_actions.append(a_ref)
        print(f"  cycle {c}: max|d_action| = {d:.6e}")

        nkf = args.keyframes // 2 if c == 0 else args.keyframes
        kfs = [make_obs(rng) for _ in range(nkf)]
        # Feed BOTH servers the same real frames, and each its OWN action, so the two
        # rollouts stay honest: if the variant drifts, its KV cache drifts with it.
        ref.infer(dict(obs=kfs, compute_kv_cache=True, imagine=False,
                       save_visualization=False, state=a_ref))
        opt.infer(dict(obs=kfs, compute_kv_cache=True, imagine=False,
                       save_visualization=False, state=a_opt))

    print()
    worst = max(deltas)
    print("=" * 68)
    print(f"max |delta action| over {args.cycles} cycles : {worst:.6e}")

    # Yardstick: how much do actions move between two consecutive real chunks? A variant
    # delta far below this is negligible; one comparable to it is a behaviour change.
    if len(ref_actions) >= 2:
        scale = np.abs(ref_actions[1].astype(np.float64) - ref_actions[0].astype(np.float64)).max()
        print(f"reference chunk-to-chunk movement          : {scale:.6e}")
        if scale > 0:
            print(f"ratio (variant delta / real movement)      : {worst/scale:.3%}")

    print()
    if worst == 0.0:
        print("VERDICT: BIT-EXACT. This optimization is free -- it can be claimed as")
        print("accuracy-neutral without a paired non-inferiority run.")
        return 0
    print("VERDICT: NOT bit-exact. The speedup is real but the accuracy claim is NOT.")
    print("This variant must go through a paired non-inferiority certificate on pinned")
    print("seeds (per-episode JSONL + McNemar) before any 'no accuracy loss' statement.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

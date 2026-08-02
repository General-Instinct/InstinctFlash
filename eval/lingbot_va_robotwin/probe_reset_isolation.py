#!/usr/bin/env python3
"""Does episode N leak into episode N+1?

E1 makes `_reset` clear logical KV state in place instead of reallocating the pools, so captured
graphs survive. That is only safe if a reset really does erase the previous episode. The k/v pools
are deliberately NOT zeroed -- 30 layers x 9792 slots is ~3.5 GB of pointless writes, and the ring
only reads [start, start+count), which is empty after a reset. That is a reachability ARGUMENT, and
arguments are what this file exists to replace.

Method. Two identically configured servers:

    A : episode with prompt PRIOR, reset, then episode with prompt TARGET
    B :                                    episode with prompt TARGET   (fresh, never saw PRIOR)

If reset isolation holds, A's TARGET actions are bitwise identical to B's. If any residue survives
-- stale k/v that becomes reachable, a ring counter that did not fully rewind, a mask bit -- A and
B diverge, and the difference is exactly the leak.

Both servers must run with --deterministic-seed, or the noise draws differ and the comparison is
meaningless.

Usage:
    python probe_reset_isolation.py --a-port 29058 --b-port 29059 --cycles 4
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "/home/ubuntu/RoboTwin")
from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

CAMS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

PRIOR = "Use the right arm to place the block into the container"
TARGET = "Use the left arm to lift the plastic drink bottle head-up"


def make_obs(rng, h=240, w=320):
    return {k: rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for k in CAMS}


def run_episode(client, prompt, cycles, keyframes, obs_seed):
    """One episode from a fresh reset. Returns the action chunk of every cycle."""
    client.infer(dict(reset=True, prompt=prompt, save_visualization=False))
    rng = np.random.default_rng(obs_seed)
    first_obs = make_obs(rng)
    actions = []
    for c in range(cycles):
        a = client.infer(dict(obs=first_obs, prompt=prompt,
                              save_visualization=False))["action"]
        actions.append(a)
        nkf = keyframes // 2 if c == 0 else keyframes
        kfs = [make_obs(rng) for _ in range(nkf)]
        client.infer(dict(obs=kfs, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=a))
    return actions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--a-port", type=int, required=True, help="server that runs PRIOR then TARGET")
    ap.add_argument("--b-port", type=int, required=True, help="server that runs TARGET only")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--keyframes", type=int, default=8)
    ap.add_argument("--prior-cycles", type=int, default=3)
    args = ap.parse_args()

    A = WebsocketClientPolicy(host=args.host, port=args.a_port)
    B = WebsocketClientPolicy(host=args.host, port=args.b_port)
    print(f"A = :{args.a_port} (PRIOR then TARGET)   B = :{args.b_port} (TARGET only)")

    print(f"\n[A] episode 1, prompt={PRIOR!r}, {args.prior_cycles} cycles "
          f"-- fills the KV pool with residue")
    run_episode(A, PRIOR, args.prior_cycles, args.keyframes, obs_seed=99)

    print(f"[A] episode 2, prompt={TARGET!r}  (after reset)")
    a_actions = run_episode(A, TARGET, args.cycles, args.keyframes, obs_seed=0)

    print(f"[B] episode 1, prompt={TARGET!r}  (fresh server)")
    b_actions = run_episode(B, TARGET, args.cycles, args.keyframes, obs_seed=0)

    print()
    worst = 0.0
    for c, (a, b) in enumerate(zip(a_actions, b_actions)):
        d = np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
        worst = max(worst, d)
        print(f"  cycle {c}: max|d_action| = {d:.6e}")

    scale = 0.0
    if len(b_actions) >= 2:
        scale = np.abs(b_actions[1].astype(np.float64)
                       - b_actions[0].astype(np.float64)).max()

    print("\n" + "=" * 68)
    print(f"max |delta| second-episode vs fresh : {worst:.6e}")
    if scale:
        print(f"chunk-to-chunk movement             : {scale:.6e}")
        print(f"ratio                               : {worst/scale:.3%}")
    print()
    if worst == 0.0:
        print("VERDICT: RESET IS ISOLATED. A prior episode leaves no observable trace, so\n"
              "reusing the pools in place is safe and captured graphs may cross a reset.")
        return 0
    print("VERDICT: LEAK. The second episode differs from a fresh one, so `_reset` is not\n"
          "fully clearing state. Graphs MUST NOT be preserved across resets until this is 0.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
